"""Engineering constraints, exact leave-one-out coverage and multi-map plans."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsu_coverage_study import make_plan, parser, requirements_for, run
from src.core.coverage_requirements import CoverageRequirements
import test_constrained_coverage as coverage_tests


class RSURequirementsTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.], [3., 0., 1.]])
        self.poses = np.array([[0., 2., 3., 90., -20.], [2., 2., 4., 90., -20.]])

    def spec(self, **values):
        return dict(schema='constrained_coverage_v2', budget_k=2, **values)

    def test_failure_metric_needs_identity_not_density_alone(self):
        evaluator = CoverageRequirements(self.spec(constraints=dict(min_single_failure_coverage_pct=40.))).bind(self.targets)
        density = np.array([1, 1, 2, 0])
        a = np.array([[1, 1, 1, 0], [0, 0, 1, 0]], dtype=bool)
        b = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=bool)
        first = evaluator.evaluate(self.poses, density, sensor_masks=a)
        second = evaluator.evaluate(self.poses, density, sensor_masks=b)
        self.assertEqual(first.coverage_pct, second.coverage_pct)
        self.assertEqual(first.overlap_ratio, second.overlap_ratio)
        self.assertEqual(first.robustness['single_failure_coverage_pct'], 25.)
        self.assertEqual(second.robustness['single_failure_coverage_pct'], 50.)
        self.assertFalse(first.feasible)
        self.assertTrue(second.feasible)
        with self.assertRaisesRegex(ValueError, 'per-sensor masks'):
            evaluator.evaluate(self.poses, density)
        with self.assertRaisesRegex(ValueError, 'sum to density'):
            evaluator.evaluate(self.poses, density, sensor_masks=np.ones((2, 4)))

    def test_failure_metrics_match_explicit_removal_including_regions(self):
        spec = self.spec(constraints=dict(min_single_failure_coverage_pct=0.),
            regions=[dict(name='critical', bounds=[0., -1., 1., 1.], min_single_failure_coverage_pct=50.)])
        evaluator = CoverageRequirements(spec).bind(self.targets)
        for flat in itertools.product((0, 1), repeat=8):
            masks = np.array(flat).reshape(2, 4)
            result = evaluator.evaluate(self.poses, masks.sum(axis=0), sensor_masks=masks)
            values = [100*np.count_nonzero(np.delete(masks, i, axis=0).sum(axis=0))/4 for i in range(2)]
            self.assertEqual(result.robustness['coverage_after_each_failure_pct'], values)
            regional = [100*np.count_nonzero(np.delete(masks, i, axis=0)[:, :2].sum(axis=0))/2 for i in range(2)]
            self.assertEqual(result.regions['critical']['single_failure_coverage_pct'], min(regional))

    def test_hub_reachability_requires_joint_capacity_assignment(self):
        hub = dict(name='cabinet', xy=[1., 2.], max_horizontal_distance_m=2., capacity=1)
        spec = self.spec(constraints=dict(service_hubs=[hub]))
        evaluator = CoverageRequirements(spec).bind(self.targets)
        result = evaluator.evaluate(self.poses, np.array([1, 1, 0, 0]))
        self.assertIn('service_connection', result.violations)
        self.assertEqual(result.infrastructure['matched_sensors'], 1)
        spec['constraints']['service_hubs'].append(dict(hub, name='backup', xy=[0., 2.], max_horizontal_distance_m=.1))
        result = CoverageRequirements(spec).bind(self.targets).evaluate(self.poses, np.array([1, 1, 0, 0]))
        self.assertTrue(result.feasible)
        self.assertEqual(len(set(result.infrastructure['assigned_hubs'])), 2)
        self.assertEqual(result.infrastructure['assigned_hubs'], ['backup', 'cabinet'])
        spec['constraints']['service_hubs'][1]['xy'] = [20., 20.]
        self.assertFalse(CoverageRequirements(spec).bind(self.targets).evaluate(
            self.poses, np.array([1, 1, 0, 0])).feasible)

    def test_height_whitelist_and_v1_opt_in_boundary(self):
        spec = self.spec(constraints=dict(mounting_z_range_m=[3., 3.],
            allowed_installation_regions=[[-1., 1., 1., 3.]]))
        result = CoverageRequirements(spec).bind(self.targets).evaluate(self.poses, np.array([1, 1, 0, 0]))
        self.assertIn('mounting_height', result.violations)
        self.assertIn('installation_whitelist', result.violations)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            CoverageRequirements(dict(spec, schema='constrained_coverage_v1'))

    def test_new_fields_fail_closed(self):
        invalid = [dict(mounting_z_range_m=[5., 3.]), dict(mounting_z_range_m=[3., float('nan')]),
            dict(allowed_installation_regions=[]), dict(service_hubs=[]),
            dict(min_single_failure_coverage_pct=101.),
            dict(service_hubs=[dict(name='x', xy=[0., 0.], max_horizontal_distance_m=5., capacity=True)])]
        for constraints in invalid:
            with self.subTest(constraints=constraints), self.assertRaises(ValueError):
                CoverageRequirements(self.spec(constraints=constraints))


class RSUStudyTests(unittest.TestCase):
    def test_plan_has_distinct_geometry_and_real_training_ladder_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'absent'
            args = parser().parse_args(['--output', str(output), '--library', str(output / 'missing')])
            with patch('rsu_coverage_study.subprocess.run', side_effect=AssertionError('No run')), \
                    redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(run(args), 0)
            plan = json.loads(stream.getvalue())
            self.assertFalse(output.exists())
        self.assertEqual(plan['requested_pairs'], 720)
        self.assertEqual(plan['requested_method_runs'], 1440)
        self.assertEqual(len({j['target_id'] for j in plan['jobs']}), 8)
        self.assertEqual({j['steps'] for j in plan['jobs']}, {4096, 20000, 100000})
        self.assertEqual({j['K'] for j in plan['jobs']}, {3, 4})
        self.assertEqual(len({j['id'] for j in plan['jobs']}), 720)
        self.assertTrue(all(j['plan']['job']['budget_k'] == j['K'] for j in plan['jobs']))
        self.assertEqual(plan['split'], 'development')

    def test_profiles_have_geometry_only_regions_and_nested_constraints(self):
        plan = make_plan(parser().parse_args(['--target-ids', '430001', '430004',
            '--seeds', '0', '--step-budgets', '4096', '--k-values', '4']))
        for tid in (430001, 430004):
            jobs = [j for j in plan['jobs'] if j['target_id'] == tid]
            a, b, c = [j['plan']['requirements'] for j in jobs]
            self.assertEqual(a['weights'], b['weights'])
            self.assertEqual(b['weights'], c['weights'])
            self.assertNotIn('service_hubs', a['constraints'])
            self.assertIn('service_hubs', b['constraints'])
            self.assertNotIn('min_single_failure_coverage_pct', b['constraints'])
            self.assertIn('min_single_failure_coverage_pct', c['constraints'])
            self.assertEqual(len(c['regions']), 4 if tid == 430001 else 5)
        for flags in (['--target-ids', '1'], ['--seeds', '0', '0'], ['--step-budgets', '128']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                make_plan(parser().parse_args(flags))

    def test_existing_output_and_report_execute_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                run(parser().parse_args(['--execute', '--output', temp]))
            with self.assertRaises(ValueError):
                run(parser().parse_args(['--report-only', '--execute', '--output', temp]))


class RSUIntegrationTests(unittest.TestCase):
    def setUp(self):
        coverage_tests.ConstrainedRefinementTests.setUp(self)

    def test_failure_aware_masks_are_independently_validated_for_both_methods(self):
        from src.methods.constrained_refinement import refine_constrained
        spec = dict(schema='constrained_coverage_v2', budget_k=2,
            constraints=dict(min_single_failure_coverage_pct=0., mounting_z_range_m=[3., 6.],
                service_hubs=[dict(name='cabinet', xy=[50., 50.], max_horizontal_distance_m=40., capacity=2)]))
        for method in ('local_search', 'mappo'):
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / method
                _, result = refine_constrained(self.initial, self.scenario, self.cfg, spec, method=method,
                    steps=8, max_evaluations=5, time_limit_sec=30., radius_m=2., output=output)
                self.assertTrue(result['validated_direct_raycast'])
                self.assertTrue(result['assessments']['deployment']['robustness'])
                with np.load(output / 'deployment_coverage.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['sensor_masks'].sum(axis=0), data['density'])
                if method == 'mappo':
                    self.assertEqual(result['training_steps'], 8)
                    self.assertEqual(result['ppo_updates'], 2)


if __name__ == '__main__':
    unittest.main()

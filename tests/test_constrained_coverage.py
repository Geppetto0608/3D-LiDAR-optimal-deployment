"""Shared objective, hard constraints and opt-in refinement regression tests."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.candidate import Candidate, CandidateSet
from src.core.coverage_requirements import CoverageRequirements
from src.core.evaluation import evaluate_selection
from src.core.models import PlacementProblem
from src.core.objective import ObjectiveWeights
from prototype_offline import width_scenario
from src.environment import MapGeometry, load_config
from src.methods.constrained_refinement import (InitialInfeasible, RefinementSession,
    _ConstrainedAgent, _evaluate_policy, _joint_step, _poses, refine_constrained)
from src.methods.local_search import _ActionDriver
from src.methods.mappo import _snapshot_agent_states


def specification(**updates):
    return dict(schema='constrained_coverage_v1', budget_k=2, **updates)


class CoverageRequirementsTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]])
        self.poses = np.array([[0., 2., 3., 90., -20.], [2., 2., 4., 90., -20.]])

    def test_common_objective_parity_for_every_small_selection(self):
        weights = dict(coverage_weight=1., importance_weight=2., overlap_penalty=3.,
                       installation_cost_weight=.5, sensor_count_penalty=.2)
        spec = specification(weights=weights,
            cost=dict(base_per_sensor=2., per_height_m=1.),
            regions=[dict(name='priority', bounds=[2., -1., 3., 1.], importance_factor=5.)])
        evaluator = CoverageRequirements(spec).bind(self.targets)
        masks = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
        candidates = CandidateSet([Candidate(i, *pose, mask)
            for i, (pose, mask) in enumerate(zip(self.poses, masks))], self.targets, '', '', {})
        problem = PlacementProblem(candidates, 2, ObjectiveWeights(**weights),
                                   [1., 1., 5.], [5., 6.])
        for k in range(3):
            for ids in itertools.combinations(range(2), k):
                density = masks[list(ids)].sum(axis=0) if ids else np.zeros(3, dtype=int)
                actual = evaluator.evaluate(self.poses[list(ids)], density)
                expected = evaluate_selection(problem, ids)
                self.assertAlmostEqual(actual.objective_score, expected.objective_score)
                self.assertEqual(actual.coverage_pct, expected.coverage_pct)
                self.assertEqual(actual.components, expected.components)

    def test_overlap_ratio_is_targets_not_excess_observations(self):
        evaluator = CoverageRequirements(dict(schema='constrained_coverage_v1', budget_k=3)).bind(self.targets)
        poses = np.vstack([self.poses, [4., 2., 3., 90., -20.]])
        result = evaluator.evaluate(poses, np.array([3, 1, 0]))
        self.assertEqual(result.components['overlap'], 2)
        self.assertAlmostEqual(result.overlap_ratio, 1 / 3)
        self.assertAlmostEqual(result.coverage_pct, 200 / 3)

    def test_cost_region_constraints_and_empty_regions(self):
        spec = specification(cost=dict(base_per_sensor=2., per_height_m=1.,
            regions=[dict(bounds=[-.5, 1., .5, 3.], surcharge=4.)]),
            constraints=dict(max_installation_cost=10., min_xy_separation_m=3.,
                             min_coverage_pct=90., max_overlap_ratio=.2),
            regions=[dict(name='critical', bounds=[0., -1., 1., 1.],
                          min_coverage_pct=100., min_overlap_ratio=.5)])
        result = CoverageRequirements(spec).bind(self.targets).evaluate(self.poses, np.array([1, 0, 2]))
        self.assertFalse(result.feasible)
        for name in ('installation_cost', 'min_xy_separation', 'coverage', 'overlap_max',
                     'region:critical:coverage', 'region:critical:overlap_min'):
            self.assertIn(name, result.violations)
        self.assertEqual(result.components['installation_cost'], 15.)
        with self.assertRaisesRegex(ValueError, 'no targets'):
            CoverageRequirements(specification(regions=[dict(name='empty', bounds=[10, 10, 11, 11])])).bind(self.targets)

    def test_pole_capacity_and_forbidden_installation(self):
        spec = specification(constraints=dict(max_sensors_per_pole=1,
            forbidden_regions=[[0., 1., 1., 3.]]))
        poses = self.poses.copy()
        poses[1, :2] = poses[0, :2]
        result = CoverageRequirements(spec).bind(self.targets).evaluate(poses, np.array([2, 0, 0]))
        self.assertIn('pole_capacity', result.violations)
        self.assertIn('forbidden_installation', result.violations)
        self.assertNotIn('duplicate_pose', result.violations)
        poses[1] = poses[0]
        self.assertIn('duplicate_pose', CoverageRequirements(spec).bind(self.targets).evaluate(
            poses, np.array([2, 0, 0])).violations)

    def test_invalid_specifications_and_density_fail_closed(self):
        bad = [specification(unknown=1), specification(constraints=dict(min_overlap_ratio=1.1)),
               specification(cost=dict(base_per_sensor=-1)),
               specification(constraints=dict(min_overlap_ratio=.8, max_overlap_ratio=.2)),
               specification(constraints=dict(max_sensors_per_pole=True)),
               specification(weights=dict(coverage_weight=float('nan'))),
               specification(regions=[dict(name='x', bounds=[1, 0, 0, 1])])]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises((ValueError, TypeError)):
                CoverageRequirements(spec)
        evaluator = CoverageRequirements(specification()).bind(self.targets)
        for density in ([1, 0], [1, -.1, 0], [1, .5, 0], [3, 0, 0]):
            with self.subTest(density=density), self.assertRaises(ValueError):
                evaluator.evaluate(self.poses, np.array(density))


class ConstrainedRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = deepcopy(load_config())
        self.cfg.mappo.update(local_map_size=3, ppo_update_timestep=4, ppo_k_epochs=1,
            ppo_minibatch_size=8, eval_interval=4, eval_steps=2)
        self.scenario = width_scenario(10)
        self.initial = np.array([[30.25, 42.25, 3, 90, -20],
                                 [70.25, 57.25, 3, 270, -20]], dtype=np.float32)
        self.cpu = patch('torch.cuda.is_available', return_value=False)
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.no_cache = patch('src.methods.mappo.bake_visibility_map', side_effect=AssertionError('No full cache'))
        self.no_cache.start()
        self.addCleanup(self.no_cache.stop)

    def test_joint_rejection_retains_sampled_actions_and_restores_evaluation(self):
        requirements = CoverageRequirements(specification(cost=dict(per_height_m=1.),
            constraints=dict(max_installation_cost=6., min_xy_separation_m=3., max_sensors_per_pole=1)))
        session = RefinementSession(self.initial, self.scenario, self.cfg, requirements, 2.)
        driver = _ActionDriver(session.env.device)
        driver.action = 9
        agents = [_ConstrainedAgent(session, driver, pose) for pose in self.initial]
        before = _snapshot_agent_states(agents)
        previous = session.current
        previous_density = session.density.copy()
        reward, rejected, proposed = _joint_step(session, agents)
        self.assertEqual(reward, -1.)
        self.assertTrue(rejected)
        self.assertIn('installation_cost', proposed.violations)
        np.testing.assert_array_equal(_poses(agents), self.initial)
        for agent in agents:
            self.assertEqual(agent.consume_transition()[2], 9)
        agents[0].yaw += 5.
        assessment, density = session.assess(_poses(agents))
        session.set_current(assessment, density)
        previous, previous_density = session.current, session.density.copy()
        previous_poses = _poses(agents).copy()
        previous_hint = None if session.params.uncovered_centroid is None else session.params.uncovered_centroid.copy()
        _, final, invalid = _evaluate_policy(session, agents, before)
        self.assertEqual(invalid, 2)
        self.assertTrue(final.feasible)
        self.assertEqual(session.current, previous)
        np.testing.assert_array_equal(session.density, previous_density)
        np.testing.assert_array_equal(_poses(agents), previous_poses)
        np.testing.assert_array_equal(session.params.uncovered_centroid, previous_hint)

    def test_infeasible_initial_is_not_silently_relaxed(self):
        spec = specification(cost=dict(base_per_sensor=2.), constraints=dict(max_installation_cost=1.))
        for method in ('local_search', 'mappo'):
            with self.subTest(method=method), self.assertRaises(InitialInfeasible) as error:
                refine_constrained(self.initial, self.scenario, self.cfg, spec, method=method, steps=8)
            self.assertIn('installation_cost', error.exception.assessment.violations)

    def test_both_methods_share_objective_and_independent_validation(self):
        spec = specification(weights=dict(coverage_weight=1., overlap_penalty=.2, installation_cost_weight=2.),
            cost=dict(base_per_sensor=10., per_height_m=1.),
            constraints=dict(max_installation_cost=30., min_xy_separation_m=3., max_sensors_per_pole=1))
        initial_reports = []
        for method in ('local_search', 'mappo'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / method
                poses, meta = refine_constrained(self.initial, self.scenario, self.cfg, spec,
                    method=method, output=root, steps=8, max_evaluations=15, radius_m=2., seed=7)
                a, b = meta['assessments']['initial'], meta['assessments']['deployment']
                initial_reports.append(a)
                self.assertGreaterEqual(b['objective_score'], a['objective_score'] - 1e-6)
                self.assertTrue(b['feasible'])
                self.assertFalse(meta['full_visibility_cache'])
                self.assertLessEqual(np.linalg.norm(poses[:, :2] - self.initial[:, :2], axis=1).max(), 2.00001)
                np.testing.assert_array_equal(np.load(root / 'initial.npy'), self.initial)
                if method == 'mappo':
                    self.assertEqual(meta['training_steps'], 8)
                    self.assertEqual(meta['ppo_updates'], 2)
                    self.assertTrue(meta['assessments']['final_policy']['feasible'])
                    checkpoint = torch.load(root / 'policy.pt', weights_only=True)
                    self.assertEqual(checkpoint['requirements'], spec)
                    with np.load(root / 'train_log.npz') as log:
                        self.assertEqual(len(log['step']), 8)
                        self.assertTrue(all(np.isfinite(v).all() for v in log.values()))
                    self.assertTrue(all(np.isfinite(v) for d in meta['ppo_diagnostics'] for v in d.values()))
                with self.assertRaises(FileExistsError):
                    refine_constrained(self.initial, self.scenario, self.cfg, spec, output=root)
        self.assertEqual(*initial_reports)

    def test_local_search_selects_objective_not_coverage(self):
        initial = self.initial[:1].copy()
        initial[0, 2] = 4.
        targets = MapGeometry(self.scenario).target_points(self.cfg.target['coverage_z_slices'])
        n = len(targets)
        spec = specification(weights=dict(installation_cost_weight=10. * n), cost=dict(per_height_m=1.))
        def density(env, poses):
            return np.full(n, int(poses[0, 2] > 3.))
        def mask(provider, pose):
            return np.full(n, pose[2] > 3., dtype=bool)
        with patch('src.methods.mappo.LegacyMAPPOEnv.coverage_density', density), \
             patch('src.core.visibility.MeshVisibility.mask', mask):
            poses, meta = refine_constrained(initial, self.scenario, self.cfg, spec,
                radius_m=1., max_evaluations=80)
        self.assertEqual(poses[0, 2], 3.)
        a, b = meta['assessments']['initial'], meta['assessments']['deployment']
        self.assertLess(b['coverage_pct'], a['coverage_pct'])
        self.assertGreater(b['objective_score'], a['objective_score'])

    def test_deadline_can_expire_before_training(self):
        _, meta = refine_constrained(self.initial.tolist(), self.scenario, self.cfg, specification(),
                                    method='mappo', steps=8, time_limit_sec=1e-9)
        self.assertEqual(meta['training_steps'], 0)
        self.assertEqual(meta['ppo_updates'], 0)
        self.assertEqual(meta['stop_reason'], 'time_limit')
        self.assertTrue(meta['assessments']['deployment']['feasible'])


class ConstrainedRunnerTests(unittest.TestCase):
    def test_default_plan_has_no_output_or_execution(self):
        from prototype_constrained import parser, run
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'absent'
            args = parser().parse_args(['--output', str(output), '--library', str(output / 'missing_library')])
            with patch('prototype_constrained.make_initial', side_effect=AssertionError('No transfer')), \
                 patch('prototype_constrained.subprocess.run', side_effect=AssertionError('No process')), \
                 redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(run(args), 0)
            self.assertFalse(output.exists())
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan['status'], 'planned')
            self.assertEqual(plan['job']['budget_k'], 3)
            self.assertFalse(plan['detection'])

    def test_invalid_limits_and_existing_output_are_rejected(self):
        from prototype_constrained import make_plan, parser, run
        for flags in (['--steps', '0'], ['--seconds', 'nan'], ['--seed', '-1'],
                      ['--methods', 'mappo', 'mappo'], ['--target-id', '999']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                make_plan(parser().parse_args(flags))
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'existing.txt'
            marker.write_text('unchanged')
            with self.assertRaises(FileExistsError):
                run(parser().parse_args(['--execute', '--output', temp]))
            self.assertEqual(marker.read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main(verbosity=2)

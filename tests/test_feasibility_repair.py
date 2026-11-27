"""Feasibility-first repair, shared starts and original transfer anchors."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype_offline import width_scenario
from src.core.coverage_requirements import CoverageRequirements
from src.environment import load_config
from src.methods.constrained_refinement import RefinementSession, _ConstrainedAgent, refine_constrained
from src.methods.feasibility_repair import _necessary_failures, repair_initial_placement
from src.methods.local_search import _ActionDriver


def spec(**updates):
    return dict(schema='constrained_coverage_v1', budget_k=2, **updates)


class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = deepcopy(load_config())
        self.cfg.mappo.update(local_map_size=3, ppo_update_timestep=4, ppo_k_epochs=1,
            ppo_minibatch_size=8, eval_interval=4, eval_steps=2)
        self.scenario = width_scenario(10)
        self.initial = np.array([[30.25, 42.25, 3, 90, -20],
                                 [70.25, 57.25, 3, 270, -20]], dtype=np.float32)
        for target, kwargs in [('torch.cuda.is_available', dict(return_value=False)),
            ('src.methods.mappo.bake_visibility_map', dict(side_effect=AssertionError('No full cache'))),
            ('src.methods.constrained_refinement.SharedMAPPO', dict(side_effect=AssertionError('No training during repair')))]:
            mocked = patch(target, **kwargs)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_residual_decreases_before_violation_disappears(self):
        evaluator = CoverageRequirements(spec(cost=dict(per_height_m=1.),
            constraints=dict(max_installation_cost=7.))).bind(np.array([[0., 0., 1.]]))
        a = self.initial.copy()
        a[:, 2] = 5.
        b = a.copy()
        b[0, 2] = 4.
        first, second = [evaluator.evaluate(p, [0]) for p in (a, b)]
        self.assertEqual(first.violations, second.violations)
        self.assertGreater(first.constraint_residuals['installation_cost'],
                           second.constraint_residuals['installation_cost'])
        self.assertTrue(all(v >= 0 and np.isfinite(v) for v in first.constraint_residuals.values()))

    def test_feasible_start_is_unchanged(self):
        poses, report = repair_initial_placement(self.initial.tolist(), self.scenario, self.cfg, spec())
        np.testing.assert_array_equal(poses, self.initial)
        self.assertEqual(report['status'], 'already_feasible')
        self.assertEqual(report['neighbor_evaluations'], 0)
        self.assertTrue(report['validated_direct_raycast'])

    def test_cost_repair_can_require_multiple_moves(self):
        initial = self.initial.copy()
        initial[:, 2] = 4.
        requirements = spec(cost=dict(per_height_m=1.), constraints=dict(max_installation_cost=6.))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'repair'
            poses, report = repair_initial_placement(initial, self.scenario, self.cfg, requirements,
                output=root, max_evaluations=100, radius_m=2.)
            self.assertEqual(report['status'], 'repaired')
            self.assertFalse(report['assessments']['initial']['feasible'])
            self.assertTrue(report['assessments']['best_candidate']['feasible'])
            np.testing.assert_array_equal(poses[:, 2], [3., 3.])
            np.testing.assert_array_equal(np.load(root / 'initial.npy'), initial)
            np.testing.assert_array_equal(np.load(root / 'feasible.npy'), poses)
            residuals = [row['total_violation'] for row in report['trace']]
            self.assertTrue(all(a >= b for a, b in zip(residuals, residuals[1:])))
            with self.assertRaises(FileExistsError):
                repair_initial_placement(initial, self.scenario, self.cfg, requirements, output=root)

    def test_forbidden_start_can_move_but_anchor_does_not_reset(self):
        requirements = spec(constraints=dict(forbidden_regions=[[29., 41., 31., 43.]]))
        poses, report = repair_initial_placement(self.initial, self.scenario, self.cfg, requirements,
                                                max_evaluations=50, radius_m=2.)
        self.assertEqual(report['status'], 'repaired')
        self.assertLessEqual(np.linalg.norm(poses[:, :2] - self.initial[:, :2], axis=1).max(), 2.00001)
        session = RefinementSession(poses, self.scenario, self.cfg, CoverageRequirements(requirements),
                                    2., anchor_poses=self.initial)
        agent = _ConstrainedAgent(session, _ActionDriver(session.env.device), poses[0], self.initial[0])
        _, mask = agent.get_state_and_mask()
        outward = 1 if poses[0, 0] > self.initial[0, 0] else 2
        self.assertEqual(mask[outward], 0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'refine'
            final, meta = refine_constrained(poses, self.scenario, self.cfg, requirements,
                anchor_poses=self.initial, radius_m=2., max_evaluations=5, output=root)
            np.testing.assert_array_equal(np.load(root / 'anchors.npy'), self.initial)
            self.assertTrue(meta['assessments']['deployment']['feasible'])
            self.assertLessEqual(np.linalg.norm(final[:, :2] - self.initial[:, :2], axis=1).max(), 2.00001)
        bad = poses.copy()
        bad[0, 0] = self.initial[0, 0] + 4.
        with self.assertRaisesRegex((ValueError, RuntimeError), 'radius'):
            refine_constrained(bad, self.scenario, self.cfg, requirements, anchor_poses=self.initial, radius_m=2.)

    def test_failure_is_not_an_infeasibility_certificate(self):
        requirements = spec(constraints=dict(forbidden_regions=[[29., 41., 31., 43.]]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'repair'
            poses, report = repair_initial_placement(self.initial, self.scenario, self.cfg, requirements,
                radius_m=1., max_evaluations=40, output=root)
            self.assertIsNone(poses)
            self.assertEqual(report['status'], 'no_feasible_found')
            self.assertFalse(report['certified_infeasible_fixed_count'])
            self.assertTrue((root / 'best_candidate.npy').exists())
            self.assertFalse((root / 'feasible.npy').exists())

    def test_proven_necessary_conditions_are_reported_separately(self):
        specifications = [spec(cost=dict(per_height_m=1.), constraints=dict(max_installation_cost=5.)),
            dict(schema='constrained_coverage_v1', budget_k=1),
            spec(constraints=dict(min_xy_separation_m=100.))]
        for requirements in specifications:
            with self.subTest(specification=requirements):
                poses, report = repair_initial_placement(self.initial, self.scenario, self.cfg, requirements,
                                                        radius_m=2., max_evaluations=5)
                self.assertIsNone(poses)
                self.assertEqual(report['status'], 'infeasible_fixed_count')
                self.assertTrue(report['certified_infeasible_fixed_count'])
                self.assertTrue(report['necessary_condition_failures'])
                self.assertEqual(report['neighbor_evaluations'], 0)
        _, report = repair_initial_placement(self.initial[:1], self.scenario, self.cfg,
                                             spec(constraints=dict(min_overlap_ratio=.1)))
        self.assertTrue(report['certified_infeasible_fixed_count'])

    def test_necessary_bounds_account_for_pose_tolerances(self):
        distance = float(np.linalg.norm(self.initial[0, :2].astype(float) - self.initial[1, :2]))
        session = SimpleNamespace(initial=self.initial, anchors=self.initial, radius_m=2., config=self.cfg,
            env=SimpleNamespace(zs=np.array([3., 4., 5., 6.])),
            evaluator=SimpleNamespace(spec=spec(constraints=dict(min_xy_separation_m=distance + 4. + 5e-6))))
        self.assertFalse(_necessary_failures(session))
        session.evaluator.spec = spec(cost=dict(per_height_m=1.),
                                      constraints=dict(max_installation_cost=6. - 1e-5))
        self.assertFalse(_necessary_failures(session))

    def test_evaluation_and_time_limits_fail_closed(self):
        initial = self.initial.copy()
        initial[:, 2] = 4.
        requirements = spec(cost=dict(per_height_m=1.), constraints=dict(max_installation_cost=6.))
        for kwargs, reason in [(dict(max_evaluations=1), 'evaluation_limit'),
                               (dict(time_limit_sec=1e-9), 'time_limit')]:
            with self.subTest(kwargs=kwargs):
                poses, report = repair_initial_placement(initial, self.scenario, self.cfg, requirements, **kwargs)
                self.assertIsNone(poses)
                self.assertEqual(report['status'], 'no_feasible_found')
                self.assertEqual(report['stop_reason'], reason)
                self.assertFalse(report['certified_infeasible_fixed_count'])
        for kwargs in (dict(max_evaluations=0), dict(time_limit_sec=float('nan')), dict(radius_m=0.)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                repair_initial_placement(initial, self.scenario, self.cfg, requirements, **kwargs)


class RepairRunnerTests(unittest.TestCase):
    def test_opt_in_plan_is_read_only_and_preserves_defaults(self):
        from prototype_constrained import parser, run
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'absent'
                args = parser().parse_args(['--output', str(root)] + (['--repair-initial'] if enabled else []))
                with redirect_stdout(io.StringIO()) as stream, \
                     patch('prototype_constrained.subprocess.run', side_effect=AssertionError('No worker')):
                    self.assertEqual(run(args), 0)
                self.assertFalse(root.exists())
                self.assertEqual(json.loads(stream.getvalue())['initial_repair']['enabled'], enabled)
        with self.assertRaises(ValueError):
            run(parser().parse_args(['--repair-initial', '--repair-evaluations', '0']))

    def test_failed_repair_stops_before_either_optimizer(self):
        from prototype_constrained import parser, run
        for status in ('no_feasible_found', 'infeasible_fixed_count'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'run'
                args = parser().parse_args(['--output', str(root), '--repair-initial', '--execute'])
                runtime = dict(status=status, returncode=2, worker_wall_sec=.1)
                with patch('prototype_constrained.freeze_inputs'), \
                     patch('prototype_constrained.subprocess.run', return_value=SimpleNamespace(stdout='')), \
                     patch('prototype_constrained.launch_worker', return_value=(dict(status=status), runtime)) as launch, \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(run(args), 2)
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(launch.call_args.args[-1], 'repair')
                manifest = json.loads((root / 'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'no_feasible_initial')
                self.assertTrue(all(job['status'] == 'not_run' for job in manifest['jobs'].values()))
                self.assertFalse((root / 'summary.csv').exists())
                self.assertFalse((root / 'local_search').exists())
                self.assertFalse((root / 'mappo').exists())

    def test_worker_rejects_tampered_shared_initial_before_transfer(self):
        from prototype_constrained import worker
        from prototype_offline import digest, save
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'repair').mkdir()
            artifact = root / 'repair/feasible.npy'
            np.save(artifact, np.zeros((2, 5), dtype=np.float32))
            save(root / 'manifest.json', dict(frozen_hashes={'repair/feasible.npy': digest(artifact)}))
            np.save(artifact, np.ones((2, 5), dtype=np.float32))
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), \
                 patch('torch.set_num_interop_threads'), \
                 patch('prototype_constrained.make_initial', side_effect=AssertionError('Must fail before transfer')), \
                 self.assertRaisesRegex(RuntimeError, 'Frozen input changed'):
                worker(root, 'local_search')


if __name__ == '__main__':
    unittest.main(verbosity=2)

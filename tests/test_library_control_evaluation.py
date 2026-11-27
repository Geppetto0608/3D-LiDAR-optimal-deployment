"""Matched-query accounting, decision isolation and exact-backend checks."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_transfer_study as runner
from src.core import library_control_evaluation as evaluation


class MatchedQueryTests(unittest.TestCase):
    def problem(self):
        return dict(geometry=SimpleNamespace(create_environment_mesh=lambda: SimpleNamespace(
            vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=int))),
            poses=np.asarray([[i, 0, 3, 0, -10] for i in range(4)], dtype=float),
            targets=np.asarray([[i, 1, 1.5] for i in range(4)], dtype=float),
            pool_ids=[0, 1], profile=None, k_values=[1, 2], context=dict(track='A'),
            transfers=[dict(status='complete', selected_ids=[0, 1], budget_k=2, num_selected=2)])

    def planner(self, events):
        def plan(poses, targets, profile, q, ks, method, *, seed, enumeration_limit):
            self.assertEqual(q, 2)
            self.assertEqual(ks, [1, 2])
            events.append('plan_' + method)
            return dict(method=method, seed=seed, selected_ids=[0, 2], query_order=[0, 2],
                        planning_sec=0., optical_candidate_evaluations=0, optical_point_tests=0)
        return plan

    def provider(self, events):
        masks = np.asarray([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=bool)

        class Provider:
            def __init__(self, mesh, targets, profile, work):
                events.append('scene')
                self.work = work

            def mask(self, pose):
                events.append(int(pose[0]))
                self.work.add('direct_pose_evaluations')
                self.work.add('rays_submitted', 4)
                return masks[int(pose[0])].copy()
        return Provider

    def test_all_plans_precede_queries_all_decisions_precede_reference(self):
        events = []
        real = evaluation.build_frontier

        def traced(*args, **kwargs):
            events.append(kwargs['context'].get('arm_id', 'full_reference'))
            return real(*args, **kwargs)

        with patch.object(evaluation, 'plan_control', side_effect=self.planner(events)), \
                patch.object(evaluation, 'MeshVisibility', self.provider(events)), \
                patch.object(evaluation, 'build_frontier', side_effect=traced):
            report = evaluation.evaluate_matched_controls(self.problem(), [0, 1])['report']
        self.assertTrue(all(str(e).startswith('plan_') for e in events[:5]))
        self.assertEqual(events[5], 'scene')
        self.assertEqual(len(report['arms']), 6)
        self.assertEqual(len(report['comparison']), 12)
        self.assertTrue(report['validation']['passed'])
        for arm in report['arms']:
            self.assertLess(events.index(arm['arm_id']), events.index('full_reference'))
            self.assertEqual(arm['work']['direct_pose_evaluations'], 2)
            self.assertTrue(arm['frontier']['enumeration']['completed'])
            self.assertEqual(len(arm['query_plan']['selected_ids']), 2)
            self.assertTrue(all(row['validated_direct_raycast'] for row in arm['frontier']['frontier']))
        rows = [r for r in report['comparison'] if r['budget_k'] == 2]
        self.assertEqual(rows[0]['covered_count'], 3)
        self.assertTrue(all(row['covered_count'] == 4 for row in rows[1:]))
        self.assertEqual(report['work']['full_reference']['direct_pose_evaluations'], 4)
        json.dumps(report, allow_nan=False)

    def test_plan_uses_neither_ranker_nor_physics(self):
        with patch.object(evaluation, 'plan_control', side_effect=AssertionError('no ranking')),\
                patch.object(evaluation, 'MeshVisibility', side_effect=AssertionError('no physics')):
            plan = evaluation.plan_matched_controls(self.problem(), [0, 1])
        self.assertFalse(plan['execute'])
        self.assertEqual(plan['expected_arm_count'], 6)

    def test_invalid_seeds_or_approximate_solver_rejected(self):
        for seeds in ([], [0], [1, 0], [0, 0], [True, 1], [-1, 0], [0, 2**32],
                      [None, 0], [[0], 1], ['0', 1], list(range(101))):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                evaluation.plan_matched_controls(self.problem(), seeds)
        with self.assertRaisesRegex(ValueError, 'exact subset'):
            evaluation.plan_matched_controls(self.problem(), [0, 1], enumeration_limit=1)

    def test_invalid_query_plan_fails_before_any_physical_call(self):
        for selected in ([0], [0, 0], [0, 4], [2, 0]):
            with self.subTest(selected=selected), \
                    patch.object(evaluation, 'plan_control', return_value=dict(
                        method='geometry_single', seed=None, selected_ids=selected)), \
                    patch.object(evaluation, 'MeshVisibility', side_effect=AssertionError('no physics')), \
                    self.assertRaises(ValueError):
                evaluation.evaluate_matched_controls(self.problem(), [0, 1])

    def test_underreported_physical_queries_rejected(self):
        with patch.object(evaluation, 'plan_control', side_effect=self.planner([])), \
                patch.object(evaluation, 'MeshVisibility', side_effect=lambda *a, **kw:
                             SimpleNamespace(mask=lambda pose: np.ones(4, bool))), \
                self.assertRaisesRegex(RuntimeError, 'count differs'):
            evaluation.evaluate_matched_controls(self.problem(), [0, 1])

    def test_reference_mask_disagreement_rejected(self):
        calls = [0]
        original = self.provider([])

        def factory(*args, **kwargs):
            calls[0] += 1
            obj = original(*args, **kwargs)
            if calls[0] == 7:
                obj.mask = lambda pose: np.zeros(4, bool)
            return obj

        with patch.object(evaluation, 'plan_control', side_effect=self.planner([])), \
                patch.object(evaluation, 'MeshVisibility', side_effect=factory), \
                self.assertRaisesRegex(RuntimeError, 'disagree'):
            evaluation.evaluate_matched_controls(self.problem(), [0, 1])

    def test_nonboolean_physical_masks_are_not_silently_coerced(self):
        with patch.object(evaluation, 'plan_control', side_effect=self.planner([])), \
                patch.object(evaluation, 'MeshVisibility', side_effect=lambda *a, **kw:
                             SimpleNamespace(mask=lambda pose: np.ones(4, dtype=int))), \
                self.assertRaisesRegex(ValueError, 'boolean masks'):
            evaluation.evaluate_matched_controls(self.problem(), [0, 1])

    def test_arm_execution_order_does_not_change_scores(self):
        with patch.object(evaluation, 'plan_control', side_effect=self.planner([])), \
                patch.object(evaluation, 'MeshVisibility', self.provider([])):
            original_plans = evaluation._query_plans
            forward = evaluation.evaluate_matched_controls(self.problem(), [0, 1])['report']
            with patch.object(evaluation, '_query_plans', side_effect=lambda *args:
                              list(reversed(original_plans(*args)))):
                backward = evaluation.evaluate_matched_controls(self.problem(), [0, 1])['report']
        key = lambda row: (row['arm_id'], row['budget_k'])
        self.assertEqual(sorted(forward['comparison'], key=key), sorted(backward['comparison'], key=key))

    def test_opt_in_cli_dry_run_has_no_writes_or_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target.json'
            target.write_text('{}')
            output = Path(directory) / 'absent'
            with patch.object(runner, 'prepare', return_value=self.problem()), \
                    patch.object(evaluation, 'plan_control', side_effect=AssertionError('no ranking')), \
                    patch.object(runner.subprocess, 'run', side_effect=AssertionError('no worker')), \
                    redirect_stdout(io.StringIO()):
                plan = runner.main(['--track', 'A', '--source', directory, '--target-json', str(target),
                    '--control-seeds', '0', '1', '--output', str(output)])
            self.assertFalse(output.exists())
            self.assertEqual(plan['control_seeds'], [0, 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Ordering, domain, certification-scope and persistence guards for target tests."""
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
from src.core import library_transfer_evaluation as evaluation


class TransferEvaluationTests(unittest.TestCase):
    def problem(self, pool=(0, 1)):
        return dict(geometry=SimpleNamespace(create_environment_mesh=lambda: SimpleNamespace(
            vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=int))),
            poses=np.asarray([[i, 0, 3, 0, -10] for i in range(4)], dtype=float),
            targets=np.asarray([[i, 1, 1.5] for i in range(4)], dtype=float),
            pool_ids=list(pool), profile=None, k_values=[1, 2], context=dict(track='A'),
            transfers=[dict(status='complete', selected_ids=list(pool), budget_k=2, num_selected=len(pool)),
                       dict(status='unavailable', reason='fixture: no correspondence')])

    def provider(self, events):
        masks = np.asarray([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=bool)

        class FakeVisibility:
            def __init__(self, mesh, targets, profile, work):
                events.append('scene')
                self.work = work

            def mask(self, pose):
                self.work.add('rays_submitted', 4)
                self.work.add('pose_evaluations')
                events.append(int(pose[0]))
                return masks[int(pose[0])].copy()

        return FakeVisibility

    def test_online_finishes_before_reference_and_gap_uses_full_domain(self):
        events = []
        real = evaluation.build_frontier

        def traced(*args, **kwargs):
            events.append(kwargs['context']['evaluation_domain'])
            return real(*args, **kwargs)

        problem = self.problem()
        with patch.object(evaluation, 'MeshVisibility', self.provider(events)), \
                patch.object(evaluation, 'build_frontier', side_effect=traced):
            result = evaluation.evaluate_problem(problem)
        report = result['report']
        self.assertEqual(events[:4], ['scene', 0, 1, 'transferred_pool'])
        self.assertEqual(report['comparison'][1]['library_pool_coverage_pct'], 75.)
        self.assertEqual(report['comparison'][1]['full_reference_coverage_pct'], 100.)
        self.assertEqual(report['comparison'][1]['pool_certified_shortfall_upper_count'], 1)
        self.assertEqual(report['pool_frontier']['frontier'][1]['certification'], 'discrete_optimal')
        self.assertNotEqual(report['pool_frontier']['domain_sha256'], report['full_frontier']['domain_sha256'])
        self.assertEqual(report['work']['library_pool']['pose_evaluations'], 2)
        self.assertEqual(report['work']['full_reference']['pose_evaluations'], 4)
        self.assertNotIn('target_coverage_pct', report['transfers'][1])
        self.assertNotIn('target_coverage_pct', problem['transfers'][0])
        self.assertTrue(report['validation']['passed'])

    def test_empty_pool_is_feasible_zero_without_fake_transfer_success(self):
        with patch.object(evaluation, 'MeshVisibility', self.provider([])):
            report = evaluation.evaluate_problem(self.problem(()))['report']
        self.assertEqual(report['pool_frontier']['frontier'][0]['poses'], [])
        self.assertEqual(report['comparison'][1]['library_pool_coverage_pct'], 0)
        self.assertEqual(report['comparison'][1]['full_reference_coverage_pct'], 100)
        self.assertEqual(report['work']['library_pool'], {})

    def test_plan_never_raycast_or_mesh(self):
        problem = self.problem()
        problem['geometry'] = None
        with patch.object(evaluation, 'MeshVisibility', side_effect=AssertionError('no raycast')):
            plan = evaluation.plan_problem(problem)
        self.assertFalse(plan['execute'])
        self.assertEqual(plan['subset_counts']['full_reference'], 11)

    def test_malformed_domain_ids_budgets_and_records_rejected(self):
        variants = []
        for pool in ([1, 0], [True], [4], [0, 0]):
            problem = self.problem()
            problem['pool_ids'] = pool
            variants.append(problem)
        problem = self.problem()
        problem['transfers'][0]['selected_ids'] = [2]
        variants.append(problem)
        problem = self.problem()
        problem['k_values'] = [True]
        variants.append(problem)
        problem = self.problem()
        problem['poses'][0, 0] = float('nan')
        variants.append(problem)
        problem = self.problem()
        problem['targets'][0] = problem['targets'][1]
        variants.append(problem)
        for problem in variants:
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                evaluation.plan_problem(problem)

    def test_reference_disagreement_fails(self):
        count = [0]
        factory = self.provider([])

        def mismatched(*args, **kwargs):
            count[0] += 1
            obj = factory(*args, **kwargs)
            if count[0] == 2:
                obj.mask = lambda pose: np.zeros(4, dtype=bool)
            return obj

        with patch.object(evaluation, 'MeshVisibility', side_effect=mismatched), self.assertRaises(RuntimeError):
            evaluation.evaluate_problem(self.problem())

    def test_no_enumeration_retains_honest_scope(self):
        with patch.object(evaluation, 'MeshVisibility', self.provider([])):
            report = evaluation.evaluate_problem(self.problem(), enumeration_limit=0)['report']
        self.assertFalse(report['full_frontier']['enumeration']['completed'])
        self.assertEqual(report['full_frontier']['frontier'][0]['bound_source'], 'union_and_single_counts')

    def test_fallback_preserves_transferred_layout_better_than_greedy(self):
        problem = self.problem((0, 1, 2))
        problem['poses'] = problem['poses'][:3]
        problem['targets'] = np.asarray([[i, 1, 1.5] for i in range(6)])
        problem['transfers'] = [dict(status='complete', selected_ids=[1, 2], budget_k=2, num_selected=2)]
        masks = np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 1]], dtype=bool)
        with patch.object(evaluation, 'MeshVisibility', side_effect=lambda *a, **kw:
                          SimpleNamespace(mask=lambda pose: masks[int(pose[0])].copy())):
            report = evaluation.evaluate_problem(problem, enumeration_limit=0)['report']
        self.assertEqual(report['pool_frontier']['frontier'][1]['covered_count'], 6)
        self.assertEqual(report['full_frontier']['frontier'][1]['covered_count'], 5)
        self.assertEqual(report['full_frontier']['frontier'][1]['upper_bound_count'], 6)
        self.assertEqual(report['pool_frontier']['frontier'][1]['full_candidate_ids'], [1, 2])

    def test_runner_plan_does_not_write_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target.json'
            target.write_text('{}')
            output = Path(directory) / 'absent'
            with patch.object(runner, 'prepare', return_value=self.problem()), \
                    patch.object(runner.subprocess, 'run', side_effect=AssertionError('no worker')), \
                    redirect_stdout(io.StringIO()):
                plan = runner.main(['--track', 'A', '--source', directory, '--target-json', str(target),
                                    '--output', str(output)])
            self.assertFalse(output.exists())
            self.assertFalse(plan['execute'])

    def test_runner_output_and_worker_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                runner.execute('A', directory, {}, [1], directory)
            with self.assertRaises(ValueError):
                runner.execute('A', directory, {}, [1], Path(directory) / 'nested')
        with self.assertRaises(ValueError):
            runner.main(['--worker-root', '/tmp/not-a-target'])

    def test_worker_rejects_implementation_or_input_changes_during_evaluation(self):
        for changed in ('source/frozen.py', 'inputs/target.json'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'source').mkdir()
                (root / 'inputs/library').mkdir(parents=True)
                for name, content in [('source/frozen.py', '# original\n'), ('inputs/target.json', '{}'),
                                      ('inputs/settings.json', json.dumps(dict(track='A', k_values=[1], enumeration_limit=1)))]:
                    (root / name).write_text(content)
                manifest = dict(schema=runner.SCHEMA, status='prepared', library_hashes={},
                    source_hashes={'frozen.py': runner.digest(root / 'source/frozen.py')},
                    input_hashes={name: runner.digest(root / name) for name in ('inputs/target.json', 'inputs/settings.json')})
                runner.save(root / 'manifest.json', manifest)

                def mutate(*args):
                    (root / changed).write_text('changed')
                    return dict(report=dict(phase_wall_sec={}, plan={}))

                with patch.object(runner, 'BASE', root / 'source'), \
                        patch.object(runner, 'prepare', return_value=self.problem()), \
                        patch.object(runner, 'evaluate_problem', side_effect=mutate), \
                        self.assertRaisesRegex(ValueError, 'Frozen input/source changed'):
                    runner._worker(root)
                self.assertEqual(json.loads((root / 'manifest.json').read_text())['status'], 'failed')
                self.assertFalse((root / 'report.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

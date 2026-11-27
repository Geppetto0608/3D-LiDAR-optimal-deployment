"""Source-only marginal-response compression and opt-in frozen study regression."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import connected_constraint_study as runner
import connected_corridor_study as corridor
from src.core.coverage_requirements import CoverageRequirements
from src.methods.source_modes import response_source_alternatives
from validate_connected_constraints import audit, response_pool


class SourceResponseTests(unittest.TestCase):
    def fixture(self):
        canonical = np.array([[s, -6, z, yaw, pitch] for s in (2, 6, 10)
            for z, yaw, pitch in ((3, 0, -20), (3, 0, -10), (5, 45, -20), (5, 45, -10))], float)
        masks = np.random.default_rng(197).random((len(canonical), 37)) < .55
        scores = masks.sum(axis=1)-.2*(10+canonical[:, 2])-.3
        return canonical, masks, scores

    def test_all_source_marginals_match_shared_objective_including_cost_and_overlap(self):
        canonical, masks, _ = self.fixture()
        spec = dict(schema='constrained_coverage_v2', budget_k=2,
            weights=dict(coverage_weight=1.4, importance_weight=.3, overlap_penalty=.17,
                         installation_cost_weight=.2, sensor_count_penalty=.3),
            cost=dict(base_per_sensor=10, per_height_m=1), constraints={})
        targets = np.column_stack([np.arange(masks.shape[1]), np.zeros((masks.shape[1], 2))])
        req = CoverageRequirements(spec).bind(targets)
        singles = np.array([req.evaluate(canonical[[i]], masks[i].astype(int),
            sensor_masks=masks[[i]]).objective_score for i in range(len(canonical))])
        coef = 1.4+.3+.17
        for i in range(len(canonical)):
            for j in range(len(canonical)):
                if np.array_equal(canonical[i, :2], canonical[j, :2]):
                    continue
                pair = masks[[i, j]]
                delta = req.evaluate(canonical[[i, j]], pair.sum(axis=0), sensor_masks=pair).objective_score-singles[j]
                self.assertAlmostEqual(delta, singles[i]-coef*np.count_nonzero(masks[i] & masks[j]))
        actual = response_source_alternatives(canonical, masks, singles, 3, intersection_weight=coef)
        self.assertEqual(actual, response_pool(canonical, masks, singles, 3, coef))
        for trace in actual[1]:
            ids = np.flatnonzero(np.all(canonical[:, :2] == trace['site'], axis=1))
            contexts = trace['context_ids']
            values = np.array([[singles[i] if j is None else
                req.evaluate(canonical[[i, j]], masks[[i, j]].sum(axis=0),
                    sensor_masks=masks[[i, j]]).objective_score-singles[j]
                for j in contexts] for i in ids])
            chosen = []
            for step in trace['steps']:
                chosen.append(int(np.flatnonzero(ids == step['selected_id'])[0]))
                regret = values.max(axis=0)-values[chosen].max(axis=0)
                self.assertAlmostEqual(step['worst_regret'], regret.max())
                self.assertAlmostEqual(step['mean_regret'], regret.mean())

    def test_same_height_yaw_alternate_pitch_can_survive_without_pitch_rule(self):
        canonical = np.array([[2, -6, 3, 0, p] for p in (-20, -15, -10)]
            + [[6, -6, 3, 0, p] for p in (-20, -15, -10)], float)
        masks = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1],
                          [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]], bool)
        scores = masks.sum(axis=1).astype(float)
        ids, _ = response_source_alternatives(canonical, masks, scores, 2, intersection_weight=1.05)
        self.assertEqual(ids[:2], [0, 2])
        # Attitude labels do not enter response ranking at a fixed site.
        canonical[:, 4] = [0, -1, -2, 0, -1, -2]
        self.assertEqual(ids, response_source_alternatives(canonical, masks, scores, 2, intersection_weight=1.05)[0])

    def test_nested_deterministic_equal_budgets_nonmutation_and_monotone_envelopes(self):
        canonical, masks, scores = self.fixture()
        originals = [x.copy() for x in (canonical, masks, scores)]
        previous = None
        for count in range(1, 5):
            result = response_source_alternatives(canonical, masks, scores, count, intersection_weight=1.05)
            self.assertEqual(result, response_pool(canonical, masks, scores, count, 1.05))
            ids, traces = result
            self.assertEqual(len(ids), count*3)
            for trace in traces:
                self.assertEqual(trace['context_count'], 9)
                self.assertTrue(all(a['worst_regret']+1e-10 >= b['worst_regret']
                    for a, b in zip(trace['steps'], trace['steps'][1:])))
            if previous is not None:
                for site in range(3):
                    self.assertEqual(previous[site*(count-1):(site+1)*(count-1)], ids[site*count:site*count+count-1])
            previous = ids
        for a, b in zip((canonical, masks, scores), originals):
            np.testing.assert_array_equal(a, b)

    def test_empty_support_and_single_site_ties_use_score_then_id(self):
        canonical = np.array([[2, -6, 3, 0, p] for p in (-20, -15, -10)], float)
        masks = np.zeros((3, 10), dtype=bool)
        ids, traces = response_source_alternatives(canonical, masks, np.array([-3., -2., -2.]), 3,
                                                  intersection_weight=1.)
        self.assertEqual(ids, [1, 2, 0])
        self.assertEqual(traces[0]['context_ids'], [None])
        self.assertTrue(all(s['worst_regret'] == 0 for s in traces[0]['steps']))

    def test_invalid_layout_values_and_budget_rejected(self):
        canonical, masks, scores = self.fixture()
        for c, m, s, n, w in [(canonical, masks.astype(int), scores, 2, 1.),
            (canonical, masks[:, :0], scores, 2, 1.), (canonical, masks, scores, True, 1.),
            (canonical, masks, scores, 5, 1.), (canonical, masks, scores, 2, -1.),
            (canonical, masks, scores, 2, float('nan')), (canonical, masks[:-1], scores, 2, 1.),
            (canonical, masks, scores[:-1], 2, 1.), (canonical[:0], masks[:0], scores[:0], 1, 1.)]:
            with self.subTest(n=n, w=w, shape=m.shape), self.assertRaises(ValueError):
                response_source_alternatives(c, m, s, n, intersection_weight=w)

    def test_opt_in_plan_has_no_raycasts_writes_or_default_method_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'uncreated'
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Raycast in plan')):
                with redirect_stdout(io.StringIO()) as stream:
                    runner.main(['--study', str(ROOT / 'configs/connected_constraint_holdout.json'),
                                 '--response-pool', '--output', str(output)])
            self.assertFalse(output.exists())
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan['method_runs'], 216)
            self.assertEqual(plan['source_alternatives_per_site'], 4)
            self.assertEqual(plan['source_geometry']['road_width_m'], 7)
            self.assertEqual(runner.study_methods(), runner.METHODS)

    def test_frozen_small_real_study_independent_selection_and_no_feasible_pair(self):
        base = deepcopy(corridor.load_suite(ROOT / 'configs/connected_corridor_pilot.json'))
        base['geometry'].update(length_m=40, bend_length_m=12, mesh_step_m=1)
        base['candidate'].update(station_step_m=10)
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        study = dict(schema='connected_constraint_holdout_v1', alternatives_per_site=2,
            targets=[dict(name='development', road_width_m=8, turn_deg=30, bend_length_m=12)],
            profiles=[dict(name='base', interface={}, forbidden_installation=[]),
                dict(name='blocked', interface={}, forbidden_installation=[dict(station_range_m=[0, 40], side='both')])])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'base.json').write_text(json.dumps(base))
            (tmp / 'study.json').write_text(json.dumps(study))
            output = tmp / 'run'
            command = [sys.executable, str(ROOT / 'connected_constraint_study.py'), '--response-pool',
                '--base-suite', str(tmp / 'base.json'), '--study', str(tmp / 'study.json'),
                '--output', str(output), '--execute']
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            logs = ''.join(p.read_text() for p in (output / 'worker.log', output / 'audit.log') if p.exists())
            self.assertEqual(run.returncode, 0, run.stderr+logs)
            checked = json.loads((output / 'independent_validation.json').read_text())
            self.assertEqual(checked['method_rows'], 18)
            self.assertEqual(checked['reference_optima_checked'], 4)
            self.assertEqual(checked['source_pool_optima_checked'], 6)
            self.assertTrue(checked['source_response_selection_verified'])
            blocked = json.loads((output / 'case_01_00/response_joint/result.json').read_text())
            self.assertIsNone(blocked['assessment'])
            self.assertEqual(blocked['work']['unique_candidate_queries'], 0)
            self.assertFalse((output / 'case_01_00/response_joint/deployment.npz').exists())
            blocked_ls = json.loads((output / 'case_01_00/response_joint_ls/result.json').read_text())
            self.assertEqual(blocked_ls['status'], 'not_run_no_feasible_library_pair')
            # Alter the trace and update its hash: the independent reconstruction must still fail.
            path = output / 'source_pools.json'
            value = json.loads(path.read_text())
            value['response_traces'][0]['steps'][0]['worst_regret'] += 1
            path.write_text(json.dumps(value))
            manifest_path = output / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['source_pools_sha256'] = corridor.digest(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'regret traces'):
                audit(output)
            rejected = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Information budgets, domain separation and a frozen real-raycast smoke run."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from itertools import product
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
import connected_corridor_study as corridor
import connected_hierarchy_study as runner
from src.environment import get_lidar_profile, load_config
from src.methods.connected_hierarchy import (BudgetedObservations, SelectionContext, optical_predictions,
    predicted_partner, refine_poses, select_pool, upper_select)
from src.roadside.corridor_requirements import CorridorRequirements
from validate_connected_hierarchy import audit


class HierarchyTests(unittest.TestCase):
    def fixture(self, constraints=None):
        canonical = np.array(list(product((2., 6.), (-6., 6.), (3., 5.), (-30., 30.), (-20., -10.))))
        targets = np.column_stack([np.arange(12), np.zeros(12), np.ones(12)])
        coordinates = np.column_stack([np.arange(12), np.ones(12), np.ones(12)])
        spec = dict(schema='constrained_coverage_v2', budget_k=2,
            weights=dict(coverage_weight=1.2, importance_weight=.3, overlap_penalty=.2,
                         sensor_count_penalty=.1, installation_cost_weight=.2),
            cost=dict(base_per_sensor=10, per_height_m=1), constraints=constraints or {})
        req = CorridorRequirements(spec, targets, coordinates, canonical, canonical,
            dict(name='toy', interface={}, forbidden_installation=[]), length_m=12, interface_half_length_m=2)
        masks = np.random.default_rng(92).random((len(canonical), len(targets))) < .6
        prior = masks.sum(axis=1)-.2*canonical[:, 2]
        sensor = get_lidar_profile('VELODYNE64_R30', load_config(ROOT/'configs/experiment.yaml'))
        return SelectionContext(canonical, canonical.copy(), targets, prior, req, sensor), masks

    def test_reject_query_before_callback_per_site_global_frozen_and_invalid_ids(self):
        context, masks = self.fixture()
        called = []
        def query(i):
            called.append(i)
            return masks[i]
        obs = BudgetedObservations(context, query, per_site=1, max_extra=2)
        obs(0)
        obs(0)
        for i in (1, True, -1, .5, len(masks)):
            with self.assertRaises(ValueError):
                obs(i)
        other = context.groups[1][0]
        obs(other)
        with self.assertRaises(ValueError):
            obs(context.groups[2][0])
        self.assertEqual(called, [0, other])
        self.assertEqual(obs.report()['extra_candidate_queries'], 2)
        with self.assertRaises(KeyError):
            obs.measured(2)
        obs.frozen = True
        np.testing.assert_array_equal(obs(0), masks[0])
        with self.assertRaises(ValueError):
            obs(context.groups[3][0])
        with self.assertRaises(ValueError):
            obs(0)[0] = False

    def test_fork_work_counters_are_not_double_counted_or_duplicate_keywords(self):
        context, masks = self.fixture()
        base = BudgetedObservations(context, lambda i: masks[i])
        base(0)
        work = base.report(dict(unique_candidate_queries=1, direct_pose_evaluations=1, rays_submitted=7))
        child = BudgetedObservations(context, lambda i: masks[i], initial=base.masks, initial_work=work, max_extra=1)
        child(0)
        child(1)
        report = child.report(dict(unique_candidate_queries=1, direct_pose_evaluations=1, rays_submitted=11))
        self.assertEqual(report, dict(unique_candidate_queries=2, base_candidate_queries=1,
            extra_candidate_queries=1, direct_pose_evaluations=2, rays_submitted=18))
        self.assertEqual(set(base.masks), {0})

    def test_bad_callback_initial_observation_and_budgets_fail_closed(self):
        context, masks = self.fixture()
        for value in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                BudgetedObservations(context, lambda i: masks[i], max_extra=value)
        for initial in ({0: masks[0].astype(int)}, {-1: masks[0]}, {0: masks[0][:-1]}):
            with self.assertRaises(ValueError):
                BudgetedObservations(context, lambda i: masks[i], initial=initial)
        obs = BudgetedObservations(context, lambda i: masks[i].astype(int))
        with self.assertRaises(ValueError):
            obs(0)

    def test_source_and_geometry_selection_have_same_measured_budget_no_oracle(self):
        context, masks = self.fixture()
        static = [i for group in context.groups for i in group[:2]]
        for kind in ('source_response', 'geometry_single'):
            obs = BudgetedObservations(context, lambda i: masks[i], per_site=2)
            result = select_pool(kind, context, obs, 2, static_ids=static)
            self.assertEqual(len(result['candidate_ids']), 8)
            self.assertEqual(len(result['events']), 8)
            self.assertTrue(obs.frozen)
            self.assertEqual(result['optical_point_tests'], 0 if kind == 'source_response' else len(masks)*12)
        obs = BudgetedObservations(context, lambda i: masks[i], per_site=2)
        obs(0)
        with self.assertRaises(ValueError):
            select_pool('probe_context', context, obs, 2)

    def test_probe_context_rounds_use_only_earlier_measured_partners(self):
        context, masks = self.fixture()
        obs = BudgetedObservations(context, lambda i: masks[i], per_site=3)
        result = select_pool('probe_context', context, obs, 3)
        self.assertEqual(len(result['candidate_ids']), 12)
        for snapshot in result['contexts']:
            prior = {e['candidate_id'] for e in result['events'] if e['phase'] == 'seed'
                or e['phase'].startswith('round_') and int(e['phase'].split('_')[1]) < snapshot['round']}
            self.assertEqual(set(snapshot['measured_context_ids']), prior)
            for decision in result['decisions']:
                if decision['round'] == snapshot['round'] and decision['measured_partner_id'] is not None:
                    self.assertIn(decision['measured_partner_id'], prior)
        obs2 = BudgetedObservations(context, lambda i: masks[i], per_site=3)
        second = select_pool('probe_context', context, obs2, 3)
        self.assertEqual(result['decisions'], second['decisions'])
        self.assertEqual(result['candidate_ids'], second['candidate_ids'])

    def test_predicted_partner_score_matches_shared_union_overlap_cost_objective(self):
        context, masks = self.fixture()
        known = [g[0] for g in context.groups[1:]]
        obs = BudgetedObservations(context, lambda i: masks[i])
        for i in known:
            obs(i)
        predicted = optical_predictions(context)
        cid = context.groups[0][0]
        value, partner = predicted_partner(cid, known, predicted, obs, context)
        expected = []
        for i in known:
            pair = np.array([predicted[cid], masks[i]])
            expected.append(context.requirements.evaluate(context.poses[[cid, i]], pair.sum(axis=0), sensor_masks=pair).objective_score)
        self.assertAlmostEqual(value, max(expected))
        self.assertIn(partner, known)
        with self.assertRaises(KeyError):
            predicted_partner(cid, [known[0]+1], predicted, obs, context)

    def test_prediction_cannot_certify_a_physically_infeasible_upper_pair(self):
        context, masks = self.fixture(dict(min_coverage_pct=100))
        masks[:] = False
        obs = BudgetedObservations(context, lambda i: masks[i], per_site=2)
        selection = select_pool('probe_context', context, obs, 2)
        result = upper_select(selection['candidate_ids'], context, obs, np.repeat([0, 1], 16), 'free')
        self.assertIsNone(result['selected_ids'])
        self.assertIsNone(result['assessment'])
        self.assertEqual(refine_poses(result, context, obs, station_radius_m=0)['status'], 'not_run_no_feasible_upper')

    def test_upper_free_can_choose_two_same_module_and_never_queries_unknown(self):
        context, masks = self.fixture()
        masks[:] = False
        first, second, third = [g[0] for g in context.groups[:3]]
        masks[first, :6] = True
        masks[second, 6:] = True
        obs = BudgetedObservations(context, lambda i: masks[i], initial={i: masks[i] for i in (first, second, third)})
        obs.frozen = True
        modules = np.repeat([0, 1], 16)
        free = upper_select([first, second, third], context, obs, modules, 'free')
        quota = upper_select([first, second, third], context, obs, modules, 'one_per_module')
        self.assertEqual(set(free['selected_ids']), {first, second})
        self.assertGreater(free['assessment']['objective_score'], quota['assessment']['objective_score'])
        self.assertEqual(obs.events, [])
        with self.assertRaises(KeyError):
            upper_select([first, first+1], context, obs, modules, 'free')

    def test_fixed_sites_lower_preserves_allocation_and_budget_limited_is_explicit(self):
        context, masks = self.fixture()
        ids = [0, 16]
        initial_obs = BudgetedObservations(context, lambda i: masks[i], initial={i: masks[i] for i in ids})
        initial = upper_select(ids, context, initial_obs, np.repeat([0, 1], 16), 'free')
        for cap in (0, 2, 16):
            obs = BudgetedObservations(context, lambda i: masks[i], initial=initial_obs.masks, max_extra=cap)
            result = refine_poses(initial, context, obs, station_radius_m=0)
            self.assertLessEqual(len(obs.events), cap)
            np.testing.assert_array_equal(context.canonical[result['selected_ids'], :2], context.canonical[ids, :2])
            self.assertGreaterEqual(result['assessment']['objective_score'], initial['assessment']['objective_score'])
            for a, b in zip(result['trace'], result['trace'][1:]):
                self.assertGreater(b['objective_score'], a['objective_score'])
            if cap == 0:
                self.assertEqual(result['stopping_reason'], 'query_budget_limited')

    def test_lower_trace_moves_one_adjacent_coordinate_and_keeps_original_radius(self):
        context, masks = self.fixture()
        masks[:] = False
        masks[16, :6] = True
        masks[1, 6:] = True
        initial_obs = BudgetedObservations(context, lambda i: masks[i], initial={i: masks[i] for i in (0, 16)})
        initial = upper_select([0, 16], context, initial_obs, np.repeat([0, 1], 16), 'free')
        obs = BudgetedObservations(context, lambda i: masks[i], initial=initial_obs.masks, max_extra=16)
        result = refine_poses(initial, context, obs, station_radius_m=4)
        self.assertGreater(len(result['trace']), 1)
        for a, b in zip(result['trace'], result['trace'][1:]):
            before, after = context.canonical[a['selected_ids']], context.canonical[b['selected_ids']]
            self.assertEqual(np.count_nonzero(before != after), 1)
            self.assertEqual(set(np.flatnonzero((before != after).any(axis=0))), {4})
        np.testing.assert_array_equal(result['original_anchors'], context.canonical[[0, 16], :2])

    def test_plan_has_no_writes_raycast_or_default_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'not_created'
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Plan used rays')):
                with redirect_stdout(io.StringIO()) as output:
                    runner.main(['--output', str(path)])
            self.assertFalse(path.exists())
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['conditions'], 24)
            self.assertEqual(plan['method_outputs'], 480)
            self.assertEqual(plan['lower_extra_queries'], 16)
            with self.assertRaises(ValueError):
                runner.main(['--lower-extra-queries', '-1'])
            with self.assertRaises(ValueError):
                runner.main(['--worker-root', str(path)])

    def test_frozen_real_smoke_and_independent_audit_rejects_rehashed_tampering(self):
        base = deepcopy(corridor.load_suite(ROOT/'configs/connected_corridor_pilot.json'))
        base['geometry'].update(length_m=40, bend_length_m=12, mesh_step_m=1)
        base['candidate']['station_step_m'] = 10
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        study = dict(schema='connected_constraint_holdout_v1', alternatives_per_site=2,
            targets=[dict(name='development', road_width_m=8, turn_deg=30, bend_length_m=12)],
            profiles=[dict(name='base', interface={}, forbidden_installation=[]),
                dict(name='blocked', interface={}, forbidden_installation=[dict(station_range_m=[0, 40], side='both')])])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp/'base.json').write_text(json.dumps(base))
            (tmp/'study.json').write_text(json.dumps(study))
            root = tmp/'run'
            command = [sys.executable, str(ROOT/'connected_hierarchy_study.py'), '--base-suite', str(tmp/'base.json'),
                '--study', str(tmp/'study.json'), '--lower-extra-queries', '4', '--output', str(root), '--execute']
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            logs = ''.join(p.read_text() for p in (root/'worker.log', root/'audit.log') if p.exists())
            self.assertEqual(run.returncode, 0, run.stderr+logs)
            validation = json.loads((root/'independent_validation.json').read_text())
            self.assertEqual(validation['method_rows'], 40)
            self.assertEqual(validation['reference_optima_checked'], 4)
            self.assertEqual(validation['queried_pool_optima_checked'], 12)
            self.assertEqual(validation['lower_policy_replays'], 24)
            blocked = json.loads((root/'case_01_00/probe_context__free__fixed_site/result.json').read_text())
            self.assertIsNone(blocked['assessment'])
            self.assertEqual(blocked['work']['unique_candidate_queries'], 0)
            # Update ordinary hashes too: replay must catch a scientifically inconsistent record.
            path = root/'case_00_00/selection_probe_context/selection.json'
            value = json.loads(path.read_text())
            value['contexts'][0]['measured_context_ids'].append(999)
            path.write_text(json.dumps(value))
            cm_path = root/'case_00_00/manifest.json'
            cm = json.loads(cm_path.read_text())
            cm['online_hashes']['selection_probe_context/selection.json'] = corridor.digest(path)
            cm_path.write_text(json.dumps(cm))
            with self.assertRaisesRegex(ValueError, 'selection replay'):
                audit(root)
            rejected = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

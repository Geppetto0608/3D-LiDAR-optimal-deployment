"""Pair-aware static ranking and same-schedule predicted-feedback controls."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geometry_placement as runner
import geometry_mechanism_report as reporting
import test_connected_hierarchy as fixtures
import test_geometry_placement as workers
from src.methods.connected_hierarchy import BudgetedObservations, select_pool
from src.methods.geometry_controls import (plan_geometry_control, select_geometry_control,
    CONTROL_METHODS, CONTROL_TIMINGS)


class GeometryControlTests(unittest.TestCase):
    def fixture(self, constraints=None):
        return fixtures.HierarchyTests().fixture(constraints)

    def test_static_pair_ranking_matches_exhaustive_optical_pairs(self):
        for constraints in ({}, dict(min_coverage_pct=70, max_overlap_ratio=.5,
                min_xy_separation_m=5, max_installation_cost=28, min_single_failure_coverage_pct=30),
                dict(min_coverage_pct=100, max_installation_cost=0)):
            context, predicted = self.fixture(constraints)
            context.requirements.importance = np.linspace(1, 3, len(context.targets))
            context.requirements.profile['interface'] = dict(min_coverage_pct=50, max_overlap_ratio=.75)
            with patch('src.methods.geometry_controls.optical_predictions', return_value=predicted):
                plan = plan_geometry_control('static_pair', context, 2)
                scalar = plan_geometry_control('static_pair', context, 2, partner_engine='scalar')
            self.assertEqual({k:v for k,v in plan.items() if k != 'partner_engine'},
                {k:v for k,v in scalar.items() if k != 'partner_engine'})
            expected = {}
            for cid in np.flatnonzero(context.allowed):
                options = []
                for other in np.flatnonzero(context.allowed):
                    if context.site_ids[cid] == context.site_ids[other]:
                        continue
                    masks = predicted[[cid, other]]
                    value = context.requirements.evaluate(context.poses[[cid, other]], masks.sum(axis=0), sensor_masks=masks)
                    if value.feasible:
                        options.append((-round(value.objective_score, 8), int(other), value.objective_score))
                best = min(options) if options else None
                expected[cid] = (best[2], best[1]) if best else (None, None)
            for row in plan['rankings']:
                for cid, score, partner in zip(row['candidate_ids'], row['predicted_j'], row['predicted_partner_ids']):
                    self.assertEqual((score, partner), expected[cid])
                order = sorted(row['candidate_ids'], key=lambda i: (expected[i][0] is None,
                    -round(expected[i][0], 8) if expected[i][0] is not None else 0., -plan['prior_values'][i], i))
                self.assertEqual(row['candidate_ids'], order)

    def test_controls_never_plan_from_actual_masks_or_source_scores(self):
        context, truth = self.fixture()
        predicted = truth.copy()
        for kind in CONTROL_METHODS:
            plans = []
            for actual in (truth, ~truth, np.zeros_like(truth)):
                ready = []
                def plan(*args, **kwargs):
                    value = plan_geometry_control(*args, **kwargs)
                    ready.extend(item['candidate_id'] for item in value['query_plan'])
                    return value
                called = []
                def query(cid):
                    self.assertTrue(ready, 'Actual query before plan completion')
                    self.assertEqual(cid, ready[len(called)])
                    called.append(cid)
                    return actual[cid]
                context.source_prior[:] = np.nan
                obs = BudgetedObservations(context, query, per_site=2)
                with patch('src.methods.geometry_controls.optical_predictions', return_value=predicted):
                    with patch('src.methods.geometry_controls.plan_geometry_control', side_effect=plan):
                        result = select_geometry_control(kind, context, obs, 2)
                self.assertEqual(len(called), 2*len(context.groups))
                self.assertEqual(obs.report()['unique_candidate_queries'], len(called))
                for cid in result['candidate_ids']:
                    np.testing.assert_array_equal(obs.measured(cid), actual[cid])
                plans.append({k:v for k,v in result.items() if k not in CONTROL_TIMINGS})
            self.assertEqual(plans[0], plans[1])
            self.assertEqual(plans[0], plans[2])

    def test_no_feedback_equals_adaptive_when_prediction_is_truth(self):
        for constraints in ({}, dict(min_coverage_pct=80, max_overlap_ratio=.4)):
            context, predicted = self.fixture(constraints)
            previous = set()
            for budget in (1, 2, 4):
                with patch('src.methods.geometry_controls.optical_predictions', return_value=predicted):
                    plan = plan_geometry_control('no_feedback', context, budget)
                    scalar = plan_geometry_control('no_feedback', context, budget, partner_engine='scalar')
                self.assertEqual({k:v for k,v in plan.items() if k != 'partner_engine'},
                    {k:v for k,v in scalar.items() if k != 'partner_engine'})
                obs = BudgetedObservations(context, lambda i: predicted[i], per_site=budget)
                with patch('src.methods.connected_hierarchy.optical_predictions', return_value=predicted):
                    adaptive = select_pool('probe_context', context, obs, budget,
                        probe_prior='geometry', fast_screen=True, partner_engine='batch')
                self.assertEqual(plan['query_plan'], [{k:e[k] for k in ('candidate_id','phase')} for e in adaptive['events']])
                decisions = deepcopy(adaptive['decisions'])
                contexts = deepcopy(adaptive['contexts'])
                for item in decisions:
                    item['predicted_partner_id'] = item.pop('measured_partner_id')
                for item in contexts:
                    item['predicted_context_ids'] = item.pop('measured_context_ids')
                self.assertEqual(plan['decisions'], decisions)
                self.assertEqual(plan['contexts'], contexts)
                ids = {e['candidate_id'] for e in plan['query_plan']}
                self.assertTrue(previous <= ids)
                previous = ids

    def test_empty_domain_ties_nested_ranking_and_invalid_budgets(self):
        context, predicted = self.fixture()
        for kind in CONTROL_METHODS:
            with patch('src.methods.geometry_controls.optical_predictions', return_value=np.zeros_like(predicted)):
                plans = [plan_geometry_control(kind, context, b) for b in (1,2,4)]
            pools = [{e['candidate_id'] for e in p['query_plan']} for p in plans]
            self.assertTrue(pools[0] <= pools[1] <= pools[2])
            for budget in (True, 0, -1, 1.5, 99):
                with self.assertRaises(ValueError):
                    plan_geometry_control(kind, context, budget)
            obs = BudgetedObservations(context, lambda i: predicted[i], per_site=2)
            obs.get(context.groups[0][0])
            with self.assertRaises(ValueError):
                select_geometry_control(kind, context, obs, 2)
        context.allowed[:] = False
        context.groups = [[] for _ in context.groups]
        for kind in CONTROL_METHODS:
            obs = BudgetedObservations(context, lambda i: self.fail('Forbidden query'), per_site=2)
            result = select_geometry_control(kind, context, obs, 2)
            self.assertEqual(result['candidate_ids'], [])
            self.assertEqual(result['virtual_candidate_masks'], 0)
            self.assertEqual(result['directed_candidate_partner_tests'], 0)

    def test_report_separates_infeasibility_and_rejects_incomplete_budget_grid(self):
        rows = []
        for case in ('good','bad'):
            for method in (*reporting.METHODS, 'full'):
                value = 10. if case=='good' or method=='full' else None
                rows.append(dict(case=case, method=method, query_budget=None if method=='full' else 1,
                    feasible=value is not None, objective_score=value, candidate_queries=8 if method=='full' else 4,
                    rays_submitted=100, cold_ready_median_sec=2, worker_wall_median_sec=3))
        paired, summary = reporting.aggregate(rows, ['good','bad'], [1])
        self.assertEqual(len(paired), 6)
        self.assertEqual(summary['groups']['1']['methods']['no_feedback']['feasible_cases'], 1)
        self.assertEqual(summary['groups']['1']['contrasts']['adaptive_batch_vs_no_feedback'],
            dict(tie=1, both_infeasible=1))
        with self.assertRaises(ValueError):
            reporting.aggregate(rows[:-1], ['good','bad'], [1])
        bad = deepcopy(rows)
        bad[0]['candidate_queries'] += 1
        with self.assertRaisesRegex(ValueError, 'Unmatched'):
            reporting.aggregate(bad, ['good','bad'], [1])

    def test_real_workers_audit_report_and_rehashed_plan_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = workers.GeometryPlacementTests().run_smoke(Path(tmp),
                [*reporting.METHODS, 'full'], [1,2], 1, 'mechanism')
            validation = runner.read(root/'independent_validation.json')
            self.assertEqual(validation['workers'], 18)
            summary = reporting.report(root)
            self.assertEqual(summary['seed_parity_checks'], 2)
            self.assertEqual(summary['nested_pool_checks'], 8)
            manifest = runner.read(root/'manifest.json')
            job = next(j for j in manifest['jobs'] if j['arm']['method']=='no_feedback')
            path = root/'runs'/job['id']/'selection.json'
            value = runner.read(path)
            value['query_plan'][0]['candidate_id'] += 1
            runner.corridor.save(path, value)
            job['output_hashes']['selection.json'] = runner.corridor.digest(path)
            runner.corridor.save(root/'manifest.json', manifest)
            from validate_geometry_placement import audit
            with self.assertRaisesRegex(ValueError, 'Policy replay'):
                audit(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)

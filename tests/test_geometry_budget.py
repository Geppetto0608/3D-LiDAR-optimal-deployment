"""True random-pool controls and paired equal-query reporting."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geometry_placement as runner
from geometry_budget_report import aggregate, report
from src.methods.connected_hierarchy import BudgetedObservations, select_pool
import test_connected_hierarchy as hierarchy_tests
import test_geometry_placement as placement_tests
from validate_geometry_placement import audit


class GeometryBudgetTests(unittest.TestCase):
    def test_random_is_nonadaptive_no_optical_source_or_coverage_ranking(self):
        context, masks = hierarchy_tests.HierarchyTests().fixture()
        outputs = []
        for truth in (masks, ~masks, np.zeros_like(masks)):
            context.source_prior[:] = np.nan
            observed = BudgetedObservations(context, lambda i: truth[i], per_site=3)
            with patch('src.methods.connected_hierarchy.optical_predictions', side_effect=AssertionError('Optics')):
                with patch('src.methods.connected_hierarchy.predicted_partner', side_effect=AssertionError('Adaptive')):
                    value = select_pool('random_pool', context, observed, 3, random_seed=4)
            outputs.append(value['candidate_ids'])
            self.assertEqual((value['optical_point_tests'], value['contexts'], value['decisions']), (0, [], []))
            self.assertEqual(len(value['events']), 12)
            self.assertTrue(observed.frozen)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_random_sampling_is_without_replacement_reproducible_and_nested(self):
        context, masks = hierarchy_tests.HierarchyTests().fixture()
        pools = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            orders = [rng.permutation(g).tolist() for g in context.groups if g]
            previous = set()
            for budget in (1, 2, 4):
                observed = BudgetedObservations(context, lambda i: masks[i], per_site=budget)
                value = select_pool('random_pool', context, observed, budget, random_seed=seed)
                expected = sorted(i for group in orders for i in group[:budget])
                self.assertEqual(value['candidate_ids'], expected)
                self.assertTrue(previous <= set(expected))
                self.assertEqual(len(expected), len(set(expected)))
                previous = set(expected)
            pools.append(tuple(sorted(previous)))
        self.assertGreater(len(set(pools)), 1)

    def test_invalid_seeds_and_options_fail_before_query(self):
        context, _ = hierarchy_tests.HierarchyTests().fixture()
        for seed in (None, True, -1, 2**32, 1.5):
            observed = BudgetedObservations(context, lambda _: self.fail('Unexpected query'), per_site=1)
            with self.assertRaises(ValueError):
                select_pool('random_pool', context, observed, 1, random_seed=seed)
        for options in (dict(probe_prior='random'), dict(fast_screen=True), dict(partner_engine='batch'),
                dict(prediction_transform=lambda x:x)):
            observed = BudgetedObservations(context, lambda _: self.fail('Unexpected query'), per_site=1)
            with self.assertRaises(ValueError):
                select_pool('random_pool', context, observed, 1, random_seed=0, **options)

    def test_cli_seed_grid_is_distinct_from_repeats_and_plan_writes_nothing(self):
        design, base = placement_tests.GeometryPlacementTests().inputs()
        for seeds in ([], [1, 0], [0, 0], [True], [-1], [2**32]):
            with self.assertRaises(ValueError):
                runner.validate_inputs(design, base, ['random'], [1], 1, seeds)
        with self.assertRaises(ValueError):
            runner.validate_inputs(design, base, ['static'], [1], 1, [0])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as captured:
            runner.main(['--methods', 'adaptive_batch', 'static', 'random', 'full',
                '--query-budgets', '1', '2', '4', '--output', str(Path(tmp)/'absent')])
            self.assertFalse((Path(tmp)/'absent').exists())
        import json
        plan = json.loads(captured.getvalue())
        self.assertEqual((plan['fresh_workers'], plan['repeats'], len(plan['arms'])), (264, 1, 22))
        self.assertEqual(sum(a['method']=='random' for a in plan['arms']), 15)

    def rows(self):
        rows = []
        for case, a, b, random, full in [('a', 8., 7., [2., 6.], 10.), ('b', None, 3., [None, 5.], 6.)]:
            for method, seed, value in [('adaptive_batch', None, a), ('static', None, b),
                    ('random', 0, random[0]), ('random', 1, random[1]), ('full', None, full)]:
                rows.append(dict(case=case, method=method, random_seed=seed,
                    query_budget=None if method=='full' else 1, feasible=value is not None,
                    objective_score=value, candidate_queries=12 if method=='full' else 2,
                    rays_submitted=30, cold_ready_median_sec=1.))
        return rows

    def test_reporting_keeps_failures_and_averages_seeds_within_case(self):
        paired, summary = aggregate(self.rows(), ['a', 'b'], [1], [0, 1])
        group = summary['groups']['1']
        random = group['methods']['random']
        self.assertEqual((summary['conditions'], len(paired)), (2, 2))
        self.assertEqual(random['mean_case_feasible_fraction'], .75)
        self.assertEqual(random['mean_case_feasible_j_gap'], 3.5)
        self.assertEqual(random['all_feasible_cases'], 1)
        self.assertEqual(paired[0]['random_feasible_mean_j'], 4.)
        self.assertAlmostEqual(paired[0]['random_feasible_sd_j'], np.sqrt(8))
        self.assertIsNone(paired[1]['adaptive_minus_random_mean_j'])
        self.assertEqual(group['complete_random_mean_cases'], 1)
        self.assertEqual(group['adaptive_vs_static'], dict(win=1, other_only_feasible=1))
        self.assertEqual(group['adaptive_vs_random_seed'], dict(win=2, both_infeasible=1, other_only_feasible=1))

    def test_report_rejects_missing_duplicate_unmatched_and_impossible_quality(self):
        original = self.rows()
        invalid = [original[:-1], original+[original[0]]]
        for field, value in [('candidate_queries', 3), ('objective_score', 11.), ('feasible', False)]:
            rows = deepcopy(original)
            rows[0][field] = value
            invalid.append(rows)
        for rows in invalid:
            with self.assertRaises(ValueError):
                aggregate(rows, ['a', 'b'], [1], [0, 1])

    def test_real_random_workers_audit_report_and_rehashed_trace_tamper(self):
        design, base = placement_tests.GeometryPlacementTests().inputs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner.corridor.save(tmp/'design.json', design)
            runner.corridor.save(tmp/'base.json', base)
            root = tmp/'study'
            cmd = [sys.executable, str(ROOT/'geometry_placement.py'), '--study', str(tmp/'design.json'),
                '--base-suite', str(tmp/'base.json'), '--methods', 'adaptive_batch', 'static', 'random', 'full',
                '--query-budgets', '1', '2', '--random-seeds', '0', '1', '--output', str(root), '--execute']
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=240)
            errors = ''.join(p.read_text() for p in root.glob('logs/*.stderr')) if result.returncode else ''
            if result.returncode and (root/'audit.log').exists():
                errors += (root/'audit.log').read_text()
            self.assertEqual(result.returncode, 0, result.stderr+errors)
            self.assertEqual(runner.read(root/'independent_validation.json')['workers'], 18)
            summary = report(root)
            self.assertEqual(summary['nested_pool_checks'], 8)
            self.assertEqual(summary['groups']['1']['methods']['random']['all_feasible_cases'], 1)
            manifest = runner.read(root/'manifest.json')
            job = next(j for j in manifest['jobs'] if j['arm']['method']=='random')
            path = root/'runs'/job['id']/'selection.json'
            value = runner.read(path)
            value['random_seed'] += 1
            runner.corridor.save(path, value)
            job['output_hashes']['selection.json'] = runner.corridor.digest(path)
            runner.corridor.save(root/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Policy replay'):
                audit(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)

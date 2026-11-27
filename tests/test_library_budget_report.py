"""Regression checks for independent metrics, certificates and aggregation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_budget_report as report


class BudgetReportTest(unittest.TestCase):
    def test_metrics_independent_counts_and_hard_violations(self):
        poses = np.array([[0, 0, 3, 0, 0], [4, 0, 3, 0, 0], [0, 0, 5, 0, 0]], float)
        masks = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], bool)
        critical = np.array([1, 0, 0, 1], bool)
        req = dict(overlap_ratio=.25, critical_overlap_ratio=0., min_separation_m=4.)
        m = report.metrics(poses, masks, critical, [0, 1], 2, req, np.array([1., 1.1, 1.2]), 2., .8)
        self.assertEqual(m['covered_count'], 3)
        self.assertEqual(m['overlap_count'], 1)
        self.assertEqual(m['critical_covered_count'], 1)
        self.assertEqual(set(m['violations']), {'critical_coverage', 'installation_cost'})
        duplicate = report.metrics(poses, masks, critical, [0, 2], 2, req)
        self.assertIn('one_sensor_per_pole', duplicate['violations'])
        self.assertIn('minimum_separation', duplicate['violations'])

    def test_disjoint_bound_proves_cost_infeasibility(self):
        masks = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]], bool)
        groups = [np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool)]
        costs = np.array([2., 2., 1., 1.])
        p = report.disjoint_critical_bound(masks, groups, costs, 2, 3., 4)
        self.assertTrue(p['infeasible_proved'])
        self.assertEqual(p['critical_upper_count'], 3)
        q = report.disjoint_critical_bound(masks, groups, costs, 2, 4., 4)
        self.assertFalse(q['infeasible_proved'])
        self.assertEqual(q['critical_upper_count'], 4)

    def test_bound_rejects_invalid_relaxation_assumptions(self):
        groups = [np.array([1, 0], bool), np.array([0, 1], bool)]
        with self.assertRaisesRegex(ValueError, 'multiple groups'):
            report.disjoint_critical_bound(np.ones((1, 2), bool), groups, np.array([1.]), 2, 3., 2)
        with self.assertRaisesRegex(ValueError, 'necessary'):
            report.disjoint_critical_bound(np.eye(2, dtype=bool), groups, np.ones(2), 2, 3., 1)

    def test_multi_group_pose_exclusion_requires_valid_upper(self):
        groups = [np.arange(30) // 10 == i for i in range(3)]
        masks = np.zeros((7, 30), bool)
        for i in range(3):
            masks[i, i * 10:(i + 1) * 10] = True
            masks[i + 3, i * 10:i * 10 + 5] = True
        masks[6, [0, 10]] = True
        p = report.disjoint_critical_bound(masks, groups, np.array([2., 2., 2., 1., 1., 1., 1.]), 3, 5., 29)
        self.assertTrue(p['infeasible_proved'])
        self.assertEqual(p['excluded_multi_group_poses'], 1)
        self.assertEqual(p['multi_group_solution_upper'], 22)
        self.assertEqual(p['critical_upper_count'], 25)
        self.assertEqual(p['global_critical_upper_count'], 25)
        self.assertEqual(p['missing_group_upper_count'], 20)
        cheap = report.disjoint_critical_bound(masks, groups, np.array([2., 2., 2., 1., 1., 1., 1.]), 3, 3., 29)
        self.assertEqual(cheap['critical_upper_count'], 15)
        self.assertEqual(cheap['global_critical_upper_count'], 22)

    def test_csv_missing_not_zero_and_float_tolerance(self):
        report.check_csv(dict(x='', y='1.20000000001', f='True'), dict(x=None, y=1.2, f=True))
        with self.assertRaises(ValueError):
            report.check_csv(dict(x='0'), dict(x=None))
        with self.assertRaises(ValueError):
            report.same(float('nan'), 1., 'finite')

    def test_seed_first_and_frontier_time_not_k_weighted(self):
        rows, times = [], []
        methods = ['library_prefix', 'library_diverse', 'geometry_union', 'geometry_portfolio', 'random']
        for case, ks, duration in [('a', [2, 3, 4], 10.), ('b', [6, 8], 30.)]:
            for method in methods:
                seeds = [0, 1, 2] if method == 'random' else [None]
                for seed in seeds:
                    times.append(dict(case=case, geometry=case, track='A', scene='dense', profile='cost', method=method,
                        seed=seed, online_all_k_sec=duration, planning_sec=1., visibility_sec=1., rays_submitted=20))
                    for k in ks:
                        value = 60. + 10 * seed if seed is not None else 75.
                        rows.append(dict(case=case, geometry=case, track='A', scene='dense', profile='cost', budget_k=k,
                            method=method, seed=seed, status='feasible', coverage_pct=value, critical_coverage_pct=80.,
                            overlap_pct=5., installation_cost=2., num_selected=2, loss_lower_pp=0., loss_upper_pp=10.,
                            online_through_k_sec=duration, reference_coverage_pct=80., actual_q=100))
        refs = pd.DataFrame([dict(status='feasible', solver_status=2, discrete_optimal=True,
                                 near_optimal_certified=True, certified_gap=0.)])
        with tempfile.TemporaryDirectory() as folder:
            s = report.summarize(pd.DataFrame(rows), pd.DataFrame(times), refs, Path(folder))
            random = next(r for r in s['methods'] if r['method'] == 'random')
            self.assertEqual(random['coverage_pct'], 70.)
            self.assertTrue(all(r['online_all_k_sec'] == 20. for r in s['timings']))
            self.assertEqual(s['common_feasible_conditions'], 5)

    def test_output_guard_keeps_raw_root_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'analysis'
            with self.assertRaises(ValueError):
                report.main(['--root', str(root), '--output', str(output)])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

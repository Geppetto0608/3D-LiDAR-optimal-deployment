"""Small real solver checks, never the expanded performance study."""
from itertools import combinations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.methods.coverage_cut_milp import certified_stagnation_stop, solve_cut_milp
from src.methods.library_scalable import coverage_metrics, geometry_plan, optimize_frontier
from src.environment import LidarProfile


class ExpandedSolverTests(unittest.TestCase):
    def test_stagnation_is_not_certificate(self):
        policy = dict(minimum_seconds=300., stagnation_seconds=120., relative_gap=.01)
        self.assertFalse(certified_stagnation_stop(1700, 0, 90, 100, policy))
        self.assertFalse(certified_stagnation_stop(299, 0, 99, 99.5, policy))
        self.assertFalse(certified_stagnation_stop(400, 350, 99, 99.5, policy))
        self.assertFalse(certified_stagnation_stop(400, 0, 99, float('inf'), policy))
        self.assertFalse(certified_stagnation_stop(400, 0, 99, 98, policy))
        self.assertTrue(certified_stagnation_stop(400, 0, 99, 99.5, policy))

    def test_overlap_denominators(self):
        masks = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=bool)
        result = coverage_metrics(masks, [0, 1])
        self.assertEqual(result['coverage_pct'], 75.)
        self.assertEqual(result['overlap_pct'], 25.)
        self.assertAlmostEqual(result['overlap_of_covered_pct'], 100 / 3)
        self.assertEqual(result['excess_observations'], 1)
        self.assertIsNone(coverage_metrics(masks, [])['overlap_of_covered_pct'])

    def test_k8_frontier_and_bruteforce(self):
        rng = np.random.default_rng(31)
        masks = rng.random((8, 30)) < .25
        poses = np.column_stack([np.arange(8), np.zeros((8, 4))])
        targets = np.column_stack([np.arange(30), np.zeros((30, 2))])
        report = optimize_frontier(poses, masks, targets, list(range(1, 9)), seconds=3., alternatives=4)
        previous = 0
        for row in report['frontier']:
            k = row['budget_k']
            optimum = max(int(np.any(masks[list(ids)], axis=0).sum())
                for size in range(k + 1) for ids in combinations(range(8), size))
            self.assertEqual(row['covered_count'], optimum)
            self.assertGreaterEqual(row['covered_count'], previous)
            self.assertLessEqual(row['num_selected'], k)
            self.assertTrue(row['discrete_optimal'])
            self.assertTrue(row['near_optimal_certified'])
            for solution in row['alternatives']:
                self.assertEqual(coverage_metrics(masks, solution['selected_ids'])['overlap_pct'], solution['overlap_pct'])
            previous = row['covered_count']

    def test_empty_query_pool(self):
        report = optimize_frontier(np.empty((0, 5)), np.empty((0, 3), bool),
                                   np.eye(3), [1, 8], seconds=1.)
        self.assertEqual([r['covered_count'] for r in report['frontier']], [0, 0])
        self.assertTrue(all(r['discrete_optimal'] for r in report['frontier']))

    def test_solver_trace_and_guard(self):
        masks = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], bool)
        policy = dict(minimum_seconds=0., stagnation_seconds=0., relative_gap=.01, interval_seconds=.01)
        result = solve_cut_milp(np.packbits(masks, axis=1), 4, 2, [0], seconds=3., swap_seconds=0., stop_policy=policy)
        trace = result['quality_time_trace']
        self.assertGreaterEqual(len(trace), 2)
        self.assertEqual(trace[-1]['covered_count'], result['objective'])
        self.assertEqual([r['covered_count'] for r in trace], sorted(r['covered_count'] for r in trace))
        self.assertTrue(all(r['upper_bound_count'] + 1e-5 >= r['covered_count'] for r in trace))
        with self.assertRaises(ValueError):
            solve_cut_milp(np.packbits(masks, axis=1), 4, 2, [0], seconds=1.,
                           stop_policy=dict(policy, minimum_seconds=2.))

    def test_geometry_query_budget(self):
        poses = np.array([[i * 5., 0., 3., 90., -10.] for i in range(8)])
        targets = np.array([[i * 3., 4., 1.5] for i in range(10)])
        profile = LidarProfile('test', 360., 26.9, 30.)
        for method in ['geometry_single', 'geometry_union', 'geometry_portfolio_milp', 'random']:
            proposal = geometry_plan(poses, targets, profile, 5, [1, 2, 8], method, seconds=.1, seed=0)
            self.assertEqual(len(proposal['selected_ids']), 5)
            self.assertEqual(len(set(proposal['selected_ids'])), 5)
            self.assertGreaterEqual(proposal['planning_sec'], 0)


if __name__ == '__main__':
    unittest.main()

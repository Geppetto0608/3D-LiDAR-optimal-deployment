"""Exact submodular inequalities and lazy MILP versus exhaustive search."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods.coverage_cut_milp import PackedCoverage, solve_cut_milp, swap_start


class CoverageCutTests(unittest.TestCase):
    def test_cuts_upper_bound_all_subsets_and_are_tight_at_source(self):
        rng = np.random.default_rng(17)
        masks = rng.random((6, 13)) < .35
        oracle = PackedCoverage(np.packbits(masks, axis=1), 13)
        sets = [list(s) for k in range(7) for s in itertools.combinations(range(6), k)]
        for source in sets:
            gains = oracle.gains(oracle.union(source))
            base = oracle.count(source)
            self.assertEqual(base+int(gains[source].sum()), base)
            for target in sets:
                self.assertGreaterEqual(base+int(gains[target].sum()), oracle.count(target))

    def test_solver_equals_brute_force(self):
        rng = np.random.default_rng(26)
        for _ in range(3):
            masks = rng.random((8, 21)) < .25
            for k in [1, 2, 3]:
                optimum = max(int(masks[list(s)].any(axis=0).sum()) for n in range(1,k+1)
                              for s in itertools.combinations(range(8),n))
                r = solve_cut_milp(np.packbits(masks, axis=1), 21, k, [0], seconds=10, swap_seconds=0)
                self.assertEqual(r['objective'], optimum)
                self.assertAlmostEqual(r['certified_upper_bound'], optimum)
                self.assertTrue(r['discrete_optimal'])

    def test_padding_rejected(self):
        with self.assertRaisesRegex(ValueError, 'padding'):
            PackedCoverage(np.array([[255]], dtype=np.uint8), 3)

    def test_swap_preserves_budget_and_improves(self):
        oracle = PackedCoverage(np.packbits(np.eye(5, dtype=bool), axis=1), 5)
        ids, record = swap_start(oracle, 3, [0], seconds=1)
        self.assertEqual(len(ids), 3)
        self.assertEqual(record['coverage_count'], 3)


if __name__ == '__main__':
    unittest.main()

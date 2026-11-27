"""Negative controls for the independent experiment-validation harness."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_rsu_stress import compare_archive, plan_signature, require


class StressValidationTest(unittest.TestCase):
    def setUp(self):
        self.problem = dict(poses=np.array([[0., 0., 3., 0., -10.], [4., 0., 3., 0., -10.]]),
                            targets=np.arange(27.).reshape(9, 3), critical=np.ones(9, dtype=bool))
        self.masks = np.array([[True] * 9, [False] * 9])
        self.archive = dict(self.problem, packed_masks=np.packbits(self.masks, axis=1))
        self.provider = SimpleNamespace(mask=lambda pose: self.masks[int(pose[0] / 4)])

    def test_full_archive_matches(self):
        result = compare_archive(self.problem, self.archive, self.provider)
        self.assertEqual(result['candidate_masks'], 2)
        self.assertEqual(result['candidate_target_pairs'], 18)
        self.assertEqual(result['visible_edges'], 9)
        self.assertEqual(result['mismatches'], 0)

    def test_corrupted_visibility_is_rejected(self):
        self.archive['packed_masks'][1, 0] ^= 128
        with self.assertRaisesRegex(ValueError, 'candidate=1'):
            compare_archive(self.problem, self.archive, self.provider)

    def test_nonzero_padding_is_rejected(self):
        self.archive['packed_masks'][0, -1] |= 1
        with self.assertRaisesRegex(ValueError, 'Direct mask differs'):
            compare_archive(self.problem, self.archive, self.provider)

    def test_target_or_critical_change_is_rejected(self):
        for name in ('poses', 'targets', 'critical'):
            with self.subTest(name=name):
                modified = dict(self.archive)
                modified[name] = self.archive[name].copy()
                modified[name].flat[0] = not modified[name].flat[0] if name == 'critical' else 99.
                with self.assertRaisesRegex(ValueError, 'Regenerated ' + name):
                    compare_archive(self.problem, modified, self.provider)

    def test_checks_are_not_python_assert_statements(self):
        with self.assertRaises(ValueError):
            require(False, 'bad input')

    def test_plan_comparison_excludes_only_timing(self):
        a = [dict(selected_ids=[1, 2], actual_q=2, planning_sec=1.)]
        b = [dict(selected_ids=[1, 2], actual_q=2, planning_sec=9.)]
        self.assertEqual(plan_signature(a), plan_signature(b))
        b[0]['selected_ids'] = [1, 3]
        self.assertNotEqual(plan_signature(a), plan_signature(b))


if __name__ == '__main__':
    unittest.main()

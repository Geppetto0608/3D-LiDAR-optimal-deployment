"""Contract validation and bounded source-library frontier tests."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.deployment_library import build_frontier, minimum_sensor_bracket


class DeploymentLibraryTest(unittest.TestCase):
    def setUp(self):
        self.poses = np.array([[i, 0., 3., 90., -10.] for i in range(3)])
        self.targets = np.array([[i, 1., 1.5] for i in range(4)])
        self.masks = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]], dtype=bool)

    def report(self, **kwargs):
        return build_frontier(self.poses, self.masks, self.targets, [1, 2, 3], **kwargs)

    def test_exact_saturation_and_minimum(self):
        result = self.report()
        self.assertEqual([r['covered_count'] for r in result['frontier']], [2, 4, 4])
        self.assertEqual([r['num_selected'] for r in result['frontier']], [1, 2, 2])
        self.assertEqual([r['certified_gap'] for r in result['frontier']], [0., 0., 0.])
        self.assertEqual(minimum_sensor_bracket(result, 4)['status'], 'certified_minimum')
        self.assertEqual(minimum_sensor_bracket(result, 4)['upper_bound_sensors'], 2)
        self.assertEqual(minimum_sensor_bracket(result, 0)['upper_bound_sensors'], 0)
        json.dumps(result, allow_nan=False)

    def test_invalid_arrays_and_budgets(self):
        for ks in ([], [2, 1], [1, 1], [True], [0], [1.5]):
            with self.assertRaises(ValueError):
                build_frontier(self.poses, self.masks, self.targets, ks)
        for masks in (self.masks.astype(int), self.masks[:, :2]):
            with self.assertRaises(ValueError):
                build_frontier(self.poses, masks, self.targets, [1])
        for options in ({'enumeration_limit': -1}, {'alternatives_per_k': 0},
                        {'context': {'nan': float('nan')}}):
            with self.assertRaises(ValueError):
                self.report(**options)
        with self.assertRaises(ValueError):
            build_frontier(np.repeat(self.poses[:1], 3, axis=0), self.masks, self.targets, [1])

    def test_proposals_cannot_smuggle_certificates_or_bad_masks(self):
        for proposal in ({'selected_ids': [0, 0]}, {'selected_ids': [3]},
                         {'selected_ids': [0, 1], 'budget_k': 1},
                         {'selected_ids': [0], 'covered_count': 4},
                         {'selected_ids': [0], 'upper_bound': 2},
                         {'selected_ids': [True]}):
            with self.assertRaises(ValueError):
                self.report(proposals=[proposal])

    def test_empty_or_zero_coverage_domain(self):
        for poses, masks in ((self.poses, np.zeros_like(self.masks)),
                             (np.empty((0, 5)), np.empty((0, 4), dtype=bool))):
            result = build_frontier(poses, masks, self.targets, [1, 3], enumeration_limit=0)
            self.assertTrue(all(r['num_selected'] == 0 for r in result['frontier']))
            self.assertTrue(all(r['certified_gap'] == 0 for r in result['frontier']))
            bracket = minimum_sensor_bracket(result, 1)
            self.assertIsNone(bracket['upper_bound_sensors'])
            self.assertEqual(bracket['status'], 'unattainable_in_declared_domain')
            self.assertEqual(result['minimum_to_candidate_ceiling']['upper_bound_sensors'], 0)

    def test_ceiling_is_not_necessarily_all_targets(self):
        masks = self.masks.copy()
        masks[:, -1] = False
        result = build_frontier(self.poses, masks, self.targets, [1, 2, 3])
        self.assertEqual(result['coverage_ceiling_count'], 3)
        self.assertEqual(result['minimum_to_candidate_ceiling']['upper_bound_sensors'], 2)
        self.assertEqual(result['minimum_to_all_targets']['status'], 'unattainable_in_declared_domain')

    def test_domain_fingerprint_covers_masks_targets_and_context(self):
        original = self.report(context={'track': 'A'})['domain_sha256']
        self.assertEqual(original, self.report(context={'track': 'A'})['domain_sha256'])
        self.assertNotEqual(original, self.report(context={'track': 'B'})['domain_sha256'])
        changed = self.masks.copy()
        changed[0, 2] = True
        self.assertNotEqual(original, build_frontier(self.poses, changed, self.targets, [1],
                                                   context={'track': 'A'})['domain_sha256'])
        self.assertNotEqual(original, build_frontier(self.poses, self.masks, self.targets + 1.,
                                                   [1], context={'track': 'A'})['domain_sha256'])

    def test_minimum_bracket_rejects_invalid_threshold_and_inconsistency(self):
        result = self.report()
        for required in (-1, 5, True, 1.5):
            with self.assertRaises(ValueError):
                minimum_sensor_bracket(result, required)
        broken = copy.deepcopy(result)
        broken['frontier'][-1]['upper_bound_count'] = 0
        with self.assertRaises(ValueError):
            minimum_sensor_bracket(broken, 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)

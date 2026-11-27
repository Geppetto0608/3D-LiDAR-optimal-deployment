"""Independent finite-domain audit; no geometry, raycasting, solver, or files.

Output assumptions: IDs index the supplied rows; alternatives include the
incumbent first; coverage percentages use all targets; gaps are relative to the
incumbent. Proposals carry layouts and optional checked counts, never solver
certificates. Thresholds are integer covered counts. Physical verification and
frozen source/config/geometry provenance belong to the separate builder.
"""

from copy import deepcopy
from itertools import combinations
import json
from math import comb
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.deployment_library import build_frontier, minimum_sensor_bracket


GREEDY_TRAP = (
    (1, 1, 1, 1, 0, 0),
    (1, 1, 0, 0, 1, 0),
    (0, 0, 1, 1, 0, 1),
)
DISJOINT = (
    (1, 1, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0),
    (0, 0, 0, 0, 1, 1),
)


def domain(binary_masks):
    masks = np.asarray(binary_masks, dtype=bool)
    m, n = masks.shape
    poses = np.asarray(
        [[10.0 * i, 5.0, 3.0, 30.0 * i, -10.0] for i in range(m)],
        dtype=float,
    ).reshape(m, 5)
    targets = np.asarray([[float(j), 0.0, 0.5] for j in range(n)])
    return poses, masks, targets


def covered_count(masks, ids):
    # Set union is deliberately independent of the core's numpy reductions.
    return len({j for i in ids for j, covered in enumerate(masks[i]) if covered})


def oracle(masks, k):
    layouts = [
        (ids, covered_count(masks, ids))
        for size in range(min(k, len(masks)) + 1)
        for ids in combinations(range(len(masks)), size)
    ]
    return sorted(layouts, key=lambda item: (-item[1], len(item[0]), item[0]))


def solution_key(solution):
    return (-solution['covered_count'], solution['num_selected'],
            tuple(solution['selected_ids']))


class LibraryContractAuditTests(unittest.TestCase):
    def assert_solution(self, solution, poses, masks, k):
        fields = {'selected_ids', 'poses', 'num_selected', 'covered_count',
                  'coverage_pct', 'origin'}
        self.assertTrue(fields <= solution.keys())
        ids = solution['selected_ids']
        self.assertIsInstance(ids, list)
        self.assertEqual(ids, sorted(set(ids)))
        self.assertTrue(all(type(i) is int and 0 <= i < len(poses) for i in ids))
        self.assertEqual(solution['num_selected'], len(ids))
        self.assertLessEqual(len(ids), k)
        self.assertEqual(solution['poses'], poses[ids].tolist())
        count = covered_count(masks, ids)
        self.assertEqual(solution['covered_count'], count)
        self.assertAlmostEqual(solution['coverage_pct'], 100.0 * count / masks.shape[1])
        self.assertIsInstance(solution['origin'], str)
        self.assertTrue(solution['origin'])

    def assert_report(self, report, poses, masks, targets, ks, keep):
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual(report['schema'], 'deployment_library_frontier_v1')
        self.assertEqual(report['budget_mode'], 'at_most_k')
        self.assertEqual(report['objective'], 'unweighted_union_count')
        self.assertEqual(report['candidate_count'], len(poses))
        self.assertEqual(report['target_count'], len(targets))
        self.assertEqual(report['k_values'], list(ks))
        self.assertIsInstance(report['context'], dict)
        self.assertRegex(report['domain_sha256'], r'^[0-9a-f]{64}$')
        ceiling = covered_count(masks, range(len(masks)))
        self.assertEqual(report['coverage_ceiling_count'], ceiling)
        self.assertAlmostEqual(report['coverage_ceiling_pct'], 100.0 * ceiling / len(targets))
        self.assertEqual(len(report['frontier']), len(ks))
        previous_count, previous_solutions = 0, []
        for k, row in zip(ks, report['frontier']):
            self.assertEqual(row['budget_k'], k)
            self.assert_solution(row, poses, masks, k)
            count = row['covered_count']
            optimum = oracle(masks, k)[0][1]
            self.assertLessEqual(count, optimum)
            self.assertGreaterEqual(row['upper_bound_count'], optimum)
            self.assertLessEqual(row['upper_bound_count'], ceiling)
            self.assertEqual(row['marginal_covered_count'], count - previous_count)
            self.assertGreaterEqual(count, previous_count)
            for previous in previous_solutions:
                self.assertLessEqual(solution_key(row), solution_key(previous))
            if row['upper_bound_count'] == count:
                self.assertEqual(row['certification'], 'discrete_optimal')
                self.assertEqual(count, optimum)
                self.assertEqual(row['certified_gap'], 0)
            else:
                self.assertEqual(row['certification'], 'best_known')
                if count:
                    self.assertAlmostEqual(row['certified_gap'],
                                           (row['upper_bound_count'] - count) / count)
                else:
                    self.assertIsNone(row['certified_gap'])
            alternatives = row['alternatives']
            self.assertGreaterEqual(len(alternatives), 1)
            self.assertLessEqual(len(alternatives), keep)
            self.assertEqual(alternatives[0]['selected_ids'], row['selected_ids'])
            seen = set()
            for alternative in alternatives:
                self.assert_solution(alternative, poses, masks, k)
                self.assertLessEqual(solution_key(row), solution_key(alternative))
                ids = tuple(alternative['selected_ids'])
                self.assertNotIn(ids, seen)
                seen.add(ids)
            previous_solutions.extend([row, *alternatives])
            previous_count = count
        for field, required in [('minimum_to_candidate_ceiling', ceiling),
                                ('minimum_to_all_targets', len(targets))]:
            self.assertEqual(report[field], minimum_sensor_bracket(report, required))
            self.assertEqual(report[field]['domain_sha256'], report['domain_sha256'])

    def build(self, binary_masks, ks, **kwargs):
        poses, masks, targets = domain(binary_masks)
        report = build_frontier(poses, masks, targets, ks, **kwargs)
        self.assert_report(report, poses, masks, targets, ks,
                           kwargs.get('alternatives_per_k', 3))
        return report

    def test_complete_enumeration_matches_independent_oracle_and_ties(self):
        fixtures = [GREEDY_TRAP, DISJOINT,
                    ((1, 1, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0)),
                    ((0, 0, 0), (0, 0, 0))]
        for fixture in fixtures:
            with self.subTest(masks=fixture):
                report = self.build(fixture, [1, 2, 4], alternatives_per_k=20)
                masks = np.asarray(fixture, dtype=bool)
                self.assertTrue(report['enumeration']['completed'])
                for row in report['frontier']:
                    expected = oracle(masks, row['budget_k'])
                    self.assertEqual(row['selected_ids'], list(expected[0][0]))
                    self.assertEqual(row['covered_count'], expected[0][1])
                    actual = [(tuple(a['selected_ids']), a['covered_count'])
                              for a in row['alternatives']]
                    self.assertEqual(actual, expected[:20])

    def test_at_most_k_retains_fewer_sensors_after_saturation(self):
        fixture = ((1, 1, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0))
        for limit in (0, 100):
            with self.subTest(enumeration_limit=limit):
                report = self.build(fixture, [1, 2, 3, 5], enumeration_limit=limit)
                for row in report['frontier']:
                    self.assertEqual(row['selected_ids'], [0])
                    self.assertEqual(row['num_selected'], 1)
                self.assertEqual([r['marginal_covered_count'] for r in report['frontier']],
                                 [3, 0, 0, 0])

    def test_empty_deployment_beats_useless_sensors_and_handles_empty_domain(self):
        for fixture in (np.zeros((3, 4), dtype=bool), np.zeros((0, 4), dtype=bool)):
            for limit in (0, 100):
                with self.subTest(candidate_count=len(fixture), enumeration_limit=limit):
                    report = self.build(fixture, [1, 3], enumeration_limit=limit)
                    for row in report['frontier']:
                        self.assertEqual(row['selected_ids'], [])
                        self.assertEqual(row['poses'], [])
                        self.assertEqual(row['num_selected'], 0)
                        self.assertEqual(row['coverage_pct'], 0)
                    minimum = report['minimum_to_candidate_ceiling']
                    self.assertEqual(minimum['status'], 'certified_minimum')
                    self.assertEqual(minimum['upper_bound_sensors'], 0)

    def test_imported_incumbent_survives_worse_larger_budget_proposals(self):
        fixture = (
            (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
            (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0),
            (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
        )
        proposals = [
            dict(selected_ids=[4, 2, 3], budget_k=3, covered_count=12, origin='imported_best'),
            dict(selected_ids=[0, 1], budget_k=4, covered_count=9, origin='later_worse'),
            dict(selected_ids=[4, 3, 2, 1, 0], budget_k=5, origin='redundant'),
            dict(selected_ids=[], budget_k=5, origin='empty_import'),
        ]
        baseline = self.build(fixture, [1, 3, 4, 5], enumeration_limit=0)
        self.assertEqual(baseline['frontier'][2]['covered_count'], 11)
        outputs = []
        for ordering in (proposals, list(reversed(proposals))):
            report = self.build(fixture, [1, 3, 4, 5], enumeration_limit=0,
                                proposals=ordering, alternatives_per_k=8)
            self.assertEqual([r['covered_count'] for r in report['frontier']], [6, 12, 12, 12])
            for row in report['frontier'][1:]:
                self.assertEqual(row['selected_ids'], [2, 3, 4])
            outputs.append([r['selected_ids'] for r in report['frontier']])
        self.assertEqual(outputs[0], outputs[1])

    def test_proposal_eligibility_uses_actual_size_not_source_budget(self):
        report = self.build(GREEDY_TRAP, [1, 2, 3], enumeration_limit=0,
                            proposals=[dict(selected_ids=[2, 1], budget_k=7,
                                            origin='saved_at_larger_budget')])
        self.assertEqual(report['frontier'][0]['selected_ids'], [0])
        self.assertEqual(report['frontier'][1]['selected_ids'], [1, 2])
        self.assertEqual(report['frontier'][2]['selected_ids'], [1, 2])

    def test_non_enumeration_bounds_dominate_bruteforce_optimum(self):
        fixtures = [GREEDY_TRAP, DISJOINT,
                    ((1, 1, 0, 0), (1, 0, 1, 0), (0, 1, 1, 0)),
                    ((0, 0, 0), (0, 0, 0))]
        for fixture in fixtures:
            with self.subTest(masks=fixture):
                report = self.build(fixture, [1, 2, 3, 5], enumeration_limit=0)
                self.assertFalse(report['enumeration']['completed'])
                self.assertEqual(report['enumeration']['limit'], 0)

    def test_greedy_trap_is_best_known_without_a_matching_bound(self):
        report = self.build(GREEDY_TRAP, [1, 2], enumeration_limit=0)
        row = report['frontier'][1]
        self.assertEqual(row['selected_ids'], [0, 1])
        self.assertEqual(row['covered_count'], 5)
        self.assertEqual(row['upper_bound_count'], 6)
        self.assertEqual(row['certification'], 'best_known')
        self.assertAlmostEqual(row['certified_gap'], 0.2)
        repeated = self.build(GREEDY_TRAP, [1, 2], enumeration_limit=0)
        self.assertEqual(report['frontier'], repeated['frontier'])

    def test_enumeration_guard_counts_empty_subset_and_is_all_or_nothing(self):
        total = sum(comb(3, size) for size in range(3))
        self.assertEqual(total, 7)
        for limit, completed, count in [(total - 1, False, 5), (total, True, 6)]:
            with self.subTest(limit=limit):
                report = self.build(GREEDY_TRAP, [1, 2], enumeration_limit=limit)
                self.assertEqual(report['enumeration'], dict(
                    completed=completed, subset_count=total, limit=limit,
                    reason='complete' if completed else 'subset_limit'))
                self.assertEqual(report['frontier'][1]['covered_count'], count)

    def test_time_limited_layout_origin_does_not_supply_a_certificate(self):
        for origin in ('gurobi_time_limit', 'best_known_import'):
            with self.subTest(origin=origin):
                report = self.build(GREEDY_TRAP, [2], enumeration_limit=0,
                                    proposals=[dict(selected_ids=[0, 1], budget_k=2,
                                                    covered_count=5, origin=origin)])
                self.assertEqual(report['frontier'][0]['certification'], 'best_known')
                self.assertEqual(report['frontier'][0]['upper_bound_count'], 6)
        # A fresh analytic proof may certify even a time-limited source layout.
        report = self.build(GREEDY_TRAP, [2], enumeration_limit=0,
                            proposals=[dict(selected_ids=[1, 2], origin='gurobi_time_limit')])
        self.assertEqual(report['frontier'][0]['certification'], 'discrete_optimal')

    def test_solver_claims_and_incorrect_proposal_counts_are_rejected(self):
        poses, masks, targets = domain(GREEDY_TRAP)
        for field, value in [('certification', 'discrete_optimal'),
                             ('solver_status_code', 2), ('best_bound', 5),
                             ('certified_gap', 0), ('covered_count', 6)]:
            with self.subTest(field=field):
                proposal = dict(selected_ids=[0, 1], budget_k=2, origin='untrusted')
                proposal[field] = value
                with self.assertRaises(ValueError):
                    build_frontier(poses, masks, targets, [2], enumeration_limit=0,
                                   proposals=[proposal])

    def test_equal_masks_keep_geometrically_distinct_alternatives(self):
        fixture = ((1, 1, 0), (1, 1, 0), (0, 0, 1))
        for limit in (0, 100):
            with self.subTest(enumeration_limit=limit):
                report = self.build(fixture, [1], enumeration_limit=limit,
                                    alternatives_per_k=4,
                                    proposals=[dict(selected_ids=[1], origin='different_geometry')])
                row = report['frontier'][0]
                layouts = {tuple(a['selected_ids']): a for a in row['alternatives']}
                self.assertEqual(row['selected_ids'], [0])
                self.assertIn((1,), layouts)
                self.assertEqual(layouts[(0,)]['covered_count'], layouts[(1,)]['covered_count'])
                self.assertNotEqual(layouts[(0,)]['poses'], layouts[(1,)]['poses'])

    def test_alternatives_need_not_be_near_optimal(self):
        report = self.build(((1, 1, 1), (1, 0, 0), (0, 0, 0)), [1],
                            alternatives_per_k=4)
        row = report['frontier'][0]
        self.assertEqual(row['covered_count'], 3)
        self.assertTrue(any(a['covered_count'] == 0 for a in row['alternatives']))

    def test_missing_k2_does_not_prove_two_sensors_suffice(self):
        for fixture, true_minimum in [(GREEDY_TRAP, 2), (DISJOINT, 3)]:
            with self.subTest(true_minimum=true_minimum):
                report = self.build(fixture, [1, 3], enumeration_limit=0)
                masks = np.asarray(fixture, dtype=bool)
                independent_minimum = min(len(ids) for ids, count in oracle(masks, len(masks))
                                          if count == masks.shape[1])
                self.assertEqual(independent_minimum, true_minimum)
                bracket = minimum_sensor_bracket(report, masks.shape[1])
                self.assertEqual(bracket['lower_bound_sensors'], 2)
                self.assertEqual(bracket['upper_bound_sensors'], 3)
                self.assertEqual(bracket['status'], 'bounded')
                self.assertEqual([r['budget_k'] for r in report['frontier']], [1, 3])
                self.assertEqual(report['frontier'][1]['marginal_covered_count'],
                                 report['frontier'][1]['covered_count'] - report['frontier'][0]['covered_count'])

    def test_k2_proof_or_two_sensor_witness_can_close_minimum_bracket(self):
        cases = [(GREEDY_TRAP, [1, 3], 2), (DISJOINT, [1, 2, 3], 3)]
        for fixture, ks, expected in cases:
            with self.subTest(ks=ks, expected=expected):
                report = self.build(fixture, ks)
                bracket = minimum_sensor_bracket(report, 6)
                self.assertEqual(bracket['lower_bound_sensors'], expected)
                self.assertEqual(bracket['upper_bound_sensors'], expected)
                self.assertEqual(bracket['status'], 'certified_minimum')

    def test_candidate_ceiling_below_all_targets_is_not_full_coverage(self):
        report = self.build(((1, 1, 0, 0), (1, 0, 1, 0), (0, 1, 1, 0)), [1, 2, 3])
        self.assertEqual(report['coverage_ceiling_count'], 3)
        self.assertEqual(report['coverage_ceiling_pct'], 75.0)
        ceiling = report['minimum_to_candidate_ceiling']
        self.assertEqual(ceiling['status'], 'certified_minimum')
        self.assertEqual(ceiling['lower_bound_sensors'], 2)
        self.assertEqual(ceiling['upper_bound_sensors'], 2)
        all_targets = report['minimum_to_all_targets']
        self.assertEqual(all_targets['status'], 'unattainable_in_declared_domain')
        self.assertIsNone(all_targets['lower_bound_sensors'])
        self.assertIsNone(all_targets['upper_bound_sensors'])

    def test_equal_incumbent_plateau_is_not_a_candidate_ceiling_proof(self):
        fixture = (
            (1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
            (0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0),
            (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        )
        report = self.build(fixture, [2, 3], enumeration_limit=0,
                            proposals=[dict(selected_ids=[2, 3], origin='imported_plateau')])
        self.assertEqual([r['covered_count'] for r in report['frontier']], [12, 12])
        self.assertEqual(report['frontier'][1]['marginal_covered_count'], 0)
        self.assertEqual(report['frontier'][1]['certification'], 'best_known')
        self.assertEqual(report['coverage_ceiling_count'], 13)
        self.assertEqual(oracle(np.asarray(fixture, dtype=bool), 3)[0][1], 13)
        bracket = report['minimum_to_candidate_ceiling']
        self.assertEqual(bracket['lower_bound_sensors'], 3)
        self.assertIsNone(bracket['upper_bound_sensors'])
        self.assertEqual(bracket['status'], 'not_reached_in_saved_solutions')

    def test_thresholds_are_integer_counts_and_zero_requires_no_sensors(self):
        report = self.build(GREEDY_TRAP, [1, 3], enumeration_limit=0)
        zero = minimum_sensor_bracket(report, 0)
        self.assertEqual((zero['lower_bound_sensors'], zero['upper_bound_sensors']), (0, 0))
        self.assertEqual(zero['status'], 'certified_minimum')
        for invalid in (True, -1, 1.5, '100%', 7):
            with self.subTest(required=invalid):
                with self.assertRaises(ValueError):
                    minimum_sensor_bracket(report, invalid)

    def test_domain_hash_binds_masks_geometry_targets_and_context_without_mutation(self):
        poses, masks, targets = domain(GREEDY_TRAP)
        context = {'fixture': 'audit', 'provenance': {'source': 'hardcoded_masks'}}
        snapshots = (poses.copy(), masks.copy(), targets.copy(), deepcopy(context))
        report = build_frontier(poses, masks, targets, [1, 2], context=context)
        self.assert_report(report, poses, masks, targets, [1, 2], 3)
        repeated = build_frontier(poses.copy(), masks.copy(), targets.copy(), [1, 2],
                                  context=deepcopy(context))
        self.assertEqual(report['domain_sha256'], repeated['domain_sha256'])
        for field in ('poses', 'masks', 'targets', 'context'):
            with self.subTest(changed=field):
                p, m, t, c = poses.copy(), masks.copy(), targets.copy(), deepcopy(context)
                if field == 'poses':
                    p[0, 0] += 0.125
                elif field == 'masks':
                    m[0, 0] = not m[0, 0]
                elif field == 'targets':
                    t[0, 2] += 0.125
                else:
                    c['provenance']['source'] = 'another_fixture'
                changed = build_frontier(p, m, t, [1, 2], context=c)
                self.assertNotEqual(report['domain_sha256'], changed['domain_sha256'])
        for actual, expected in zip((poses, masks, targets), snapshots[:3]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(context, snapshots[3])
        report['context']['provenance']['source'] = 'changed_report_only'
        self.assertEqual(context, snapshots[3])

    def test_invalid_k_sequences_are_rejected(self):
        poses, masks, targets = domain(GREEDY_TRAP)
        for ks in ([], [0], [-1], [True], [1.0], [2, 1], [1, 1]):
            with self.subTest(k_values=ks):
                with self.assertRaises(ValueError):
                    build_frontier(poses, masks, targets, ks)


if __name__ == '__main__':
    unittest.main(verbosity=2)

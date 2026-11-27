"""Exact small-problem checks for the opt-in RSU stress benchmark."""
from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_rsu_stress as study
from src.methods.rsu_coverage_milp import assess, compress, solve_frontier
from src.roadside.rsu_stress import domain, retrieve, scene, transfer_ids


class ConstrainedCoverageTest(unittest.TestCase):
    def test_exact_against_enumeration(self):
        poses = np.array([[x, 0, z, 0, -10] for x, z in
                          [(0, 3), (0, 5), (2, 3), (4, 3), (8, 3), (12, 3), (16, 3)]], float)
        critical = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
        for seed in range(3):
            masks = np.random.default_rng(seed).random((len(poses), len(critical))) < .5
            for overlap in (0., .34):
                req = dict(overlap_ratio=overlap, critical_overlap_ratio=overlap, min_separation_m=3.)
                result = solve_frontier(poses, masks, critical, [1, 2, 3], req,
                                        seconds=10., alternatives=3, stop_gap=0.)
                for row in result['frontier']:
                    k = row['budget_k']
                    feasible = [assess(poses, masks, ids, critical, req, k)
                                for size in range(k+1) for ids in combinations(range(len(poses)), size)]
                    feasible = [v for v in feasible if v['feasible']]
                    if not feasible:
                        self.assertEqual(row['status'], 'infeasible')
                        self.assertIsNone(row['metrics'])
                    else:
                        expected = max(v['covered_count'] for v in feasible)
                        self.assertEqual(row['metrics']['covered_count'], expected)
                        self.assertTrue(row['discrete_optimal'])
                        self.assertGreaterEqual(row['upper_bound_count']+1e-6, expected)
                        for entry in row['alternatives']:
                            self.assertTrue(assess(poses, masks, entry['selected_ids'], critical, req, k)['feasible'])

    def test_one_pole_cannot_provide_redundancy(self):
        poses = np.array([[0, 0, 3, 0, -10], [0, 0, 5, 0, -10]], float)
        masks = np.ones((2, 3), bool)
        critical = np.ones(3, bool)
        req = dict(overlap_ratio=1., critical_overlap_ratio=1., min_separation_m=0.)
        row = solve_frontier(poses, masks, critical, [2], req, seconds=5.)['frontier'][0]
        self.assertEqual(row['status'], 'infeasible')
        self.assertIn('one_sensor_per_pole', assess(poses, masks, [0, 1], critical, req, 2)['violations'])

    def test_empty_pool_not_missing_zero(self):
        poses, masks, critical = np.empty((0, 5)), np.empty((0, 4), bool), np.ones(4, bool)
        for overlap in (0., .5):
            req = dict(overlap_ratio=overlap, critical_overlap_ratio=0., min_separation_m=4.)
            row = solve_frontier(poses, masks, critical, [1], req, seconds=5.)['frontier'][0]
            self.assertEqual(row['status'], 'feasible' if overlap == 0 else 'infeasible')
            if overlap == 0:
                self.assertEqual(row['metrics']['covered_count'], 0)
            else:
                self.assertIsNone(row['metrics'])

    def test_lossless_critical_compression(self):
        masks = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]], bool)
        critical = np.array([1, 0, 0, 0, 0], bool)
        matrix, weights, flags = compress(masks, critical)
        self.assertEqual(len(weights), 3)
        for selected in ([0], [1], [0, 1], []):
            density = masks[selected].sum(axis=0)
            selection = np.zeros(2); selection[selected] = 1
            grouped = matrix @ selection
            self.assertEqual(weights[grouped > 0].sum(), np.count_nonzero(density))
            self.assertEqual(weights[(grouped > 0) & flags].sum(), np.count_nonzero((density > 0) & critical))


class DomainAndPlanningTest(unittest.TestCase):
    def test_checked_in_spec_matches_defaults(self):
        path = Path(study.__file__).parent/'configs/library_rsu_stress_20261123.json'
        self.assertEqual(json.loads(path.read_text()), study.default_spec())

    def test_dry_run_no_execution_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'not_created'
            with patch.object(study, 'solve_frontier', side_effect=AssertionError('solve in dry run')):
                with patch.object(study, 'query', side_effect=AssertionError('raycast in dry run')):
                    with redirect_stdout(StringIO()):
                        study.main(['--smoke', '--output', str(output)])
            self.assertFalse(output.exists())
        with self.assertRaises(SystemExit):
            study.main(['--worker', '/tmp/no-run'])

    def test_scenes_keep_support_and_direct_visibility_matches(self):
        spec = study.smoke_spec()
        for d in spec['targets']:
            problem = domain(d, spec['sampling'])
            targets, poses = problem['targets'].copy(), problem['poses'].copy()
            masks = []
            for level in ('clear', 'mixed', 'dense'):
                mesh = scene(problem, level)
                ids = list(range(min(3, len(poses))))
                fast, _, work = study.query(mesh, problem, ids)
                direct, _, _ = study.query(mesh, problem, ids, direct=True)
                self.assertTrue(np.array_equal(fast, direct))
                self.assertGreater(work.get('rays_submitted', 0), 0)
                masks.append(fast)
                self.assertTrue(np.array_equal(problem['targets'], targets))
                self.assertTrue(np.array_equal(problem['poses'], poses))
            self.assertFalse(np.any(masks[1] & ~masks[0]))
            self.assertFalse(np.any(masks[2] & ~masks[1]))

    def test_large_map_and_geometric_transfer(self):
        spec = study.default_spec()
        sampling = study.smoke_spec()['sampling']
        for d in (spec['targets'][1], spec['targets'][-1]):
            target = domain(d, sampling)
            sources = retrieve(spec['sources'], d, 3)
            self.assertEqual(len(sources), 3)
            source = domain(sources[0], sampling)
            canonical = source['canonical']
            if isinstance(canonical, np.ndarray):
                canonical = canonical.tolist()
            data = dict(descriptor=sources[0], canonical=canonical)
            ids, records = transfer_ids(data, [[0, 1], [1, 2]], target)
            self.assertTrue(ids)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(0 <= i < len(target['poses']) for i in ids))
            self.assertEqual(len(records), 2)
            self.assertGreater(np.ptp(target['targets'][:, 0]), 100.)

    def test_shortlists_are_unique_and_nested(self):
        masks = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], bool)
        poses = np.zeros((4, 5)); poses[:, 0] = np.arange(4)*4
        for double in (False, True):
            full = study.shortlist(masks, poses, 4, double=double)
            self.assertEqual(len(set(full)), 4)
            self.assertEqual(study.shortlist(masks, poses, 2, double=double), full[:2])

    def test_solution_pool_callback_bound_regression(self):
        spec = study.smoke_spec()
        problem = domain(spec['targets'][1], spec['sampling'])
        masks = study._optical_masks(problem['poses'].astype(np.float32),
            problem['targets'].astype(np.float32), study._profile_values(problem['profile']))
        req = spec['requirements']['union']
        result = solve_frontier(problem['poses'], masks, problem['critical'], [2, 4], req,
                                seconds=2., alternatives=6, stop_min_sec=.1, stop_stagnation_sec=.1)
        for row in result['frontier']:
            k = row['budget_k']
            values = [assess(problem['poses'], masks, ids, problem['critical'], req, k)
                      for size in range(k+1) for ids in combinations(range(len(masks)), size)]
            optimum = max(v['covered_count'] for v in values if v['feasible'])
            for point in row['quality_time_trace']:
                self.assertGreaterEqual(point['upper']+1e-6, optimum)
            self.assertEqual(row['metrics']['covered_count'], optimum)


if __name__ == '__main__':
    unittest.main()

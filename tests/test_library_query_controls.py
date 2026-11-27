"""Source-free query planning, optical parity, and exact portfolio policy."""
import ast
import copy
import inspect
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment import LidarProfile
from src.methods import library_query_controls as controls


METHODS = ('geometry_single', 'geometry_union', 'geometry_portfolio', 'random')


class QueryControlTests(unittest.TestCase):
    def setUp(self):
        self.poses = np.array([[0, 0, 0, 0, 0], [0, .1, 0, 0, 0],
                               [0, 0, 0, 180, 0], [0, 0, 0, 90, 0]], float)
        self.targets = np.array([[2, 0, 0], [3, 0, 0], [4, 0, 0],
                                 [-2, 0, 0], [-3, 0, 0], [0, 2, 0]], float)
        self.profile = LidarProfile('test', 60., 60., 10.)

    def plan(self, method='geometry_union', q=2, **kwargs):
        args = dict(poses=self.poses, targets=self.targets, profile=self.profile,
                    query_budget=q, k_values=[1, 2], method=method)
        args.update(kwargs)
        return controls.plan_control(**args)

    def test_single_union_diversity_and_accounting(self):
        single = self.plan('geometry_single')
        union = self.plan('geometry_union')
        self.assertEqual(single['query_order'], [0, 1])
        self.assertEqual(union['query_order'], [0, 2])
        for result in (single, union):
            self.assertEqual(result['optical_candidate_evaluations'], 4)
            self.assertEqual(result['optical_point_tests'], 24)
            self.assertGreaterEqual(result['planning_sec'], 0.)
            self.assertIn('occlusion', json.dumps(result['semantics']))
            json.dumps(result, allow_nan=False)

    def test_random_exact_policy_and_no_optics(self):
        with patch.object(controls, 'compute_view_angles_numpy',
                          side_effect=AssertionError('random must not evaluate optics')), \
                patch.object(controls, 'build_frontier',
                             side_effect=AssertionError('random must not optimize')):
            for seed in range(30):
                expected = sorted(np.random.default_rng(seed).choice(4, 2, replace=False).tolist())
                result = self.plan('random', seed=seed)
                self.assertEqual(result['selected_ids'], expected)
                self.assertEqual(result['query_order'], expected)
                self.assertEqual(result['seed'], seed)
                self.assertEqual(result['optical_candidate_evaluations'], 0)
                self.assertEqual(result['optical_point_tests'], 0)

    def test_determinism_rng_and_inputs_untouched(self):
        before = np.random.get_state()
        poses, targets, profile = self.poses.copy(), self.targets.copy(), copy.deepcopy(self.profile)
        self.poses.setflags(write=False)
        self.targets.setflags(write=False)
        for method in METHODS:
            first, second = self.plan(method, seed=7), self.plan(method, seed=7)
            first.pop('planning_sec')
            second.pop('planning_sec')
            self.assertEqual(first, second)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(self.poses, poses)
        np.testing.assert_array_equal(self.targets, targets)
        self.assertEqual(self.profile, profile)

    def test_exact_cardinality_zero_full_and_no_gain(self):
        for method in METHODS:
            for q in range(5):
                for targets in (self.targets, self.targets + 1000):
                    with self.subTest(method=method, q=q, zero_gain=targets[0, 0] > 10):
                        result = self.plan(method, q, seed=0, targets=targets)
                        self.assertEqual(len(result['query_order']), q)
                        self.assertEqual(len(set(result['query_order'])), q)
                        self.assertEqual(result['selected_ids'], sorted(result['query_order']))
                        self.assertTrue(all(0 <= i < 4 for i in result['selected_ids']))
                        if q == 4:
                            self.assertEqual(result['selected_ids'], [0, 1, 2, 3])
        self.assertEqual(self.plan(q=4)['query_order'], [0, 2, 3, 1])
        self.assertEqual(self.plan(q=4, targets=self.targets[:5])['query_order'], [0, 2, 1, 3])
        self.assertEqual(self.plan(q=4, targets=self.targets + 1000)['query_order'], [0, 1, 2, 3])

    def test_zero_budget_skips_optics_and_frontier_even_empty_domain(self):
        with patch.object(controls, 'compute_view_angles_numpy', side_effect=AssertionError), \
                patch.object(controls, 'build_frontier', side_effect=AssertionError):
            for method in METHODS:
                for poses, targets in ((self.poses, self.targets),
                                      (np.empty((0, 5)), np.empty((0, 3))), ([], [])):
                    result = self.plan(method, 0, seed=0, poses=poses, targets=targets,
                                       enumeration_limit=0)
                    self.assertEqual(result['selected_ids'], [])
                    self.assertEqual(result['query_order'], [])
                    self.assertEqual(result['optical_candidate_evaluations'], 0)
                    self.assertEqual(result['optical_point_tests'], 0)

    def test_optical_parity_with_existing_helper_float32(self):
        from src.methods.connected_hierarchy import optical_predictions

        rng = np.random.default_rng(82)
        poses = np.column_stack((rng.normal(size=(10, 3)), rng.uniform(-180, 180, (10, 2))))
        targets = rng.normal(size=(40, 3)) * 3
        for h in (60., 180., 359.99999, 360.):
            profile = LidarProfile('test', h, 95., 5.)
            expected = optical_predictions(SimpleNamespace(
                poses=poses.astype(np.float32), targets=targets.astype(np.float32),
                allowed=np.ones(10, bool), sensor=profile))
            captured = []
            real = controls.build_frontier

            def capture(p, masks, t, ks, **kwargs):
                captured.append(masks.copy())
                return real(p, masks, t, ks, **kwargs)

            with patch.object(controls, 'build_frontier', side_effect=capture):
                result = self.plan('geometry_portfolio', poses=poses, targets=targets,
                                   profile=profile)
            np.testing.assert_array_equal(captured[0], expected)
            self.assertTrue(result['surrogate_enumeration_completed'])

    def test_optical_boundaries_without_forward_slab(self):
        poses = np.array([[0, 0, 0, 0, 0]], float)
        targets = np.array([[0, 0, 0], [1e-7, 0, 0], [.5, 0, 0], [.50001, 0, 0],
                            [2, 0, 0], [2.001, 0, 0], [-1, 0, 0], [1, 2, 0],
                            [1, 0, 2], [1, 1, 0], [1, 0, 1]], float)
        real = controls.build_frontier
        for h, expected in ((90., [2, 3, 4, 9, 10]),
                            (360., [2, 3, 4, 6, 9, 10])):
            captured = []

            def capture(p, masks, t, ks, **kwargs):
                captured.append(masks.copy())
                return real(p, masks, t, ks, **kwargs)

            with patch.object(controls, 'build_frontier', side_effect=capture):
                self.plan('geometry_portfolio', 1, poses=poses, targets=targets,
                          profile=LidarProfile('test', h, 90., 2.))
            self.assertEqual(np.flatnonzero(captured[0][0]).tolist(), expected)

    @unittest.skipUnless(importlib.util.find_spec('open3d'), 'Open3D is not installed')
    def test_actual_raycast_clear_scene_parity_and_wall_occlusion(self):
        import open3d as o3d
        from src.core.visibility import MeshVisibility

        captured = []
        real = controls.build_frontier

        def capture(p, masks, t, ks, **kwargs):
            captured.append(masks.copy())
            return real(p, masks, t, ks, **kwargs)

        with patch.object(controls, 'build_frontier', side_effect=capture):
            result = self.plan('geometry_portfolio')
        clear = o3d.geometry.TriangleMesh.create_box().translate([100., 100., 100.])
        visibility = MeshVisibility(clear, self.targets, self.profile)
        measured = np.array([visibility.mask(pose) for pose in self.poses])
        np.testing.assert_array_equal(measured, captured[0])
        wall = o3d.geometry.TriangleMesh.create_box(.1, 100., 100.).translate([1.5, -50., -50.])
        occluded = MeshVisibility(wall, self.targets, self.profile).mask(self.poses[0])
        self.assertGreater(int(captured[0][0].sum()), int(occluded.sum()))
        self.assertFalse(result['semantics']['physical_coverage_certified'])

    def test_portfolio_matches_independent_bruteforce_policy(self):
        from src.methods.connected_hierarchy import optical_predictions

        rng = np.random.default_rng(91)
        for trial in range(8):
            poses = np.column_stack((rng.uniform(-3, 3, (6, 3)), rng.uniform(-180, 180, (6, 2))))
            targets = rng.uniform(-4, 4, (12, 3))
            profile = LidarProfile('test', 160., 110., 8.)
            masks = optical_predictions(SimpleNamespace(poses=poses.astype(np.float32),
                targets=targets.astype(np.float32), allowed=np.ones(6, bool), sensor=profile))
            counts = masks.sum(axis=1)
            greedy, covered = [], np.zeros(12, bool)
            for _ in range(6):
                cid = min((i for i in range(6) if i not in greedy),
                          key=lambda i: (-int((masks[i] & ~covered).sum()), -int(counts[i]), i))
                greedy.append(cid)
                covered |= masks[cid]
            for q in range(1, 7):
                alternatives = []
                for k in (1, 2, 3):
                    layouts = [ids for size in range(k + 1) for ids in combinations(range(6), size)]
                    layouts.sort(key=lambda ids: (-int(np.any(masks[list(ids)], axis=0).sum()),
                                                   len(ids), ids))
                    alternatives.append(layouts[:q])
                order = []
                for rank in range(q):
                    for rows in alternatives:
                        for cid in rows[rank] if rank < len(rows) else ():
                            if cid not in order:
                                order.append(cid)
                order.extend(i for i in greedy if i not in order)
                result = self.plan('geometry_portfolio', q, poses=poses, targets=targets,
                                   profile=profile, k_values=[1, 2, 3])
                self.assertEqual(result['query_order'], order[:q], (trial, q))

    def test_portfolio_scan_order_and_greedy_fill(self):
        frontier = dict(enumeration=dict(completed=True), frontier=[
            dict(budget_k=1, alternatives=[dict(selected_ids=[2]), dict(selected_ids=[0])]),
            dict(budget_k=2, alternatives=[dict(selected_ids=[3, 1]), dict(selected_ids=[0, 2])])])
        with patch.object(controls, 'build_frontier', return_value=frontier) as build:
            result = self.plan('geometry_portfolio', 3)
        self.assertEqual(result['query_order'], [2, 1, 3])
        self.assertEqual(build.call_args.kwargs['alternatives_per_k'], 3)
        frontier['frontier'] = [dict(budget_k=1, alternatives=[dict(selected_ids=[2])])]
        with patch.object(controls, 'build_frontier', return_value=frontier):
            self.assertEqual(self.plan('geometry_portfolio', 4)['query_order'], [2, 0, 3, 1])

    def test_portfolio_requires_completed_enumeration(self):
        with self.assertRaisesRegex(ValueError, 'enumeration'):
            self.plan('geometry_portfolio', enumeration_limit=10)
        result = self.plan('geometry_portfolio', enumeration_limit=11)
        self.assertTrue(result['surrogate_enumeration_completed'])
        with patch.object(controls, 'build_frontier', return_value=dict(
                enumeration=dict(completed=False), frontier=[])):
            with self.assertRaisesRegex(ValueError, 'enumeration'):
                self.plan('geometry_portfolio')

    def test_invalid_seeds_and_query_budgets(self):
        for seed in (None, -1, True, np.bool_(False), 1.5, '0', np.random.default_rng(0)):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.plan('random', seed=seed)
        for method in METHODS:
            for q in (-1, 5, True, np.bool_(False), 1.5, '2', None):
                with self.subTest(method=method, q=q), self.assertRaises(ValueError):
                    self.plan(method, q, seed=0)
            with self.assertRaises(ValueError):
                self.plan(method, seed=-1)
        result = self.plan('random', np.int64(2), seed=np.int64(0))
        json.dumps(result, allow_nan=False)

    def test_invalid_geometry_profile_budgets_and_caps(self):
        bad = [dict(poses=np.repeat(self.poses[:1], 4, axis=0)),
               dict(targets=np.repeat(self.targets[:1], 6, axis=0)),
               dict(poses=self.poses[:, :3]), dict(targets=self.targets[:, :2]),
               dict(poses=np.full_like(self.poses, np.nan)),
               dict(targets=np.full_like(self.targets, np.inf)),
               dict(method='unknown'), dict(enumeration_limit=-1),
               dict(enumeration_limit=True), dict(enumeration_limit=2.5)]
        bad.extend(dict(k_values=ks) for ks in ([], [0], [65], [1, 1], [2, 1], [True], [1.5]))
        bad.extend(dict(profile=LidarProfile('bad', h, v, r)) for h, v, r in
                   ((0, 60, 10), (361, 60, 10), (60, 181, 10), (60, 0, 10),
                    (60, 60, 0), (60, 60, np.inf), (np.nan, 60, 10)))
        bad.append(dict(poses=np.column_stack((np.arange(2001), np.zeros((2001, 4))))))
        bad.append(dict(poses=np.column_stack((np.arange(2000), np.zeros((2000, 4)))),
                        targets=np.column_stack((np.arange(2001), np.zeros((2001, 2))))))
        for kwargs in bad:
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.plan(**kwargs)

    def test_public_signature_and_import_boundary(self):
        self.assertEqual(list(inspect.signature(controls.plan_control).parameters),
                         ['poses', 'targets', 'profile', 'query_budget', 'k_values',
                          'method', 'seed', 'enumeration_limit'])
        tree = ast.parse(inspect.getsource(controls))
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imports - {'__future__', 'math', 'time'},
                         {'src.environment', 'src.core.deployment_library'})
        with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('no physical query')):
            for method in METHODS:
                self.plan(method, seed=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

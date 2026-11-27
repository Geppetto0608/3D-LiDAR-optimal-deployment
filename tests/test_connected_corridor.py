"""Geometry, source separation, constrained enumeration and tiny end-to-end checks."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import connected_corridor_study as study
from src.core.coverage_requirements import CoverageRequirements
from src.core.visibility import MeshVisibility
from src.environment import ConnectedCorridorGeometry, LidarProfile, get_lidar_profile, load_config
from src.methods.connected_pairs import assess_ids, canonical_local_search, exact_pairs

ROOT = Path(__file__).resolve().parents[1]


class ConnectedCorridorTests(unittest.TestCase):
    def suite(self):
        return study.load_suite(ROOT / 'configs/connected_corridor_pilot.json')

    def tiny_suite(self):
        suite = self.suite()
        suite['geometry'].update(length_m=40, bend_length_m=20, mesh_step_m=1)
        suite['candidate'].update(station_step_m=10, heights_m=[3],
                                  relative_yaws_deg=[0], pitches_deg=[-10])
        suite['angles_deg'] = [0, 30]
        suite['interface_half_length_m'] = 5
        suite['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        return suite

    def test_centerline_continuity_and_signed_bending(self):
        geometry = ConnectedCorridorGeometry(turn_deg=60)
        starts = [(geometry.length_m - geometry.bend_length_m) / 2,
                  (geometry.length_m + geometry.bend_length_m) / 2]
        for station in starts:
            xy, heading = geometry.centerline(np.array([station - 1e-6, station + 1e-6]))
            self.assertLess(np.linalg.norm(xy[1] - xy[0]), 3e-6)
            self.assertLess(abs(heading[1] - heading[0]), 1e-6)
        s = np.linspace(0, 80, 1001)
        positive, headings = geometry.centerline(s)
        negative, opposite = ConnectedCorridorGeometry(turn_deg=-60).centerline(s)
        np.testing.assert_allclose(positive[:, 0], negative[:, 0])
        np.testing.assert_allclose(positive[:, 1], -negative[:, 1])
        np.testing.assert_allclose(headings, -opposite)
        self.assertAlmostEqual(headings[-1], np.pi / 3)
        path_length = np.linalg.norm(np.diff(positive, axis=0), axis=1).sum()
        self.assertAlmostEqual(path_length, 80, places=4)
        with self.assertRaises(ValueError):
            ConnectedCorridorGeometry(turn_deg=90, bend_length_m=5)
        with self.assertRaises(ValueError):
            ConnectedCorridorGeometry(turn_deg=30, bend_length_m=0)

    def test_target_ownership_and_candidate_transfer(self):
        suite = self.suite()
        geometry = ConnectedCorridorGeometry(**suite['geometry'], turn_deg=30)
        targets, coordinates = geometry.target_samples()
        self.assertEqual(targets.shape, (1440, 3))
        self.assertEqual(len(np.unique(targets, axis=0)), len(targets))
        self.assertEqual(int((coordinates[:, 0] < 40).sum()), 720)
        source = ConnectedCorridorGeometry(length_m=40, bend_length_m=0)
        canonical = study.candidates(source, suite['candidate'])
        transferred = canonical.copy()
        transferred[:, 0] += 40
        self.assertEqual(len(canonical), 240)
        world = geometry.poses(transferred)
        _, headings = geometry.centerline(transferred[:, 0])
        inward = np.where(transferred[:, 1] > 0, -90, 90)
        expected = (np.rad2deg(headings) + inward + transferred[:, 3]) % 360
        np.testing.assert_allclose(world[:, 3], expected)
        invalid = transferred.copy()
        invalid[0, 1] = 0
        with self.assertRaises(ValueError):
            geometry.poses(invalid)

    def test_world_rays_occlusion_and_rigid_rotation(self):
        geometry = ConnectedCorridorGeometry(turn_deg=60)
        mesh = geometry.create_environment_mesh()
        # A target on the road is visible; an exterior point is blocked by the wall.
        targets = np.array([[10, 0, 1.5], [10, 15, 1.5]], dtype=np.float32)
        pose = np.array([10, -6, 3, 90, -10.])
        profile = LidarProfile('test', 360, 180, 100)
        masks = MeshVisibility(mesh, targets, profile).mask(pose)
        np.testing.assert_array_equal(masks, [True, False])
        theta = np.deg2rad(37)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        translated = mesh.rotate(rotation, center=[0, 0, 0]).translate([11, 7, 0])
        target_rotated = targets @ rotation.T + [11, 7, 0]
        rotated_pose = np.r_[pose[:3] @ rotation.T + [11, 7, 0], pose[3] + 37, pose[4]]
        actual = MeshVisibility(translated, target_rotated, profile).mask(rotated_pose)
        np.testing.assert_array_equal(actual, masks)

    def test_exact_reference_matches_naive_shared_evaluator(self):
        rng = np.random.default_rng(41)
        masks = rng.random((8, 13)) < .6
        poses = np.array([[i*2, 0, 3 + i % 2, 90, -10] for i in range(8)], dtype=float)
        spec = dict(schema='constrained_coverage_v2', budget_k=2,
            weights=dict(coverage_weight=1, importance_weight=.2, overlap_penalty=.8,
                         installation_cost_weight=.4),
            cost=dict(base_per_sensor=10, per_height_m=1),
            constraints=dict(max_installation_cost=27, min_xy_separation_m=4,
                             min_single_failure_coverage_pct=20, max_overlap_ratio=.8))
        targets = np.column_stack([np.arange(13), np.zeros(13), np.ones(13)])
        req = CoverageRequirements(spec).bind(targets)
        for modules in (None, np.repeat([0, 1], 4)):
            valid = []
            for pair in combinations(range(8), 2):
                if modules is not None and modules[pair[0]] == modules[pair[1]]:
                    continue
                assessment = assess_ids(pair, poses, masks.__getitem__, req)
                if assessment.feasible:
                    valid.append((assessment.objective_score, pair))
            expected = sorted(valid, key=lambda value: (-value[0], value[1]))[0]
            result = exact_pairs(range(8), poses, masks.__getitem__, req, module_ids=modules)
            self.assertAlmostEqual(result['assessment']['objective_score'], expected[0])
            self.assertEqual(result['selected_ids'], list(expected[1]))
            self.assertEqual(result['combinations_considered'], 28 if modules is None else 16)
        bad = deepcopy(spec)
        bad['constraints']['max_installation_cost'] = 1
        result = exact_pairs(range(8), poses, masks.__getitem__, CoverageRequirements(bad).bind(targets))
        self.assertEqual(result['status'], 'infeasible_in_declared_grid')
        self.assertIsNone(result['selected_ids'])

    def test_exhaustive_ties_and_all_binary_small_masks(self):
        poses = np.array([[0, -6, 3, 90, -10], [3, -6, 3, 90, -10],
                          [6, 6, 3, 270, -10], [9, 6, 3, 270, -10]], dtype=float)
        req = CoverageRequirements(dict(schema='constrained_coverage_v2', budget_k=2)).bind(np.ones((2, 3)))
        for encoded in range(256):
            masks = ((encoded >> np.arange(8)) & 1).reshape(4, 2).astype(bool)
            naive = [(assess_ids(pair, poses, masks.__getitem__, req).objective_score, list(pair))
                     for pair in combinations(range(4), 2)]
            expected = sorted(naive, key=lambda value: (-value[0], value[1]))[0]
            result = exact_pairs(range(4), poses, masks.__getitem__, req)
            self.assertEqual(result['selected_ids'], expected[1])
            self.assertEqual(result['assessment']['objective_score'], expected[0])

    def test_local_search_feasibility_anchors_and_monotonicity(self):
        canonical = np.array([[s, -6, 3, 0, -10] for s in (1, 3, 5, 7)], dtype=float)
        poses = ConnectedCorridorGeometry(length_m=10, bend_length_m=0).poses(canonical)
        masks = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]], dtype=bool)
        req = CoverageRequirements(dict(schema='constrained_coverage_v2', budget_k=2,
            constraints=dict(min_xy_separation_m=2))).bind(np.ones((4, 3)))
        result = canonical_local_search([0, 3], canonical, poses, masks.__getitem__, req,
                                         station_radius_m=2)
        self.assertEqual(result['selected_ids'], [1, 2])
        self.assertEqual(result['assessment']['coverage_pct'], 100)
        self.assertTrue(all(np.diff([x['objective_score'] for x in result['trace']]) > 0))
        frozen = canonical_local_search([0, 3], canonical, poses, masks.__getitem__, req,
                                         station_radius_m=0)
        self.assertEqual(frozen['selected_ids'], [0, 3])

    def test_blind_run_spans_interface_without_wrapping(self):
        coords = np.array([[s, 1, 1.5] for s in np.arange(.5, 8, 1)])
        density = np.array([0, 0, 1, 0, 0, 0, 1, 0])
        value = study.metrics(density, coords, 4, 2, 1)
        self.assertEqual(value['max_sampled_blind_run_m'], 3)
        self.assertEqual(value['interface_coverage_pct'], 25)

    def test_plan_no_raycast_or_writes_and_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'uncreated'
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Unexpected raycast')):
                with redirect_stdout(io.StringIO()) as text:
                    study.main(['--output', str(output)])
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(text.getvalue())['target_candidates'], 480)
            output.mkdir()
            with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                study.main(['--output', str(output), '--execute'])
            with self.assertRaises(ValueError):
                study.main(['--worker-root', str(output)])

    def test_invalid_suite_limits_and_duplicate_targets(self):
        suite = self.suite()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'suite.json'
            for changed in (
                {'target_step_m': .00001},
                {'angles_deg': [0, 0]},
                {'alternatives_per_site': 0},
                {'local_station_radius_m': -1},
                {'candidate': {**suite['candidate'], 'station_step_m': .00001}},
                {'geometry': {**suite['geometry'], 'mesh_step_m': .000001}},
                {'target_heights_m': [1, 1]},
            ):
                path.write_text(json.dumps({**suite, **changed}))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    study.load_suite(path)

    def test_frozen_tiny_end_to_end(self):
        suite = self.tiny_suite()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'suite.json'
            path.write_text(json.dumps(suite))
            output = root / 'run'
            run = subprocess.run([sys.executable, str(ROOT / 'connected_corridor_study.py'),
                '--suite', str(path), '--output', str(output), '--execute'],
                cwd=ROOT, capture_output=True, text=True, timeout=90)
            log = (output / 'worker.log').read_text() if (output / 'worker.log').exists() else ''
            self.assertEqual(run.returncode, 0, run.stderr + log)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['jobs']), 2)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['method_rows'], 12)
            for row in summary['results']:
                if row['coverage_pct'] is not None:
                    self.assertTrue(row['validated_direct_raycast'])
                if row['feasible']:
                    self.assertGreaterEqual(row['objective_gap_to_grid_optimum'], -1e-7)
            for job in manifest['jobs']:
                directory = output / job['directory']
                ref = json.loads((directory / 'reference_global/result.json').read_text())
                self.assertEqual(ref['combinations_considered'], 28)
                self.assertTrue((directory / 'comparison.png').exists())
                self.assertLessEqual(
                    (directory / 'library_joint/result.json').stat().st_mtime,
                    (directory / 'reference_masks.npz').stat().st_mtime)
            self.assertTrue((output / 'source/src/environment.py').exists())
            from validate_connected_corridor import audit
            checked = audit(output)
            self.assertEqual(checked['checked_deployments'], 12)
            self.assertTrue(checked['all_reference_optima_match'])
            result_path = output / 'angle_00/reference_global/result.json'
            altered = json.loads(result_path.read_text())
            altered['assessment']['objective_score'] += 1
            result_path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'optimum mismatch'):
                audit(output)


if __name__ == '__main__':
    unittest.main(verbosity=2)

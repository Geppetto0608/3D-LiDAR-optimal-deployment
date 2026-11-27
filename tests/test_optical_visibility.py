"""Independent angle/range expectations across direct, cached and Torch paths."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.visibility import MeshVisibility
from src.environment import (OPTICAL_MODEL, MapGeometry, LidarProfile, angular_visibility_mask,
    compute_config_hash, compute_view_angles_numpy, config_hash_payload, coverage_mask_for_pose,
    get_scenario, load_config, make_cache_key, visibility_cache_path)
from src.methods.connected_hierarchy import optical_predictions
from src.methods.library_query_controls import _optical_masks
from src.methods.mappo import LegacyMAPPOEnv, LegacyMAPPOParams, _compute_view_angles_torch


class OpticalVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.array([.25, .25, 3., 0., 0.], dtype=np.float32)
        self.vectors = np.array([[.25, 0, 0], [.5, 0, 0], [-1, np.sqrt(3), 0],
            [-2, 0, 0], [0, 0, 0], [11, 0, 0], [1, 0, 2], [1, 0, 0]], dtype=np.float32)
        self.targets = self.vectors + self.pose[:3]
        self.clear = o3d.geometry.TriangleMesh.create_box().translate([100, 100, 100])
        self.config = load_config('configs/library_topology_smoke.yaml')
        self.geometry = MapGeometry(get_scenario(1))

    def expected(self, horizontal):
        # Independent unrotated analytic expectations, not the production helper.
        return np.array([True, True, horizontal >= 240, horizontal == 360,
                         False, False, False, True])

    def test_direct_cached_and_surrogates_against_analytic_points(self):
        data = dict(target_points=self.targets, zs=np.array([3.]),
                    table={(0, 0, 0): np.packbits(np.ones(len(self.targets), bool))})
        for h in (90., 180., 270., 359.999, 360.):
            with self.subTest(h=h):
                profile = LidarProfile('audit', h, 90., 10.)
                expected = self.expected(h)
                np.testing.assert_array_equal(MeshVisibility(self.clear, self.targets, profile).mask(self.pose), expected)
                np.testing.assert_array_equal(coverage_mask_for_pose(
                    self.pose, data, self.geometry, profile, self.config), expected)
                np.testing.assert_array_equal(_optical_masks(
                    self.pose[None], self.targets, (h, 90., 10.))[0], expected)
                context = SimpleNamespace(poses=self.pose[None], targets=self.targets,
                                          allowed=np.array([True]), sensor=profile)
                np.testing.assert_array_equal(optical_predictions(context)[0], expected)

    def test_rotated_pose_preserves_sensor_frame_expectations(self):
        yaw, pitch = np.deg2rad([37., -20.])
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        # Inverse of sensor yaw/pitch rotations, formed independently as a basis.
        basis = np.array([[cy * cp, -sy, -cy * sp],
                          [sy * cp, cy, -sy * sp], [sp, 0, cp]])
        targets = self.vectors @ basis.T + self.pose[:3]
        pose = np.array([*self.pose[:3], 37., -20.], dtype=np.float32)
        profile = LidarProfile('audit', 270., 90., 10.)
        np.testing.assert_array_equal(MeshVisibility(self.clear, targets, profile).mask(pose), self.expected(270.))

    def test_occlusion_still_blocks_angle_valid_targets(self):
        wall = o3d.geometry.TriangleMesh.create_box(.1, 2, 2).translate([.8, -.75, 2.])
        actual = MeshVisibility(wall, self.targets, LidarProfile('audit', 90., 90., 10.)).mask(self.pose)
        self.assertTrue(actual[0])
        self.assertFalse(actual[-1])

    def test_mappo_numpy_and_reward_torch_match_analytic_coverage(self):
        data = dict(target_points=self.targets, zs=np.array([3.]),
                    target_weights=np.ones(len(self.targets), dtype=np.float32))
        params_args = dict(num_agents=1, episodes=1, xy_step_m=2., z_min=3., z_max=3., z_step=1.,
            yaw_step_deg=5., yaw_axis_min_deg=35., yaw_axis_max_deg=145.,
            pitch_step_deg=2.5, min_pitch=-25., max_pitch=-5., edge_margin_m=0.)
        for h in (90., 270., 360.):
            self.config.raw['lidar_profiles'][self.config.lidar_profile].update(
                h_fov_deg=h, v_fov_deg=90., max_range_m=10.)
            with self.subTest(h=h), patch('torch.cuda.is_available', return_value=False):
                env = LegacyMAPPOEnv(get_scenario(1), self.config.lidar_profile, self.config,
                    LegacyMAPPOParams(**params_args), visibility_data=data,
                    visibility_query=lambda *args: np.arange(len(self.targets)))
                np.testing.assert_array_equal(env.coverage_density(self.pose[None]), self.expected(h).astype(float))
                agent = SimpleNamespace(x=0, y=0, z=3., z_idx=0, yaw=0., pitch=0.)
                _, _, info = env.calculate_reward([agent])
                self.assertAlmostEqual(info['score'], self.expected(h).sum())
                self.assertAlmostEqual(info['cov'], 100 * self.expected(h).mean())

    def test_torch_numpy_rotation_parity(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for yaw, pitch in [(0., 0.), (37., -20.)]:
                a, e, _, d = compute_view_angles_numpy(*self.vectors.T, yaw, pitch)
                v = torch.tensor(self.vectors, device=device)
                ta, te, _, td = _compute_view_angles_torch(*v.T, yaw, pitch)
                expected = angular_visibility_mask(a, e, d, np.deg2rad(135), np.deg2rad(45), 10.)
                got = angular_visibility_mask(ta, te, td, np.deg2rad(135), np.deg2rad(45), 10.)
                np.testing.assert_array_equal(got.cpu().numpy(), expected)

    def test_limited_fov_hash_separates_legacy_and_360_hash_is_preserved(self):
        scenario = get_scenario(1)
        for h in (90., 270., 360.):
            profile = LidarProfile(self.config.lidar_profile, h, 90., 10.)
            current = compute_config_hash(self.config, scenario, profile)
            path = visibility_cache_path(self.config, scenario, profile)
            key = make_cache_key(scenario, profile, self.config)
            with patch('src.environment.optical_model_metadata', return_value={}):
                old = compute_config_hash(self.config, scenario, profile)
                old_path = visibility_cache_path(self.config, scenario, profile)
                old_key = make_cache_key(scenario, profile, self.config)
            if h < 360:
                self.assertNotEqual(current, old)
                self.assertNotEqual(path, old_path)
                self.assertNotEqual(key, old_key)
                self.assertEqual(config_hash_payload(self.config, profile)['optical_model'], OPTICAL_MODEL)
            else:
                self.assertEqual(current, old)
                self.assertEqual(path, old_path)
                self.assertEqual(key, old_key)
                self.assertNotIn('optical_model', config_hash_payload(self.config, profile))


if __name__ == '__main__':
    unittest.main(verbosity=2)

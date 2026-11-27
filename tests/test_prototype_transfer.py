import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_offline import width_scenario
from prototype_transfer import (angle_delta, align_poses, interpolate, project_poses,
    transfer_raw, prepare_placements, project_to_target)
from src.environment import MapGeometry, load_config, compute_config_hash, get_lidar_profile


class WidthTransferTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.low = MapGeometry(width_scenario(7))
        self.high = MapGeometry(width_scenario(14))
        self.target = MapGeometry(width_scenario(10))
        self.poses = np.array([[30.25, 44.25, 3, 85, -20],
                              [70.25, 44.25, 3, 95, -20],
                              [50.25, 55.75, 4, 270, -15]])

    def test_width_is_only_geometry_change_and_cache_separates(self):
        self.assertEqual(self.low.num_lanes, self.high.num_lanes)
        self.assertEqual(self.high.scenario.raw['changed_fields'], ['road_width'])
        profile = get_lidar_profile(self.config.lidar_profile, self.config)
        self.assertNotEqual(compute_config_hash(self.config, self.low.scenario, profile),
                            compute_config_hash(self.config, self.high.scenario, profile))
        for bad in (float('nan'), 0, 99):
            with self.assertRaises(ValueError):
                width_scenario(bad)

    def test_matching_handles_permutation_and_side(self):
        high = transfer_raw(self.poses, self.low, self.high)[[2,0,1]]
        aligned, metadata = align_poses(self.poses, high, self.high)
        np.testing.assert_array_equal(aligned[:,1] > 50, self.poses[:,1] > 50)
        np.testing.assert_allclose(aligned[:,0], self.poses[:,0])
        self.assertEqual(len(metadata['high_indices']), 3)

    def test_interpolation_projects_to_installation_grid(self):
        high = transfer_raw(self.poses, self.low, self.high)
        result, metadata = interpolate(self.poses, high, self.low, self.high, self.target, self.config)
        self.assertAlmostEqual(metadata['alpha'], 3/7)
        self.assertTrue(all(self.target.is_installable(*p[:3]) for p in result))
        np.testing.assert_allclose(project_poses(result, self.target, self.config), result)
        self.assertEqual(len(np.unique(result, axis=0)), 3)
        with self.assertRaises(ValueError):
            interpolate(self.poses, high, self.low, self.high, MapGeometry(width_scenario(16)), self.config)

    def test_rejects_incompatible_counts_and_sides(self):
        with self.assertRaises(ValueError):
            align_poses(self.poses, self.poses[:2], self.high)
        all_south = self.poses.copy(); all_south[:,1] = 40
        with self.assertRaises(ValueError):
            align_poses(self.poses, all_south, self.high)

    def test_circular_yaw(self):
        self.assertEqual(float(angle_delta(350, 10)), 20)
        self.assertEqual(float(angle_delta(10, 350)), -20)

    def test_global_projection_respects_grid_without_branch_lock(self):
        raw = np.array([[50,50,3.4,355,-13], [50,50,3.4,355,-13]], dtype=float)
        with patch('prototype_transfer.MeshVisibility', side_effect=AssertionError('No raycasting in projection')):
            result = project_to_target(raw,self.target,self.config)
        self.assertEqual(len(np.unique(result,axis=0)),2)
        self.assertTrue(all(self.target.is_installable(*p[:3]) for p in result))
        np.testing.assert_allclose(project_to_target(result,self.target,self.config), result)
        for bad in (np.zeros((0,5)),np.zeros((2,4)),np.full((1,5),np.nan)):
            with self.assertRaises(ValueError):
                project_to_target(bad,self.target,self.config)

    def test_projection_collision_assignment_and_infeasibility(self):
        base = [(10.25,44.25,3.),(12.25,44.25,3.)]
        raw = np.array([[10.25,44.25,3,90,-20]]*2)
        with patch('prototype_transfer._base_positions', return_value=base):
            result = project_to_target(raw,self.low,self.config)
            self.assertEqual(len(np.unique(result,axis=0)),2)
            self.assertAlmostEqual(float(np.sum((result[:,:3]-raw[:,:3])**2)),4.)
        for small in ([],base[:1]):
            with patch('prototype_transfer._base_positions', return_value=small):
                with self.assertRaises(ValueError):
                    project_to_target(raw,self.low,self.config)

    def assert_geometry_transfer(self, p0, p1, low, high, target):
        with patch('prototype_transfer.MeshVisibility', side_effect=AssertionError('No coverage-based matching')):
            placements, meta = prepare_placements(p0,p1,low,high,target,self.config,matching='geometry')
        self.assertEqual(meta['status'],'available')
        self.assertEqual(meta['matching_mode'],'geometry')
        result = placements['interpolation']
        self.assertEqual(result.shape,p0.shape)
        self.assertTrue(all(target.is_installable(*p[:3]) for p in result))
        np.testing.assert_allclose(project_to_target(result,target,self.config), result)
        np.testing.assert_allclose(meta['projection_displacement_m'],
            np.linalg.norm(result[:,:3]-np.asarray(meta['unprojected_poses'])[:,:3],axis=1))
        again, _ = prepare_placements(p0,p1,low,high,target,self.config,matching='geometry')
        np.testing.assert_array_equal(again['interpolation'],result)

    def test_incompatible_junction_preserves_controls_not_fake_interpolation(self):
        low, high, target = [MapGeometry(width_scenario(w,1)) for w in (7,14,10)]
        p0 = np.array([[28.25,55.75,3,265,-20], [56.25,42.25,3,135,-10],
                       [72.25,55.75,3,265,-20]])
        p1 = np.array([[26.25,59.25,3,270,-15], [40.25,38.25,3,40,-10],
                       [74.25,40.25,3,85,-15]])
        placements, metadata = prepare_placements(p0,p1,low,high,target,self.config)
        self.assertEqual(metadata['status'], 'unavailable')
        self.assertEqual(set(placements), {'transfer_low','transfer_high'})
        self.assertIn('correspondence', metadata['reason'])
        for poses in placements.values():
            self.assertTrue(all(target.is_installable(*p[:3]) for p in poses))
        self.assert_geometry_transfer(p0,p1,low,high,target)
        with self.assertRaises(ValueError):
            prepare_placements(p0,p1,low,high,MapGeometry(width_scenario(16,1)),self.config)
        low, high, target = [MapGeometry(width_scenario(w,2)) for w in (7,14,10)]
        p0 = np.array([[28.25,44.25,3,90,-20], [43.75,58.25,3,320,-10],
                       [58.25,43.75,3,130,-10]])
        p1 = np.array([[40.25,40.25,3,45,-10], [59.25,62.25,3,195,-5],
                       [76.25,59.25,3,270,-15]])
        placements, metadata = prepare_placements(p0,p1,low,high,target,self.config)
        self.assertEqual(metadata['status'], 'unavailable')
        self.assertEqual(set(placements), {'transfer_low','transfer_high'})
        self.assertTrue(all(target.is_installable(*p[:3]) for poses in placements.values() for p in poses))
        self.assert_geometry_transfer(p0,p1,low,high,target)

    def test_junction_corner_projection_retains_original_branch(self):
        from prototype_offline import canonical
        low, high = [MapGeometry(width_scenario(w,1)) for w in (7,14)]
        p = np.array([[56.25,42.25,3,135,-10]])
        keys = canonical(p,low)
        raw = transfer_raw(p,low,high)
        self.assertNotEqual(canonical(raw,high)[0]['branch_id'],keys[0]['branch_id'])
        result = project_poses(raw,high,self.config,keys)
        self.assertEqual(canonical(result,high)[0]['branch_id'],keys[0]['branch_id'])
        self.assertEqual(canonical(result,high)[0]['installation_line_id'],keys[0]['installation_line_id'])

    def test_junction_width_transfer_and_roundtrip(self):
        from prototype_offline import canonical, absolute, describe
        for sid in (1,2):
            low, high, target = [MapGeometry(width_scenario(w,sid)) for w in (7,14,10)]
            poses = np.array([[20.25,55.75,3,270,-15], [80.25,44.25,3,90,-20],
                              [55.75,20.25,4,180,-15]])
            moved = transfer_raw(poses,low,high)
            np.testing.assert_allclose(transfer_raw(moved,high,low),poses)
            np.testing.assert_allclose(absolute(canonical(poses,low),low),poses)
            result, meta = interpolate(poses,moved[[2,0,1]],low,high,target,self.config)
            self.assertTrue(all(target.is_installable(*p[:3]) for p in result))
            self.assertEqual(meta['correspondence']['reflect_y'],False)
            self.assertEqual(len(describe(target,self.config)['branches']),sid+2)
        with self.assertRaises(ValueError):
            transfer_raw(poses,low,MapGeometry(width_scenario(10,1)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

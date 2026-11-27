"""Opt-in shifted junctions must not change the historical center-map model."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment import MapGeometry, Scenario, compute_config_hash, get_lidar_profile, load_config, load_scenarios
from prototype_offline import canonical, absolute, describe
from prototype_segments import prepare_segment_placements
from prototype_transfer import prepare_relative_placements
from src.methods.local_mappo import validate_poses


class ShapePilotTest(unittest.TestCase):
    def test_legacy_geometry_and_hash_unchanged(self):
        cfg = load_config()
        base = load_scenarios()[0]
        self.assertEqual(compute_config_hash(cfg, base, get_lidar_profile(cfg.lidar_profile, cfg)),
                         '79a7508b07b990fa01758b5c1d0320616eb73ba8')
        self.assertEqual(len(MapGeometry(base).target_points([1, 1.5, 2])), 7200)
        for sid in (1, 2):
            s = next(s for s in load_scenarios() if s.id == sid)
            a = MapGeometry(s)
            b = MapGeometry(replace(s, raw=dict(s.raw, geometry=dict(junction_x_m=50, junction_y_m=50))))
            np.testing.assert_array_equal(a.target_points([1, 1.5, 2]), b.target_points([1, 1.5, 2]))
            np.testing.assert_array_equal(np.asarray(a.create_environment_mesh().vertices), np.asarray(b.create_environment_mesh().vertices))

    def test_offset_changes_road_mesh_and_hash(self):
        cfg = load_config()
        s = Scenario(400001, 'T_JUNCTION', 10., 2, dict(geometry=dict(junction_x_m=35., junction_y_m=65.)))
        g = MapGeometry(s)
        self.assertTrue(g.is_road(35.75, 20.25))
        self.assertFalse(g.is_road(50.75, 20.25))
        self.assertTrue(g.is_road(80.25, 65.75))
        self.assertFalse(g.is_road(80.25, 50.75))
        self.assertTrue(g.is_installable(20.25, 72.25))
        profile = get_lidar_profile(cfg.lidar_profile, cfg)
        self.assertNotEqual(compute_config_hash(cfg, s, profile),
            compute_config_hash(cfg, replace(s, raw=dict(geometry=dict(junction_x_m=50., junction_y_m=50.))), profile))
        pose = np.array([[20.25, 72.25, 3, 270, -20], [42.25, 20.25, 3, 180, -20]])
        np.testing.assert_allclose(absolute(canonical(pose, g), g), pose, atol=1e-6)
        desc = describe(g, cfg)
        self.assertEqual(desc['coordinate_frame']['origin'], [35., 65., 0])
        lengths = {p['id']: p['length_m'] for p in desc['branches']}
        self.assertEqual(lengths, dict(east=65., west=35., south=65.))

    def test_shape_schema_validation(self):
        for settings in [dict(junction_x_m=float('nan')), dict(junction_y_m=99), dict(unknown=5)]:
            with self.assertRaises(ValueError):
                MapGeometry(Scenario(400001, 'T_JUNCTION', 10., 2, dict(geometry=settings)))

    def test_shape_plan_is_unique_and_dry_run_safe(self):
        from prototype_runtime import parser, jobs, map_specs
        args = parser().parse_args(['--output', 'unused', '--shape-config', 'configs/shape_pilot.json',
            '--modes', 'fraction', '--methods', 'select', 'local_search', 'mappo', '--seeds', '0'])
        self.assertEqual(len(map_specs(args)), 4)
        planned = list(jobs(args))
        self.assertEqual(len(planned), 12)
        self.assertEqual(len({j['key'] for j in planned}), 12)
        self.assertEqual({j['target_scenario_id'] for j in planned}, {400001,400002,400003,400004})

    def test_raycast_mesh_moves_with_road_predicate(self):
        import open3d as o3d
        old = MapGeometry(Scenario(1, 'T_JUNCTION', 10., 2, {}))
        shifted = MapGeometry(Scenario(400001, 'T_JUNCTION', 10., 2,
            dict(geometry=dict(junction_x_m=35., junction_y_m=65.))))
        hits = []
        for geometry in (old, shifted):
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(geometry.create_environment_mesh()))
            ray = o3d.core.Tensor([[90.,50.,20.,0.,0.,-1.]], dtype=o3d.core.Dtype.Float32)
            hits.append(float(scene.cast_rays(ray)['t_hit'].numpy()[0]))
        self.assertGreater(hits[0], 19.)
        self.assertAlmostEqual(hits[1], 4.9, places=4)

    def test_transfers_land_on_shifted_installation_lines(self):
        cfg = load_config()
        for sid, topo in [(1, 'T_JUNCTION'), (2, 'CROSS')]:
            low, high = [MapGeometry(Scenario(sid, topo, w, 2, {})) for w in (7., 14.)]
            target = MapGeometry(Scenario(400001, topo, 10., 2, dict(geometry=dict(junction_x_m=35., junction_y_m=65.))))
            p0 = np.array([[20.25, 55.75, 3, 270, -20], [80.25, 44.25, 3, 90, -20], [55.75, 20.25, 3, 180, -20]])
            p1 = p0.copy()
            p1[0, 1] += 3.5
            p1[1, 1] -= 3.5
            p1[2, 0] += 3.5
            for prepare in (prepare_relative_placements, prepare_segment_placements):
                placements, meta = prepare(p0, p1, low, high, target, cfg)
                self.assertEqual(meta['status'], 'available')
                for poses in placements.values():
                    validate_poses(poses, target, cfg)


if __name__ == '__main__':
    unittest.main()

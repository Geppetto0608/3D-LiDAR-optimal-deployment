"""No raycast or solve: discrete-domain inclusion and legacy compatibility."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_offline import explicit_scenario
from src.environment import (MapGeometry, allowed_mount_yaws_deg, installation_base_positions,
    load_config, load_scenarios, make_cache_key, validate_installation_pose)
from src.methods.mappo import PPOAgent
from src.roadside.candidate_generator import _base_positions, build_oriented_candidates


class InstallationDomainTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/experiment.yaml')
        self.geometry = MapGeometry(load_scenarios()[0])

    def test_legacy_order_unchanged(self):
        cfg = self.config
        for scenario in load_scenarios():
            g = MapGeometry(scenario)
            old, seen = [], set()
            for gx in range(int(g.map_len/g.grid_size)):
                for gy in range(int(g.map_width/g.grid_size)):
                    x, y = (gx+.5)*g.grid_size, (gy+.5)*g.grid_size
                    if not g.is_installable(x, y):
                        continue
                    key = (int(x//max(cfg.xy_step_m, g.grid_size)), int(y//max(cfg.xy_step_m, g.grid_size)))
                    if key not in seen:
                        seen.add(key)
                        old.extend((x, y, float(z)) for z in np.arange(cfg.z_min, cfg.z_max+1e-9, cfg.z_step))
            self.assertEqual(old, _base_positions(g, cfg))

    def test_full_grid_inclusion_and_movement_closure(self):
        rows = json.loads(Path('configs/library_expansion.json').read_text())
        cfg = self.config
        for family in rows['families']:
            for row in family['targets']:
                g = MapGeometry(explicit_scenario(row))
                legacy = set(_base_positions(g, cfg))
                full_cfg = copy.deepcopy(cfg)
                full_cfg.candidate['xy_domain'] = 'installable_grid'
                full = set(_base_positions(g, full_cfg))
                self.assertLess(legacy, full)
                for x, y, z in full:
                    for dx, dy in ((2,0),(-2,0),(0,2),(0,-2)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < g.map_len and 0 <= ny < g.map_width and g.is_installable(nx, ny, z):
                            self.assertIn((nx, ny, z), full)

    def test_config_hash_and_invalid_mode(self):
        cfg = self.config
        before = make_cache_key(self.geometry.scenario, cfg.lidar_profile, cfg)
        cfg.candidate['xy_domain'] = 'installable_grid'
        self.assertNotEqual(before, make_cache_key(self.geometry.scenario, cfg.lidar_profile, cfg))
        cfg.candidate['xy_domain'] = 'typo'
        with self.assertRaises(ValueError):
            installation_base_positions(self.geometry, cfg)

    def test_pose_grid_and_mappo_yaw_clamp_agree(self):
        cfg, g = self.config, self.geometry
        cfg.candidate['xy_domain'] = 'installable_grid'
        agent = PPOAgent.__new__(PPOAgent)
        agent.geometry = g
        agent.params = SimpleNamespace(yaw_axis_min_deg=cfg.yaw_axis_min_deg,
            yaw_axis_max_deg=cfg.yaw_axis_max_deg, yaw_step_deg=cfg.yaw_step_deg)
        for x, y, z in installation_base_positions(g, cfg)[::40]:
            agent.x, agent.y = int(x/g.grid_size), int(y/g.grid_size)
            for yaw in allowed_mount_yaws_deg(x,y,g,cfg.yaw_step_deg,cfg.yaw_axis_min_deg,cfg.yaw_axis_max_deg):
                agent.yaw = float(yaw)
                agent._clamp_yaw_to_target()
                validate_installation_pose([x,y,z,agent.yaw,cfg.pitch_min_deg],g,cfg)
        x,y,z = installation_base_positions(g,cfg)[0]
        yaw = allowed_mount_yaws_deg(x,y,g,cfg.yaw_step_deg,cfg.yaw_axis_min_deg,cfg.yaw_axis_max_deg)[0]
        pose = np.array([x,y,z,yaw,cfg.pitch_min_deg],dtype=float)
        for index, delta in ((0,.1),(2,.1),(3,1),(4,1)):
            invalid = pose.copy(); invalid[index] += delta
            with self.assertRaises(ValueError):
                validate_installation_pose(invalid,g,cfg)

    def test_builder_uses_full_domain_without_pruning(self):
        cfg, g = self.config, self.geometry
        cfg.candidate.update(xy_domain='installable_grid', max_orientations_per_position=0,
                             z_min=3,z_max=3,yaw_step_deg=55,pitch_step_deg=20)
        full = _base_positions(g,cfg)
        visibility = dict(target_points=np.zeros((1,3)),cache_key='toy',meta={})
        with patch('src.roadside.candidate_generator.bake_visibility_map',return_value=visibility), \
             patch('src.roadside.candidate_generator.RoadsideVisibility') as provider:
            provider.return_value.mask.return_value = np.array([True])
            candidates = build_oriented_candidates(g.scenario,cfg.lidar_profile,cfg,dedup_exact_cover=False)
        self.assertEqual(len(candidates.candidates),len(full)*3*2)
        self.assertEqual({(c.x,c.y,c.z) for c in candidates.candidates},set(full))


if __name__ == '__main__':
    unittest.main(verbosity=2)

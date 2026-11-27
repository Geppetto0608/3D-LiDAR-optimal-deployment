"""Constrained prototype solver and constraint-aware candidate dedup regression."""
from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_compatible import group_warm_start
from src.core.candidate import Candidate, CandidateSet
from src.environment import get_scenario, load_config
from src.methods.milp import _solve_with_gurobi, _validate_selection_groups
from src.roadside.candidate_generator import build_oriented_candidates


class CompatiblePrototypeTest(unittest.TestCase):
    def setUp(self):
        self.masks = np.array([[1,1,1,1,0,0],[0,0,0,0,1,1],
                               [0,0,0,0,0,0],[0,0,0,0,0,1]],dtype=bool)
        self.candidates = CandidateSet([Candidate(i,0,0,3,90,-10,m) for i,m in enumerate(self.masks)],
                                      np.zeros((6,3)),'test','test',{})
        self.groups = [([0,1],1),([2,3],1)]

    def test_group_milp_matches_constrained_brute_force(self):
        optimum = max(int(self.masks[list(ids)].any(axis=0).sum())
                      for ids in product(*[ids for ids,n in self.groups]))
        self.assertEqual(optimum,5)
        cfg = load_config()
        cfg.solver.update(time_limit_sec=10,mip_gap=0,threads=1)
        for batch in (0,2):
            cfg.solver['sparse_target_batch_size'] = batch
            result = _solve_with_gurobi(self.candidates,2,cfg,[0,2],selection_groups=self.groups)
            self.assertEqual(result['objective'],optimum)
            self.assertEqual(result['best_bound'],optimum)
            for ids,n in self.groups:
                self.assertEqual(len(set(ids).intersection(result['selected_ids'])),n)

    def test_invalid_groups_and_warm_starts_rejected(self):
        for groups,start in [([([0,1],1),([1,2,3],1)],[0,2]),
                             ([([0,1],1),([2],1)],[0,2]),
                             (self.groups,[0,1]),(self.groups,[0,2,2]),
                             ([([0,1],2),([2,3],1)],None)]:
            with self.assertRaises(ValueError):
                _validate_selection_groups(4,2,groups,start)

    def test_quota_greedy_warm_start(self):
        selected,count = group_warm_start(self.candidates,self.groups)
        self.assertEqual(count,5)
        _validate_selection_groups(4,2,self.groups,selected)
        with self.assertRaises(ValueError):
            group_warm_start(self.candidates,[([0,1,2,3],2)])

    def test_dedup_keeps_equal_masks_on_different_lines(self):
        cfg = load_config()
        cfg.candidate.update(max_orientations_per_position=0,pitch_min_deg=-10,pitch_max_deg=-10)
        visibility = dict(target_points=np.zeros((2,3)),cache_key='test',meta={'config_hash':'test'})
        with patch('src.roadside.candidate_generator.bake_visibility_map',return_value=visibility), \
             patch('src.roadside.candidate_generator._base_positions',return_value=[(1,0,3),(2,0,3),(3,0,3)]), \
             patch('src.roadside.candidate_generator.allowed_mount_yaws_deg',return_value=np.array([90.])), \
             patch('src.roadside.candidate_generator.RoadsideVisibility') as provider:
            provider.return_value.mask.return_value = np.array([True,False])
            kwargs = dict(scenario=get_scenario(0),lidar_profile=cfg.lidar_profile,config=cfg)
            legacy = build_oriented_candidates(**kwargs)
            grouped = build_oriented_candidates(**kwargs,dedup_group=lambda x,y,z:'a' if x<3 else 'b')
            filtered = build_oriented_candidates(**kwargs,position_filter=lambda x,y,z:x>1,
                                                dedup_group=lambda x,y,z:'a' if x<3 else 'b')
        self.assertEqual(len(legacy.candidates),1)
        self.assertEqual([c.x for c in grouped.candidates],[1,3])
        self.assertEqual([c.x for c in filtered.candidates],[2,3])


if __name__ == '__main__':
    unittest.main(verbosity=2)

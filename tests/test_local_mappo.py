"""On-demand visibility and fixed-start MAPPO refinement regression checks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_offline import width_scenario
from prototype_transfer import (relative_coordinates, relative_to_target,
                                prepare_relative_placements)
from src.environment import MapGeometry, get_lidar_profile, load_config
from src.methods.local_mappo import (LocalPPOAgent, OnDemandVisibility,
                                     refine_placement, validate_poses)
from src.methods.mappo import LegacyMAPPOEnv, MappoPlanRow, SharedMAPPO, _legacy_params


class LocalRefinementTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = deepcopy(load_config())
        self.cfg.mappo.update(local_map_size=3,ppo_update_timestep=4,ppo_k_epochs=1,
                              ppo_minibatch_size=8,eval_interval=4,eval_steps=2)
        self.geometry = MapGeometry(width_scenario(10))
        self.initial = np.array([[30.25,42.25,3,90,-20],[70.25,57.25,3,270,-20]],dtype=np.float32)

    def environment(self):
        row = MappoPlanRow(self.geometry.scenario.id,'STRAIGHT',self.cfg.lidar_profile,2,0,0,8,'test')
        params = _legacy_params(row,self.cfg)
        visibility = OnDemandVisibility(self.geometry,self.cfg,get_lidar_profile(self.cfg.lidar_profile,self.cfg))
        with patch('src.methods.mappo.bake_visibility_map',side_effect=AssertionError('No full cache')):
            env = LegacyMAPPOEnv(self.geometry.scenario,self.cfg.lidar_profile,self.cfg,params,
                visibility_data=visibility.metadata(params.score_base_voxel),visibility_query=visibility.query)
        return env,visibility

    def test_direct_and_on_demand_evaluators_agree_and_memoize(self):
        env,visibility = self.environment()
        covered = np.stack([visibility.provider.mask(p) for p in self.initial]).any(axis=0)
        cov,_ = env.evaluate_positions(self.initial)
        self.assertAlmostEqual(cov,100*covered.mean(),places=7)
        self.assertEqual(visibility.raycast_queries,2)
        env.evaluate_positions(self.initial)
        self.assertEqual(visibility.raycast_queries,2)
        self.assertEqual(len(env.visible_indices(0,0,0)),0)
        self.assertEqual(visibility.raycast_queries,2)

    def test_initial_pose_and_local_action_mask(self):
        env,_ = self.environment()
        dim = 5+env.params.local_map_size**2+7
        shared = SharedMAPPO(dim,11,2*dim,env.params.local_map_size,env.params,env.device)
        ag = LocalPPOAgent(env,shared,self.initial[0],1.)
        np.testing.assert_array_equal(ag.pose(),self.initial[0])
        state,mask = ag.get_state_and_mask()
        self.assertEqual(state.shape,(dim,))
        self.assertTrue(torch.isfinite(state).all())
        self.assertEqual(mask[0],1)
        self.assertEqual(mask[1:5].sum(),0)
        ag.radius_m = 2.
        _,mask = ag.get_state_and_mask()
        self.assertEqual(mask[1],1)
        ag.x += ag._xy_step_cells()
        _,mask = ag.get_state_and_mask()
        self.assertEqual(mask[1],0)
        self.assertEqual(mask[2],1)

    def test_training_updates_and_saves_initial_and_policy_separately(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.methods.mappo.bake_visibility_map',side_effect=AssertionError('No cache')), \
             patch('torch.cuda.is_available',return_value=False):
            root = Path(temp)/'run'
            poses,meta = refine_placement(self.initial,self.geometry.scenario,self.cfg,root,
                                         steps=8,seed=7,radius_m=2.)
            self.assertEqual(meta['ppo_updates'],2)
            self.assertGreaterEqual(meta['coverage_pct'],meta['initial_coverage_pct']-1e-6)
            self.assertGreater(meta['raycast_xyz_queries'],0)
            self.assertFalse(meta['full_visibility_cache'])
            self.assertFalse(meta['yaw_postprocessing'])
            np.testing.assert_array_equal(np.load(root/'initial.npy'),self.initial)
            self.assertTrue(np.all(np.linalg.norm(poses[:,:2]-self.initial[:,:2],axis=1)<=2.00001))
            validate_poses(poses,self.geometry,self.cfg)
            with np.load(root/'train_log.npz') as log:
                self.assertEqual(len(log['step']),8)
                self.assertEqual(log['eval_step'][-1],8)
                for value in log.values():
                    self.assertTrue(np.isfinite(value).all())
            for update in meta['ppo_diagnostics']:
                self.assertTrue(all(np.isfinite(v) for v in update.values()))
            checkpoint = torch.load(root/'policy.pt',weights_only=True)
            self.assertEqual(checkpoint['initial_pose'],self.initial.tolist())
            with self.assertRaises(FileExistsError):
                refine_placement(self.initial,self.geometry.scenario,self.cfg,root,steps=8)

    def test_pose_validation(self):
        validate_poses(self.initial,self.geometry,self.cfg)
        for column,value in [(0,30.),(1,50.),(2,2.),(3,270.),(4,-21.)]:
            bad = self.initial.copy(); bad[0,column] = value
            with self.assertRaises(ValueError):
                validate_poses(bad,self.geometry,self.cfg)

    def test_road_relative_anchor_and_interpolation(self):
        for sid in (1,2):
            low,high,target = [MapGeometry(width_scenario(w,sid)) for w in (7,14,10)]
            poses = np.array([[20.25,55.75,3,270,-15],[80.25,44.25,3,90,-20],
                              [55.75,20.25,4,180,-15]])
            c = relative_coordinates(poses,low)
            moved = relative_to_target(c,high)
            recovered = relative_coordinates(moved,high)
            np.testing.assert_allclose([p['s_from_anchor_m'] for p in c],
                                       [p['s_from_anchor_m'] for p in recovered])
            np.testing.assert_allclose(relative_to_target(recovered,low),poses)
            placements,meta = prepare_relative_placements(poses,moved[[2,0,1]],low,high,target,self.cfg)
            self.assertEqual(meta['status'],'available')
            np.testing.assert_allclose(placements['interpolation'],placements['transfer_low'])
            for values in placements.values():
                validate_poses(values,target,self.cfg)

    def test_incompatible_relative_interpolation_stays_unavailable(self):
        low,high,target = [MapGeometry(width_scenario(w,1)) for w in (7,14,10)]
        a = np.array([[28.25,55.75,3,265,-20],[56.25,42.25,3,135,-10],[72.25,55.75,3,265,-20]])
        b = np.array([[26.25,59.25,3,270,-15],[40.25,38.25,3,40,-10],[74.25,40.25,3,85,-15]])
        placements,meta = prepare_relative_placements(a,b,low,high,target,self.cfg)
        self.assertEqual(meta['status'],'unavailable')
        self.assertEqual(set(placements),{'transfer_low','transfer_high'})


if __name__ == '__main__':
    unittest.main(verbosity=2)

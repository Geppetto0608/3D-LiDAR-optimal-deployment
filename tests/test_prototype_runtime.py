"""Runtime accounting, segment-coordinate and bounded-search regression tests."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_offline import width_scenario
from prototype_segments import (installation_segments, segment_coordinates, segment_to_target,
                                prepare_segment_placements)
from prototype_runtime import jobs, parser, run, select_initial, summarize, validate_arguments, full_baseline
from src.core.candidate import Candidate, CandidateSet
from src.core.runtime import RuntimeMeter
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, get_lidar_profile, load_config, bake_visibility_map
from src.methods.local_mappo import refine_placement, validate_poses
from src.methods.local_search import refine_local_search


class PrototypeRuntimeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = deepcopy(load_config())
        self.cfg.mappo.update(local_map_size=3, ppo_update_timestep=4, ppo_k_epochs=1,
            ppo_minibatch_size=8, eval_interval=4, eval_steps=2)
        self.geometry = MapGeometry(width_scenario(10))
        self.poses = np.array([[30.25, 42.25, 3, 90, -20], [70.25, 57.25, 3, 270, -20]], dtype=np.float32)

    def test_segment_roundtrip_and_projection(self):
        for sid in (0, 1, 2):
            low, high, target = [MapGeometry(width_scenario(w, sid)) for w in (7, 14, 10)]
            poses = np.array([[20.25, 55.75, 3, 270, -15], [80.25, 44.25, 3, 90, -20]])
            s0, s1 = installation_segments(low), installation_segments(high)
            coords = segment_coordinates(poses, low, s0)
            np.testing.assert_allclose(segment_to_target(coords, low, s0), poses, atol=1e-6)
            moved = segment_to_target(coords, high, s1)
            c1 = segment_coordinates(moved, high, s1)
            np.testing.assert_allclose([c['u'] for c in coords], [c['u'] for c in c1])
            proposed, meta = prepare_segment_placements(poses, moved, low, high, target, self.cfg)
            self.assertEqual(meta['status'], 'available')
            for result in proposed.values():
                validate_poses(result, target, self.cfg)
            self.assertTrue(all(0 <= c['u'] <= 1 for c in meta['interpolated_coordinates']))
            if sid != 0:
                # Changing arm length changes the distance from its inner endpoint.
                self.assertNotAlmostEqual(c1[0]['s_from_anchor_m'], coords[0]['s_from_anchor_m'])

    def test_work_counts_actual_rays_and_selection(self):
        work = RuntimeMeter()
        profile = get_lidar_profile(self.cfg.lidar_profile, self.cfg)
        provider = MeshVisibility(self.geometry.create_environment_mesh(),
            self.geometry.target_points(self.cfg.target['coverage_z_slices']), profile, work=work)
        origin = self.poses[0, :3]
        expected = int(((np.linalg.norm(provider.targets - origin, axis=1) <= profile.max_range_m) &
                        (np.linalg.norm(provider.targets - origin, axis=1) > 1e-6)).sum())
        provider.line_of_sight_mask(origin)
        self.assertEqual(work.counts['rays_submitted'], expected)
        self.assertEqual(work.counts['raycast_calls'], 1)
        chosen, meta = select_initial(dict(transfer_low=self.poses, transfer_high=self.poses.copy()), provider, work)
        self.assertEqual(meta['selected'], 'transfer_low')
        np.testing.assert_array_equal(chosen, self.poses)
        self.assertEqual(work.counts['deployment_evaluations'], 2)
        self.assertEqual(work.counts['direct_pose_evaluations'], 4)
        self.assertEqual(len(work.trace), 1)

    def test_local_search_respects_domain_budget_and_direct_coverage(self):
        with patch('torch.cuda.is_available', return_value=False), \
             patch('src.methods.mappo.bake_visibility_map', side_effect=AssertionError('No full cache')):
            work = RuntimeMeter()
            poses, meta = refine_local_search(self.poses, self.geometry.scenario, self.cfg,
                radius_m=2, max_evaluations=15, work=work)
            self.assertLessEqual(meta['neighbor_evaluations'], 15)
            self.assertGreaterEqual(meta['coverage_pct'], meta['initial_coverage_pct'])
            self.assertTrue(np.all(np.linalg.norm(poses[:, :2] - self.poses[:, :2], axis=1) <= 2.00001))
            validate_poses(poses, self.geometry, self.cfg)
            provider = MeshVisibility(self.geometry.create_environment_mesh(),
                self.geometry.target_points(self.cfg.target['coverage_z_slices']),
                get_lidar_profile(self.cfg.lidar_profile, self.cfg))
            direct = 100 * np.stack([provider.mask(p) for p in poses]).any(axis=0).mean()
            self.assertAlmostEqual(meta['coverage_pct'], direct)
            again, _ = refine_local_search(self.poses, self.geometry.scenario, self.cfg,
                radius_m=2, max_evaluations=15)
            np.testing.assert_array_equal(again, poses)

    def test_mappo_measured_and_zero_step_deadline(self):
        with tempfile.TemporaryDirectory() as temp, patch('torch.cuda.is_available', return_value=False):
            for limit, expected in [(None, 8), (1e-9, 0)]:
                work = RuntimeMeter()
                poses, meta = refine_placement(self.poses, self.geometry.scenario, self.cfg,
                    Path(temp)/str(limit), steps=8, seed=7, radius_m=2, work=work, time_limit_sec=limit)
                self.assertEqual(meta['training_steps'], expected)
                self.assertEqual(work.counts['joint_training_steps'], expected)
                self.assertEqual(work.counts['agent_training_transitions'], expected * 2)
                self.assertGreater(meta['coverage_pct'], 0)
                self.assertGreater(work.counts['rays_submitted'], 0)
                self.assertGreater(work.counts['reward_evaluations'], 0)
                with np.load(Path(temp)/str(limit)/'train_log.npz') as log:
                    self.assertEqual(len(log['step']), expected)
                    self.assertEqual(log['eval_step'][-1], expected)

    def test_job_matrix_and_summary(self):
        args = parser().parse_args(['--output', 'unused', '--scenarios', '0'])
        validate_arguments(args)
        planned = list(jobs(args))
        self.assertEqual(len(planned), 12)  # 2*(1+1+3) + greedy + milp
        self.assertEqual(len({j['key'] for j in planned}), 12)
        self.assertTrue(all(j['seed'] is None for j in planned if j['method'] != 'mappo'))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            planned[0]['status'] = 'failed'
            summarize(root, dict(jobs=planned))
            self.assertIn('failed', (root/'comparison.csv').read_text())

    def test_worker_cannot_bypass_execute_guard(self):
        args = parser().parse_args(['--output', 'unused', '--worker', '0'])
        with self.assertRaisesRegex(ValueError, '--execute'):
            run(args)

    def test_full_cache_counter_and_reuse(self):
        self.cfg.candidate.update(z_min=3., z_max=3.)
        geometry = self.geometry
        profile = get_lidar_profile(self.cfg.lidar_profile, self.cfg)
        work = RuntimeMeter()
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.environment.MapGeometry', return_value=geometry), \
             patch.object(geometry, 'target_points', return_value=np.array([[50., 49., 1.], [50., 51., 1.]], dtype=np.float32)):
            self.cfg.output['cache_dir'] = temp
            first = bake_visibility_map(self.cfg, geometry.scenario, profile, work=work)
            self.assertGreater(work.counts['raycast_calls'], 0)
            self.assertEqual(work.counts['rays_submitted'], 2 * work.counts['raycast_calls'])
            calls = work.counts['raycast_calls']
            second = bake_visibility_map(self.cfg, geometry.scenario, profile, work=work)
            self.assertEqual(work.counts['raycast_calls'], calls)
            self.assertEqual(work.counts['full_cache_loads'], 1)
            self.assertEqual(first['cache_key'], second['cache_key'])

    def test_baseline_wrapper_with_toy_candidates(self):
        masks = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
        candidates = CandidateSet([Candidate(i, 1., 1., 3., 90., -20., np.array(m, dtype=bool))
            for i, m in enumerate(masks)], np.zeros((4, 3)), 'toy', 'toy', dict(num_coverage_edges=6))
        args = parser().parse_args(['--output', 'unused', '--solver-seconds', '2', '--gap', '0'])
        with tempfile.TemporaryDirectory() as temp, \
             patch('src.roadside.candidate_generator.build_oriented_candidates', return_value=candidates) as builder:
            for method in ('greedy', 'milp'):
                work = RuntimeMeter()
                poses, meta = full_baseline(dict(method=method), args, Path(temp), deepcopy(self.cfg), self.geometry, work)
                self.assertLessEqual(len(poses), args.k)
                self.assertEqual(meta['coverage_pct'], 100.)
                self.assertIs(builder.call_args.kwargs['work'], work)
                if method == 'milp':
                    self.assertEqual(meta['solver']['mip_gap'], 0.)
                self.assertIn('full_visibility_and_candidates', work.stages)


if __name__ == '__main__':
    unittest.main()

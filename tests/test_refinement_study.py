"""Regression checks for pairing, correspondence diagnosis and cold transfer."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_refinement_study import (allocation_difference, make_initial, make_plan,
                                       pose_hash, summarize)
from prototype_offline import explicit_scenario, save
from src.environment import MapGeometry, load_config


class RefinementStudyTests(unittest.TestCase):
    def test_allocation_counts_not_just_unique_keys(self):
        low = {'south/left': 2, 'east/right': 1}
        high = {'south/left': 1, 'east/right': 2}
        self.assertEqual(allocation_difference(low, high), {'east/right': 1, 'south/left': -1})

    def test_single_source_does_not_load_other_prototype_or_target_archive(self):
        source = dict(id=410001, parent_scenario_id=1, topology='T_JUNCTION',
            road_width=7., num_lanes=2, provenance='synthetic_junction_offset',
            geometry=dict(junction_x_m=35., junction_y_m=55.))
        target = dict(source, id=420001, road_width=9., geometry=dict(junction_x_m=37., junction_y_m=57.))
        geometry = MapGeometry(explicit_scenario(source))
        cfg = deepcopy(load_config())
        poses = np.array([[70.25, 49.25, 3., 90., -20.]], dtype=np.float32)
        prototype = (None, {}, {}, geometry, poses, cfg)
        job = dict(initialization='transfer_low', budget_k=1, scenario=target,
                   source_roots=['low', 'forbidden_high'], reference_pose='forbidden_target')
        with patch('prototype_refinement_study.load_prototype', return_value=prototype) as load, \
             patch('src.environment.bake_visibility_map', side_effect=AssertionError('Full cache forbidden')):
            initial, _, _ = make_initial(job)
        load.assert_called_once_with('low', 1)
        self.assertEqual(initial.shape, (1, 5))

    def test_available_job_matrix_and_unavailable_not_zero(self):
        suite = json.loads(Path('configs/library_expansion.json').read_text())
        suite['families'] = suite['families'][:1]
        family = suite['families'][0]
        low_pose = np.array([[70.25, 49.25, 3., 90., -20.]])
        high_pose = np.array([[70.25, 45.25, 3., 90., -20.]])
        protos = [(None, {}, {}, MapGeometry(explicit_scenario(s)), p, None)
                  for s, p in zip(family['sources'], [low_pose, high_pose])]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root/'suite.json', suite)
            save(root/'manifest.json', dict(status='complete', plan=dict(k_values=[1])))
            for target in family['targets']:
                (root/'transfers'/f's{target["id"]}'/'k1').mkdir(parents=True)
                save(root/'transfers'/f's{target["id"]}'/'k1/report.json',
                    dict(scenario=target, budget_k=1, placements={
                        'transfer_low': dict(status='complete', positions_npy='low.npy'),
                        'transfer_high': dict(status='complete', positions_npy='high.npy'),
                        'interpolation': dict(status='unavailable', reason='test mismatch')}))
            with patch('prototype_refinement_study.load_prototype', side_effect=protos):
                jobs, diagnostics = make_plan(root, [1], [0, 1, 2])
            self.assertEqual(len(jobs), 16)
            self.assertEqual(len({j['id'] for j in jobs}), 16)
            self.assertEqual(len(diagnostics), 2)
            self.assertFalse(any(j['initialization']=='interpolation' for j in jobs))

    def test_pair_summary_rejects_different_initial_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = []
            for method, value, digest in [('local_search', 90., 'a'), ('mappo', 92., 'a')]:
                job = dict(id=method, status='complete', scenario=dict(id=1), budget_k=3,
                    initialization='transfer_low', method=method, seed=0 if method=='mappo' else None,
                    worker_wall_sec=4.)
                jobs.append(job)
                (root/'jobs'/method).mkdir(parents=True)
                save(root/'jobs'/method/'result.json', dict(coverage_pct=value, initial_coverage_pct=80.,
                    gain_pp=value-80., initial_pose_sha256=digest, online_runtime=dict(elapsed_sec=3., counts={}),
                    validation_runtime=dict(elapsed_sec=.1), refinement=dict(elapsed_sec=2.)))
            manifest = dict(jobs=jobs, status='complete', seeds=[0], limits=dict(seconds=2.))
            summarize(root, manifest)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['pairs'][0]['mappo_minus_local_search_pp'], 2.)
            self.assertIsNone(summary['pairs'][0]['mappo_sample_sd_pct'])
            path = root/'jobs/mappo/result.json'
            data = json.loads(path.read_text())
            data['initial_pose_sha256'] = 'different'
            save(path, data)
            with self.assertRaisesRegex(RuntimeError, 'identical initial'):
                summarize(root, manifest)

    def test_pose_hash_dtype_independent(self):
        a = np.array([[1., 2., 3., 4., 5.]])
        self.assertEqual(pose_hash(a), pose_hash(a.astype(np.float32)))
        self.assertNotEqual(pose_hash(a), pose_hash(a+1))


if __name__ == '__main__':
    unittest.main()

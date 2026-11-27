"""Packed candidate parity, planning safety and a small real full-grid worker."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_full_grid import parser, run, summarize
from prototype_offline import explicit_scenario
from src.environment import load_config
from src.methods.coverage_cut_milp import PackedCoverage
from src.roadside.packed_candidates import build_packed_candidates, packed_greedy


class FullGridTests(unittest.TestCase):
    def test_packed_masks_poses_and_padding(self):
        cfg=load_config()
        cfg.candidate.update(xy_domain='installable_grid',max_orientations_per_position=0,
                             pitch_min_deg=-5,pitch_max_deg=-5)
        row=json.loads(Path('configs/library_expansion.json').read_text())['families'][0]['targets'][0]
        vis=dict(target_points=np.zeros((9,3)),cache_key='toy',meta={'config_hash':'toy'})
        with tempfile.TemporaryDirectory() as directory, \
             patch('src.roadside.packed_candidates.installation_base_positions',return_value=[(1.,2.,3.),(4.,5.,3.)]), \
             patch('src.roadside.packed_candidates.allowed_mount_yaws_deg',return_value=np.array([0.,90.])), \
             patch('src.roadside.packed_candidates.bake_visibility_map',return_value=vis), \
             patch('src.roadside.packed_candidates.RoadsideVisibility') as provider:
            provider.return_value.mask.side_effect=lambda pose: np.arange(9) % 3 == int(pose[0]) % 3
            poses,masks,targets,stats=build_packed_candidates(explicit_scenario(row),cfg,Path(directory)/'archive')
            self.assertEqual(poses.shape,(4,5))
            self.assertEqual(masks.shape,(4,2))
            self.assertEqual(stats['num_candidates'],4)
            for pose,mask in zip(poses,masks):
                np.testing.assert_array_equal(np.unpackbits(mask)[:9],provider.return_value.mask(pose))
            oracle=PackedCoverage(masks,len(targets))
            self.assertEqual(oracle.count(packed_greedy(masks,9,2)),3)
            with self.assertRaises(FileExistsError):
                build_packed_candidates(explicit_scenario(row),cfg,Path(directory)/'archive')

    def test_greedy_union_and_at_most(self):
        masks=np.packbits(np.eye(5,dtype=bool),axis=1)
        ids=packed_greedy(masks,5,3)
        self.assertEqual(ids,[0,1,2])
        self.assertEqual(PackedCoverage(masks,5).count(ids),3)
        self.assertEqual(packed_greedy(np.zeros((2,1),dtype=np.uint8),5,3),[])

    def test_planning_does_not_execute_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'new'
            with patch('prototype_full_grid.subprocess.run') as subprocess, contextlib.redirect_stdout(io.StringIO()):
                run(parser().parse_args(['--output',str(output)]))
            subprocess.assert_not_called()
            self.assertFalse(output.exists())

    def test_pending_not_counted_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            summarize(root,[dict(scenario={'id':1},k_values=[3,4],status='pending')])
            value=json.loads((root/'summary.json').read_text())
            self.assertEqual((value['completed'],value['expected']),(0,2))
            self.assertFalse((root/'summary.csv').exists())

    def test_small_real_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=load_config()
            cfg.candidate.update(z_min=3,z_max=3,yaw_step_deg=110,pitch_step_deg=20)
            cfg.target['coverage_z_slices']=[1.5]
            (root/'config.yaml').write_text(yaml.safe_dump(cfg.raw))
            row=json.loads(Path('configs/library_expansion.json').read_text())['families'][0]['targets'][0]
            (root/'scenario.json').write_text(json.dumps(row))
            output=root/'run'
            args=parser().parse_args(['--config',str(root/'config.yaml'),'--worker',str(root/'scenario.json'),
                '--output',str(output),'--k-values','1','--seconds','2','--swap-seconds','0','--execute'])
            with patch('prototype_transfer.export_view'), contextlib.redirect_stdout(io.StringIO()):
                run(args)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete')
            result=json.loads((output/'k1/result.json').read_text())
            self.assertTrue(result['validated_direct_raycast'])
            self.assertFalse(result['archive_reused'])
            masks=np.load(output/'candidates/packed_masks.npy',mmap_mode='r')
            oracle=PackedCoverage(masks,result['target_count'])
            self.assertEqual(result['objective'],int(oracle.gains(oracle.union([])).max()))


if __name__=='__main__':
    unittest.main(verbosity=2)

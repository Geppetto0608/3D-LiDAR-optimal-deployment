"""Saved-array report must reject wrong targets, masks, or candidate poses."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_full_grid_report import Archive


class ArchiveReportTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.poses=np.array([[.25,.25,3,0,-5],[.25,.25,3,90,-5],[2.25,.25,3,0,-5]],dtype=np.float32)
        self.targets=np.arange(9).reshape(3,3)
        np.save(root/'poses.npy',self.poses)
        np.save(root/'targets.npy',self.targets)
        np.save(root/'packed_masks.npy',np.packbits([[1,0,0],[0,1,0],[0,0,1]],axis=1))
        self.archive=Archive(root)

    def test_valid_union_and_wrapped_yaw(self):
        poses=self.poses[[0,2]].copy();poses[0,3]=360
        self.archive.check(poses,self.targets,np.array([1,0,1]),200/3,2)

    def test_target_mismatch(self):
        with self.assertRaisesRegex(ValueError,'Target'):
            self.archive.check(self.poses[:1],self.targets+1,np.array([1,0,0]),100/3,1)

    def test_density_and_coverage_mismatch(self):
        with self.assertRaisesRegex(ValueError,'density'):
            self.archive.check(self.poses[:1],self.targets,np.array([0,1,0]),100/3,1)
        with self.assertRaisesRegex(ValueError,'coverage'):
            self.archive.check(self.poses[:1],self.targets,np.array([1,0,0]),0,1)

    def test_missing_pose_and_budget(self):
        poses=self.poses[:1].copy();poses[0,3]=45
        with self.assertRaisesRegex(ValueError,'Pose missing'):
            self.archive.check(poses,self.targets,np.array([1,0,0]),100/3,1)
        with self.assertRaisesRegex(ValueError,'Invalid deployment'):
            self.archive.check(self.poses,self.targets,np.ones(3),100,1)


if __name__=='__main__':
    unittest.main(verbosity=2)

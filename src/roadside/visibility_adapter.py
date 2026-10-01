"""Expose legacy grid visibility through the common mask API."""
from src.environment import coverage_mask_for_pose


class RoadsideVisibility:
    def __init__(self, visibility, geometry, profile, config):
        self.visibility = visibility
        self.geometry = geometry
        self.profile = profile
        self.config = config

    def mask(self, pose):
        return coverage_mask_for_pose(pose, self.visibility, self.geometry, self.profile, self.config)

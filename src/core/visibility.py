"""Mesh visibility for explicit poses, using legacy optical conventions."""
from typing import Protocol
import numpy as np
from src.environment import compute_view_angles_numpy


class VisibilityProvider(Protocol):
    def mask(self, pose: np.ndarray) -> np.ndarray: ...


class MeshVisibility:
    def __init__(self, mesh, targets, profile):
        import open3d as o3d
        self.o3d = o3d
        self.targets = np.asarray(targets, dtype=np.float32)
        self.profile = profile
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def mask(self, pose):
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape != (5,) or not np.isfinite(pose).all():
            raise ValueError('Expected finite [x,y,z,yaw,pitch] pose; roll support is pending')
        vectors = self.targets - pose[:3]
        az, el, forward, distance = compute_view_angles_numpy(*vectors.T, pose[3], pose[4])
        half_h, half_v = np.deg2rad([self.profile.h_fov_deg, self.profile.v_fov_deg]) / 2
        optical = ((np.abs(az) <= half_h) & (np.abs(el) <= half_v)
                   & (distance <= self.profile.max_range_m) & (distance > 1e-6))
        if half_h < np.pi - 1e-6:
            optical &= forward > 0.5
        result = np.zeros(len(distance), dtype=bool)
        idx = np.flatnonzero(optical)
        if not len(idx):
            return result
        rays = np.concatenate([np.broadcast_to(pose[:3], (len(idx), 3)),
                               vectors[idx] / distance[idx, None]], axis=1).astype(np.float32)
        hits = self.scene.cast_rays(self.o3d.core.Tensor(rays))['t_hit'].numpy()
        result[idx] = (~np.isfinite(hits)) | (hits >= distance[idx] - 1e-3)
        return result

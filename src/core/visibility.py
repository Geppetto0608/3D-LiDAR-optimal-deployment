"""Mesh visibility for explicit poses, using shared angular/range optics."""
from typing import Protocol
import numpy as np
from src.environment import angular_visibility_mask, compute_view_angles_numpy


class VisibilityProvider(Protocol):
    def mask(self, pose: np.ndarray) -> np.ndarray: ...


class MeshVisibility:
    def __init__(self, mesh, targets, profile, *, work=None):
        import open3d as o3d
        self.o3d = o3d
        self.targets = np.asarray(targets, dtype=np.float32)
        self.profile = profile
        self.work = work
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def _cast(self, rays):
        if self.work is not None:
            self.work.add('raycast_calls')
            self.work.add('rays_submitted', len(rays))
        return self.scene.cast_rays(self.o3d.core.Tensor(rays))['t_hit'].numpy()

    def line_of_sight_mask(self, origin):
        """Range-limited LOS without orientation, for on-demand road observations."""
        if self.work is not None:
            self.work.add('los_queries')
        origin = np.asarray(origin, dtype=np.float32)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('Expected finite XYZ origin')
        vectors = self.targets - origin
        distance = np.linalg.norm(vectors, axis=1)
        idx = np.flatnonzero((distance > 1e-6) & (distance <= self.profile.max_range_m))
        result = np.zeros(len(distance), dtype=bool)
        if len(idx):
            rays = np.concatenate([np.broadcast_to(origin, (len(idx), 3)),
                                   vectors[idx] / distance[idx, None]], axis=1).astype(np.float32)
            hits = self._cast(rays)
            result[idx] = (~np.isfinite(hits)) | (hits >= distance[idx] - 1e-3)
        return result

    def mask(self, pose):
        if self.work is not None:
            self.work.add('direct_pose_evaluations')
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape != (5,) or not np.isfinite(pose).all():
            raise ValueError('Expected finite [x,y,z,yaw,pitch] pose; roll support is pending')
        vectors = self.targets - pose[:3]
        az, el, _, distance = compute_view_angles_numpy(*vectors.T, pose[3], pose[4])
        half_h, half_v = np.deg2rad([self.profile.h_fov_deg, self.profile.v_fov_deg]) / 2
        optical = angular_visibility_mask(az, el, distance, half_h, half_v, self.profile.max_range_m)
        result = np.zeros(len(distance), dtype=bool)
        idx = np.flatnonzero(optical)
        if not len(idx):
            return result
        rays = np.concatenate([np.broadcast_to(pose[:3], (len(idx), 3)),
                               vectors[idx] / distance[idx, None]], axis=1).astype(np.float32)
        hits = self._cast(rays)
        result[idx] = (~np.isfinite(hits)) | (hits >= distance[idx] - 1e-3)
        return result

"""Opt-in opaque boxes for library target evaluation, without changing domains."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


SCHEMA = 'library_scene_occlusion_v1'


def validate_spec(spec):
    if (not isinstance(spec, dict) or set(spec) != {'schema', 'boxes'}
            or spec['schema'] != SCHEMA or not isinstance(spec['boxes'], list)
            or len(spec['boxes']) > 100):
        raise ValueError('Expected versioned occlusion spec with at most 100 boxes')
    result = deepcopy(spec)
    for box in result['boxes']:
        if not isinstance(box, dict) or set(box) != {'center_m', 'size_m', 'yaw_deg'}:
            raise ValueError('Each box requires center_m, size_m, yaw_deg only')
        for key in ('center_m', 'size_m'):
            values = box[key]
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(v) not in (int, float) for v in values)
                    or not np.isfinite(values).all()):
                raise ValueError('Box coordinates must be three finite numbers')
        yaw = box['yaw_deg']
        if type(yaw) not in (int, float) or not np.isfinite(yaw):
            raise ValueError('Box yaw must be finite')
        if np.any(np.asarray(box['size_m']) <= 0):
            raise ValueError('Box dimensions must be positive')
    return result


def _rotation(yaw):
    angle = np.deg2rad(yaw)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@dataclass(frozen=True)
class OccludedGeometry:
    base: object
    boxes: tuple

    def create_environment_mesh(self):
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh(self.base.create_environment_mesh())
        for box in self.boxes:
            size = np.asarray(box['size_m'], dtype=float)
            obstacle = o3d.geometry.TriangleMesh.create_box(*size)
            obstacle.translate(-size / 2)
            obstacle.rotate(_rotation(box['yaw_deg']), center=(0., 0., 0.))
            obstacle.translate(box['center_m'])
            mesh += obstacle
        return mesh


def with_occlusion(problem, spec):
    spec = validate_spec(spec)
    if 'scene_occlusion' in problem['context']:
        raise ValueError('Occlusion already attached to this problem')
    for box in spec['boxes']:
        for key in ('poses', 'targets'):
            points = np.asarray(problem[key])[:, :3]
            local = (points - box['center_m']) @ _rotation(box['yaw_deg'])
            inside = np.all(np.abs(local) <= np.asarray(box['size_m']) / 2 + 1e-6, axis=1)
            if np.any(inside):
                raise ValueError(f'Occluder intersects {key}; domains must remain unchanged')
    # Only the physical mesh changes; planning receives the original arrays/pool.
    return dict(problem, geometry=OccludedGeometry(problem['geometry'], tuple(spec['boxes'])),
                context=dict(problem['context'], scene_occlusion=deepcopy(spec)))

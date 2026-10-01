"""Application-neutral mesh inputs in metres and legacy pose5 convention."""
import hashlib
import json
import numpy as np
from src.core.candidate import Candidate, CandidateSet
from src.core.models import PlacementProblem
from src.core.visibility import MeshVisibility
from src.environment import LidarProfile


def build_mesh_problem(data):
    import open3d as o3d
    vertices = np.asarray(data['vertices'], dtype=float)
    triangles = np.asarray(data['triangles'])
    poses = np.asarray(data['poses'], dtype=float)
    targets = np.asarray(data['targets'], dtype=float)
    for name, array, cols in [('vertices', vertices, 3), ('poses', poses, 5), ('targets', targets, 3)]:
        if array.ndim != 2 or array.shape[1] != cols or not len(array) or not np.isfinite(array).all():
            raise ValueError(f'{name}: expected finite nonempty [N,{cols}]')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles) or triangles.dtype.kind not in 'iu':
        raise ValueError('triangles: expected integer [N,3]')
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError('Triangle index out of bounds')
    if len(poses) > 2000 or len(targets) > 20000 or len(poses)*len(targets) > 20000000:
        raise ValueError('Interactive limit: 2000 poses, 20000 targets, 20M pose-target pairs')
    profile = LidarProfile(**data['sensor'])
    if not (0 < profile.h_fov_deg <= 360 and 0 < profile.v_fov_deg <= 180 and np.isfinite(profile.max_range_m) and profile.max_range_m > 0):
        raise ValueError('Invalid sensor FOV/range')
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    provider = MeshVisibility(mesh, targets, profile)
    candidates = [Candidate(i, *p, provider.mask(p)) for i, p in enumerate(poses)]
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    candidate_set = CandidateSet(candidates, targets, digest, digest, {})
    problem = PlacementProblem(candidate_set, 1, importance=data.get('importance'), costs=data.get('costs'))
    return problem, mesh

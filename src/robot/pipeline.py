import numpy as np
from src.core.candidate import Candidate, CandidateSet
from src.core.models import PlacementProblem
from src.core.objective import ObjectiveWeights
from src.core.visibility import MeshVisibility
from src.environment import get_lidar_profile


def build_problem(config, budget_k):
    import open3d as o3d
    robot = config.raw['robot']
    if robot.get('platform_mesh'):
        mesh = o3d.io.read_triangle_mesh(robot['platform_mesh'])
        if mesh.is_empty():
            raise ValueError('Platform mesh is empty')
    else:
        dimensions = np.asarray(robot['box_dimensions_m'], dtype=float)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
            raise ValueError('Box dimensions must be three positive finite metres')
        mesh = o3d.geometry.TriangleMesh.create_box(*dimensions)
    poses = np.asarray(robot['mount_poses'], dtype=np.float32)
    targets = np.asarray(robot['target_points'], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 5 or not np.isfinite(poses).all():
        raise ValueError('mount_poses must be finite [M,5] approved sensor origins outside the body')
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
        raise ValueError('target_points must be finite [N,3]')
    provider = MeshVisibility(mesh, targets, get_lidar_profile(config.lidar_profile, config))
    candidates = [Candidate(i, *map(float, pose), provider.mask(pose)) for i, pose in enumerate(poses)]
    candidate_set = CandidateSet(candidates, targets, '', '', {'num_candidates': len(candidates), 'num_targets': len(targets)})
    return PlacementProblem(candidate_set, budget_k, ObjectiveWeights(**config.raw.get('objective', {})),
                            importance=robot.get('target_weights'), costs=robot.get('installation_costs'))

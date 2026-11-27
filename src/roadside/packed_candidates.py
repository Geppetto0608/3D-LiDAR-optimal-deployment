"""Stream full-grid poses and packed masks without a dense candidate matrix."""
from pathlib import Path
import time

import numpy as np

from src.environment import (MapGeometry, allowed_mount_yaws_deg, bake_visibility_map,
    get_lidar_profile, installation_base_positions)
from src.roadside.candidate_generator import _frange
from src.roadside.visibility_adapter import RoadsideVisibility


def build_packed_candidates(scenario, config, root, *, work=None, progress=None):
    if config.xy_domain != 'installable_grid' or config.max_orientations_per_position != 0:
        raise ValueError('Full installable grid without orientation pruning required')
    if config.target.get('road_detection', 1) != 1:
        raise ValueError('Only binary union coverage supported')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    geometry = MapGeometry(scenario)
    profile = get_lidar_profile(config.lidar_profile, config)
    bases = installation_base_positions(geometry, config)
    pitches = list(_frange(config.pitch_min_deg, config.pitch_max_deg, config.pitch_step_deg))
    yaws = [allowed_mount_yaws_deg(x,y,geometry,config.yaw_step_deg,
            config.yaw_axis_min_deg,config.yaw_axis_max_deg) for x,y,_ in bases]
    count = sum(len(v)*len(pitches) for v in yaws)
    if not count:
        raise ValueError('Empty candidate domain')
    start = time.perf_counter()
    visibility = bake_visibility_map(config, scenario, profile, work=work)
    cache_sec = time.perf_counter()-start
    targets = np.asarray(visibility['target_points'])
    if not len(targets):
        raise ValueError('Empty target domain')
    np.save(root/'targets.npy', targets)
    poses = np.lib.format.open_memmap(root/'poses.npy', mode='w+', dtype=np.float32, shape=(count,5))
    packed = np.lib.format.open_memmap(root/'packed_masks.npy', mode='w+', dtype=np.uint8,
                                     shape=(count,(len(targets)+7)//8))
    provider = RoadsideVisibility(visibility,geometry,profile,config)
    start = time.perf_counter()
    index, edges = 0, 0
    for base_index, ((x,y,z), angles) in enumerate(zip(bases,yaws)):
        for yaw in angles:
            for pitch in pitches:
                pose = [x,y,z,float(yaw),pitch]
                mask = provider.mask(pose)
                poses[index] = pose
                packed[index] = np.packbits(mask)
                edges += int(mask.sum())
                index += 1
                if work is not None:
                    work.add('candidate_pose_evaluations')
        if progress is not None and ((base_index+1) % 128 == 0 or index == count):
            progress(index,count)
    if index != count:
        raise RuntimeError('Candidate count mismatch')
    poses.flush()
    packed.flush()
    return poses, packed, targets, dict(num_candidates=count,num_targets=len(targets),
        num_base_positions=len(bases),num_coverage_edges=edges,cache_sec=cache_sec,
        candidate_mask_write_sec=time.perf_counter()-start, packed_mask_bytes=packed.nbytes,
        top_m=0, xy_domain=config.xy_domain, exact_mask_dedup=False,
        cache_key=visibility['cache_key'],config_hash=visibility['meta']['config_hash'])


def packed_greedy(masks, targets, k):
    from src.methods.coverage_cut_milp import PackedCoverage
    oracle = PackedCoverage(masks,targets)
    ids = []
    for _ in range(k):
        gains = oracle.gains(oracle.union(ids))
        gains[ids] = -1
        index = int(np.argmax(gains))
        if gains[index] <= 0:
            break
        ids.append(index)
    return ids

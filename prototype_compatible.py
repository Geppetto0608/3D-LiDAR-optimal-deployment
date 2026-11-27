"""Build a width prototype constrained to a reference installation-line allocation."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from itertools import permutations
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import yaml

from prototype_offline import canonical, certified, describe, digest, save
from prototype_transfer import load_prototype, export_view
from src.core.visibility import MeshVisibility
from src.environment import evaluate_coverage, get_lidar_profile, visibility_cache_path
from src.methods.local_mappo import validate_poses
from src.methods.milp import _solve_with_gurobi
from src.roadside.candidate_generator import _base_positions, build_oriented_candidates


def line_key(pose, geometry):
    p = canonical([pose], geometry)[0]
    return p['branch_id'], p['installation_line_id']


def allocation(poses, geometry):
    return Counter(line_key(p,geometry) for p in poses)


def group_warm_start(candidate_set, groups):
    """Best of greedy group orders; one candidate per group, no target-map leakage."""
    if len(groups) > 6 or any(count != 1 for ids,count in groups):
        raise ValueError('Pilot warm start supports at most six groups with quota one')
    packed = np.stack([np.packbits(c.covered) for c in candidate_set.candidates])
    bitcount = np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    best_ids, best_count = None, -1
    for order in permutations(range(len(groups))):
        union = np.zeros(packed.shape[1],dtype=np.uint8)
        selected = []
        for index in order:
            ids = np.asarray(groups[index][0],dtype=int)
            if not len(ids):
                raise ValueError('Required installation group has no candidates')
            gain = bitcount[packed[ids] & ~union].sum(axis=1)
            winner = int(ids[int(np.argmax(gain))])
            selected.append(winner)
            union |= packed[winner]
        covered = int(bitcount[union].sum())
        if covered > best_count:
            best_ids, best_count = selected, covered
    return best_ids, best_count


def run(args):
    if not np.isfinite(args.time_limit) or args.time_limit <= 0:
        raise ValueError('Positive finite solver time limit required')
    low, high = load_prototype(args.reference,args.k), load_prototype(args.template,args.k)
    geometry, cfg = high[3], high[5]
    if low[3].scenario.topology != geometry.scenario.topology or low[3].scenario.road_width >= geometry.scenario.road_width:
        raise ValueError('Reference/template must have the same topology and increasing widths')
    for field in ('candidate','target','mappo'):
        if getattr(cfg,field) != getattr(low[5],field):
            raise ValueError(f'Reference/template {field} mismatch')
    if cfg.max_orientations_per_position != 0:
        raise ValueError('Auxiliary prototype requires full orientations, topM=0')
    for field in ('map_len','map_width','grid_size','sidewalk_width','building_height','median_width','road_centers_y','num_lanes'):
        if getattr(geometry,field) != getattr(low[3],field):
            raise ValueError(f'Only width changes supported: {field}')
    profile = get_lidar_profile(cfg.lidar_profile,cfg)
    if asdict(profile) != asdict(get_lidar_profile(low[5].lidar_profile,low[5])):
        raise ValueError('Sensor profiles differ')
    validate_poses(low[4],low[3],low[5])
    quotas = allocation(low[4],low[3])
    if len(low[4]) != args.k or any(v != 1 for v in quotas.values()):
        raise ValueError('Current pilot supports exactly one sensor per reference installation group')

    @lru_cache(maxsize=None)
    def group_for_xyz(x,y,z):
        return line_key([x,y,z,0,0],geometry)

    available = Counter(group_for_xyz(*p) for p in _base_positions(geometry,cfg))
    if any(available[k] < v for k,v in quotas.items()):
        raise ValueError('Required target installation group is empty')
    quota_records = [dict(branch_id=k[0],installation_line_id=k[1],count=v,
                          base_positions=available[k]) for k,v in sorted(quotas.items())]
    plan = dict(scenario_id=geometry.scenario.id,topology=geometry.scenario.topology,
                reference_width=low[3].scenario.road_width,target_width=geometry.scenario.road_width,
                budget_k=args.k,allocation=quota_records,topM=0,
                solver_time_limit_sec=args.time_limit,soft_mem_limit_gb=90,threads=1,
                dedup_scope='same installation group only; quota=1')
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:
        return
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    manifest = dict(schema_version='compatible_prototype_v1',status='running',plan=plan,
        started_at=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        optimality_domain='full declared discrete grid with exact installation-group allocation',
        reference_result_sha256=digest(low[0]/f'k{args.k}/result.json'),
        reference_pose_sha256=digest(low[0]/low[1]['positions_npy']),
        template_result_sha256=digest(high[0]/f'k{args.k}/result.json'),
        reference_root=str(low[0].resolve()),template_root=str(high[0].resolve()))
    save(root/'manifest.json',manifest)
    try:
        source_paths = [Path(__file__),Path('prototype_offline.py'),Path('prototype_transfer.py'),
                        Path('data/scenarios/yeongjong_final.json'),*Path('src').rglob('*.py')]
        hashes = {}
        for path in source_paths:
            hashes[str(path)] = digest(path)
            dest = root/'source'/(path.name if path.is_absolute() else path)
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
        manifest['source_hashes'] = hashes
        # Only the physical visibility cache is reused, never globally deduplicated poses.
        previous = json.loads((high[0]/'manifest.json').read_text())
        if previous['source_hashes'].get('src/environment.py') != digest('src/environment.py'):
            raise ValueError('Cannot reuse visibility cache after environment source changes')
        _, old_cache = visibility_cache_path(cfg,geometry.scenario,profile)
        if not old_cache.exists():
            raise FileNotFoundError(old_cache)
        (root/'cache').mkdir()
        shutil.copy2(old_cache,root/'cache'/old_cache.name)
        manifest['visibility_cache_reuse'] = dict(source=str(old_cache),sha256=digest(old_cache),
                                                   environment_sha256=digest('src/environment.py'))
        cfg.raw['output'] = dict(cache_dir=str(root/'cache'),results_dir=str(root))
        cfg.solver.update(time_limit_sec=args.time_limit,mip_gap=0.,threads=1,soft_mem_limit_gb=90,
                          sparse_target_batch_size=128,nodefile_dir=str(root/'nodefiles'),
                          log_file=str(root/'gurobi.log'))
        (root/'config.yaml').write_text(yaml.safe_dump(cfg.raw,sort_keys=False),encoding='utf-8')
        save(root/'geometry.json',describe(geometry,cfg))
        save(root/'allocation.json',quota_records)
        stage = time.perf_counter()
        print('Building full orientations within required installation groups',flush=True)
        candidates = build_oriented_candidates(geometry.scenario,cfg.lidar_profile,cfg,
            position_filter=lambda x,y,z: group_for_xyz(x,y,z) in quotas,dedup_group=group_for_xyz)
        manifest['candidate_build_sec'] = time.perf_counter()-stage
        manifest['candidate_stats'] = candidates.stats
        save(root/'manifest.json',manifest)
        if candidates.stats['num_coverage_edges'] > 1_500_000_000:
            raise RuntimeError('Edge safety limit exceeded without pruning candidates')
        ids_by_group = {key:[] for key in sorted(quotas)}
        for c in candidates.candidates:
            ids_by_group[group_for_xyz(c.x,c.y,c.z)].append(c.candidate_id)
        groups = [(ids,quotas[key]) for key,ids in ids_by_group.items()]
        start_ids,start_count = group_warm_start(candidates,groups)
        manifest['warm_start'] = dict(selected_ids=start_ids,covered_count=start_count,
                                      method='best quota-greedy order')
        save(root/'manifest.json',manifest)
        print(f'Candidates={len(candidates.candidates)}, edges={candidates.stats["num_coverage_edges"]}, warm coverage={start_count}',flush=True)
        metadata = _solve_with_gurobi(candidates,args.k,cfg,start_ids,selection_groups=groups)
        selected = metadata['selected_ids']
        poses = candidates.positions_array(selected)
        validate_poses(poses,geometry,cfg)
        if allocation(poses,geometry) != quotas:
            raise RuntimeError('Solver selection violated reference group allocation')
        result_eval = evaluate_coverage(poses,geometry.scenario,cfg.lidar_profile,cfg)
        provider = MeshVisibility(geometry.create_environment_mesh(),candidates.target_points,profile)
        density = np.stack([provider.mask(p) for p in poses]).sum(axis=0)
        union = np.any([candidates.candidates[i].covered for i in selected],axis=0)
        if not np.array_equal(density>0,union) or not np.array_equal(result_eval.covered_mask,union):
            raise RuntimeError('Candidate/cache/direct raycast mismatch')
        count = int(union.sum())
        if metadata['objective'] is not None and not np.isclose(metadata['objective'],count,atol=1e-5,rtol=0):
            raise RuntimeError('Solver objective mismatch')
        out = root/f'k{args.k}'; out.mkdir()
        np.save(out/'poses.npy',poses)
        save(out/'canonical_poses.json',canonical(poses,geometry))
        problem = dict(config_hash=candidates.config_hash,allocation=quota_records,dedup='within_group',topM=0)
        record = dict(schema_version='prototype_result_v1',scenario_id=geometry.scenario.id,budget_k=args.k,
            num_selected=len(poses),coverage_pct=100*count/len(union),covered_count=count,num_targets=len(union),
            overlap_target_count=int((density>=2).sum()),uncovered_count=int((density==0).sum()),
            certification='constrained_discrete_optimal' if certified(metadata,count) else 'constrained_best_known',
            optimality_domain=manifest['optimality_domain'],budget_mode='exact_reference_allocation',
            installation_allocation=quota_records,solver=metadata,positions_npy=f'k{args.k}/poses.npy',
            cache_key=candidates.cache_key,config_hash=candidates.config_hash,
            problem_hash=hashlib.sha256(json.dumps(problem,sort_keys=True).encode()).hexdigest())
        save(out/'result.json',record)
        save(out/'validation.json',dict(passed=True,exact_allocation=True,direct_cached_candidate_parity=True))
        export_view(out/'view.html',geometry.create_environment_mesh(),candidates.target_points,density,poses,
                    'Constrained auxiliary prototype',record['coverage_pct'])
        manifest.update(status='complete',result=record)
        print(json.dumps(record),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=str(exc))
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        manifest['wall_time_sec'] = time.perf_counter()-started
        save(root/'manifest.json',manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--template',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--k',type=int,default=3)
    parser.add_argument('--time-limit',type=float,default=300)
    parser.add_argument('--execute',action='store_true')
    run(parser.parse_args())

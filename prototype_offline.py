"""Road prototype library. No training; explicit --execute required."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

from src.environment import MapGeometry, Scenario, get_scenario, get_lidar_profile, load_config, evaluate_coverage
from src.roadside.candidate_generator import build_oriented_candidates, planned_candidate_signature
from src.methods.greedy import greedy_select
from src.methods.milp import _solve_with_gurobi
from src.core.candidate import Candidate, CandidateSet


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def width_scenario(width=None, scenario_id=0):
    if scenario_id not in (0, 1, 2):
        raise ValueError('Supported parent scenarios: 0, 1, 2')
    base = get_scenario(scenario_id)
    if width is None or float(width) == base.road_width:
        return base
    width = float(width)
    if not math.isfinite(width) or not 2 <= width <= 60:
        raise ValueError('Synthetic straight-road width must be within 2..60 metres')
    sid = 100000 + scenario_id * 100000 + int(round(width * 1000))
    raw = dict(base.raw, id=sid, road_width=width,
               provenance='synthetic_width_variant', parent_scenario_id=scenario_id,
               changed_fields=['road_width'], num_lanes_note='Held fixed; not inferred from width')
    return Scenario(sid, base.topology, width, base.num_lanes, raw)


def explicit_scenario(row):
    """Validate an opt-in synthetic junction without changing legacy scenarios."""
    if type(row.get('id')) is not int or row['id'] < 400000:
        raise ValueError('Explicit scenario ID must be an integer >= 400000')
    parent = row.get('parent_scenario_id')
    if parent not in (1, 2) or row.get('topology') != {1: 'T_JUNCTION', 2: 'CROSS'}[parent]:
        raise ValueError('Explicit scenario parent/topology mismatch')
    width = row.get('road_width')
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or not 2 <= width <= 60:
        raise ValueError('Road width must be finite and within 2..60 metres')
    if type(row.get('num_lanes')) is not int or row['num_lanes'] < 1:
        raise ValueError('Positive integer lane count required')
    if not row.get('geometry') or row.get('provenance') != 'synthetic_junction_offset':
        raise ValueError('Explicit synthetic geometry and provenance required')
    scenario = Scenario(row['id'], row['topology'], float(width), row['num_lanes'], row)
    MapGeometry(scenario)
    return scenario


def branch_frame(x, y, geometry):
    normal = geometry.road_normal_yaw_deg(x, y)
    if geometry.scenario.topology == 'STRAIGHT':
        return 'main', 0.
    if normal in (90., 270.):
        return ('east', 0.) if x >= geometry.junction_x_m else ('west', 180.)
    return ('north', 90.) if y >= geometry.junction_y_m else ('south', 270.)


def canonical(poses, geometry):
    if geometry.scenario.topology != 'STRAIGHT':
        result = []
        for x, y, z, yaw, pitch in poses:
            branch, heading = branch_frame(x, y, geometry)
            rad = np.deg2rad(heading)
            s = (x-geometry.junction_x_m)*np.cos(rad) + (y-geometry.junction_y_m)*np.sin(rad)
            d = -(x-geometry.junction_x_m)*np.sin(rad) + (y-geometry.junction_y_m)*np.cos(rad)
            result.append(dict(branch_id=branch, heading_deg=heading,
                installation_line_id='left' if d > 0 else 'right', s_m=float(s), d_m=float(d),
                z_m=float(z), relative_yaw_deg=float((yaw-heading+180)%360-180), pitch_deg=float(pitch)))
        return result
    return [dict(branch_id='main', installation_line_id='north' if y > geometry.road_centers_y[0] else 'south',
                 s_m=float(x-geometry.map_len/2), d_m=float(y-geometry.road_centers_y[0]), z_m=float(z),
                 relative_yaw_deg=float((yaw+180)%360-180), pitch_deg=float(pitch))
            for x,y,z,yaw,pitch in poses]


def absolute(canonical_poses, geometry):
    if geometry.scenario.topology != 'STRAIGHT':
        return np.array([[geometry.junction_x_m+p['s_m']*np.cos(np.deg2rad(p['heading_deg']))-p['d_m']*np.sin(np.deg2rad(p['heading_deg'])),
            geometry.junction_y_m+p['s_m']*np.sin(np.deg2rad(p['heading_deg']))+p['d_m']*np.cos(np.deg2rad(p['heading_deg'])),
            p['z_m'], (p['relative_yaw_deg']+p['heading_deg'])%360, p['pitch_deg']]
            for p in canonical_poses], dtype=float).reshape(-1,5)
    return np.array([[p['s_m']+geometry.map_len/2, p['d_m']+geometry.road_centers_y[0], p['z_m'],
                      p['relative_yaw_deg']%360, p['pitch_deg']] for p in canonical_poses], dtype=float).reshape(-1,5)


def describe(geometry, config):
    width = geometry.scenario.road_width
    cy = geometry.road_centers_y[0]
    result = dict(schema_version='straight_geometry_v1', scenario_id=geometry.scenario.id,
        topology=geometry.scenario.topology, provenance=dict(type=geometry.scenario.raw.get('provenance', 'osm_metadata_parametric'),
            scenario_fields=['id','topology','road_width','num_lanes'],
            not_used=['OSM polylines','actual building footprints','latitude/longitude']),
        geometry=asdict(geometry),
        coordinate_frame=dict(origin=[geometry.map_len/2,cy,0], heading_deg=0,
            s='signed longitudinal metres along +X', d='signed transverse metres along +Y',
            yaw='relative to +X, degrees; wrapped to [-180,180)'),
        branches=[dict(id='main',width_m=width,length_m=geometry.map_len,heading_deg=0,
            endpoints=[[0,cy],[geometry.map_len,cy]])],
        installation_lines=[dict(id=side,branch_id='main',nominal_y_m=cy+sign*geometry.sidewalk_center_offset,
            tolerance_m=geometry.grid_size/2) for side,sign in [('north',1),('south',-1)]],
        assumed_fields=['map dimensions','sidewalk width','building zones and height','median width','straight centerline'],
        candidate=config.candidate, target=config.target,
        discretization='existing cell centers and first feasible cell per XY bin; not continuous space',
        objective='unweighted binary union coverage', budget_mode='at_most')
    if geometry.scenario.topology != 'STRAIGHT':
        arms = [('east',0), ('west',180), ('south',270)]
        if geometry.scenario.topology == 'CROSS':
            arms.append(('north',90))
        cx, cy = geometry.junction_x_m, geometry.junction_y_m
        lengths = dict(east=geometry.map_len-cx, west=cx, south=cy, north=geometry.map_width-cy)
        result.update(schema_version='junction_geometry_v1',
            coordinate_frame=dict(origin=[cx,cy,0], s='outward longitudinal metres',
                d='signed left-normal metres', yaw='relative to branch heading',
                assignment='nearest valid road normal; horizontal tie break'),
            branches=[dict(id=name, heading_deg=h, width_m=width, length_m=lengths[name],
                endpoints=[[cx,cy],[cx+lengths[name]*np.cos(np.deg2rad(h)),cy+lengths[name]*np.sin(np.deg2rad(h))]]) for name,h in arms],
            installation_lines=[dict(branch_id=name,id=side,nominal_d_m=sign*geometry.sidewalk_center_offset,
                tolerance_m=geometry.grid_size/2) for name,h in arms for side,sign in [('left',1),('right',-1)]])
    return result


def certified(metadata, covered_count):
    return (metadata.get('solver_status_code') == 2
            and metadata.get('objective') is not None
            and metadata.get('best_bound') is not None
            and abs(metadata['objective']-covered_count)<1e-5
            and abs(metadata['best_bound']-covered_count)<1e-5)


def reuse_candidates(source, root, plan):
    source = Path(source)
    previous = json.loads((source/'manifest.json').read_text())
    if previous['plan']['config_hash'] != plan['config_hash']:
        raise ValueError('Candidate reuse rejected: configuration mismatch')
    for name, expected in previous['source_hashes'].items():
        if name.startswith('src/') and name != 'src/methods/milp.py':
            if digest(name) != expected:
                raise ValueError(f'Candidate reuse rejected: source changed: {name}')
    archive = source/'candidates.npz'
    with np.load(archive, allow_pickle=False) as data:
        poses = data['poses']
        targets = data['target_points']
        masks = np.unpackbits(data['packed_masks'], axis=1, count=len(targets)).astype(bool)
    if masks.shape != (len(poses), len(targets)) or poses.shape[1:] != (5,):
        raise ValueError('Invalid archived candidate dimensions')
    stats = previous['candidate_stats']
    if len(poses) != stats['num_candidates'] or int(masks.sum()) != stats['num_coverage_edges']:
        raise ValueError('Archived candidate statistics mismatch')
    shutil.copytree(source/'cache', root/'cache')
    shutil.copy2(archive, root/'candidates.npz')
    provenance = dict(source_root=str(source.resolve()), archive_sha256=digest(archive),
        manifest_sha256=digest(source/'manifest.json'),
        cache_sha256={p.name:digest(p) for p in (root/'cache').iterdir() if p.is_file()})
    candidates = [Candidate(i, *map(float, pose), masks[i]) for i, pose in enumerate(poses)]
    return CandidateSet(candidates, targets, plan['cache_key'], plan['config_hash'], stats), provenance


def run(args):
    config = load_config(args.config)
    if getattr(args, 'xy_domain', None) is not None:
        config.candidate['xy_domain'] = args.xy_domain
    config.candidate['max_orientations_per_position'] = 0
    config.solver.update(solver='gurobi', threads=1, time_limit_sec=args.time_limit,
                         mip_gap=0., soft_mem_limit_gb=90)
    config.solver['sparse_target_batch_size'] = args.target_batch_size
    if config.target.get('road_detection', 1) != 1:
        raise ValueError('Only binary union coverage is supported')
    scenario_json = getattr(args, 'scenario_json', None)
    if scenario_json and args.road_width is not None:
        raise ValueError('Use scenario-json or road-width, not both')
    scenario = (explicit_scenario(json.loads(Path(scenario_json).read_text()))
                if scenario_json else width_scenario(args.road_width, args.scenario_id))
    geometry = MapGeometry(scenario)
    profile = get_lidar_profile(config.lidar_profile, config)
    plan = planned_candidate_signature(scenario, config.lidar_profile, config)
    print(json.dumps(dict(plan=plan,k_values=args.k_values,solver=config.solver),indent=2),flush=True)
    if not args.execute:
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(schema_version='prototype_pilot_v1', status='running',
        command=[sys.executable,*sys.argv],
        created_at=datetime.now(timezone.utc).isoformat(), scenario_id=scenario.id,k_values=args.k_values,
        plan=plan,results=[], objective='unweighted coverage', optimality_domain='declared discrete grid',
        exact_mask_dedup=True, dedup_note='preserves pure union objective under at-most-K; alternate equal masks are not stored')
    try:
        # A fresh run-local cache prevents reuse of caches from older geometry code.
        config.raw.setdefault('output',{}).update(cache_dir=str(root/'cache'),results_dir=str(root))
        config.solver['nodefile_dir']=str(root/'nodefiles')
        config.raw.update(scenario_ids=[scenario.id], k_values=args.k_values)
        manifest['plan']=planned_candidate_signature(scenario, config.lidar_profile, config)
        (root/'config.yaml').write_text(yaml.safe_dump(config.raw,sort_keys=False),encoding='utf-8')
        source_dir=root/'source';source_dir.mkdir()
        hashes={}
        paths = [Path(__file__).resolve(),Path(args.config),Path('data/scenarios/yeongjong_final.json'),*Path('src').rglob('*.py')]
        if scenario_json:
            paths.append(Path(scenario_json))
        for path in paths:
            label=path.name if path.is_absolute() else str(path)
            dest=source_dir/label;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest);hashes[label]=digest(path)
        manifest['source_hashes']=hashes
        save(root/'manifest.json',manifest)
        descriptor=describe(geometry,config)
        descriptor['sensor']=asdict(profile)
        save(root/'geometry.json',descriptor)
        mesh=geometry.create_environment_mesh()
        np.savez_compressed(root/'mesh.npz',vertices=np.asarray(mesh.vertices),triangles=np.asarray(mesh.triangles))
        t0=time.monotonic()
        if args.reuse_candidates:
            print('Validating and loading archived candidates/cache', flush=True)
            candidates, provenance = reuse_candidates(args.reuse_candidates, root, manifest['plan'])
            manifest['candidate_reuse'] = provenance
        else:
            print('Building full-orientation candidates and fresh visibility cache',flush=True)
            candidates=build_oriented_candidates(scenario,config.lidar_profile,config)
        manifest['candidate_build_sec']=time.monotonic()-t0
        manifest['candidate_stats']=candidates.stats
        save(root/'manifest.json',manifest)
        print(json.dumps(candidates.stats),flush=True)
        if not args.reuse_candidates:
            poses=candidates.positions_array(range(len(candidates.candidates)))
            packed=np.stack([np.packbits(c.covered) for c in candidates.candidates])
            np.savez_compressed(root/'candidates.npz',poses=poses,packed_masks=packed,target_points=candidates.target_points)
            del packed
        for k in args.k_values:
            t0=time.monotonic();out=root/f'k{k}';out.mkdir()
            warm=greedy_select(candidates,k)
            if k==1 or warm.covered_mask.all():
                count=int(warm.covered_mask.sum())
                metadata=dict(selected_ids=warm.selected_ids,
                    solver_status='ANALYTIC_EXACT_K1' if k==1 else 'ANALYTIC_FULL_COVERAGE',
                    solver_status_code=2,objective=count,best_bound=count,mip_gap=0.,solve_time_sec=0.,
                    solver_backend='enumeration' if k==1 else 'coverage_upper_bound')
            else:
                # Sparse assembly has multiple edge arrays alive before Gurobi's memory limit applies.
                if candidates.stats['num_coverage_edges']>args.max_edges:
                    raise RuntimeError('Sparse matrix safety guard reached; no candidate pruning applied')
                print(f'Solving K={k}, time limit={args.time_limit}s, MIPGap=0',flush=True)
                config.solver['log_file'] = str(out/'gurobi.log')
                metadata=_solve_with_gurobi(candidates,k,config,warm.selected_ids)
                metadata['solver_backend']='gurobi'
            selected=metadata['selected_ids']
            selected_poses=candidates.positions_array(selected).reshape(-1,5)
            evaluation=evaluate_coverage(selected_poses,scenario,config.lidar_profile,config)
            density=np.zeros(len(candidates.target_points),dtype=int)
            for i in selected:
                density+=candidates.candidates[i].covered
            covered=int((density>0).sum())
            mapped=canonical(selected_poses,geometry)
            recovered=absolute(mapped,geometry)
            roundtrip=(np.allclose(recovered[:,:3],selected_poses[:,:3]) and
                       np.allclose((recovered[:,3]-selected_poses[:,3]+180)%360-180,0) and
                       np.allclose(recovered[:,4],selected_poses[:,4]))
            valid=(covered==evaluation.covered_count and len(selected)<=k and len(set(selected))==len(selected)
                   and roundtrip and all(geometry.is_installable(*p[:3]) for p in selected_poses))
            if metadata.get('objective') is not None:
                valid=valid and abs(metadata['objective']-covered)<1e-5
            if not valid:
                raise RuntimeError('Saved selection failed common coverage/canonical/solver validation')
            np.save(out/'poses.npy',selected_poses)
            save(out/'canonical_poses.json',mapped)
            record=dict(schema_version='prototype_result_v1',scenario_id=scenario.id,budget_k=k,num_selected=len(selected),
                covered_count=covered,num_targets=len(density),coverage_pct=100*covered/len(density),
                overlap_target_count=int((density>=2).sum()),uncovered_count=int((density==0).sum()),
                certification='discrete_optimal' if certified(metadata,covered) else 'best_known',
                solver=metadata,cache_key=candidates.cache_key,config_hash=candidates.config_hash,
                wall_time_sec=time.monotonic()-t0,positions_npy=f'k{k}/poses.npy')
            save(out/'result.json',record)
            save(out/'validation.json',dict(passed=True,canonical_roundtrip=True,common_evaluator=True,solver_objective_checked=metadata.get('objective') is not None))
            manifest['results'].append(record)
            save(root/'manifest.json',manifest)
            print(json.dumps(record),flush=True)
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error=str(exc))
        raise
    finally:
        manifest['total_wall_time_sec']=time.monotonic()-started
        save(root/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiment.yaml')
    parser.add_argument('--scenario-id',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--scenario-json',help='Explicit synthetic junction JSON; overrides scenario-id')
    parser.add_argument('--road-width',type=float,help='Synthetic width variant of scenario 0; all other geometry fields fixed')
    parser.add_argument('--output',default='results/prototype_s0_pilot')
    parser.add_argument('--k-values',type=int,nargs='+',default=[1,2])
    parser.add_argument('--time-limit',type=float,default=300)
    parser.add_argument('--max-edges',type=int,default=250_000_000)
    parser.add_argument('--target-batch-size',type=int,default=128)
    parser.add_argument('--reuse-candidates',help='Reuse a trusted pilot archive after config/source checks')
    parser.add_argument('--xy-domain', choices=['bin_representatives', 'installable_grid'],
                        help='Geometry-only XYZ domain; default preserves the input config')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.k_values!=sorted(set(args.k_values)) or any(k<1 or k>6 for k in args.k_values):
        parser.error('Use ascending unique K values within 1..6')
    if not math.isfinite(args.time_limit) or args.time_limit<=0:
        parser.error('Time limit must be positive and finite')
    if args.target_batch_size < 1 or args.target_batch_size > 256 or args.max_edges < 1:
        parser.error('Target batch size must be 1..256 and max edges positive')
    run(args)

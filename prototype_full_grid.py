"""Fresh full-grid packed candidates and sequential cut-MILP references.

Default plans only. Execute freezes source/inputs and starts one worker per map.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from prototype_offline import digest, explicit_scenario, save
from src.environment import MapGeometry, get_lidar_profile, load_config, validate_installation_pose
from src.roadside.candidate_generator import planned_candidate_signature


def now():
    return datetime.now(timezone.utc).isoformat()


def configuration(path):
    cfg = load_config(path)
    cfg.candidate.update(xy_domain='installable_grid',max_orientations_per_position=0)
    if cfg.target.get('road_detection',1) != 1:
        raise ValueError('Binary union required')
    return cfg


def summarize(root, jobs):
    rows = []
    for job in jobs:
        for k in job['k_values']:
            path = root/f"s{job['scenario']['id']}"/f'k{k}'/'result.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if not result['validated_direct_raycast']:
                raise ValueError('Unvalidated result')
            rows.append(dict(scenario_id=job['scenario']['id'],K=k,coverage_pct=result['coverage_pct'],
                objective=result['objective'],upper_bound=result['certified_upper_bound'],
                relative_gap=result['certified_gap'],discrete_optimal=result['discrete_optimal'],
                solver_status=result['solver_status'],candidate_count=result['candidate_count'],
                solver_sec=result['solver_sec'],optimization_sec=result['total_sec'],
                greedy_sec=result['greedy_sec'],cache_and_candidates_sec=result['cache_and_candidates_sec'],
                result_json=str(path)))
    if rows:
        with (root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(root/'summary.json',dict(completed=len(rows),expected=sum(len(j['k_values']) for j in jobs),
        results=rows,note='Fresh full-grid reference. Per-map cache/candidate cost shared across K; '
                         'not an automatically controlled speedup comparison.'))


def worker(args):
    from src.core.runtime import RuntimeMeter
    from src.core.visibility import MeshVisibility
    from src.methods.coverage_cut_milp import solve_cut_milp
    from src.roadside.packed_candidates import build_packed_candidates, packed_greedy
    from prototype_transfer import export_view
    root=Path(args.output).resolve()
    cfg=configuration(args.config)
    scenario=explicit_scenario(json.loads(Path(args.worker).read_text()))
    plan=planned_candidate_signature(scenario,cfg.lidar_profile,cfg)
    print(json.dumps(plan),flush=True)
    if not args.execute:
        return
    root.mkdir(parents=True,exist_ok=False)
    cfg.output.update(cache_dir=str(root/'cache'),results_dir=str(root))
    (root/'config.yaml').write_text(yaml.safe_dump(cfg.raw,sort_keys=False))
    save(root/'scenario.json',scenario.raw)
    manifest=dict(status='running',stage='visibility',started_at=now(),scenario_id=scenario.id,
        k_values=args.k_values,plan=plan,results=[],online_pose_warm_start=False)
    save(root/'manifest.json',manifest)
    meter=RuntimeMeter()
    try:
        def progress(done,total):
            manifest.update(stage='candidate_masks',candidates_written=done,candidates_total=total)
            save(root/'manifest.json',manifest)
            print(f'Candidates {done}/{total}',flush=True)
        start=time.perf_counter()
        poses,masks,targets,stats=build_packed_candidates(scenario,cfg,root/'candidates',work=meter,progress=progress)
        build_sec=time.perf_counter()-start
        manifest.update(candidate_stats=stats,cache_and_candidates_sec=build_sec,
            archive_hashes={p.name:digest(p) for p in (root/'candidates').glob('*.npy')})
        save(root/'manifest.json',manifest)
        geometry=MapGeometry(scenario)
        mesh=geometry.create_environment_mesh()
        provider=MeshVisibility(mesh,targets,get_lidar_profile(cfg.lidar_profile,cfg))
        for k in args.k_values:
            directory=root/f'k{k}';directory.mkdir()
            manifest.update(stage='greedy',current_k=k);save(root/'manifest.json',manifest)
            start=time.perf_counter()
            initial=packed_greedy(masks,len(targets),k)
            greedy_sec=time.perf_counter()-start
            manifest.update(stage='cut_milp');save(root/'manifest.json',manifest)
            print(f'Start cut-MILP K={k}',flush=True)
            result=solve_cut_milp(masks,len(targets),k,initial,seconds=args.seconds,
                swap_seconds=args.swap_seconds,soft_mem_gb=90,threads=1,log_file=directory/'gurobi.log')
            selected=np.asarray(poses[result['selected_ids']])
            for pose in selected:
                validate_installation_pose(pose,geometry,cfg)
            start=time.perf_counter()
            density=np.zeros(len(targets),dtype=np.int32)
            for pose in selected:
                density += provider.mask(pose)
            if int((density>0).sum()) != result['objective']:
                raise RuntimeError('Independent raycast disagrees with packed coverage')
            result.update(coverage_pct=100*result['objective']/len(targets),greedy_sec=greedy_sec,
                cache_and_candidates_sec=build_sec,validation_sec=time.perf_counter()-start,
                validated_direct_raycast=True,xy_domain=cfg.xy_domain,archive_reused=False,
                online_pose_warm_start=False,
                semantics='Full installable 0.5m XYZ grid, all configured yaw/pitch, no dedup/pruning, '
                          'unweighted union at-most-K. Cache/candidates built once per map for both K.')
            np.save(directory/'poses.npy',selected)
            np.savez_compressed(directory/'coverage.npz',targets=targets,density=density)
            export_view(directory/'view.html',mesh,targets,density,selected,
                        f'Full-grid S{scenario.id} K={k}',result['coverage_pct'])
            save(directory/'result.json',result)
            manifest['results'].append(dict(k=k,coverage_pct=result['coverage_pct'],
                relative_gap=result['certified_gap'],result_json=str(directory/'result.json')))
            save(root/'manifest.json',manifest)
            print(f'Completed K={k}: {result["coverage_pct"]:.6f}%, gap={result["certified_gap"]}',flush=True)
        manifest.update(status='complete',stage='complete',completed_at=now())
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        manifest['runtime']=meter.report()
        save(root/'manifest.json',manifest)


def run(args):
    if not np.isfinite([args.seconds,args.swap_seconds]).all() or args.seconds<=0 or args.swap_seconds<0:
        raise ValueError('Invalid finite time budget')
    if not args.k_values or any(k<1 for k in args.k_values) or len(set(args.k_values))!=len(args.k_values):
        raise ValueError('Unique positive K values required')
    if args.worker:
        return worker(args)
    cfg=configuration(args.config)
    suite=json.loads(Path(args.suite).read_text())
    jobs=[dict(scenario=row,k_values=args.k_values,status='pending')
          for family in suite['families'] for row in family['targets']]
    if len({j['scenario']['id'] for j in jobs})!=len(jobs):
        raise ValueError('Duplicate scenario IDs')
    plans=[planned_candidate_signature(explicit_scenario(j['scenario']),cfg.lidar_profile,cfg) for j in jobs]
    print(json.dumps(dict(maps=len(jobs),solves=len(jobs)*len(args.k_values),plans=plans,execute=args.execute),indent=2),flush=True)
    if not args.execute:
        return
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'inputs').mkdir();(root/'logs').mkdir()
    (root/'inputs/config.yaml').write_text(yaml.safe_dump(cfg.raw,sort_keys=False))
    shutil.copy2(args.suite,root/'inputs/suite.json')
    for job in jobs:
        save(root/'inputs'/f"s{job['scenario']['id']}.json",job['scenario'])
    for path in [*Path('.').glob('prototype_*.py'),*Path('src').rglob('*.py')]:
        dest=root/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    hashes={str(p.relative_to(root)):digest(p) for folder in ('source','inputs')
            for p in (root/folder).rglob('*') if p.is_file()}
    manifest=dict(status='running',started_at=now(),command=[sys.executable,*sys.argv],jobs=jobs,
        source_input_hashes=hashes,solver_seconds=args.seconds,swap_seconds=args.swap_seconds,
        threads=1,soft_mem_gb=90,xy_domain='installable_grid',online_inputs_used=False)
    save(root/'manifest.json',manifest)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (root/'packages.txt').open('w') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    try:
        for job,plan in zip(jobs,plans):
            for path,expected in hashes.items():
                if digest(root/path)!=expected:
                    raise RuntimeError(f'Frozen input/source changed: {path}')
            mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            if int(mem['MemAvailable'].split()[0])*1024 < 96*1024**3:
                raise RuntimeError('Less than 96 GiB available; refusing 90 GB soft-limit worker')
            targets=len(MapGeometry(explicit_scenario(job['scenario'])).target_points(cfg.target['coverage_z_slices']))
            needed=plan['num_candidates_planned']*((targets+7)//8+20)
            if shutil.disk_usage(root).free < needed*2+5*1024**3:
                raise RuntimeError('Insufficient disk headroom')
            sid=job['scenario']['id'];job.update(status='running',started_at=now())
            save(root/'manifest.json',manifest)
            cmd=[sys.executable,'-u','prototype_full_grid.py','--worker',str(root/'inputs'/f's{sid}.json'),
                 '--config',str(root/'inputs/config.yaml'),'--output',str(root/f's{sid}'),
                 '--seconds',str(args.seconds),'--swap-seconds',str(args.swap_seconds),
                 '--k-values',*map(str,args.k_values)]
            subprocess.run(cmd,cwd=root/'source',env=env,check=True)
            start=time.perf_counter()
            with (root/'logs'/f's{sid}.log').open('w') as f:
                subprocess.run(cmd+['--execute'],cwd=root/'source',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            job.update(status='complete',completed_at=now(),worker_wall_sec=time.perf_counter()-start)
            save(root/'manifest.json',manifest);summarize(root,jobs)
        manifest.update(status='complete',completed_at=now())
    except BaseException as exc:
        for job in jobs:
            if job['status']=='running':job['status']='failed'
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        save(root/'manifest.json',manifest);summarize(root,jobs)


def parser():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/experiment.yaml')
    p.add_argument('--suite',default='configs/library_expansion.json')
    p.add_argument('--output',required=True)
    p.add_argument('--k-values',type=int,nargs='+',default=[3,4])
    p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--swap-seconds',type=float,default=30)
    p.add_argument('--worker',help='Internal single-map JSON worker')
    p.add_argument('--execute',action='store_true')
    return p


if __name__=='__main__':
    run(parser().parse_args())

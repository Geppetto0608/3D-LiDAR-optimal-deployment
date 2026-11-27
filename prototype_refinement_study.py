"""Paired cold-map local refinement study; never build a target visibility table.

Run without --execute to inspect the plan. Each job uses a fresh CPU process.
Archived target poses are used only for validation after online optimization.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from prototype_library import load_suite
from prototype_offline import canonical, digest, explicit_scenario, save
from prototype_segments import (installation_segments, prepare_segment_placements,
                                segment_coordinates, segment_to_target)
from prototype_transfer import export_view, load_prototype, project_poses
from src.core.runtime import RuntimeMeter
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, get_lidar_profile
from src.methods.local_mappo import refine_placement, validate_poses
from src.methods.local_search import refine_local_search


def now():
    return datetime.now(timezone.utc).isoformat()


def pose_hash(poses):
    return hashlib.sha256(np.asarray(poses, dtype='<f4').tobytes()).hexdigest()


def allocation(poses, geometry):
    return dict(sorted(Counter(f'{p["branch_id"]}/{p["installation_line_id"]}'
                               for p in canonical(poses, geometry)).items()))


def allocation_difference(low, high):
    return {key: high.get(key, 0) - low.get(key, 0)
            for key in sorted(low.keys() | high.keys())
            if high.get(key, 0) != low.get(key, 0)}


def make_plan(library, k_values, seeds):
    library = Path(library).resolve()
    suite = load_suite(json.loads((library/'suite.json').read_text()))
    manifest = json.loads((library/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Source library must be complete')
    jobs, diagnostics = [], []
    for family in suite['families']:
        roots = [library/'library'/f's{row["id"]}' for row in family['sources']]
        for k in manifest['plan']['k_values']:
            prototypes = [load_prototype(root, k) for root in roots]
            groups = [allocation(p[4], p[3]) for p in prototypes]
            delta = allocation_difference(*groups)
            for target in family['targets']:
                report_path = library/'transfers'/f's{target["id"]}'/f'k{k}'/'report.json'
                report = json.loads(report_path.read_text())
                if report['scenario'] != target or report['budget_k'] != k:
                    raise ValueError('Archived target definition differs from suite')
                interpolation = report['placements']['interpolation']
                diagnostics.append(dict(scenario_id=target['id'], budget_k=k,
                    low_count=len(prototypes[0][4]), high_count=len(prototypes[1][4]),
                    low_allocation=groups[0], high_allocation=groups[1], high_minus_low=delta,
                    interpolation_status=interpolation['status'], reason=interpolation.get('reason'),
                    diagnosis=('sensor_count_mismatch' if len(prototypes[0][4]) != len(prototypes[1][4])
                               else 'branch_side_allocation_mismatch' if delta else 'allocation_compatible'),
                    report=str(report_path)))
                if k not in k_values:
                    continue
                for name, result in report['placements'].items():
                    if result['status'] != 'complete':
                        continue
                    for method, seed in [('local_search', None), *[('mappo', s) for s in seeds]]:
                        identity = f's{target["id"]}_k{k}_{name}_{method}'
                        if seed is not None:
                            identity += f'_seed{seed}'
                        jobs.append(dict(id=identity, scenario=target, budget_k=k,
                            initialization=name, method=method, seed=seed,
                            source_roots=list(map(str, roots)),
                            reference_pose=str(report_path.parent/result['positions_npy']),
                            status='pending'))
    if not set(k_values).issubset(manifest['plan']['k_values']) or not jobs:
        raise ValueError('Requested K is absent from library')
    return jobs, diagnostics


def make_initial(job):
    """No target result, cached mask, or coverage-based initialization selection."""
    name, k = job['initialization'], job['budget_k']
    geometry = MapGeometry(explicit_scenario(job['scenario']))
    if name in ('transfer_low', 'transfer_high'):
        p = load_prototype(job['source_roots'][int(name == 'transfer_high')], k)
        cfg = p[5]
        coords = segment_coordinates(p[4], p[3], installation_segments(p[3]))
        raw = segment_to_target(coords, geometry, installation_segments(geometry))
        initial = project_poses(raw, geometry, cfg, coords)
    elif name == 'interpolation':
        low, high = [load_prototype(root, k) for root in job['source_roots']]
        cfg = low[5]
        for key in ('candidate', 'target', 'mappo'):
            if getattr(cfg, key) != getattr(high[5], key):
                raise ValueError(f'Incompatible source config: {key}')
        if get_lidar_profile(cfg.lidar_profile, cfg) != get_lidar_profile(high[5].lidar_profile, high[5]):
            raise ValueError('Incompatible sensor profiles')
        placements, _ = prepare_segment_placements(low[4], high[4], low[3], high[3], geometry, cfg)
        initial = placements[name]
    else:
        raise ValueError(f'Unknown initialization: {name}')
    validate_poses(initial, geometry, cfg)
    if len(initial) > k:
        raise ValueError('Initial deployment exceeds K')
    return initial, geometry, cfg


def worker(root, index):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Study worker requires CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads((root/'manifest.json').read_text())
    if not 0 <= index < len(manifest['jobs']):
        raise ValueError('Worker index out of bounds')
    for path, expected in manifest['frozen_hashes'].items():
        if digest(root/path) != expected:
            raise RuntimeError(f'Frozen input changed: {path}')
    job = manifest['jobs'][index]
    out = root/'jobs'/job['id']
    out.mkdir(parents=True, exist_ok=False)
    limits = manifest['limits']
    work = RuntimeMeter()
    with work.stage('source_read_geometry_transfer'):
        initial, geometry, cfg = make_initial(job)
    with work.stage('refinement'):
        if job['method'] == 'mappo':
            poses, meta = refine_placement(initial, geometry.scenario, cfg, out/'mappo',
                steps=limits['steps'], seed=job['seed'], radius_m=limits['radius_m'],
                time_limit_sec=limits['seconds'], work=work)
        else:
            poses, meta = refine_local_search(initial, geometry.scenario, cfg,
                radius_m=limits['radius_m'], max_evaluations=limits['max_evaluations'],
                time_limit_sec=limits['seconds'], work=work)
    online = work.report()
    # Archived target data enters only after the online result has been fixed.
    reference = np.load(job['reference_pose'], allow_pickle=False)
    if not np.array_equal(initial, reference):
        raise RuntimeError('Regenerated initial deployment differs from frozen reference')
    validate_poses(poses, geometry, cfg)
    if poses.shape != initial.shape or np.any(np.linalg.norm(poses[:, :2]-initial[:, :2], axis=1) > limits['radius_m']+1e-5):
        raise RuntimeError('Refinement changed sensor count or escaped anchor radius')
    audit = RuntimeMeter()
    with audit.stage('independent_direct_validation'):
        mesh = geometry.create_environment_mesh()
        provider = MeshVisibility(mesh, geometry.target_points(cfg.target['coverage_z_slices']),
                                  get_lidar_profile(cfg.lidar_profile, cfg), work=audit)
        def evaluate(value):
            density = np.stack([provider.mask(p) for p in value]).sum(axis=0)
            return 100 * float((density > 0).mean()), density
        initial_cov, _ = evaluate(initial)
        cov, density = evaluate(poses)
        if abs(cov-meta['coverage_pct']) > 1e-5 or abs(initial_cov-meta['initial_coverage_pct']) > 1e-5:
            raise RuntimeError('Internal and direct coverage disagree')
        if cov + 1e-5 < initial_cov:
            raise RuntimeError('Refinement lost the retained initial deployment')
    validation = audit.report()
    np.save(out/'initial.npy', initial)
    np.save(out/'deployment.npy', poses)
    np.savez_compressed(out/'coverage.npz', targets=provider.targets, density=density)
    result = dict(schema='paired_refinement_result_v1', job=job,
        initial_pose_sha256=pose_hash(initial), coverage_pct=cov,
        initial_coverage_pct=initial_cov, gain_pp=cov-initial_cov,
        num_selected=len(poses), online_runtime=online, validation_runtime=validation,
        refinement=meta, full_target_cache=False, checkpoint_reused=False,
        initialization_selection='none; each available initialization evaluated independently',
        timing_note='Soft refinement deadline; flush/evaluation/checkpoint may overrun. '
                    'Online includes source read, transfer and refinement, excludes import, validation and HTML.',
        completed_at=now())
    export_view(out/'view.html', mesh, provider.targets, density, poses, job['id'], cov)
    save(out/'result.json', result)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, manifest):
    raw, groups, views = [], defaultdict(dict), []
    for job in manifest['jobs']:
        path = root/'jobs'/job['id']/'result.json'
        if job['status'] != 'complete':
            continue
        r = json.loads(path.read_text())
        row = dict(scenario_id=job['scenario']['id'], K=job['budget_k'],
            initialization=job['initialization'], method=job['method'], seed=job['seed'],
            initial_coverage_pct=r['initial_coverage_pct'], coverage_pct=r['coverage_pct'], gain_pp=r['gain_pp'],
            online_sec=r['online_runtime']['elapsed_sec'], refinement_sec=r['refinement']['elapsed_sec'],
            rays_submitted=r['online_runtime']['counts'].get('rays_submitted', 0),
            raycast_calls=r['online_runtime']['counts'].get('raycast_calls', 0),
            deployment_evaluations=r['online_runtime']['counts'].get('deployment_evaluations', 0),
            worker_wall_sec=job['worker_wall_sec'], validation_sec=r['validation_runtime']['elapsed_sec'],
            soft_limit_overrun_sec=max(0., r['refinement']['elapsed_sec']-manifest['limits']['seconds']),
            actual_steps=r['refinement'].get('training_steps'),
            final_policy_coverage_pct=r['refinement'].get('final_policy_coverage_pct'),
            initial_pose_sha256=r['initial_pose_sha256'])
        raw.append(row)
        group = groups[(row['scenario_id'], row['K'], row['initialization'])]
        group.setdefault(job['method'], []).append(row)
        views.append(f'<tr><td>{html.escape(job["id"])}</td><td>{row["coverage_pct"]:.3f}</td>'
                     f'<td>{row["online_sec"]:.2f}</td><td><a href="jobs/{job["id"]}/view.html">3D</a></td></tr>')
    paired = []
    for (sid, k, init), group in groups.items():
        rows = group.get('mappo', [])
        ls = group.get('local_search', [])
        if len(rows) != len(manifest['seeds']) or len(ls) != 1:
            continue
        if len({r['initial_pose_sha256'] for r in rows+ls}) != 1:
            raise RuntimeError('Paired methods did not use identical initial poses')
        values = [r['coverage_pct'] for r in rows]
        paired.append(dict(scenario_id=sid, K=k, initialization=init,
            initial_coverage_pct=ls[0]['initial_coverage_pct'], local_search_coverage_pct=ls[0]['coverage_pct'],
            mappo_mean_pct=float(np.mean(values)), mappo_sample_sd_pct=float(np.std(values, ddof=1)) if len(values)>1 else None,
            mappo_minus_local_search_pp=float(np.mean(values))-ls[0]['coverage_pct'],
            local_search_online_sec=ls[0]['online_sec'],
            mappo_mean_online_sec=float(np.mean([r['online_sec'] for r in rows])),
            trials=len(rows)))
    write_csv(root/'raw_results.csv', raw)
    write_csv(root/'paired_summary.csv', paired)
    save(root/'summary.json', dict(status=manifest['status'], completed=len(raw), expected=len(manifest['jobs']),
        job_status_counts=dict(Counter(j['status'] for j in manifest['jobs'])),
        complete_pairs=len(paired), pairs=paired,
        note='Development maps, no held-out generalization or full-target MILP optimality claim. '
             'MAPPO best-seen deployment, not final policy performance. Unavailable interpolation is not zero coverage.'))
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Paired refinement study</title>'
        '<h1>Paired cold-map refinement</h1><p>Status: '+html.escape(manifest['status'])+
        '</p><p>Development study. No full-target MILP. Coverage is best retained deployment.</p>'
        '<p><a href="manifest.json">All jobs / failures / pending</a> | '
        '<a href="raw_results.csv">Raw CSV</a> | <a href="paired_summary.csv">Paired CSV</a> | '
        '<a href="diagnostics.json">Correspondence diagnosis</a></p><table><tr><th>Job</th>'
        '<th>Coverage %</th><th>Online seconds</th><th>View</th></tr>'+''.join(views)+'</table>', encoding='utf-8')


def run(args):
    if (args.k_values != sorted(set(args.k_values)) or not args.seeds or
            len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds)):
        raise ValueError('Unique ascending K and unique nonnegative seeds required')
    if any(not math.isfinite(v) or v <= 0 for v in (args.seconds, args.radius_m)) or args.steps < 1 or args.max_evaluations < 1:
        raise ValueError('Positive finite limits required')
    jobs, diagnostics = make_plan(args.library, args.k_values, args.seeds)
    if args.scenario_ids is not None:
        known = {j['scenario']['id'] for j in jobs}
        if not set(args.scenario_ids).issubset(known):
            raise ValueError('Unknown target scenario')
        jobs = [j for j in jobs if j['scenario']['id'] in args.scenario_ids]
    limits = dict(seconds=args.seconds, radius_m=args.radius_m, steps=args.steps, max_evaluations=args.max_evaluations)
    print(json.dumps(dict(jobs=len(jobs), initializations=len(jobs)//(1+len(args.seeds)), limits=limits,
        interpolation_status=dict(Counter(d['interpolation_status'] for d in diagnostics)), execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    frozen = {}
    def freeze(path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        frozen[str(destination.relative_to(root))] = digest(destination)
    code_paths = [*Path('.').glob('prototype_*.py'), *Path('src').rglob('*.py')]
    for path in code_paths:
        freeze(path, root/'source'/path)
    copied = {}
    for job in jobs:
        new_roots = []
        for path in job['source_roots']:
            source = Path(path)
            destination = root/'inputs'/source.name
            for filename in ('geometry.json', 'config.yaml', f'k{job["budget_k"]}/result.json'):
                freeze(source/filename, destination/filename)
            info = json.loads((source/f'k{job["budget_k"]}/result.json').read_text())
            freeze(source/info['positions_npy'], destination/info['positions_npy'])
            new_roots.append(str(destination))
            copied[str(source)] = str(destination)
        job['source_roots'] = new_roots
        ref = Path(job['reference_pose'])
        dest = root/'validation_references'/f's{job["scenario"]["id"]}'/f'k{job["budget_k"]}'/ref.name
        freeze(ref, dest)
        job['reference_pose'] = str(dest)
    for filename in ('suite.json', 'manifest.json'):
        freeze(Path(args.library)/filename, root/'inputs'/f'library_{filename}')
    manifest = dict(schema='paired_refinement_study_v1', status='running', started_at=now(),
        command=[sys.executable, *sys.argv], limits=limits, seeds=args.seeds, jobs=jobs,
        frozen_hashes=frozen, source_library=str(Path(args.library).resolve()), copied_sources=copied,
        hardware=dict(platform=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count()),
        execution='Sequential fresh CPU processes, torch/BLAS/OpenMP threads=1; shared host not isolated')
    save(root/'manifest.json', manifest)
    save(root/'diagnostics.json', diagnostics)
    write_csv(root/'diagnostics.csv', [{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                                       for k, v in row.items()} for row in diagnostics])
    with (root/'packages.txt').open('w') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    try:
        for index, job in enumerate(jobs):
            job['status'] = 'running'
            manifest['current'] = job['id']
            save(root/'manifest.json', manifest)
            print(f'[{index+1}/{len(jobs)}] {job["id"]}', flush=True)
            start = time.monotonic()
            with (root/'logs'/f'{job["id"]}.log').open('w') as log:
                process = subprocess.run([sys.executable, '-u', 'prototype_refinement_study.py',
                    '--output', str(root), '--worker', str(index), '--execute'], cwd=root/'source',
                    env=env, stdout=log, stderr=subprocess.STDOUT)
            job['worker_wall_sec'] = time.monotonic()-start
            job['returncode'] = process.returncode
            job['status'] = 'complete' if process.returncode == 0 else 'failed'
            save(root/'manifest.json', manifest)
            summarize(root, manifest)
        manifest.update(status='complete' if all(j['status']=='complete' for j in jobs) else 'failed',
                        completed_at=now(), current=None)
    except BaseException as exc:
        manifest.update(status='interrupted', error=repr(exc), completed_at=now())
        raise
    finally:
        save(root/'manifest.json', manifest)
        summarize(root, manifest)
    if manifest['status'] != 'complete':
        raise RuntimeError('Study contains failed jobs; inspect manifest and logs')


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--library', default='results/prototype_library_shapes_20261008')
    p.add_argument('--output', required=True)
    p.add_argument('--k-values', type=int, nargs='+', default=[3, 4])
    p.add_argument('--scenario-ids', type=int, nargs='+')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--radius-m', type=float, default=10.)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--max-evaluations', type=int, default=100000)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', type=int)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.worker is not None:
        if not args.execute:
            raise ValueError('Worker requires --execute and an existing study manifest')
        worker(Path(args.output).resolve(), args.worker)
    else:
        run(args)

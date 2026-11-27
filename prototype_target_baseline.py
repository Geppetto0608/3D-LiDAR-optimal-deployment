"""Full-orientation, cold-target MILP references for a frozen refinement study.

Reuses prototype_offline.py unchanged, from the online study's source snapshot.
No online deployment is supplied to the MILP warm start. Default is planning only.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import csv
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from prototype_offline import digest, save


def now():
    return datetime.now(timezone.utc).isoformat()


def plan(study):
    study = Path(study).resolve()
    manifest = json.loads((study/'manifest.json').read_text())
    if manifest['status'] != 'complete' or any(j['status'] != 'complete' for j in manifest['jobs']):
        raise ValueError('Online study must be complete before target baseline generation')
    grouped = {}
    for job in manifest['jobs']:
        row = job['scenario']
        item = grouped.setdefault(row['id'], dict(scenario=row, k_values=set(),
            config=str(Path(job['source_roots'][0])/'config.yaml')))
        if item['scenario'] != row:
            raise ValueError('Conflicting target geometry')
        item['k_values'].add(job['budget_k'])
    return [dict(item, k_values=sorted(item['k_values'])) for _, item in sorted(grouped.items())]


def command(root, item, seconds, execute=False):
    sid = item['scenario']['id']
    result = [sys.executable, '-u', 'prototype_offline.py',
        '--config', str(root/'inputs'/f's{sid}_config.yaml'),
        '--scenario-json', str(root/'inputs'/f's{sid}.json'),
        '--output', str(root/'baselines'/f's{sid}'),
        '--k-values', *map(str, item['k_values']), '--time-limit', str(seconds),
        '--target-batch-size', '128', '--max-edges', '1500000000']
    if execute:
        result.append('--execute')
    return result


def available_memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) * 1024


def check_online_domain(study, original):
    from prototype_offline import explicit_scenario
    from src.environment import MapGeometry, load_config
    from src.roadside.candidate_generator import _base_positions
    from src.methods.local_mappo import validate_poses
    # Fail rather than inspect using a different geometry/candidate implementation.
    for name in ('src/environment.py', 'src/roadside/candidate_generator.py', 'src/methods/local_mappo.py'):
        if digest(name) != digest(study/'source'/name):
            raise ValueError(f'Domain validation source differs from online snapshot: {name}')
    domains, result = {}, {}
    for job in original['jobs']:
        sid = job['scenario']['id']
        if sid not in domains:
            cfg = load_config(Path(job['source_roots'][0])/'config.yaml')
            geometry = MapGeometry(explicit_scenario(job['scenario']))
            domains[sid] = (cfg, geometry, set(_base_positions(geometry, cfg)))
        cfg, geometry, positions = domains[sid]
        poses = np.load(study/'jobs'/job['id']/'deployment.npy', allow_pickle=False)
        validate_poses(poses, geometry, cfg)
        outside = [p.tolist() for p in poses if tuple(map(float, p[:3])) not in positions]
        result[job['id']] = dict(inside_milp_xyz_domain=not outside,
            outside_xyz_count=len(outside), outside_poses=outside)
    return result


def report(root, manifest):
    """Join results only after solving; online poses never inform optimization."""
    study = Path(manifest['online_study'])
    online = json.loads((root/'online_summary.json').read_text())['pairs']
    domains = json.loads((root/'online_domain.json').read_text())
    pairs, baseline_rows = [], []
    for item in manifest['targets']:
        if item['status'] != 'complete':
            continue
        sid = item['scenario']['id']
        directory = root/'baselines'/f's{sid}'
        data = json.loads((directory/'manifest.json').read_text())
        if data['status'] != 'complete' or data['k_values'] != item['k_values']:
            raise ValueError('Incomplete baseline worker')
        # Load only small arrays. Packed coverage masks remain in the archive.
        with np.load(directory/'candidates.npz', allow_pickle=False) as archive:
            targets = archive['target_points']
        for k in item['k_values']:
            value = json.loads((directory/f'k{k}/result.json').read_text())
            checked = json.loads((directory/f'k{k}/validation.json').read_text())
            if not checked['passed'] or value['num_targets'] != len(targets):
                raise ValueError('Baseline validation or target count mismatch')
            solver = value['solver']
            baseline = dict(scenario_id=sid, K=k, milp_coverage_pct=value['coverage_pct'],
                certification=value['certification'], solver_status=solver['solver_status'],
                objective=solver.get('objective'), best_bound=solver.get('best_bound'), mip_gap=solver.get('mip_gap'),
                candidate_count=data['candidate_stats']['num_candidates'],
                candidate_build_sec=data['candidate_build_sec'], k_work_sec=value['wall_time_sec'],
                candidate_plus_k_work_sec=data['candidate_build_sec']+value['wall_time_sec'],
                solver_sec=solver.get('solve_time_sec'), map_worker_wall_sec=item['worker_wall_sec'])
            baseline_rows.append(baseline)
            for row in online:
                if row['scenario_id'] != sid or row['K'] != k:
                    continue
                identity = f's{sid}_k{k}_{row["initialization"]}'
                # Target samples must be identical, not merely the same count.
                for name in ['local_search', *[f'mappo_seed{s}' for s in manifest['seeds']]]:
                    path = study/'jobs'/f'{identity}_{name}'/'coverage.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        if not np.array_equal(targets, saved['targets']):
                            raise ValueError('Online/baseline target coordinates differ')
                pairs.append(dict(**baseline, initialization=row['initialization'],
                    local_search_inside_milp_domain=domains[f'{identity}_local_search']['inside_milp_xyz_domain'],
                    all_mappo_trials_inside_milp_domain=all(domains[f'{identity}_mappo_seed{s}']['inside_milp_xyz_domain']
                                                         for s in manifest['seeds']),
                    initial_coverage_pct=row['initial_coverage_pct'],
                    local_search_coverage_pct=row['local_search_coverage_pct'],
                    mappo_mean_pct=row['mappo_mean_pct'], mappo_sample_sd_pct=row['mappo_sample_sd_pct'],
                    local_search_minus_milp_pp=row['local_search_coverage_pct']-value['coverage_pct'],
                    mappo_minus_milp_pp=row['mappo_mean_pct']-value['coverage_pct'],
                    local_search_online_sec=row['local_search_online_sec'],
                    mappo_mean_online_sec=row['mappo_mean_online_sec']))
    for name, rows in [('baselines.csv', baseline_rows), ('comparison.csv', pairs)]:
        if rows:
            with (root/name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    save(root/'comparison.json', dict(status=manifest['status'], pairs=pairs,
        expected_baselines=sum(len(t['k_values']) for t in manifest['targets']),
        completed_baselines=len(baseline_rows),
        note='Time-limited best feasible references unless certified. Full orientation on declared 2m XY grid, '
             'not a continuous optimum. Runtime is single shared-host measurement. '
             'Some online positions are outside this XY candidate domain; see domain flags. '
             'candidate_plus_k_work_sec excludes archive serialization/import and includes final cached validation; '
             'map_worker_wall_sec includes both K and all archive overhead. No controlled speedup claim.'))
    table = ''.join(f'<tr><td>{r["scenario_id"]}</td><td>{r["K"]}</td><td>{html.escape(r["initialization"])}</td>'
        f'<td>{r["milp_coverage_pct"]:.3f}</td><td>{r["local_search_coverage_pct"]:.3f}</td>'
        f'<td>{r["mappo_mean_pct"]:.3f}</td><td>{r["certification"]}</td></tr>' for r in pairs)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Cold target MILP references</title>'
        '<h1>Cold target MILP references</h1><p>Status: '+html.escape(manifest['status'])+'</p>'
        '<p>Full orientation / declared discrete grid. Time-limited feasible is not an optimum.</p>'
        '<p>Some online deployments are outside the MILP XY candidate domain. '
        'This is a quality comparison, not a global optimality gap certificate.</p>'
        '<p><a href="manifest.json">All jobs</a> | <a href="comparison.csv">Comparison CSV</a> | '
        '<a href="baselines.csv">Time, bounds and status</a></p>'
        '<table><tr><th>Map</th><th>K</th><th>Initialization</th><th>MILP %</th>'
        '<th>Local search %</th><th>MAPPO mean %</th><th>Certification</th></tr>'+table+'</table>', encoding='utf-8')


def run(args):
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('Positive finite solver time required')
    study = Path(args.study).resolve()
    targets = plan(study)
    print(json.dumps(dict(target_maps=len(targets), solves=sum(len(t['k_values']) for t in targets),
        seconds=args.seconds, top_m=0, threads=1, mip_gap=0, soft_mem_gb=90,
        new_cache_per_target=True, online_warm_start=False, execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    (root/'logs').mkdir()
    original = json.loads((study/'manifest.json').read_text())
    for relative, expected in original['frozen_hashes'].items():
        if digest(study/relative) != expected:
            raise RuntimeError(f'Online frozen source/input changed: {relative}')
    shutil.copytree(study/'source', root/'source', ignore=shutil.ignore_patterns('__pycache__'))
    # The reused runner snapshots this metadata file even for explicit target JSON.
    dest = root/'source/data/scenarios/yeongjong_final.json'
    dest.parent.mkdir(parents=True)
    shutil.copy2('data/scenarios/yeongjong_final.json', dest)
    shutil.copy2(__file__, root/'source/prototype_target_baseline.py')
    shutil.copy2(study/'summary.json', root/'online_summary.json')
    save(root/'online_domain.json', check_online_domain(study, original))
    frozen = {str(p.relative_to(root)): digest(p) for p in (root/'source').rglob('*') if p.is_file()}
    frozen['online_summary.json'] = digest(root/'online_summary.json')
    frozen['online_domain.json'] = digest(root/'online_domain.json')
    for item in targets:
        sid = item['scenario']['id']
        save(root/'inputs'/f's{sid}.json', item['scenario'])
        shutil.copy2(item['config'], root/'inputs'/f's{sid}_config.yaml')
        item['status'] = 'pending'
    for p in (root/'inputs').iterdir():
        frozen[str(p.relative_to(root))] = digest(p)
    online_hashes = {str(p): digest(p) for p in [study/'manifest.json', study/'summary.json',
        *[study/'jobs'/j['id']/f for j in original['jobs']
          for f in ('result.json', 'coverage.npz', 'deployment.npy')]]}
    manifest = dict(schema='cold_target_baselines_v1', status='running', started_at=now(),
        command=[sys.executable, *sys.argv], online_study=str(study), targets=targets,
        frozen_hashes=frozen, online_result_hashes=online_hashes, seeds=original['seeds'])
    save(root/'manifest.json', manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with (root/'packages.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    try:
        for item in targets:
            for path, expected in frozen.items():
                if digest(root/path) != expected:
                    raise RuntimeError(f'Frozen baseline input changed: {path}')
            if available_memory() < 96 * 1024**3:
                raise RuntimeError('Less than 96 GiB available; refusing to start a 90 GB soft-limit solve')
            sid = item['scenario']['id']
            manifest['current'] = sid
            item.update(status='running', started_at=now())
            save(root/'manifest.json', manifest)
            # Run the actual offline dry-run before each expensive fresh-cache job.
            subprocess.run(command(root, item, args.seconds), cwd=root/'source', env=env, check=True)
            print(f'Start target {sid}, K={item["k_values"]}', flush=True)
            start = time.monotonic()
            with (root/'logs'/f's{sid}.log').open('w') as log:
                subprocess.run(command(root, item, args.seconds, True), cwd=root/'source',
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            item.update(status='complete', completed_at=now(), worker_wall_sec=time.monotonic()-start)
            save(root/'manifest.json', manifest)
            for path, expected in online_hashes.items():
                if digest(path) != expected:
                    raise RuntimeError(f'Online evidence changed: {path}')
            report(root, manifest)
        manifest.update(status='complete', current=None, completed_at=now())
    except BaseException as exc:
        for item in targets:
            if item['status'] == 'running':
                item['status'] = 'failed'
        manifest.update(status='failed', error=repr(exc), completed_at=now())
        raise
    finally:
        save(root/'manifest.json', manifest)
        report(root, manifest)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='results/prototype_refinement_paired_20261011')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--execute', action='store_true')
    run(parser.parse_args())

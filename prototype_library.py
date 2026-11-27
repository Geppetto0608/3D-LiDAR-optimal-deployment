"""Expand an offline library, then evaluate transfers on separate cold maps.

No MAPPO or target-map MILP runs here. Existing libraries are not overwritten.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from prototype_offline import describe, digest, explicit_scenario, save
from prototype_segments import prepare_segment_placements
from prototype_transfer import export_view, load_prototype
from src.core.runtime import RuntimeMeter
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, get_lidar_profile
from src.methods.local_mappo import validate_poses


def now():
    return datetime.now(timezone.utc).isoformat()


def load_suite(data):
    if data.get('schema') != 'prototype_library_expansion_v1' or data.get('split') != 'development':
        raise ValueError('Expected a development library expansion suite')
    if not data.get('families'):
        raise ValueError('Empty library suite')
    names, ids, geometries = set(), set(), set()
    for family in data['families']:
        name = family['name']
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in names:
            raise ValueError('Family names must be unique safe identifiers')
        names.add(name)
        if len(family['sources']) != 2 or not family['targets']:
            raise ValueError('Each family requires two sources and at least one target')
        low, high = family['sources']
        if not low['road_width'] < high['road_width']:
            raise ValueError('Sources must have increasing widths')
        for row in family['sources'] + family['targets']:
            scenario = explicit_scenario(row)
            g = MapGeometry(scenario)
            signature = (scenario.topology, scenario.road_width, scenario.num_lanes,
                         g.junction_x_m, g.junction_y_m)
            if row['id'] in ids or signature in geometries:
                raise ValueError('Duplicate ID or geometry; source/target leakage prohibited')
            ids.add(row['id'])
            geometries.add(signature)
            if row['topology'] != low['topology'] or row['num_lanes'] != low['num_lanes']:
                raise ValueError('Family topology and lane count must agree')
        for row in family['targets']:
            if not low['road_width'] < row['road_width'] < high['road_width']:
                raise ValueError('Target widths must be interior; extrapolation is unsupported')
    return data


def offline_command(args, root, row):
    return [sys.executable, '-u', 'prototype_offline.py', '--config', args.config,
            '--scenario-json', str(root/'inputs'/f's{row["id"]}.json'),
            '--output', str(root/'library'/f's{row["id"]}'),
            '--k-values', *map(str, args.k_values), '--time-limit', str(args.solver_seconds),
            '--max-edges', str(args.max_edges), '--target-batch-size', '128', '--execute']


def evaluate_target(directory, row, k, low, high):
    """Only geometric source poses enter this path, never archived source masks."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    work = RuntimeMeter()
    cfg = low[5]
    for field in ('candidate', 'target', 'mappo'):
        if getattr(cfg, field) != getattr(high[5], field):
            raise ValueError(f'Prototype configuration mismatch: {field}')
    profile = get_lidar_profile(cfg.lidar_profile, cfg)
    if profile != get_lidar_profile(high[5].lidar_profile, high[5]):
        raise ValueError('Prototype sensor mismatch')
    with work.stage('geometry_and_transfer'):
        geometry = MapGeometry(explicit_scenario(row))
        placements, matching = prepare_segment_placements(low[4], high[4], low[3], high[3], geometry, cfg)
    with work.stage('fresh_scene'):
        mesh = geometry.create_environment_mesh()
        provider = MeshVisibility(mesh, geometry.target_points(cfg.target['coverage_z_slices']), profile, work=work)
    record = dict(scenario=row, budget_k=k, matching=matching,
                  sources=[dict(root=str(p[0]), certification=p[1]['certification'],
                                result_sha256=digest(p[0]/f'k{k}/result.json')) for p in (low, high)],
                  full_target_cache=False, mappo_executed=False, placements={})
    views = []
    for name in ('transfer_low', 'transfer_high', 'interpolation'):
        if name not in placements:
            record['placements'][name] = dict(status='unavailable', reason=matching.get('reason'))
            continue
        poses = placements[name]
        with work.stage('direct_evaluation'):
            validate_poses(poses, geometry, cfg)
            if len(poses) > k:
                raise ValueError('Transferred pose exceeds sensor budget')
            work.add('deployment_evaluations')
            density = (np.stack([provider.mask(p) for p in poses]).sum(axis=0)
                       if len(poses) else np.zeros(len(provider.targets), dtype=int))
            coverage = 100 * float((density > 0).mean())
        record['placements'][name] = dict(status='complete', coverage_pct=coverage,
            covered_count=int((density > 0).sum()), target_count=len(density), num_selected=len(poses),
            positions_npy=f'{name}.npy', view=f'{name}.html')
        np.save(directory/f'{name}.npy', poses)
        np.savez_compressed(directory/f'{name}_coverage.npz', targets=provider.targets, density=density)
        views.append((name, density, poses, coverage))
    record['runtime'] = work.report()
    # Export is excluded from the measured transfer/evaluation interval.
    for name, density, poses, coverage in views:
        export_view(directory/f'{name}.html', mesh, provider.targets, density, poses,
                    f'S{row["id"]} K={k} {name}', coverage)
    save(directory/'geometry.json', describe(geometry, cfg))
    save(directory/'report.json', record)
    return record


def write_index(root, manifest):
    rows = []
    for entry in manifest['transfers']:
        for name, result in entry['placements'].items():
            path = f'transfers/s{entry["scenario_id"]}/k{entry["k"]}/{name}.html'
            value = (f'<a href="{path}">{result["coverage_pct"]:.3f}%</a>'
                     if result['status'] == 'complete' else html.escape(result.get('reason') or 'unavailable'))
            rows.append(f'<tr><td>{entry["scenario_id"]}</td><td>{entry["k"]}</td>'
                        f'<td>{name}</td><td>{value}</td></tr>')
    (root/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Prototype Library Expansion</title><style>body{font-family:sans-serif;margin:24px}'
        'td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style>'
        '<h1>Prototype Library Expansion</h1><p>Status: ' + html.escape(manifest['status']) + '</p>'
        '<p>Development maps. No MAPPO or target MILP. Unavailable correspondence is not a zero score.</p>'
        '<table><tr><th>Scenario</th><th>K</th><th>Method</th><th>Coverage / reason</th></tr>'
        + ''.join(rows) + '</table></html>', encoding='utf-8')


def run(args):
    suite = load_suite(json.loads(Path(args.suite).read_text()))
    if args.k_values != sorted(set(args.k_values)) or any(k < 1 or k > 6 for k in args.k_values):
        raise ValueError('Use ascending unique K within 1..6')
    if not math.isfinite(args.solver_seconds) or args.solver_seconds <= 0 or args.max_edges < 1:
        raise ValueError('Positive finite solver time and edge limit required')
    root = Path(args.output)
    plan = dict(source_maps=sum(len(f['sources']) for f in suite['families']),
                new_target_maps=sum(len(f['targets']) for f in suite['families']),
                k_values=args.k_values, topm=0, solver_seconds=args.solver_seconds,
                mip_gap=0, memory_gb=90, sequential=True, mappo=False,
                missing_correspondence='record unavailable; no silent auxiliary constraints')
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    (root/'logs').mkdir()
    save(root/'suite.json', suite)
    paths = [Path(args.config), Path(args.suite), *Path('.').glob('prototype_*.py'),
             Path('data/scenarios/yeongjong_final.json'), *Path('src').rglob('*.py')]
    hashes = {}
    for path in paths:
        hashes[str(path)] = digest(path)
        dest = root/'source'/(path.name if path.is_absolute() else path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    manifest = dict(schema='prototype_library_expansion_run_v1', status='running',
        started_at=now(), command=[sys.executable, *sys.argv], plan=plan, source_hashes=hashes,
        library=[], transfers=[], current=None)
    save(root/'manifest.json', manifest)
    def check_sources():
        for path, expected in hashes.items():
            if digest(path) != expected:
                raise RuntimeError(f'Source changed during library generation: {path}')
    started = time.monotonic()
    try:
        for family in suite['families']:
            for row in family['sources']:
                check_sources()
                save(root/'inputs'/f's{row["id"]}.json', row)
                command = offline_command(args, root, row)
                manifest['current'] = dict(stage='offline', scenario_id=row['id'], command=command)
                save(root/'manifest.json', manifest)
                print(f'Offline library scenario {row["id"]}', flush=True)
                with (root/'logs'/f's{row["id"]}.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                library_root = root/'library'/f's{row["id"]}'
                library_manifest = json.loads((library_root/'manifest.json').read_text())
                if library_manifest['status'] != 'complete':
                    raise RuntimeError('Incomplete source library')
                manifest['library'].append(dict(scenario=row, root=str(library_root),
                    candidate_stats=library_manifest['candidate_stats'], results=library_manifest['results'],
                    candidate_archive_sha256=digest(library_root/'candidates.npz')))
                save(root/'manifest.json', manifest)
            # Complete each family's transfers before starting another large source build.
            for row in family['targets']:
                for k in args.k_values:
                    check_sources()
                    sources = [load_prototype(root/'library'/f's{s["id"]}', k) for s in family['sources']]
                    manifest['current'] = dict(stage='transfer', scenario_id=row['id'], k=k)
                    save(root/'manifest.json', manifest)
                    directory = root/'transfers'/f's{row["id"]}'/f'k{k}'
                    try:
                        report = evaluate_target(directory, row, k, *sources)
                    except ValueError as exc:
                        # A collapsed/invalid projection must not stop other library families.
                        directory.mkdir(parents=True, exist_ok=True)
                        report = dict(scenario=row, budget_k=k, error=repr(exc), placements={
                            name: dict(status='failed', reason=str(exc)) for name in
                            ('transfer_low', 'transfer_high', 'interpolation')})
                        save(directory/'report.json', report)
                        print(f'Transfer failed S{row["id"]} K{k}: {exc}', flush=True)
                    manifest['transfers'].append(dict(scenario_id=row['id'], k=k, placements=report['placements']))
                    save(root/'manifest.json', manifest)
                    write_index(root, manifest)
        manifest['unavailable_interpolations'] = sum(
            r['placements']['interpolation']['status'] == 'unavailable' for r in manifest['transfers'])
        manifest['failed_transfers'] = sum(
            r['placements']['interpolation']['status'] == 'failed' for r in manifest['transfers'])
        manifest.update(status='complete_with_failures' if manifest['failed_transfers'] else 'complete', current=None)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest.update(finished_at=now(), wall_time_sec=time.monotonic()-started)
        save(root/'manifest.json', manifest)
        write_index(root, manifest)


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--suite', default='configs/library_expansion.json')
    p.add_argument('--config', default='configs/experiment.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--k-values', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--solver-seconds', type=float, default=300)
    p.add_argument('--max-edges', type=int, default=1_500_000_000)
    p.add_argument('--execute', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())

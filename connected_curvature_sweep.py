"""Controlled curvature and source-pool sweep using one fixed source library."""
from argparse import ArgumentParser, SUPPRESS
from copy import deepcopy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import connected_corridor_study as corridor
from src.environment import ConnectedCorridorGeometry, get_lidar_profile, load_config


BASE = Path(__file__).resolve().parent
SOURCE_FILES = corridor.SOURCE_FILES + [
    'connected_curvature_sweep.py', 'tests/validate_connected_corridor.py',
    'tests/validate_connected_curvature.py',
]


def variant_suite(base, sweep, bend, alternatives):
    suite = deepcopy(base)
    suite['geometry']['bend_length_m'] = bend
    suite['alternatives_per_site'] = alternatives
    suite['angles_deg'] = sweep['angles_deg'].copy()
    return corridor.validate_suite(suite)


def validate_sweep(sweep, base):
    expected = {'schema', 'angles_deg', 'bend_lengths_m', 'alternatives_per_site'}
    if (not isinstance(sweep, dict) or set(sweep) != expected
            or sweep['schema'] != 'connected_curvature_sweep_v1'):
        raise ValueError('Invalid curvature sweep schema/keys')
    for key in ('angles_deg', 'bend_lengths_m', 'alternatives_per_site'):
        values = sweep[key]
        if (not isinstance(values, list) or not values
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)
                or not np.isfinite(values).all() or len(set(values)) != len(values)):
            raise ValueError(f'Expected unique finite values for {key}')
    if len(sweep['angles_deg']) > 8:
        raise ValueError('At most 8 signed-angle samples in this sweep')
    if any(v <= 0 for v in sweep['bend_lengths_m']):
        raise ValueError('Positive transition lengths required')
    available = (len(base['candidate']['heights_m']) * len(base['candidate']['relative_yaws_deg'])
                 * len(base['candidate']['pitches_deg']))
    if any(not isinstance(v, int) or not 1 <= v <= available for v in sweep['alternatives_per_site']):
        raise ValueError('Alternative counts must be integers within source orientations per site')
    conditions = len(sweep['angles_deg']) * len(sweep['bend_lengths_m']) * len(sweep['alternatives_per_site'])
    if conditions > 64:
        raise ValueError('Bounded sweep limit: 64 conditions')
    for bend in sweep['bend_lengths_m']:
        for count in sweep['alternatives_per_site']:
            variant_suite(base, sweep, bend, count)
    return sweep


def plan_jobs(sweep):
    return [dict(bend_length_m=bend, alternatives_per_site=count, angle_deg=angle,
        group=f'b{bi:02d}_a{ai:02d}', directory=f'runs/b{bi:02d}_a{ai:02d}/angle_{i:02d}',
        status='pending')
        for bi, bend in enumerate(sweep['bend_lengths_m'])
        for ai, count in enumerate(sweep['alternatives_per_site'])
        for i, angle in enumerate(sweep['angles_deg'])]


def aggregate(root, sweep, jobs):
    rows = []
    groups = dict.fromkeys(job['group'] for job in jobs)
    for group in groups:
        path = root / 'runs' / group / 'comparison.csv'
        if not path.exists():
            continue
        current = [job for job in jobs if job['group'] == group]
        jobs_by_angle = {float(job['angle_deg']): job for job in current}
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                job = jobs_by_angle[float(row['angle_deg'])]
                if job['status'] != 'complete':
                    continue
                rows.append({**row, 'bend_length_m': job['bend_length_m'],
                    'alternatives_per_site': job['alternatives_per_site'],
                    'result_json': str(Path('runs') / group / row['result_json'])})
    if not rows:
        return
    with (root / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for count in sweep['alternatives_per_site']:
        selected = [r for r in rows if r['method'] == 'library_joint'
                    and r['alternatives_per_site'] == count]
        feasible = [r for r in selected if r['feasible'] == 'True']
        gaps = [float(r['objective_gap_to_grid_optimum']) for r in feasible]
        summary.append(dict(alternatives_per_site=count, completed=len(selected), feasible=len(feasible),
            matches_grid_optimum=sum(abs(v) <= 1e-7 for v in gaps),
            mean_objective_gap=float(np.mean(gaps)) if gaps else None,
            max_objective_gap=max(gaps, default=None),
            min_interface_coverage_pct=min((float(r['interface_coverage_pct']) for r in feasible), default=None),
            candidate_queries=sorted(set(int(r['candidate_queries']) for r in selected))))
    pairs = {(float(r['bend_length_m']), float(r['angle_deg']), r['alternatives_per_site']): {}
             for r in rows}
    for row in rows:
        pairs[float(row['bend_length_m']), float(row['angle_deg']), row['alternatives_per_site']][row['method']] = row
    comparisons = []
    for (bend, angle, count), values in pairs.items():
        if len(values) != len(corridor.METHODS):
            continue
        initial, library = values['independent_transfer'], values['library_joint']
        reference, local = values['reference_global'], values['library_joint_ls']
        comparisons.append(dict(bend_length_m=bend, angle_deg=angle, alternatives_per_site=count,
            radius_m=bend / abs(np.deg2rad(angle)) if angle else None,
            transfer_feasible=initial['feasible'], transfer_coverage_pct=initial['coverage_pct'],
            transfer_interface_pct=initial['interface_coverage_pct'], library_feasible=library['feasible'],
            library_coverage_pct=library['coverage_pct'], library_interface_pct=library['interface_coverage_pct'],
            library_j_gap=library['objective_gap_to_grid_optimum'], library_ls_j_gap=local['objective_gap_to_grid_optimum'],
            reference_coverage_pct=reference['coverage_pct'], reference_interface_pct=reference['interface_coverage_pct']))
    with (root / 'paired.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    corridor.save(root / 'summary.json', dict(completed_conditions=sum(j['status'] == 'complete' for j in jobs),
        expected_conditions=len(jobs), method_rows=len(rows), library_summaries=summary,
        note='One fixed straight source; signed curvature samples; deterministic pool-size ablation. '
             'Means use feasible completed geometry cases only; no independent-trial inference.'))
    lines = ['# Curvature and Library Alternatives', '',
        'One source library is built once and reused for all conditions.',
        'Length, width, K, sensor model, grid, J and requirements are fixed.',
        'Transition length and source alternatives per site are the only swept variables.',
        'Each condition has separate per-method target queries and a later exhaustive reference build.', '',
        '| Bend m | Angle | Alternatives/site | Transfer coverage % | Library coverage % | Reference coverage % | Library J gap | Library + LS J gap |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        def display(key):
            return '-' if row[key] == '' or row[key] is None else f'{float(row[key]):.3f}'
        lines.append(f"| {row['bend_length_m']:g} | {row['angle_deg']:g} | {row['alternatives_per_site']} | "
            f"{display('transfer_coverage_pct')} | {display('library_coverage_pct')} | "
            f"{display('reference_coverage_pct')} | {display('library_j_gap')} | {display('library_ls_j_gap')} |")
    lines += ['', 'Raw methods: comparison.csv; matched conditions: paired.csv; counts: summary.json.',
        'Plots: coverage.png and objective_gap.png. Individual scenes: runs/<group>/angle_XX/comparison.png.',
        'Exact optima refer only to fixed K=2 and this coarse finite mesh/candidate/target grid.',
        'Interface constraints, new obstacles, periodic wrapping and MAPPO are subsequent experiments.',
        'Coverage/overlap/cost/failure thresholds are illustrative, not operational requirements.',
        'Timing includes repeated reference builds and shared process/import state; no controlled speedup claim.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    return rows


def plot_summary(root, sweep, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angles = sweep['angles_deg']
    colors = ['#147d92', '#cf593d', '#5b8e45', '#8d5d91']
    for filename, metrics in [('coverage.png', [('coverage_pct', 'Global coverage (%)'),
                                               ('interface_coverage_pct', 'Interface coverage (%)')]),
                              ('objective_gap.png', [('objective_gap_to_grid_optimum', 'J shortfall: library'),
                                                     ('objective_gap_to_grid_optimum', 'J shortfall: library + LS')])]:
        fig, axes = plt.subplots(2, len(angles), figsize=(4*len(angles), 7), squeeze=False,
                                 constrained_layout=True)
        for column, angle in enumerate(angles):
            for row_number, (metric, label) in enumerate(metrics):
                ax = axes[row_number, column]
                method = 'library_joint_ls' if filename.startswith('objective') and row_number else 'library_joint'
                for index, count in enumerate(sweep['alternatives_per_site']):
                    use = [r for r in rows if float(r['angle_deg']) == angle
                           and r['alternatives_per_site'] == count and r['method'] == method]
                    by_length = {float(r['bend_length_m']): r for r in use}
                    values = [float(by_length[b][metric]) if by_length[b][metric] else np.nan
                              for b in sweep['bend_lengths_m']]
                    ax.plot(sweep['bend_lengths_m'], values, marker='o', color=colors[index % len(colors)],
                            label=f'{count} alternatives/site')
                if filename.startswith('coverage'):
                    for name, style, text in [('reference_global', '--', 'Grid optimum'),
                                               ('independent_transfer', ':', 'Independent transfer')]:
                        use = [r for r in rows if float(r['angle_deg']) == angle
                               and r['alternatives_per_site'] == sweep['alternatives_per_site'][0]
                               and r['method'] == name]
                        by_length = {float(r['bend_length_m']): r for r in use}
                        ax.plot(sweep['bend_lengths_m'], [float(by_length[b][metric])
                            if by_length[b][metric] else np.nan for b in sweep['bend_lengths_m']],
                            color='#303030' if name.startswith('reference') else '#888888',
                            linestyle=style, marker='x', label=text)
                ax.set_title(f'{angle:g} deg')
                ax.set_xlabel('Transition length (m)')
                ax.set_ylabel(label)
                ax.set_xticks(sweep['bend_lengths_m'])
                ax.invert_xaxis()
                ax.grid(alpha=.2)
                if column == 0:
                    ax.legend(fontsize=8)
        fig.savefig(root / filename, dpi=160)
        plt.close(fig)


def worker(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        if corridor.digest(BASE / name) != expected:
            raise RuntimeError(f'Frozen source mismatch: {name}')
    for name, expected in manifest['input_hashes'].items():
        if corridor.digest(root / name) != expected:
            raise RuntimeError(f'Frozen input mismatch: {name}')
    base = corridor.load_suite(root / 'inputs/base_suite.json')
    sweep = validate_sweep(json.loads((root / 'inputs/sweep.json').read_text()), base)
    config = load_config(root / 'inputs/experiment.yaml')
    profile = get_lidar_profile(config.lidar_profile, config)
    started = time.perf_counter()
    try:
        manifest.update(status='running', stage='source_library', worker_started_at=corridor.now())
        corridor.save(root / 'manifest.json', manifest)
        source_suite = deepcopy(base)
        source_suite['alternatives_per_site'] = max(sweep['alternatives_per_site'])
        canonical, _, best = corridor.source_library(source_suite, profile, root / 'library')
        with np.load(root / 'library/library.npz', allow_pickle=False) as data:
            scores = data['scores'].copy()
        for job in manifest['jobs']:
            suite = variant_suite(base, sweep, job['bend_length_m'], job['alternatives_per_site'])
            directory = root / 'runs' / job['group']
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory / 'suite.json').exists():
                corridor.save(directory / 'suite.json', suite)
                selected = corridor.select_source_alternatives(canonical, scores, job['alternatives_per_site'])
                corridor.save(directory / 'source_selection.json', dict(source='../../library/library.npz',
                    alternative_ids=selected, best_source_id=best,
                    selection='Per-site ranking from the single fixed source, before target queries'))
            else:
                selected = json.loads((directory / 'source_selection.json').read_text())['alternative_ids']
            job['status'] = 'running'
            manifest.update(stage='connected_target', current_job=job['directory'])
            corridor.save(root / 'manifest.json', manifest)
            print(f"Start bend={job['bend_length_m']:g}, angle={job['angle_deg']:g}, "
                  f"alternatives={job['alternatives_per_site']}", flush=True)
            corridor.run_angle(suite, profile, canonical, selected, best, job['angle_deg'], root / job['directory'])
            job['status'] = 'complete'
            corridor.save(root / 'manifest.json', manifest)
            group_jobs = [dict(angle_deg=j['angle_deg'], directory=Path(j['directory']).name, status=j['status'])
                          for j in manifest['jobs'] if j['group'] == job['group']]
            corridor.summarize(directory, suite, group_jobs)
            aggregate(root, sweep, manifest['jobs'])
            print(f"Complete {sum(j['status'] == 'complete' for j in manifest['jobs'])}/{len(manifest['jobs'])}", flush=True)
        rows = aggregate(root, sweep, manifest['jobs'])
        plot_summary(root, sweep, rows)
        manifest.update(status='complete', stage='complete', finished_at=corridor.now())
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['worker_wall_sec'] = time.perf_counter() - started
        corridor.save(root / 'manifest.json', manifest)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', default=str(BASE / 'configs/connected_curvature_sweep.json'))
    parser.add_argument('--base-suite', default=str(BASE / 'configs/connected_corridor_pilot.json'))
    parser.add_argument('--config', default=str(BASE / 'configs/experiment.yaml'))
    parser.add_argument('--output', default='results/connected_curvature_sweep')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-root', help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return worker(Path(args.worker_root).resolve())
    base = corridor.load_suite(args.base_suite)
    sweep = validate_sweep(json.loads(Path(args.sweep).read_text()), base)
    jobs = plan_jobs(sweep)
    source_geometry = ConnectedCorridorGeometry(**{**base['geometry'],
        'length_m': base['geometry']['length_m']/2, 'bend_length_m': 0})
    canonical = corridor.candidates(source_geometry, base['candidate'])
    sites = len(np.unique(canonical[:, :2], axis=0))
    config = load_config(args.config)
    plan = dict(geometries=len(sweep['angles_deg'])*len(sweep['bend_lengths_m']),
        conditions=len(jobs), method_runs=len(jobs)*len(corridor.METHODS), source_builds=1,
        source_candidates=len(canonical), target_candidates=2*len(canonical),
        source_alternative_sizes={str(n): n*sites for n in sweep['alternatives_per_site']},
        target_pool_sizes={str(n): 2*n*sites for n in sweep['alternatives_per_site']},
        sweep=sweep, sensor=asdict(get_lidar_profile(config.lidar_profile, config)), execute=args.execute)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'inputs').mkdir()
    for origin, name in [(args.sweep, 'sweep.json'), (args.base_suite, 'base_suite.json'),
                         (args.config, 'experiment.yaml')]:
        shutil.copy2(origin, root / 'inputs' / name)
    for name in SOURCE_FILES:
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, destination)
    packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
                              capture_output=True, text=True).stdout
    (root / 'packages.txt').write_text(packages)
    corridor.save(root / 'manifest.json', dict(status='prepared', created_at=corridor.now(), plan=plan,
        command=sys.argv if argv is None else argv, jobs=jobs,
        source_hashes={name: corridor.digest(root / 'source' / name) for name in SOURCE_FILES},
        input_hashes={f'inputs/{name}': corridor.digest(root / 'inputs' / name)
                      for name in ('sweep.json', 'base_suite.json', 'experiment.yaml')}))
    command = [sys.executable, '-u', str(root / 'source/connected_curvature_sweep.py'),
               '--worker-root', str(root), '--execute']
    with (root / 'worker.log').open('w') as log:
        completed = subprocess.run(command, cwd=root / 'source', stdout=log, stderr=subprocess.STDOUT)
    try:
        if completed.returncode:
            raise RuntimeError(f'Worker failed; see {root / "worker.log"}')
        audit = subprocess.run([sys.executable, str(root / 'source/tests/validate_connected_curvature.py'),
            str(root)], capture_output=True, text=True)
        if audit.returncode:
            (root / 'audit.log').write_text(audit.stdout + audit.stderr)
            raise RuntimeError(f'Independent audit failed; see {root / "audit.log"}')
        corridor.save(root / 'independent_validation.json', json.loads(audit.stdout))
    except BaseException as exc:
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest.update(status='failed', error=repr(exc))
        corridor.save(root / 'manifest.json', manifest)
        raise
    print(f'Completed and independently audited: {root / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()

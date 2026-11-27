"""Explain completed sweep results using saved arrays; no optimization/raycast."""
from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    validation = json.loads((root / 'independent_validation.json').read_text())
    if (manifest['status'] != 'complete' or validation['manifest_sha256'] != digest(root / 'manifest.json')
            or not validation['all_reference_optima_match']):
        raise ValueError('A completed, independently audited sweep is required')
    if output == root or root.is_relative_to(output):
        raise ValueError('Derived report must have its own output directory')
    input_paths = [root / p for p in ('manifest.json', 'comparison.csv', 'paired.csv',
        'summary.json', 'independent_validation.json', 'inputs/base_suite.json',
        'inputs/sweep.json', 'library/library.npz')]
    with np.load(root / 'library/library.npz', allow_pickle=False) as data:
        source, scores = data['canonical'], data['scores']
    pool_min = min(job['alternatives_per_site'] for job in manifest['jobs'])
    geometries = [job for job in manifest['jobs'] if job['alternatives_per_site'] == pool_min]
    by_angle = {}
    for job in geometries:
        by_angle.setdefault(job['angle_deg'], []).append(job)
    ranks, visibility = [], []
    for angle, jobs in by_angle.items():
        jobs = sorted(jobs, key=lambda j: -j['bend_length_m'])
        first_masks = None
        for job in jobs:
            directory = root / job['directory']
            result_path = directory / 'reference_global/result.json'
            result = json.loads(result_path.read_text())
            input_paths.extend([result_path, directory / 'reference_masks.npz', directory / 'geometry.npz'])
            with np.load(directory / 'reference_masks.npz', allow_pickle=False) as data:
                masks = data['masks']
            with np.load(directory / 'geometry.npz', allow_pickle=False) as data:
                target_canonical = data['canonical_candidates']
            if first_masks is None:
                first_masks = masks.copy()
            changed = int(np.count_nonzero(masks != first_masks))
            visibility.append(dict(bend_length_m=job['bend_length_m'], angle_deg=angle,
                comparison_bend_length_m=jobs[0]['bend_length_m'], changed_entries=changed,
                total_candidate_target_entries=int(masks.size), changed_fraction=changed/masks.size))
            for agent, cid in enumerate(result['selected_ids'] or []):
                sid = cid % len(source)
                ids = np.flatnonzero(np.all(source[:, :2] == source[sid, :2], axis=1))
                ranked = ids[np.argsort(-scores[ids], kind='stable')]
                rank = int(np.flatnonzero(ranked == sid)[0]) + 1
                ranks.append(dict(bend_length_m=job['bend_length_m'], angle_deg=angle,
                    sensor_index=agent, target_candidate_id=cid, source_candidate_id=sid,
                    target_station_m=float(target_canonical[cid, 0]), source_station_m=float(source[sid, 0]),
                    side_lateral_m=float(source[sid, 1]), height_m=float(source[sid, 2]),
                    relative_yaw_deg=float(source[sid, 3]), pitch_deg=float(source[sid, 4]),
                    source_rank_at_site=rank, source_score=float(scores[sid])))
    with (root / 'comparison.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((root / 'summary.json').read_text())
    library_rows = [row for row in rows if row['method'] == 'library_joint']
    coverages = [float(row['coverage_pct']) for row in library_rows if row['coverage_pct']]
    interfaces = [float(row['interface_coverage_pct']) for row in library_rows if row['interface_coverage_pct']]
    local_gains = 0
    for row in library_rows:
        matching = [other for other in rows if other['method'] == 'library_joint_ls'
            and all(other[key] == row[key] for key in ('bend_length_m', 'angle_deg', 'alternatives_per_site'))]
        if matching and matching[0]['objective_score'] and row['objective_score']:
            local_gains += float(matching[0]['objective_score']) > float(row['objective_score']) + 1e-8
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'reference_source_ranks.csv', ranks)
    write_csv(output / 'visibility_changes.csv', visibility)
    plot(output, rows, manifest['plan']['sweep'])
    missing = [row for row in ranks if row['source_rank_at_site'] > max(manifest['plan']['sweep']['alternatives_per_site'])]
    lines = ['# Curvature Sweep Findings', '',
        f"Completed conditions: {summary['completed_conditions']}; method results: {summary['method_rows']}.",
        f"Independent audit: {validation['checked_deployments']} saved deployments and "
        f"{validation['reference_optima_checked']} exhaustive references matched.", '',
        '| Source alternatives/site | Target pose queries | Exact J matches / geometry cases | Mean J gap | Max J gap |',
        '|---:|---:|---:|---:|---:|']
    for item in summary['library_summaries']:
        lines.append(f"| {item['alternatives_per_site']} | {','.join(map(str,item['candidate_queries']))} | "
            f"{item['matches_grid_optimum']}/{item['completed']} | "
            f"{item['mean_objective_gap']:.6f} | {item['max_objective_gap']:.6f} |")
    lines.extend(['',
        f'Recorded library global coverage range: {min(coverages):.6f}-{max(coverages):.6f}%; '
        f'interface coverage range: {min(interfaces):.6f}-{max(interfaces):.6f}%.',
        'Detailed geometry-matched coverage and J gaps are in ../paired.csv.',
        f'Library-initialized LS improved J in {local_gains} of {len(library_rows)} conditions.', '',
        f'{len(missing)} recorded reference sensor poses are absent from every tested source pool.',
        f"Their source ranks at the same site are {sorted(set(row['source_rank_at_site'] for row in missing))}; "
        f"heights {sorted(set(row['height_m'] for row in missing))}m and relative yaws "
        f"{sorted(set(row['relative_yaw_deg'] for row in missing))}deg.",
        'This explains omission of those recorded layouts; it is not a characterization of all optimal layouts.',
        'The summary table reports whether additional target queries improve best J.', '',
        'The visibility_changes.csv comparison confirms that the executed geometry changes target masks.',
        f"Maximum changed candidate-target fraction from the 20m geometry: {max(v['changed_fraction'] for v in visibility):.8f}.",
        'This fraction is a mask-entry difference, not a coverage loss, a ray count or a detection metric.', '',
        'The base requirements are recorded in ../inputs/base_suite.json; thresholds remain illustrative.',
        'Sampled interface coverage alone does not establish a maximum reusable curvature.',
        'Next conditions should add explicit connection-band redundancy and installation exclusions.',
        'Source alternatives should also be tested for different height/side/response modes rather than '
        'only more prefixes of the isolated-source score.', '',
        'All cases are synthetic development geometries. Numeric constraints are illustrative. '
        'No MAPPO, continuous-domain certificate or cold-start speedup conclusion is provided.', ''])
    (output / 'REPORT.md').write_text('\n'.join(lines))
    provenance = dict(report_code_sha256=digest(__file__),
        input_hashes={str(path.relative_to(root)): digest(path) for path in input_paths},
        scope='Post-run source-rank and geometry sensitivity diagnostics; stored arrays only')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return output


def plot(output, rows, sweep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angles = sweep['angles_deg']
    colors, markers = ['#147d92', '#cf593d', '#5b8e45'], ['o', 's', '^']
    for filename, metrics in [('coverage.png', [('coverage_pct', 'Global coverage (%)'),
                                               ('interface_coverage_pct', 'Interface coverage (%)')]),
                              ('objective_gap.png', [('objective_gap_to_grid_optimum', 'J gap: library'),
                                                     ('objective_gap_to_grid_optimum', 'J gap: library + LS')])]:
        fig, axes = plt.subplots(2, len(angles), figsize=(4*len(angles), 7), squeeze=False,
                                 constrained_layout=True)
        for row_number, (metric, ylabel) in enumerate(metrics):
            method = 'library_joint_ls' if filename.startswith('objective') and row_number else 'library_joint'
            finite = [float(r[metric]) for r in rows if r[metric] and r['method'] == method]
            if filename.startswith('coverage'):
                finite += [float(r[metric]) for r in rows if r[metric] and r['method'] == 'reference_global']
                pad = max(.05, (max(finite) - min(finite))*.15) if max(finite) > min(finite) else 1.
                bounds = (max(0, min(finite)-pad), min(100, max(finite)+pad))
            else:
                bounds = (0, max(1, max(finite)*1.1))
            for column, angle in enumerate(angles):
                ax = axes[row_number, column]
                for index, count in enumerate(sweep['alternatives_per_site']):
                    use = [r for r in rows if float(r['angle_deg']) == angle
                           and int(r['alternatives_per_site']) == count and r['method'] == method]
                    by_length = {float(r['bend_length_m']): r for r in use}
                    ax.plot(sweep['bend_lengths_m'], [float(by_length[b][metric]) for b in sweep['bend_lengths_m']],
                        marker=markers[index % len(markers)], color=colors[index % len(colors)],
                        markerfacecolor='none', label=f'{count} alternatives/site')
                if filename.startswith('coverage'):
                    use = [r for r in rows if float(r['angle_deg']) == angle and r['method'] == 'reference_global'
                           and int(r['alternatives_per_site']) == min(sweep['alternatives_per_site'])]
                    by_length = {float(r['bend_length_m']): r for r in use}
                    ax.plot(sweep['bend_lengths_m'], [float(by_length[b][metric]) for b in sweep['bend_lengths_m']],
                            marker='x', linestyle='--', color='#303030', label='Grid optimum')
                ax.set(title=f'{angle:g} deg', xlabel='Transition length (m)', ylabel=ylabel, ylim=bounds)
                ax.set_xticks(sweep['bend_lengths_m'])
                ax.invert_xaxis()
                ax.grid(alpha=.2)
                if column == 0:
                    ax.legend(fontsize=8, loc='center right')
        fig.suptitle('Controlled curvature sweep: fixed source and sensor budget')
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(generate(args.root, args.output or args.root / 'diagnostics'))

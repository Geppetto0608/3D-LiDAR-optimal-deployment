"""Derived plots and feasibility summary for a completed connection study."""
from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ('independent_transfer', 'top_joint', 'diverse_joint', 'top_joint_ls',
           'diverse_joint_ls', 'reference_one_per_module', 'reference_global')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    validation = json.loads((root / 'independent_validation.json').read_text())
    if (manifest['status'] != 'complete' or validation['manifest_sha256'] != digest(root / 'manifest.json')
            or not validation['all_reference_optima_match']):
        raise ValueError('Completed independently audited study required')
    if output == root or root.is_relative_to(output):
        raise ValueError('Use a separate derived output directory')
    inputs = [root / name for name in ('manifest.json', 'independent_validation.json',
        'comparison.csv', 'paired.csv', 'inputs/study.json', 'source_pools.json')]
    with (root / 'comparison.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (root / 'paired.csv').open(newline='') as handle:
        pairs = list(csv.DictReader(handle))
    by_method = {name: [] for name in METHODS}
    for row in rows:
        path = root / row['result_json']
        result = json.loads(path.read_text())
        inputs.append(path)
        assessment = result['assessment']
        if row['feasible'] != str(assessment is not None and assessment['feasible']):
            raise ValueError('CSV feasibility disagrees with original result')
        if assessment is not None:
            for key in ('objective_score', 'coverage_pct'):
                if abs(float(row[key])-assessment[key]) > 1e-8:
                    raise ValueError('CSV metrics disagree with original result')
        by_method[row['method']].append(row)
    output.mkdir(parents=True, exist_ok=False)
    plot_overview(output / 'overview.png', rows, manifest)
    profiles = [p['name'] for p in manifest['plan']['profiles']]
    example_profile = 'combined' if 'combined' in profiles else profiles[-1]
    examples = [j for j in manifest['jobs'] if j['profile'] == example_profile]
    example = max(examples, key=lambda j: j['angle_deg'])
    scene_inputs = plot_scene(output / 'scene.png', root / example['directory'], example)
    inputs.extend(scene_inputs)
    summary = []
    for name, values in by_method.items():
        valid = [r for r in values if r['feasible'] == 'True']
        summary.append(dict(method=name, feasible=len(valid), conditions=len(values),
            grid_optimum_matches=sum(abs(float(r['objective_gap'])) <= 1e-7 for r in valid),
            min_coverage_pct=min((float(r['coverage_pct']) for r in valid), default=None),
            max_coverage_pct=max((float(r['coverage_pct']) for r in valid), default=None)))
    differences = [float(p['diverse_minus_top_j']) for p in pairs if p['diverse_minus_top_j']]
    incomparable = sum(not p['diverse_minus_top_j'] for p in pairs)
    held_out = manifest['plan'].get('study_schema') == 'connected_constraint_holdout_v1'
    initial_bad = [r for r in by_method['independent_transfer'] if r['feasible'] != 'True']
    lines = ['# Connection Constraint Findings', '',
        f"Conditions: {len(manifest['jobs'])}; saved deployments checked: {validation['checked_deployments']}; "
        f"feasible deployments: {validation['feasible_deployments']}.",
        f"Independent global/quota references checked: {validation['reference_optima_checked']}; "
        f"source-pool optima checked: {validation['source_pool_optima_checked']}.", '',
        '| Method | Feasible / conditions | Matches full-grid J | Coverage range % (feasible only) |',
        '|---|---:|---:|---:|']
    for value in summary:
        bounds = ('-' if value['min_coverage_pct'] is None else
                  f"{value['min_coverage_pct']:.3f}-{value['max_coverage_pct']:.3f}")
        lines.append(f"| {value['method']} | {value['feasible']}/{value['conditions']} | "
                     f"{value['grid_optimum_matches']} | {bounds} |")
    lines.extend(['',
        f"Matched source-pool comparison: diverse improves J in {sum(v > 1e-7 for v in differences)} conditions, "
        f"ties in {sum(abs(v) <= 1e-7 for v in differences)}, and is lower in {sum(v < -1e-7 for v in differences)}.",
        f"No paired feasible score exists in {incomparable} conditions; these are not ties or zero scores.",
        f"Original independent transfer violates constraints in {len(initial_bad)} conditions. "
        'Those poses are retained as infeasible measurements, never treated as zero-coverage successes.',
        'The installation exclusions are defined by road station and side, with inclusive interval boundaries.',
        'Overlap profiles require at least 95% interface coverage and 50% of its sampled targets seen by both sensors.',
        'All thresholds are illustrative. The two source rules query the same number of eligible poses per condition; '
        'submitted ray counts and elapsed time can differ.', '',
        'The diversity rule chooses the best source pitch per height/yaw mode, then uses normalized '
        'height/yaw max-min selection seeded by source J. No target optimum enters this selection.',
        ('The source rule and numeric requirements were fixed before these synthetic target queries. '
         f"The {len(manifest['plan'].get('targets', []))} geometry tuples were not used to develop that rule; "
         'widths, angles and bend lengths vary together, '
         'so their individual causal effects are not identified. This is not real-road generalization.' if held_out else
         'These geometries were used in development. They are not an untouched generalization set.'),
        'Exact J matches concern fixed K=2 in the declared finite grid and static sampled scene.',
        'No policy is trained here; MAPPO advantage, continuous-road optimality and cold-start speedup are untested.', '',
        f"The scene illustration uses profile {example['profile']}, angle {example['angle_deg']:g}deg, "
        'selected for showing the combined constraints. Aggregate conclusions use every condition.', ''])
    (output / 'REPORT.md').write_text('\n'.join(lines))
    provenance = dict(report_code_sha256=digest(__file__),
        input_hashes={str(p.relative_to(root)): digest(p) for p in inputs}, selected_scene=example,
        scope='Stored-result analysis only; no new optimization/raycast or changes to original JSON')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return output


def plot_overview(path, rows, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jobs = manifest['jobs']
    field = 'target_name' if 'target_name' in jobs[0] else 'angle_deg'
    lookup = {(r['profile'], str(r[field]), r['method']): r for r in rows}
    profile_map = {p['name']: p for p in manifest['plan']['profiles']}
    x = np.arange(len(jobs))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    metrics = [('objective_gap', 'J shortfall to grid optimum'), ('coverage_pct', 'Global coverage (%)'),
               ('interface_overlap_ratio', 'Interface double coverage (%)')]
    for axis, (metric, ylabel) in zip(axes, metrics):
        finite = []
        for method, offset, color, label in [('top_joint', -.18, '#178395', 'Source-score top 4'),
            ('diverse_joint', .18, '#ce6846', 'Height/yaw diverse 4')]:
            selected = [lookup[j['profile'], str(j[field]), method] for j in jobs]
            values = [float(r[metric]) if r['feasible'] == 'True' and r[metric] else np.nan for r in selected]
            if metric == 'interface_overlap_ratio':
                values = np.asarray(values)*100
            axis.bar(x+offset, values, width=.34, color=color, label=label)
            finite.extend(v for v in values if np.isfinite(v))
        if metric == 'coverage_pct' and finite:
            axis.set_ylim(max(0, min(finite)-.3), 100)
        elif metric == 'interface_overlap_ratio':
            minimum = [100*profile_map[j['profile']]['interface'].get('min_overlap_ratio', 0) for j in jobs]
            axis.scatter(x, minimum, marker='_', s=130, color='#242424', label='Required minimum', zorder=3)
            axis.set_ylim(0, 100)
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(loc='upper right', fontsize=9)
        for i in range(1, len(jobs)):
            if jobs[i]['profile'] != jobs[i-1]['profile']:
                axis.axvline(i-.5, color='#bbbbbb', linewidth=.7)
    labels = [f"{j['profile']}\n{j['angle_deg']:g} deg" +
              (f"\nw={j['road_width_m']:g}\nb={j['bend_length_m']:g}" if 'target_name' in j else '') for j in jobs]
    axes[-1].set_xticks(x, labels, fontsize=8)
    fig.suptitle('Matched source-pool size with connection redundancy and mounting exclusions')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scene(path, directory, job):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    geometry_path = directory / 'geometry.npz'
    with np.load(geometry_path, allow_pickle=False) as data:
        poses, targets, canonical = data['poses'], data['targets'], data['canonical_candidates']
        vertices, allowed = data['vertices'], data['allowed_candidates']
        _, sites = np.unique(poses[:, :2], axis=0, return_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    inputs = [geometry_path]
    for ax, method in zip(axes, ['independent_transfer', 'diverse_joint', 'reference_global']):
        result_path = directory / method / 'result.json'
        result = json.loads(result_path.read_text())
        inputs.append(result_path)
        ax.scatter(vertices[:, 0], vertices[:, 1], color='#bbbbbb', s=1)
        ax.scatter(poses[sites[~allowed[sites]], 0], poses[sites[~allowed[sites]], 1],
                   marker='x', s=22, color='#bd4560', label='Excluded mount')
        ax.scatter(poses[sites[allowed[sites]], 0], poses[sites[allowed[sites]], 1],
                   facecolors='none', edgecolors='#657e70', s=18, label='Eligible mount')
        if result['selected_ids'] is not None:
            archive = directory / method / 'deployment.npz'
            inputs.append(archive)
            with np.load(archive, allow_pickle=False) as data:
                density = data['density']
            visible = np.isclose(targets[:, 2], 1.5)
            ax.scatter(targets[visible, 0], targets[visible, 1], s=3,
                       c=np.where(density[visible] > 0, '#178395', '#ce6846'))
            selected = poses[result['selected_ids']]
            ax.scatter(selected[:, 0], selected[:, 1], marker='^', color='#202020', s=70, label='LiDAR')
            ax.set_title(f"{method}\nfeasible={result['assessment']['feasible']}, "
                         f"interface overlap={100*result['interface_overlap_ratio']:.1f}%", fontsize=10)
        else:
            ax.set_title(f"{method}\n{result['status']}", fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Profile {job['profile']}, {job['angle_deg']:g} deg; forbidden mounts remain forbidden after transfer")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return inputs


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(generate(args.root, args.output or args.root / 'analysis'))

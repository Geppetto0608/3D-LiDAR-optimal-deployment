"""Compare completed shape studies without training or optimization."""
from argparse import ArgumentParser
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


def collect(roots):
    rows, references, seen = [], {}, set()
    for root in roots:
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'Incomplete experiment: {root}')
        validation = json.loads((root / 'validation.json').read_text())
        if validation['status'] != 'passed':
            raise ValueError(f'Unvalidated experiment: {root}')
        k = manifest['arguments']['k']
        for job in manifest['jobs']:
            directory = root / job['key']
            record = json.loads((directory / 'result.json').read_text())
            if record['status'] != 'complete':
                raise ValueError(f'Incomplete result: {directory}')
            shape = record['target_scenario_id']
            geometry = json.loads((directory / 'geometry.json').read_text())
            config = yaml.safe_load((directory / 'config.yaml').read_text())
            sensor = (config['lidar_profile'], config['lidar_profiles'][config['lidar_profile']])
            with np.load(directory / 'coverage.npz', allow_pickle=False) as data:
                targets = data['targets']
                signature = (k, geometry['geometry'], geometry['target'], sensor, targets.shape,
                             hashlib.sha256(targets.tobytes()).hexdigest())
                coverage = 100 * float((data['density'] > 0).mean())
            if shape in references and references[shape] != signature:
                raise ValueError(f'Geometry/target/budget mismatch for shape {shape}')
            references[shape] = signature
            if not np.isclose(coverage, record['coverage_pct'], atol=1e-8, rtol=0):
                raise ValueError(f'Coverage metadata mismatch: {directory}')
            meta = record['method_metadata']
            method = record['method']
            if method == 'mappo':
                method += f"_{meta['training_steps']}"
            identity = (shape, method, record['seed'], record['repeat'])
            if identity in seen:
                raise ValueError(f'Duplicate comparison result: {identity}')
            seen.add(identity)
            solver = meta.get('solver') or {}
            rows.append(dict(shape_id=shape, shape=record['shape_name'], method=method, k=k,
                seed=record['seed'], repeat=record['repeat'], coverage_pct=coverage,
                final_policy_coverage_pct=meta.get('final_policy_coverage_pct'),
                online_wall_sec=record['runtime']['elapsed_sec'],
                solver_status=solver.get('solver_status'), objective=solver.get('objective'),
                best_bound=solver.get('best_bound'), mip_gap=solver.get('mip_gap'),
                solver_time_sec=solver.get('solve_time_sec'),
                candidate_count=meta.get('candidate_stats', {}).get('num_candidates'),
                result_json=str(directory / 'result.json'), view=str(directory / 'deployment.html')))
    return rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.roots)
    expected = {'select', 'local_search', 'mappo_256', 'mappo_4096', 'greedy', 'milp'}
    shapes = sorted({r['shape_id'] for r in rows})
    if len(shapes) != 4:
        raise ValueError('This report expects the four-map development suite')
    for sid in shapes:
        subset = [r for r in rows if r['shape_id'] == sid]
        if len(subset) != len(expected) or {r['method'] for r in subset} != expected:
            raise ValueError(f'Incomplete method set for {sid}')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'method_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = ['select', 'local_search', 'mappo_256', 'mappo_4096', 'greedy', 'milp']
    labels = ['Transfer selection', 'Local search (128 evals)', 'MAPPO (256 steps)',
              'MAPPO (4096 steps)', 'GREEDY (top-10)', 'MILP (top-10, 300s)']
    colors = ['#666666', '#008a88', '#bd88b5', '#9b296a', '#3679b4', '#b77718']
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout='constrained')
    x = np.arange(len(shapes))
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = [next(r for r in rows if r['shape_id'] == sid and r['method'] == method)
                  for sid in shapes]
        for ax, field in zip(axes, ['coverage_pct', 'online_wall_sec']):
            ax.bar(x + (i - 2.5) * .13, [r[field] for r in values], width=.12,
                   label=label, color=color)
    axes[0].set(ylim=(0, 105), ylabel='Coverage (%)',
                title='Best retained deployments: different search budgets and feasible spaces')
    axes[0].legend(ncol=3, fontsize=9, loc='lower left')
    axes[1].set(yscale='log', ylabel='Online wall time (s, log scale)',
                title='Includes preprocessing; excludes worker startup and final figure export')
    axes[1].set_xticks(x, [next(r['shape'] for r in rows if r['shape_id'] == sid) for sid in shapes])
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / f'method_comparison.{suffix}', dpi=180)
    plt.close(fig)
    body = []
    for r in sorted(rows, key=lambda r: (r['shape_id'], methods.index(r['method']))):
        view = html.escape(os.path.relpath(r['view'], args.output), quote=True)
        body.append(f'<tr><td>{html.escape(r["shape"])}</td><td><a href="{view}">{r["method"]}</a></td>'
            f'<td>{r["coverage_pct"]:.3f}</td><td>{r["online_wall_sec"]:.2f}</td>'
            f'<td>{r["solver_status"] or ""}</td><td>{r["mip_gap"] if r["mip_gap"] is not None else ""}</td></tr>')
    (args.output / 'method_comparison.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Shape baseline comparison</title><style>body{font-family:sans-serif;margin:24px}'
        'td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style>'
        '<h1>Shape baseline comparison</h1><p>Development maps, K=3, MAPPO seed=0. '
        'Best retained deployment is not final policy performance. Top-10 MILP is not a global '
        'upper bound for MAPPO. Times are single measurements with unequal budgets, not a '
        'controlled speedup claim.</p><p><a href="method_comparison.csv">CSV</a> | '
        '<a href="method_comparison.pdf">PDF</a></p>'
        '<img src="method_comparison.png" alt="Coverage and online runtime" style="max-width:100%">'
        '<div style="overflow-x:auto"><table><tr><th>Map</th><th>Method / 3D</th>'
        '<th>Coverage (%)</th><th>Online seconds</th><th>Solver status</th><th>Relative gap</th></tr>'
        + ''.join(body) + '</table></div></html>', encoding='utf-8')
    (args.output / 'comparison_provenance.json').write_text(json.dumps(dict(
        command=[sys.executable, *sys.argv], report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for r in rows for p in [Path(r['result_json'])]}, rows=len(rows)), indent=2))
    print(f'{len(rows)} checked records exported to {args.output}')


if __name__ == '__main__':
    main()

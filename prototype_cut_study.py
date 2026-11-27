"""Sequential archived-candidate cut-MILP diagnostics, with a frozen source copy."""
from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prototype_offline import digest, save


def summarize(root, jobs):
    rows = []
    for job in jobs:
        if job['status'] != 'complete':
            continue
        r = json.loads((Path(job['output'])/'result.json').read_text())
        rows.append(dict(scenario_id=job['scenario_id'], K=job['k'],
            previous_coverage_pct=r['prior_coverage_pct'], coverage_pct=r['coverage_pct'],
            objective=r['objective'], upper_bound=r['certified_upper_bound'],
            relative_gap=r['certified_gap'], discrete_optimal=r['discrete_optimal'],
            solver_bound=r['solver_best_bound'], bound_source=r['bound_source'],
            solver_status=r['solver_status'], swap_objective=r['swap']['coverage_count'],
            swap_sec=r['swap']['elapsed_sec'], solver_sec=r['solver_sec'],
            archive_load_sec=r['archive_load_sec'], total_optimization_sec=r['total_sec'],
            cuts=r['cuts'], result_json=str(Path(job['output'])/'result.json')))
    if rows:
        with (root/'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save(root/'summary.json', dict(completed=len(rows), expected=len(jobs), results=rows,
        note='Warm archive solver diagnosis, not a cold-map runtime benchmark. '
             'Domain remains the existing discrete candidate archive; no online pose warm start.'))


def run(args):
    baseline = Path(args.baselines).resolve()
    root = Path(args.output).resolve()
    jobs = [dict(scenario_id=sid, k=k, baseline=str(baseline/f's{sid}'),
        output=str(root/f's{sid}_k{k}'), status='pending')
        for sid in (420001, 420002, 420003, 420004) for k in (3, 4)]
    for job in jobs:
        previous = json.loads((Path(job['baseline'])/'manifest.json').read_text())
        if previous['status'] != 'complete' or job['k'] not in previous['k_values']:
            raise ValueError('Incomplete baseline archive')
        if not (Path(job['baseline'])/'candidates.npz').is_file():
            raise ValueError('Missing candidate archive')
    if args.reuse:
        reuse = Path(args.reuse).resolve()
        m = json.loads((reuse/'manifest.json').read_text())
        r = json.loads((reuse/'result.json').read_text())
        if m['status'] != 'complete' or not r['validated_direct_raycast']:
            raise ValueError('Reuse requires a completed, validated diagnostic')
        if r['parameters']['seconds'] != 300 or r['parameters']['swap_seconds'] != 30:
            raise ValueError('Reused run has different solver budgets')
        for name in ('src/methods/coverage_cut_milp.py', 'src/environment.py'):
            if digest(name) != m['source_hashes'][name]:
                raise ValueError('Reused diagnostic has a different implementation')
        match = next(j for j in jobs if j['scenario_id']==m['scenario_id'] and j['k']==m['k'])
        if Path(m['baseline']) != Path(match['baseline']) or digest(Path(match['baseline'])/'candidates.npz') != m['archive_sha256']:
            raise ValueError('Reused diagnostic is from a different candidate archive')
        match.update(status='complete', output=str(reuse), reused=True)
    print(json.dumps(jobs, indent=2), flush=True)
    if not args.execute:
        return
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    paths = [*Path('.').glob('prototype_*.py'), *Path('src').rglob('*.py')]
    for path in paths:
        dest = root/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    hashes = {str(p.relative_to(root)):digest(p) for p in (root/'source').rglob('*.py')}
    manifest = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv], jobs=jobs, source_hashes=hashes)
    save(root/'manifest.json', manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        for job in jobs:
            if job['status'] == 'complete':
                continue
            for path, value in hashes.items():
                if digest(root/path) != value:
                    raise RuntimeError(f'Frozen source changed: {path}')
            job['status']='running'
            save(root/'manifest.json', manifest)
            cmd=[sys.executable,'-u','prototype_milp_diagnostic.py','--baseline',job['baseline'],
                 '--output',job['output'],'--k',str(job['k']),'--seconds','300','--swap-seconds','30']
            subprocess.run(cmd, cwd=root/'source', env=env, check=True)
            print(f'Start {job["scenario_id"]} K={job["k"]}', flush=True)
            with (root/'logs'/f's{job["scenario_id"]}_k{job["k"]}.log').open('w') as log:
                subprocess.run(cmd+['--execute'], cwd=root/'source', env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            job['status']='complete'
            save(root/'manifest.json',manifest)
            summarize(root,jobs)
        manifest.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        for job in jobs:
            if job['status']=='running': job['status']='failed'
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        save(root/'manifest.json',manifest)
        summarize(root,jobs)


if __name__ == '__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--baselines',default='results/prototype_target_full_20261012/baselines')
    p.add_argument('--output',required=True)
    p.add_argument('--reuse')
    p.add_argument('--execute',action='store_true')
    run(p.parse_args())

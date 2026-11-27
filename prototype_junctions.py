"""Sequential junction width pilot, K=3; no MAPPO training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from prototype_offline import digest, save


def report(root, transfer_s1='transfer_s1', transfer_s2='transfer_s2'):
    """Summarize existing artifacts without running any solver or training."""
    summary = dict(training=False, budget_k=3, sources={}, offline=[], transfers=[])
    lines = ['# Junction Width Pilot (K=3)', '',
        'Synthetic width changes of parametric scenarios 1 and 2, not reconstructed OSM maps.',
        'No MAPPO training, held-out MILP reference, or near-optimality claim.', '',
        '## Offline Library', '',
        '| Scenario | Width (m) | Coverage (%) | Certification | Solver status |',
        '|---|---:|---:|---|---|']
    for sid in (1,2):
        for width in (7,14):
            path = root/f's{sid}_w{width}'/'k3/result.json'
            if not path.exists():
                lines.append(f'| {sid} | {width} | TBD | missing | missing |')
                continue
            item = json.loads(path.read_text())
            summary['sources'][str(path.relative_to(root))] = digest(path)
            summary['offline'].append(dict(scenario_id=sid,width_m=width,result=item))
            lines.append(f"| {sid} | {width} | {item['coverage_pct']:.6f} | {item['certification']} | {item['solver']['solver_status']} |")
    lines += ['', '## Held-out 10 m Map', '',
        '| Scenario | Method | Coverage (%) | 3D |', '|---|---|---:|---|']
    for sid, name in ((1,transfer_s1),(2,transfer_s2)):
        directory = (root/name).resolve()
        relative = directory.relative_to(root.resolve())
        path = directory/'report.json'
        if not path.exists():
            lines.append(f'| {sid} | TBD | TBD | missing report |')
            continue
        item = json.loads(path.read_text())
        summary['sources'][str(path.relative_to(root.resolve()))] = digest(path)
        summary['transfers'].append(dict(scenario_id=sid,report=str(relative/'report.json'),data=item))
        if item['interpolation'].get('status') == 'unavailable':
            lines.append(f'| {sid} | interpolation unavailable: incompatible installation lines | N/A | N/A |')
        for method, result in item['results'].items():
            lines.append(f"| {sid} | {method} | {result['coverage_pct']:.6f} | [Open]({relative}/{method}.html) |")
    lines += ['', 'Original task exit codes remain in `manifest.json`; unavailable interpolation is not a zero score.',
        'A `partial` transfer report preserves valid single-source controls but is not successful interpolation.',
        'Offline 300 s is the solver limit, not total wall time; interrupted presolve may overrun it.',
        'Exact commands and source snapshots are recorded in the referenced manifests/reports.', '']
    save(root/'summary.json',summary)
    (root/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--transfer-s1', default='transfer_s1')
    parser.add_argument('--transfer-s2', default='transfer_s2')
    args = parser.parse_args()
    root = Path(args.output)
    if args.report_only:
        if args.execute:
            parser.error('--report-only cannot be combined with --execute')
        report(root,args.transfer_s1,args.transfer_s2)
        return
    tasks = []
    for sid in (1,2):
        for width in (7,14):
            tasks.append((f's{sid}_w{width}', [sys.executable,'-u','prototype_offline.py',
                '--scenario-id',str(sid),'--road-width',str(width),'--k-values','3',
                '--output',str(root/f's{sid}_w{width}'),'--max-edges','2000000000',
                '--time-limit','300']))
        tasks.append((f'transfer_s{sid}',[sys.executable,'-u','prototype_transfer.py',
            '--low',str(root/f's{sid}_w7'),'--high',str(root/f's{sid}_w14'),
            '--width','10','--k','3','--output',str(root/f'transfer_s{sid}')]))
    if not args.execute:
        for label, cmd in tasks:
            print(label, ' '.join(cmd))
            if 'prototype_offline.py' in cmd:
                subprocess.run(cmd,check=True)
        return
    root.mkdir(parents=True,exist_ok=False)
    manifest = dict(status='running',tasks=[], training=False)
    save(root/'manifest.json',manifest)
    for label, cmd in tasks:
        command = [*cmd,'--execute']
        print('START',label,flush=True)
        with (root/f'{label}.log').open('w') as log:
            code = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        task = dict(name=label,command=command,returncode=code)
        if label.startswith('transfer') and (root/label/'report.json').exists():
            task['report_status'] = json.loads((root/label/'report.json').read_text())['status']
        manifest['tasks'].append(task)
        save(root/'manifest.json',manifest)
        print('END',label,code,flush=True)
        if code and not label.startswith('transfer'):
            manifest['status']='failed';save(root/'manifest.json',manifest)
            raise SystemExit(code)
    manifest['status']='complete' if all(t['returncode']==0 and t.get('report_status','complete')=='complete'
        for t in manifest['tasks']) else 'partial'
    save(root/'manifest.json',manifest)
    report(root)


if __name__ == '__main__':
    main()

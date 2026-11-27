"""Small A/B new-target evaluation, using immutable new-format source libraries.

No execution without --execute. Output must be new. Each run snapshots source
library inputs and implementation before launching a sequential frozen worker.
"""
from argparse import ArgumentParser, SUPPRESS
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import csv
import json
import shutil
import subprocess
import sys
import time

import numpy as np

from prototype_offline import digest, save
from src.core.library_transfer_evaluation import evaluate_problem, plan_problem


BASE = Path(__file__).resolve().parent
SCHEMA = 'library_target_run_v1'
ARTIFACTS = ('report.json', 'comparison.csv', 'target.npz', 'mesh.npz')


def prepare(track, source, target, k_values, occlusion=None, installation=None):
    if track == 'A':
        from library_topology_transfer import prepare_problem
    elif track == 'B':
        from library_component_transfer import prepare_problem
    else:
        raise ValueError('Expected track A or B')
    problem = prepare_problem(source, target, k_values)
    if installation is not None:
        from src.core.library_installation import with_installation
        problem = with_installation(problem, installation)
    if occlusion is not None:
        from src.core.scene_occlusion import with_occlusion
        problem = with_occlusion(problem, occlusion)
    return problem


def source_names():
    return ['library_transfer_study.py', 'library_topology_transfer.py', 'library_component_transfer.py',
            'library_topology.py', 'library_components.py', 'prototype_offline.py',
            'prototype_transfer.py', 'prototype_segments.py',
            *sorted(str(p.relative_to(BASE)) for p in (BASE / 'src').rglob('*.py'))]


def _file_hashes(directory):
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError('Expected a physical source directory')
    files = sorted(p for p in directory.rglob('*') if p.is_file() or p.is_symlink())
    if any(p.is_symlink() for p in files):
        raise ValueError('Source snapshots may not contain symlinks')
    return {str(p.relative_to(directory)): digest(p) for p in files}


def _verify(directory, hashes):
    directory = Path(directory).resolve()
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid snapshot path')
        path = directory / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f'Frozen input/source changed: {name}')


def _worker(root):
    root = Path(root).resolve()
    if BASE != root / 'source':
        raise ValueError('Worker must execute frozen source')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'prepared':
        raise ValueError('Worker requires a newly prepared run')
    if any((root / name).exists() for name in ARTIFACTS):
        raise FileExistsError('Refusing existing target artifacts')
    _verify(root, manifest['input_hashes'])
    _verify(BASE, manifest['source_hashes'])
    _verify(root / 'inputs/library', manifest['library_hashes'])
    started = time.perf_counter()
    try:
        manifest['status'] = 'running'
        save(root / 'manifest.json', manifest)
        settings = json.loads((root / 'inputs/settings.json').read_text())
        target = json.loads((root / 'inputs/target.json').read_text())
        t0 = time.perf_counter()
        occlusion = (json.loads((root / 'inputs/occlusion.json').read_text())
                     if settings.get('occlusion') else None)
        installation = (json.loads((root / 'inputs/installation.json').read_text())
                        if settings.get('installation') else None)
        problem = prepare(settings['track'], root / 'inputs/library', target, settings['k_values'],
                          occlusion, installation)
        prepare_sec = time.perf_counter() - t0
        if settings.get('control_seeds') is None:
            result = evaluate_problem(problem, settings['enumeration_limit'])
        else:
            from src.core.library_control_evaluation import evaluate_matched_controls
            result = evaluate_matched_controls(problem, settings['control_seeds'], settings['enumeration_limit'])
        if installation is not None:
            from src.core.library_installation import validate_result
            validate_result(result, installation)
        result['report']['phase_wall_sec']['source_load_and_geometry_transfer'] = prepare_sec
        result['report']['plan']['execute'] = True
        _verify(root / 'inputs/library', manifest['library_hashes'])
        _verify(BASE, manifest['source_hashes'])
        _verify(root, manifest['input_hashes'])
        np.savez_compressed(root / 'target.npz', **result['archive'])
        np.savez_compressed(root / 'mesh.npz', **result['mesh'])
        save(root / 'report.json', result['report'])
        rows = result['report']['comparison']
        with (root / 'comparison.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest.update(status='complete', artifact_hashes={name: digest(root / name) for name in ARTIFACTS},
            packages={name: metadata.version(name) for name in ('numpy', 'open3d', 'PyYAML')},
            original_source_certificates_reused=False, source_snapshot_unchanged=True)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest.update(worker_work_sec=time.perf_counter() - started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        save(root / 'manifest.json', manifest)


def execute(track, source, target, k_values, output, *, enumeration_limit=100000, command=None,
            control_seeds=None, occlusion=None, installation=None):
    output = Path(output).expanduser().absolute()
    source = Path(source).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output must be new: {output}')
    if source in output.resolve().parents:
        raise ValueError('Output cannot be inside the source library')
    before = _file_hashes(source)
    problem = prepare(track, source, target, k_values, occlusion, installation)
    if control_seeds is None:
        plan = plan_problem(problem, enumeration_limit)
    else:
        from src.core.library_control_evaluation import plan_matched_controls
        plan = plan_matched_controls(problem, control_seeds, enumeration_limit)
    plan['execute'] = True
    names = source_names()
    contents = {name: (BASE / name).read_bytes() for name in names}
    protocol = (BASE / 'LIBRARY_QUERY_CONTROLS.md').read_bytes() if control_seeds is not None else None
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    (output / 'inputs').mkdir()
    shutil.copytree(source, output / 'inputs/library')
    save(output / 'inputs/target.json', target)
    settings = dict(track=track, k_values=k_values, enumeration_limit=enumeration_limit)
    input_names = ['inputs/target.json', 'inputs/settings.json']
    if installation is not None:
        save(output / 'inputs/installation.json', installation)
        settings['installation'] = True
        input_names.append('inputs/installation.json')
    if occlusion is not None:
        save(output / 'inputs/occlusion.json', occlusion)
        settings['occlusion'] = True
        input_names.append('inputs/occlusion.json')
    if control_seeds is not None:
        settings['control_seeds'] = list(control_seeds)
    save(output / 'inputs/settings.json', settings)
    if protocol is not None:
        (output / 'inputs/protocol.md').write_bytes(protocol)
        input_names.append('inputs/protocol.md')
    for name, content in contents.items():
        path = output / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = dict(schema=SCHEMA, status='prepared', track=track, plan=plan,
        original_library=str(source), original_library_hashes=before, library_hashes=before,
        created_at=datetime.now(timezone.utc).isoformat(), command=command, python=sys.version,
        input_hashes={name: digest(output / name) for name in input_names},
        source_hashes={name: digest(output / 'source' / name) for name in names})
    save(output / 'manifest.json', manifest)
    started = time.perf_counter()
    try:
        _verify(output / 'inputs/library', before)
        with (output / 'worker.log').open('x') as log:
            completed = subprocess.run([sys.executable, '-B', str(output / 'source/library_transfer_study.py'),
                '--worker-root', str(output), '--execute'], cwd=output / 'source', stdout=log,
                stderr=subprocess.STDOUT, **({'timeout': 600} if control_seeds is not None else {}))
        manifest = json.loads((output / 'manifest.json').read_text())
        if completed.returncode or manifest['status'] != 'complete':
            raise RuntimeError(f'Target worker failed: {output / "worker.log"}')
        _verify(output / 'source', manifest['source_hashes'])
        _verify(output, manifest['input_hashes'])
        _verify(output / 'inputs/library', manifest['library_hashes'])
        if _file_hashes(source) != before:
            raise RuntimeError('Original source library changed during target evaluation')
        manifest['original_source_unchanged'] = True
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['worker_process_wall_sec'] = time.perf_counter() - started
        save(output / 'manifest.json', manifest)
    return output


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=['A', 'B'])
    parser.add_argument('--source')
    parser.add_argument('--target-json')
    parser.add_argument('--occlusion-json', help='Opt-in versioned opaque boxes; no domain changes')
    parser.add_argument('--installation-json', help='Opt-in closed XY forbidden regions; no mesh changes')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--enumeration-limit', type=int, default=100000)
    parser.add_argument('--control-seeds', type=int, nargs='+',
                        help='Opt-in matched-query controls with these random seeds; no controls by default')
    parser.add_argument('--output', default='results/library_target_smoke')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-root', help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return _worker(args.worker_root)
    if not args.track or not args.source or not args.target_json:
        parser.error('--track, --source and --target-json are required')
    target = json.loads(Path(args.target_json).read_text())
    occlusion = json.loads(Path(args.occlusion_json).read_text()) if args.occlusion_json else None
    installation = json.loads(Path(args.installation_json).read_text()) if args.installation_json else None
    problem = prepare(args.track, args.source, target, args.k_values, occlusion, installation)
    if args.control_seeds is None:
        plan = plan_problem(problem, args.enumeration_limit)
    else:
        from src.core.library_control_evaluation import plan_matched_controls
        plan = plan_matched_controls(problem, args.control_seeds, args.enumeration_limit)
    plan['execute'] = args.execute
    print(json.dumps(plan, indent=2, allow_nan=False))
    if not args.execute:
        return plan
    root = execute(args.track, args.source, target, args.k_values, args.output,
        enumeration_limit=args.enumeration_limit, control_seeds=args.control_seeds, occlusion=occlusion,
        installation=installation,
        command=[sys.executable, str(BASE / 'library_transfer_study.py'), *(sys.argv[1:] if argv is None else argv)])
    print(f'Completed: {root / "report.json"}')
    return root


if __name__ == '__main__':
    main()

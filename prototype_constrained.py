"""Opt-in coverage/cost/constraint refinement. Default: read-only plan, not training."""
from argparse import ArgumentParser, SUPPRESS
from copy import deepcopy
from datetime import datetime, timezone
import csv
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
from prototype_offline import digest, save
from prototype_refinement_study import make_initial, pose_hash
from prototype_transfer import load_prototype
from src.core.coverage_requirements import CoverageRequirements
from src.core.runtime import RuntimeMeter
from src.methods.constrained_refinement import InitialInfeasible, refine_constrained
from src.methods.feasibility_repair import repair_initial_placement


def now():
    return datetime.now(timezone.utc).isoformat()


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--library', default='results/prototype_library_shapes_20261008')
    p.add_argument('--suite', default='configs/library_expansion.json')
    p.add_argument('--target-id', type=int, default=420002)
    p.add_argument('--initialization', choices=('transfer_low', 'transfer_high', 'interpolation'), default='transfer_low')
    p.add_argument('--requirements', default='configs/coverage_constraints_example.json')
    p.add_argument('--methods', nargs='+', choices=('local_search', 'mappo'), default=['local_search', 'mappo'])
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--evaluations', type=int, default=2000)
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--radius-m', type=float, default=10.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--repair-initial', action='store_true')
    p.add_argument('--repair-evaluations', type=int, default=1000)
    p.add_argument('--repair-seconds', type=float, default=30.)
    p.add_argument('--output', default='results/constrained_coverage_demo')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker-root', help=SUPPRESS)
    p.add_argument('--worker-method', choices=('repair', 'local_search', 'mappo'), help=SUPPRESS)
    return p


def make_plan(args, *, suite_data=None, requirements_spec=None):
    requirements = CoverageRequirements(requirements_spec if requirements_spec is not None else
                                        json.loads(Path(args.requirements).read_text()))
    suite = load_suite(suite_data if suite_data is not None else json.loads(Path(args.suite).read_text()))
    if (args.steps < 1 or args.evaluations < 1 or not math.isfinite(args.seconds) or args.seconds <= 0
            or not math.isfinite(args.radius_m) or args.radius_m <= 0 or args.seed < 0
            or args.seed >= 2 ** 32 or len(args.methods) != len(set(args.methods))
            or args.repair_evaluations < 1 or not math.isfinite(args.repair_seconds) or args.repair_seconds <= 0):
        raise ValueError('Positive finite limits, unique methods and seed in [0, 2**32) required')
    matches = [(f, t) for f in suite['families'] for t in f['targets'] if t['id'] == args.target_id]
    if len(matches) != 1:
        raise ValueError('Target ID must occur exactly once in the geometry-only suite')
    family, target = matches[0]
    roots = [str(Path(args.library).resolve() / 'library' / f's{row["id"]}') for row in family['sources']]
    return dict(schema='constrained_refinement_run_v1', status='planned',
        job=dict(scenario=target, source_roots=roots, source_scenarios=family['sources'],
                 initialization=args.initialization, budget_k=requirements.specification['budget_k']),
        requirements=requirements.specification, methods=args.methods,
        limits=dict(steps=args.steps, max_evaluations=args.evaluations, time_limit_sec=args.seconds,
                    radius_m=args.radius_m, seed=args.seed),
        initial_repair=dict(enabled=args.repair_initial, max_evaluations=args.repair_evaluations,
                            time_limit_sec=args.repair_seconds),
        target_full_cache=False, pretrained_policy=False, detection=False, sequential_cpu=True,
        interpretation='Illustrative requirements, not deployment standards or a performance benchmark')


def freeze_inputs(root, plan):
    repo = Path(__file__).resolve().parent
    paths = sorted(repo.glob('*.py')) + sorted((repo / 'src').rglob('*.py'))
    paths += [repo / 'configs/experiment.yaml', repo / 'data/scenarios/yeongjong_final.json']
    for path in paths:
        destination = root / 'source' / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    save(root / 'requirements.json', plan['requirements'])
    save(root / 'settings.json', dict(limits=plan['limits'], methods=plan['methods'],
                                    initial_repair=plan['initial_repair']))
    frozen_job = deepcopy(plan['job'])
    indices = [0, 1] if frozen_job['initialization'] == 'interpolation' else [int(frozen_job['initialization'] == 'transfer_high')]
    originals = []
    for index in indices:
        original = Path(frozen_job['source_roots'][index])
        k = frozen_job['budget_k']
        proto = load_prototype(original, k)
        scenario = proto[3].scenario
        expected = frozen_job['source_scenarios'][index]
        if (scenario.id != expected['id'] or scenario.topology != expected['topology']
                or scenario.road_width != expected['road_width'] or scenario.num_lanes != expected['num_lanes']
                or scenario.raw.get('geometry') != expected['geometry']):
            raise ValueError('Source geometry differs from the declared suite')
        pose_path = Path(proto[1]['positions_npy'])
        if pose_path.is_absolute() or '..' in pose_path.parts:
            raise ValueError('Source positions_npy must be relative to its prototype directory')
        frozen = root / 'inputs' / f's{scenario.id}'
        files = ['config.yaml', 'geometry.json', f'k{k}/result.json', str(pose_path)]
        hashes = {}
        for name in dict.fromkeys(files):
            destination = frozen / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / name, destination)
            hashes[name] = digest(destination)
        originals.append(dict(root=str(original), hashes=hashes, certification=proto[1].get('certification')))
        frozen_job['source_roots'][index] = str(frozen.relative_to(root))
    # An unused single-transfer source must not be accessed by a worker.
    for index in set(range(2)) - set(indices):
        frozen_job['source_roots'][index] = 'unused_source'
    save(root / 'job.json', frozen_job)
    plan['source_prototypes'] = originals
    plan['frozen_hashes'] = {str(path.relative_to(root)): digest(path)
        for directory in ('inputs', 'source') for path in (root / directory).rglob('*') if path.is_file()}
    for name in ('requirements.json', 'job.json', 'settings.json'):
        plan['frozen_hashes'][name] = digest(root / name)


def worker(root, method):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Worker requires an explicitly CPU-only process')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['frozen_hashes'].items():
        if digest(root / name) != expected:
            raise RuntimeError(f'Frozen input changed: {name}')
    settings = json.loads((root / 'settings.json').read_text())
    repair_settings = settings['initial_repair']
    if method == 'repair' and not repair_settings['enabled']:
        raise ValueError('Repair absent from frozen plan')
    if method != 'repair' and method not in settings['methods']:
        raise ValueError('Method absent from frozen plan')
    job = json.loads((root / 'job.json').read_text())
    job['source_roots'] = [str(root / path) for path in job['source_roots']]
    spec = json.loads((root / 'requirements.json').read_text())
    out = root / method
    meter = RuntimeMeter()
    with meter.stage('source_read_geometry_transfer'):
        transferred, geometry, config = make_initial(job)
    if method == 'repair':
        with meter.stage('shared_initial_repair_validation_save'):
            poses, result = repair_initial_placement(transferred, geometry.scenario, config, spec,
                output=out, work=meter, radius_m=settings['limits']['radius_m'],
                max_evaluations=repair_settings['max_evaluations'],
                time_limit_sec=repair_settings['time_limit_sec'])
        result.update(schema='constrained_initial_repair_v1', runtime=meter.report(),
            transferred_pose_sha256=pose_hash(transferred), finished_at=now(),
            requirements_sha256=digest(root / 'requirements.json'),
            feasible_pose_sha256=pose_hash(poses) if poses is not None else None)
        save(out / 'config.json', config.raw)
        save(out / 'result.json', result)
        print(f'initial repair: {result["status"]}', flush=True)
        return 0 if poses is not None else 2
    initial, anchors = transferred, None
    if repair_settings['enabled']:
        report = json.loads((root / 'repair/result.json').read_text())
        initial = np.load(root / 'repair/feasible.npy', allow_pickle=False)
        if (report['status'] not in ('repaired', 'already_feasible')
                or report['transferred_pose_sha256'] != pose_hash(transferred)
                or report['feasible_pose_sha256'] != pose_hash(initial)
                or report['requirements_sha256'] != digest(root / 'requirements.json')):
            raise RuntimeError('Shared repair does not match this transfer or requirements')
        anchors = transferred
    try:
        with meter.stage('refinement_validation_save'):
            _, result = refine_constrained(initial, geometry.scenario, config, spec,
                method=method, output=out, work=meter, anchor_poses=anchors, **settings['limits'])
    except InitialInfeasible as error:
        save(out / 'result.json', dict(status='initial_infeasible', assessment=error.assessment.to_dict(),
                                      initial_pose_sha256=pose_hash(initial)))
        return 2
    result.update(schema='constrained_refinement_result_v1', initial_pose_sha256=pose_hash(initial),
                  transferred_pose_sha256=pose_hash(transferred), shared_initial_repair=repair_settings['enabled'],
                  repair_worker_wall_sec=manifest.get('initial_repair', {}).get('worker_wall_sec', 0.),
                  requirements_sha256=digest(root / 'requirements.json'), runtime=meter.report(),
                  finished_at=now())
    save(out / 'config.json', config.raw)
    save(out / 'result.json', result)
    print(f'{method}: feasible J={result["assessments"]["deployment"]["objective_score"]:.3f}', flush=True)
    return 0


def launch_worker(root, env, method):
    command = [sys.executable, '-u', str(root / 'source/prototype_constrained.py'),
        '--worker-root', str(root), '--worker-method', method, '--execute']
    started = time.perf_counter()
    with (root / 'logs' / f'{method}.log').open('w') as log:
        process = subprocess.run(command, cwd=root / 'source', env=env, stdout=log, stderr=subprocess.STDOUT)
    path = root / method / 'result.json'
    result = json.loads(path.read_text()) if path.exists() else dict(status='failed')
    return result, dict(status=result['status'], returncode=process.returncode,
                        worker_wall_sec=time.perf_counter() - started)


def run(args):
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan.update(status='preparing', started_at=now(), command=sys.argv, jobs={})
    save(root / 'manifest.json', plan)
    try:
        freeze_inputs(root, plan)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        (root / 'packages.txt').write_text(packages.stdout)
        save(root / 'hardware.json', dict(platform=platform.platform(), machine=platform.machine(),
            python=sys.version, cpu_count=os.cpu_count(), thread_environment={k: env[k] for k in
                ('CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
        (root / 'logs').mkdir()
        plan['status'] = 'running'
        if plan['initial_repair']['enabled']:
            save(root / 'manifest.json', plan)
            result, runtime = launch_worker(root, env, 'repair')
            plan['initial_repair'].update(runtime)
            if runtime['returncode'] == 2 and result['status'] in ('no_feasible_found', 'infeasible_fixed_count'):
                plan['status'] = 'no_feasible_initial'
                plan['jobs'] = {name: dict(status='not_run', reason=result['status']) for name in args.methods}
                print(f'Initial repair: {result["status"]}; LS/MAPPO not run', flush=True)
                return 2
            if (runtime['returncode'] or result['status'] not in ('repaired', 'already_feasible')
                    or not result.get('validated_direct_raycast')
                    or not result['assessments']['best_candidate']['feasible']):
                raise RuntimeError(f'Initial repair failed; see {root / "logs/repair.log"}')
            for name in ('feasible.npy', 'initial.npy', 'anchors.npy', 'result.json'):
                plan['frozen_hashes'][f'repair/{name}'] = digest(root / 'repair' / name)
            print(f'Initial repair: {result["status"]}; common start frozen', flush=True)
        initial_hash = None
        rows = []
        for method in args.methods:
            save(root / 'manifest.json', plan)
            result, runtime = launch_worker(root, env, method)
            wall = runtime['worker_wall_sec']
            result_path = root / method / 'result.json'
            plan['jobs'][method] = runtime
            if runtime['returncode'] == 2 and result['status'] == 'initial_infeasible':
                print(f'{method}: initial_infeasible (see {result_path})', flush=True)
                continue
            if runtime['returncode'] or result['status'] != 'complete':
                raise RuntimeError(f'{method} failed; see {root / "logs" / (method + ".log")}')
            if initial_hash is not None and initial_hash != result['initial_pose_sha256']:
                raise RuntimeError('Methods received different initial deployments')
            initial_hash = result['initial_pose_sha256']
            a = result['assessments']['deployment']
            rows.append(dict(method=method, objective_score=a['objective_score'], coverage_pct=a['coverage_pct'],
                overlap_ratio=a['overlap_ratio'], installation_cost=a['components']['installation_cost'],
                feasible=a['feasible'], optimization_sec=result['optimization_sec'], worker_wall_sec=wall,
                repair_worker_wall_sec=result.get('repair_worker_wall_sec', 0.)))
            print(f'{method}: complete; see {result_path}', flush=True)
        if rows:
            with (root / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        plan['status'] = 'complete' if len(rows) == len(args.methods) else 'initial_infeasible'
    except BaseException as error:
        plan.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        plan['finished_at'] = now()
        save(root / 'manifest.json', plan)
    return 0 if plan['status'] == 'complete' else 2


if __name__ == '__main__':
    args = parser().parse_args()
    if args.worker_root is not None:
        if not args.execute or args.worker_method is None:
            raise SystemExit('Worker requires --execute and --worker-method')
        raise SystemExit(worker(Path(args.worker_root).resolve(), args.worker_method))
    raise SystemExit(run(args))

"""Source-free geometry placement and fresh-process comparison, fixed K=2.

Default: adaptive geometry, four queries/site, one repetition, no source assets
or full-reference generation. No-write plan unless --execute is provided.
"""
import time

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys

import numpy as np

import connected_constraint_study as constraints
import connected_corridor_study as corridor
from src.environment import get_lidar_profile, load_config
from src.methods.connected_hierarchy import BudgetedObservations, SelectionContext, select_pool, upper_select
from src.methods.geometry_controls import CONTROL_METHODS, select_geometry_control
from src.roadside.corridor_requirements import CorridorRequirements

BASE = Path(__file__).resolve().parent
METHODS = ('adaptive', 'adaptive_batch', 'static', 'random', 'full', *CONTROL_METHODS)
SOURCE_FILES = list(dict.fromkeys(constraints.SOURCE_FILES + [
    'geometry_placement.py', 'src/methods/connected_hierarchy.py',
    'tests/validate_geometry_placement.py', 'tests/test_geometry_placement.py',
    'tests/validate_connected_priors.py', 'connected_prior_ablation.py', 'connected_hierarchy_study.py',
    'connected_validation_audit.py', 'src/methods/connected_diagnostics.py',
    'tests/validate_connected_hierarchy.py', 'tests/test_connected_hierarchy.py',
    'tests/test_conditional_batch.py',
    'geometry_budget_report.py', 'tests/test_geometry_budget.py',
    'src/methods/geometry_controls.py', 'geometry_mechanism_report.py', 'tests/test_geometry_controls.py',
    'GEOMETRY_MECHANISM_STUDY.md',
]))


def read(path):
    return json.loads(Path(path).read_text())


def clock_ns():
    if not sys.platform.startswith('linux') or not hasattr(time, 'CLOCK_MONOTONIC'):
        raise RuntimeError('Cross-process cold-ready timing requires Linux CLOCK_MONOTONIC')
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def validate_inputs(design, base, methods, budgets, repeats, random_seeds=None):
    if (not isinstance(design, dict) or set(design) != {'schema', 'targets', 'profiles'}
            or design['schema'] != 'geometry_placement_v1'):
        raise ValueError('Invalid source-free geometry design')
    if (not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods)
            or type(repeats) is not int or not 1 <= repeats <= 5
            or not budgets or any(type(b) is not int or b < 1 for b in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError('Unique methods, sorted positive budgets, and 1-5 repetitions required')
    if 'random' in methods:
        random_seeds = list(range(5)) if random_seeds is None else random_seeds
        if (not random_seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in random_seeds)
                or random_seeds != sorted(set(random_seeds))):
            raise ValueError('Random seeds must be sorted unique integers in [0, 2**32)')
    elif random_seeds is not None:
        raise ValueError('Random seeds require the random method')
    study = dict(schema='connected_constraint_holdout_v1', alternatives_per_site=1,
        targets=design['targets'], profiles=design['profiles'])
    suite = constraints.validate_study(study, base)
    supported = {'max_installation_cost', 'min_xy_separation_m', 'max_sensors_per_pole',
        'mounting_z_range_m', 'min_coverage_pct', 'min_overlap_ratio', 'max_overlap_ratio',
        'min_single_failure_coverage_pct'}
    if set(suite['requirements'].get('constraints', {}))-supported:
        raise ValueError('Unsupported independent fixed-pair audit constraint')
    case_jobs = [{k:v for k,v in job.items() if k != 'status'} for job in constraints.study_jobs(study)]
    arms = []
    for method in methods:
        if method == 'full':
            continue
        for budget in budgets:
            if method == 'random':
                arms.extend(dict(name=f'random_q{budget}_s{seed}', method=method,
                    query_budget=budget, random_seed=seed) for seed in random_seeds)
            else:
                arms.append(dict(name=f'{method}_q{budget}', method=method, query_budget=budget))
    if 'full' in methods:
        arms.append(dict(name='full', method='full', query_budget=None))
    if len(case_jobs)*len(arms)*repeats > 504:
        raise ValueError('Bounded pilot allows at most 504 fresh workers')
    return suite, case_jobs, arms


def build_problem(suite, job, profile, sensor):
    geometry = constraints.target_geometry(suite, job)
    canonical = corridor.candidates(geometry, suite['candidate'])
    poses = geometry.poses(canonical)
    targets, coordinates = geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
    req = CorridorRequirements(suite['requirements'], targets, coordinates, canonical, poses, profile,
        length_m=geometry.length_m, interface_half_length_m=suite['interface_half_length_m'])
    # Legacy context requires a vector; geometry modes never consume source scores.
    context = SelectionContext(canonical, poses, targets, np.zeros(len(poses)), req, sensor)
    modules = (canonical[:, 0] >= geometry.length_m/2).astype(int)
    return geometry, context, coordinates, modules


def run_worker(input_path, config_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = read(input_path)
    suite, job, profile, arm = (data[k] for k in ('suite', 'case', 'profile', 'arm'))
    config = load_config(config_path)
    sensor = get_lidar_profile(config.lidar_profile, config)
    geometry, context, coordinates, modules = build_problem(suite, job, profile, sensor)
    mesh = geometry.create_environment_mesh()
    geometry_sec = time.perf_counter()-started
    cache = corridor.MaskCache(mesh, context.targets, sensor, context.poses)
    observed = BudgetedObservations(context, cache, per_site=arm['query_budget'])
    selection_start = time.perf_counter()
    if arm['method'] == 'full':
        for cid in np.flatnonzero(context.allowed):
            observed.get(int(cid), phase='full_reference')
        observed.frozen = True
        selection = dict(candidate_ids=sorted(observed.masks), kind='full', events=observed.events,
            contexts=[], decisions=[], optical_point_tests=0, selection_sec=time.perf_counter()-selection_start)
    elif arm['method'] in CONTROL_METHODS:
        selection = select_geometry_control(arm['method'], context, observed, arm['query_budget'])
    else:
        if arm['method'] not in ('adaptive', 'adaptive_batch', 'static', 'random'):
            raise ValueError('Unknown source-free method')
        adaptive = arm['method'] in ('adaptive', 'adaptive_batch')
        options = dict(probe_prior='geometry', fast_screen=True) if adaptive else {}
        if arm['method'] == 'adaptive_batch':
            options['partner_engine'] = 'batch'
        kind = 'probe_context' if adaptive else 'geometry_single'
        if arm['method'] == 'random':
            kind, options = 'random_pool', dict(random_seed=arm['random_seed'])
        selection = select_pool(kind,
            context, observed, arm['query_budget'], **options)
    upper_start = time.perf_counter()
    result = upper_select(selection['candidate_ids'], context, observed, modules, 'free')
    upper_sec = time.perf_counter()-upper_start
    decision_sec = time.perf_counter()-started
    completed_ns = clock_ns()
    marker = dict(event='decision_complete', completed_monotonic_ns=completed_ns, source_free=True,
        input_sha256=corridor.digest(input_path), arm=arm)
    print(json.dumps(marker), flush=True)
    # Everything below this checkpoint is excluded from cold algorithm-ready latency.
    result.update(**arm, source_free=True, decision_sec=decision_sec, geometry_sec=geometry_sec,
        upper_sec=upper_sec, optical_point_tests=selection['optical_point_tests'],
        timing_semantics='Child decision includes input/config, geometry/scene, optics/probes and exact upper '
            'selection, but excludes interpreter/import and final validation/export. Parent cold-ready adds startup/import.')
    result = corridor.persist_method(output/'solution', result, context.canonical, context.poses,
        context.targets, coordinates, observed, mesh, sensor, context.requirements, suite)
    ids = np.asarray(sorted(observed.masks), dtype=int)
    masks = np.asarray([observed.measured(i) for i in ids], bool).reshape(len(ids), len(context.targets))
    np.savez_compressed(output/'observations.npz', candidate_ids=ids, masks=masks)
    selection.update(work=observed.report(), observations_sha256=corridor.digest(output/'observations.npz'))
    corridor.save(output/'selection.json', selection)
    np.savez_compressed(output/'geometry.npz', vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles),
        canonical_candidates=context.canonical, poses=context.poses, targets=context.targets,
        target_coordinates=coordinates, module_ids=modules, allowed_candidates=context.allowed)
    corridor.save(output/'geometry.json', vars(geometry))
    corridor.save(output/'worker.json', dict(status='complete', peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        completed_monotonic_ns=completed_ns, work=observed.report(), source_free=True))


def worker_jobs(cases, arms, repeats):
    jobs = []
    for ci, case in enumerate(cases):
        for rep in range(repeats):
            offset = (ci+rep) % len(arms)
            order = arms[offset:]+arms[:offset]
            for position, arm in enumerate(order):
                name = f"{case['directory']}_r{rep}_{arm['name']}"
                jobs.append(dict(id=name, case=case['directory'], arm=arm, repeat=rep,
                    order_position=position, status='pending'))
    return jobs


def cold_timing(start_ns, end_ns, marker):
    ready = marker['completed_monotonic_ns']
    if (marker.get('event') != 'decision_complete' or marker.get('source_free') is not True
            or type(ready) is not int or not start_ns <= ready <= end_ns):
        raise ValueError('Invalid cross-process algorithm-ready marker')
    return dict(cold_ready_sec=(ready-start_ns)/1e9, worker_wall_sec=(end_ns-start_ns)/1e9,
        parent_launch_monotonic_ns=start_ns, child_ready_monotonic_ns=ready, parent_exit_monotonic_ns=end_ns,
        clock='Linux CLOCK_MONOTONIC', semantics='Launch through algorithm completion, includes interpreter/import; '
            'excludes final direct validation and artifact export. OS filesystem cache is not cleared.')


def tables(root, manifest):
    raw, groups = [], {}
    for job in manifest['jobs']:
        if job['status'] != 'complete':
            raise ValueError('Incomplete workers cannot be summarized')
        directory = root/'runs'/job['id']
        result = read(directory/'solution/result.json')
        timing = read(directory/'timing.json')
        case = next(c for c in manifest['cases'] if c['directory'] == job['case'])
        assessment = result['assessment']
        row = dict(run=job['id'], case=job['case'], profile=case['profile'], target=case['target_name'],
            **job['arm'], repeat=job['repeat'], order_position=job['order_position'],
            feasible=assessment is not None, objective_score=assessment['objective_score'] if assessment else None,
            coverage_pct=assessment['coverage_pct'] if assessment else None,
            candidate_queries=result['work']['unique_candidate_queries'], rays_submitted=result['work'].get('rays_submitted', 0),
            optical_point_tests=result['optical_point_tests'], cold_ready_sec=timing['cold_ready_sec'],
            decision_sec=result['decision_sec'], worker_wall_sec=timing['worker_wall_sec'])
        raw.append(row)
        row['random_seed'] = job['arm'].get('random_seed')
        groups.setdefault((job['case'], job['arm']['name']), []).append((row, result))
    comparisons = []
    for (case, name), values in groups.items():
        first, result = values[0]
        for row, other in values:
            for field in ('feasible', 'objective_score', 'coverage_pct', 'candidate_queries', 'rays_submitted', 'optical_point_tests'):
                if row[field] != first[field]:
                    raise ValueError(f'Non-deterministic repeated quality/work: {case}/{name}/{field}')
            if other['selected_ids'] != result['selected_ids']:
                raise ValueError('Repeated workers selected different deployment IDs')
        reference = groups.get((case, 'full'))
        ref_j = reference[0][0]['objective_score'] if reference else None
        value = first['objective_score']
        if value is not None and ref_j is not None and value > ref_j+1e-7:
            raise ValueError('Queried pool exceeds eligible full reference')
        cold = [v['cold_ready_sec'] for v, _ in values]
        comparisons.append(dict(case=case, profile=first['profile'], target=first['target'], name=name,
            method=first['method'], query_budget=first['query_budget'], random_seed=first.get('random_seed'),
            repeats=len(values), feasible=first['feasible'],
            objective_score=value, reference_j=ref_j, objective_gap=ref_j-value if value is not None and ref_j is not None else None,
            candidate_queries=first['candidate_queries'], rays_submitted=first['rays_submitted'],
            optical_point_tests=first['optical_point_tests'], cold_ready_median_sec=float(np.median(cold)),
            cold_ready_q25_sec=float(np.quantile(cold,.25)), cold_ready_q75_sec=float(np.quantile(cold,.75)),
            decision_median_sec=float(np.median([v['decision_sec'] for v,_ in values])),
            worker_wall_median_sec=float(np.median([v['worker_wall_sec'] for v,_ in values]))))
    summary = dict(conditions=len(manifest['cases']), runs=len(raw), repeats=manifest['plan']['repeats'], groups={},
        scope='Known synthetic development shapes; repetitions measure timing, not new quality samples. '
            'Fixed K=2; no source library or MAPPO. Fresh process/algorithm cache, not OS-cache eviction or server isolation.')
    for arm in manifest['plan']['arms']:
        rows = [r for r in comparisons if r['name'] == arm['name']]
        gaps = [r['objective_gap'] for r in rows if r['objective_gap'] is not None]
        equal = [r for r in rows if r['objective_gap'] is not None and abs(r['objective_gap']) <= 1e-7]
        ratios = [r['cold_ready_median_sec']/next(v['cold_ready_median_sec'] for v in comparisons
            if v['case'] == r['case'] and v['name'] == 'full') for r in equal]
        summary['groups'][arm['name']] = dict(cases=len(rows), feasible_cases=sum(r['feasible'] for r in rows),
            reference_cases=len(gaps), matches_reference=len(equal), mean_feasible_j_gap=float(np.mean(gaps)) if gaps else None,
            mean_queries=float(np.mean([r['candidate_queries'] for r in rows])),
            mean_case_median_cold_ready_sec=float(np.mean([r['cold_ready_median_sec'] for r in rows])),
            mean_case_median_decision_sec=float(np.mean([r['decision_median_sec'] for r in rows])),
            equal_j_latency_ratio_cases=len(ratios), mean_equal_j_latency_ratio=float(np.mean(ratios)) if ratios else None)
    return raw, comparisons, summary


def summarize(root, manifest):
    raw, comparison, summary = tables(root, manifest)
    for name, rows in [('runs.csv', raw), ('comparison.csv', comparison)]:
        with (root/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    corridor.save(root/'summary.json', summary)


def execute(root, design, suite, cases, arms, plan, config_path):
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs/jobs').mkdir(parents=True)
    (root/'logs').mkdir()
    corridor.save(root/'inputs/design.json', design)
    corridor.save(root/'inputs/base_suite.json', suite)
    shutil.copy2(config_path, root/'inputs/experiment.yaml')
    for name in SOURCE_FILES:
        path = root/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/name, path)
    jobs = worker_jobs(cases, arms, plan['repeats'])
    profiles = {p['name']:p for p in design['profiles']}
    for job in jobs:
        case = next(c for c in cases if c['directory'] == job['case'])
        corridor.save(root/'inputs/jobs'/f"{job['id']}.json", dict(suite=suite, case=case,
            profile=profiles[case['profile']], arm=job['arm']))
    manifest = dict(status='prepared', created_at=corridor.now(), plan=plan, cases=cases, jobs=jobs,
        source_hashes={n:corridor.digest(root/'source'/n) for n in SOURCE_FILES},
        input_hashes={str(p.relative_to(root)):corridor.digest(p) for p in sorted((root/'inputs').rglob('*')) if p.is_file()},
        hardware=dict(platform=platform.platform(), python=platform.python_version(), cpu=platform.processor(),
            affinity=sorted(os.sched_getaffinity(0)), cpu_count=os.cpu_count()), source_free=True)
    corridor.save(root/'manifest.json', manifest)
    (root/'packages.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
        capture_output=True, text=True, check=True).stdout)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':'',
        'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
    try:
        manifest['status'] = 'running'
        for job in jobs:
            job['status'] = 'running'
            corridor.save(root/'manifest.json', manifest)
            input_path = root/'inputs/jobs'/f"{job['id']}.json"
            cmd = [sys.executable, str(root/'source/geometry_placement.py'), '--worker-input', str(input_path),
                '--worker-config', str(root/'inputs/experiment.yaml'), '--worker-output', str(root/'runs'/job['id']), '--execute']
            with (root/'logs'/f"{job['id']}.stderr").open('w') as stderr:
                begin = clock_ns()
                child = subprocess.run(cmd, cwd=root/'source', env=env, stdout=subprocess.PIPE,
                    stderr=stderr, text=True, timeout=180)
                end = clock_ns()
            (root/'logs'/f"{job['id']}.stdout").write_text(child.stdout)
            if child.returncode:
                raise RuntimeError(f"Worker failed: {job['id']} (exit {child.returncode})")
            marker = json.loads(child.stdout)
            if marker.get('input_sha256') != manifest['input_hashes'][str(input_path.relative_to(root))] or marker.get('arm') != job['arm']:
                raise ValueError('Worker/input marker mismatch')
            timing = cold_timing(begin, end, marker)
            directory = root/'runs'/job['id']
            corridor.save(directory/'timing.json', timing)
            job.update(status='complete', command=cmd, output_hashes={str(p.relative_to(directory)):corridor.digest(p)
                for p in sorted(directory.rglob('*')) if p.is_file()})
            corridor.save(root/'manifest.json', manifest)
        summarize(root, manifest)
        manifest.update(status='complete', finished_at=corridor.now())
        corridor.save(root/'manifest.json', manifest)
        with (root/'audit.log').open('w') as log:
            subprocess.run([sys.executable, str(root/'source/tests/validate_geometry_placement.py'), str(root)],
                cwd=root/'source', env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        corridor.save(root/'manifest.json', manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=str(BASE/'configs/geometry_placement.json'))
    parser.add_argument('--base-suite', default=str(BASE/'configs/connected_corridor_pilot.json'))
    parser.add_argument('--config', default=str(BASE/'configs/experiment.yaml'))
    parser.add_argument('--methods', choices=METHODS, nargs='+', default=['adaptive'])
    parser.add_argument('--query-budgets', type=int, nargs='+', default=[4])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--random-seeds', type=int, nargs='+',
        help='Random pool seeds (default 0 1 2 3 4); distinct from timing repetitions')
    parser.add_argument('--output', default='results/geometry_placement_20261103')
    parser.add_argument('--execute', action='store_true')
    for flag in ('worker-input', 'worker-config', 'worker-output'):
        parser.add_argument('--'+flag, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if any((args.worker_input, args.worker_config, args.worker_output)):
        if not args.execute or not all((args.worker_input, args.worker_config, args.worker_output)):
            raise ValueError('Worker requires explicit execute/input/config/output')
        return run_worker(args.worker_input, args.worker_config, args.worker_output)
    design = read(args.study)
    suite, cases, arms = validate_inputs(design, corridor.load_suite(args.base_suite),
        args.methods, args.query_budgets, args.repeats, args.random_seeds)
    config = load_config(args.config)
    sensor = get_lidar_profile(config.lidar_profile, config)
    if sensor.h_fov_deg != 360:
        raise ValueError('Independent pilot audit requires a 360-degree profile')
    profiles = {p['name']:p for p in design['profiles']}
    for case in cases:
        _, context, _, _ = build_problem(suite, case, profiles[case['profile']], sensor)
        if any(g and len(g) < max(args.query_budgets) for g in context.groups) and any(m != 'full' for m in args.methods):
            raise ValueError('A requested query budget exceeds an eligible site domain')
    plan = dict(source_free=True, source_assets_required=False, conditions=len(cases), arms=arms,
        repeats=args.repeats, fresh_workers=len(cases)*len(arms)*args.repeats, execute=args.execute,
        reference_requested='full' in args.methods, fixed_sensor_count=2,
        order='Deterministic cyclic rotation by case and repetition', sensor=vars(sensor),
        timing='Separate child decision, cross-process cold algorithm-ready and whole-worker clocks')
    print(json.dumps(plan), flush=True)
    if args.execute:
        clock_ns()
        execute(Path(args.output).resolve(), design, suite, cases, arms, plan, args.config)


if __name__ == '__main__':
    main()

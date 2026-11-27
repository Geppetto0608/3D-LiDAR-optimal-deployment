"""Opt-in fixed-K=2 hierarchy/target-query pilot, not a learned deployment model."""
from argparse import ArgumentParser, SUPPRESS
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import connected_corridor_study as corridor
import connected_constraint_study as constraints
from src.environment import get_lidar_profile, load_config
from src.methods.connected_hierarchy import BudgetedObservations, SelectionContext, refine_poses, select_pool, upper_select
from src.methods.connected_pairs import exact_pairs
from src.methods.source_modes import response_source_alternatives
from src.roadside.corridor_requirements import CorridorRequirements


BASE = Path(__file__).resolve().parent
SELECTORS = ('source_response', 'geometry_single', 'probe_context')
ALLOCATIONS = ('one_per_module', 'free')
STAGES = ('raw', 'fixed_site', 'radius_control')
REFERENCES = ('reference_one_per_module', 'reference_global')
SOURCE_FILES = list(dict.fromkeys(constraints.SOURCE_FILES + [
    'connected_hierarchy_study.py', 'src/methods/connected_hierarchy.py',
    'tests/validate_connected_hierarchy.py',
]))


def method_name(selector, allocation, stage):
    return f'{selector}__{allocation}__{stage}'


def methods():
    return tuple(method_name(s, a, t) for s in SELECTORS for a in ALLOCATIONS for t in STAGES) + REFERENCES


def save_observations(directory, observations):
    ids = sorted(observations.masks)
    masks = np.asarray([observations.measured(i) for i in ids], dtype=bool).reshape(
        len(ids), len(observations.context.targets))
    np.savez_compressed(directory / 'observations.npz', candidate_ids=ids, masks=masks)
    return corridor.digest(directory / 'observations.npz')


def allocation_signature(ids, canonical, modules):
    return None if ids is None else [dict(module=int(modules[i]), side='left' if canonical[i, 1] > 0 else 'right',
        station_m=float(canonical[i, 0])) for i in ids]


def run_case(root, job, suite, profile, sensor, source, source_scores, static_ids, extra_queries):
    directory = root / job['directory']
    directory.mkdir()
    started = time.perf_counter()
    geometry = constraints.target_geometry(suite, job)
    mesh = geometry.create_environment_mesh()
    targets, coordinates = geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
    canonical = constraints.transfer_source(source, geometry)
    poses = geometry.poses(canonical)
    modules = np.repeat([0, 1], len(source))
    req = CorridorRequirements(suite['requirements'], targets, coordinates, canonical, poses, profile,
        length_m=geometry.length_m, interface_half_length_m=suite['interface_half_length_m'])
    context = SelectionContext(canonical, poses, targets, np.tile(source_scores, 2), req, sensor)
    static_ids = static_ids + [i+len(source) for i in static_ids]
    np.savez_compressed(directory / 'geometry.npz', vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles),
        targets=targets, target_coordinates=coordinates, canonical_candidates=canonical, poses=poses,
        module_ids=modules, allowed_candidates=context.allowed)
    corridor.save(directory / 'geometry.json', vars(geometry))
    corridor.save(directory / 'corridor_profile.json', profile)
    corridor.save(directory / 'requirements.json', suite['requirements'])
    outputs = {}
    for selector in SELECTORS:
        selection_dir = directory / f'selection_{selector}'
        selection_dir.mkdir()
        selection_start = time.perf_counter()
        physical = corridor.MaskCache(mesh, targets, sensor, poses)
        observed = BudgetedObservations(context, physical, per_site=suite['alternatives_per_site'])
        selection = select_pool(selector, context, observed, suite['alternatives_per_site'], static_ids=static_ids)
        selection.update(work=observed.report(), observations_sha256=save_observations(selection_dir, observed),
            selection_wall_sec=time.perf_counter()-selection_start, committed_at=corridor.now())
        corridor.save(selection_dir / 'selection.json', selection)
        selection_hash = corridor.digest(selection_dir / 'selection.json')
        for allocation in ALLOCATIONS:
            upper_start = time.perf_counter()
            upper = upper_select(selection['candidate_ids'], context, observed, modules, allocation)
            upper_sec = time.perf_counter()-upper_start
            for stage in STAGES:
                lower_start = time.perf_counter()
                cache = observed
                if stage == 'raw':
                    result = deepcopy(upper)
                else:
                    physical = corridor.MaskCache(mesh, targets, sensor, poses)
                    cache = BudgetedObservations(context, physical, max_extra=extra_queries,
                        initial=observed.masks, initial_work=observed.report())
                    result = refine_poses(upper, context, cache, station_radius_m=
                        0 if stage == 'fixed_site' else suite['local_station_radius_m'])
                    result['initialization'] = deepcopy(upper)
                result.update(selector=selector, upper_allocation=allocation, stage=stage,
                    selection_sha256=selection_hash, lower_query_limit=0 if stage == 'raw' else extra_queries,
                    lower_events=[] if stage == 'raw' else cache.events,
                    final_allocation=allocation_signature(result['selected_ids'], canonical, modules),
                    selection_shared_sec=selection['selection_wall_sec'], upper_shared_sec=upper_sec,
                    lower_sec=time.perf_counter()-lower_start,
                    timing_semantics='Factorial branches share measured selection and upper results; not independent cold timings')
                name = method_name(selector, allocation, stage)
                outputs[name] = corridor.persist_method(directory / name, result, canonical, poses, targets,
                    coordinates, cache, mesh, sensor, req, suite)
                observation_hash = save_observations(directory / name, cache)
                outputs[name]['observations_sha256'] = observation_hash
                corridor.save(directory / name / 'result.json', outputs[name])
    online_hashes = {str(path.relative_to(directory)): corridor.digest(path)
        for path in sorted(directory.glob('*/*')) if path.is_file()}
    case = dict(status='online_committed', online_committed_at=corridor.now(), online_hashes=online_hashes,
        geometry_sha256=corridor.digest(directory / 'geometry.npz'))
    corridor.save(directory / 'manifest.json', case)
    # No offline full target-mask archive is available to any online branch above.
    case['reference_started_at'] = corridor.now()
    cache = corridor.MaskCache(mesh, targets, sensor, poses)
    np.savez_compressed(directory / 'reference_masks.npz', masks=np.asarray([cache(i) for i in range(len(poses))]))
    allowed = np.flatnonzero(context.allowed).tolist()
    for name, quota in zip(REFERENCES, (modules, None)):
        result = (exact_pairs(allowed, poses, cache, req, module_ids=quota,
            pair_mask_filter=req.filter_visibility_pairs) if len(allowed) >= 2 else constraints.no_pair_result(len(allowed), name))
        outputs[name] = corridor.persist_method(directory / name, result, canonical, poses, targets, coordinates,
            cache, mesh, sensor, req, suite)
    representative = [method_name(s, 'free', t) for s in SELECTORS for t in ('raw', 'fixed_site')] + list(REFERENCES)
    corridor.plot_case(directory, mesh, targets, outputs, poses, job['angle_deg'], methods=representative)
    case.update(status='complete', finished_at=corridor.now(), case_wall_sec=time.perf_counter()-started,
        reference_masks_sha256=corridor.digest(directory / 'reference_masks.npz'))
    corridor.save(directory / 'manifest.json', case)


def summarize(root, jobs):
    rows = []
    for job in jobs:
        if job['status'] != 'complete':
            continue
        case = root / job['directory']
        reference = json.loads((case / 'reference_global/result.json').read_text())['assessment']
        for name in methods():
            result = json.loads((case / name / 'result.json').read_text())
            a = result['assessment']
            feasible = a is not None and a['feasible']
            gap = reference['objective_score']-a['objective_score'] if reference and feasible else None
            if gap is not None and gap < -1e-7:
                raise RuntimeError('Feasible result exceeds exhaustive reference')
            rows.append(dict(case=job['directory'], profile=job['profile'], target=job.get('target_name', str(job['angle_deg'])),
                method=name, feasible=feasible, objective_score=a['objective_score'] if a else None,
                objective_gap=gap, coverage_pct=a['coverage_pct'] if a else None,
                interface_overlap_ratio=result.get('interface_overlap_ratio'),
                candidate_queries=result['work']['unique_candidate_queries'],
                extra_queries=0 if name in REFERENCES or result.get('stage') == 'raw' else result['work']['extra_candidate_queries'],
                rays_submitted=result['work'].get('rays_submitted', 0), stopping_reason=result.get('stopping_reason', '')))
    if not rows:
        return
    with (root / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = sum(j['status'] == 'complete' for j in jobs)
    aggregates = {}
    for name in methods():
        values = [r for r in rows if r['method'] == name]
        gaps = [r['objective_gap'] for r in values if r['objective_gap'] is not None]
        aggregates[name] = dict(feasible=sum(r['feasible'] for r in values),
            matches_global=sum(abs(g) <= 1e-7 for g in gaps),
            mean_feasible_j_gap=float(np.mean(gaps)) if gaps else None,
            mean_queries=float(np.mean([r['candidate_queries'] for r in values])),
            budget_limited=sum(r['stopping_reason'] == 'query_budget_limited' for r in values))
    corridor.save(root / 'summary.json', dict(conditions=count, rows=len(rows), methods=aggregates,
        scope='Observed synthetic development conditions, fixed two modules and K=2. Illustrative requirements.'))
    lines = ['# Two-Level Placement and Metered Target Probes', '',
        f'{count}/{len(jobs)} conditions; 3 selectors x 2 upper domains x 3 stages + 2 full-grid references.',
        'Geometry and targets are known; optical predictions ignore occlusion and do not certify feasibility.',
        'Actual per-site query budgets match. Optical work, rays and elapsed time need not match.',
        'The radius control can change station/module; upper allocation labels describe initialization only.',
        'Lower branches share selection/upper observations, then have independent capped query caches.', '',
        '| Method | Feasible | Matches global J | Mean feasible J gap | Mean queries | Budget limited |',
        '|---|---:|---:|---:|---:|---:|']
    for name, value in aggregates.items():
        gap = value['mean_feasible_j_gap']
        lines.append(f"| {name} | {value['feasible']}/{count} | {value['matches_global']}/{count} | "
            f"{gap if gap is not None else '-'} | {value['mean_queries']:g} | {value['budget_limited']} |")
    lines.extend(['', 'No MAPPO, learned response transfer, continuous optimum, detection or citywide guarantee.',
        'Full references enumerate only the declared coarse fixed-K grid. Failures are not scored zero.', ''])
    (root / 'REPORT.md').write_text('\n'.join(lines))


def worker(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        if corridor.digest(BASE / name) != expected:
            raise RuntimeError(f'Frozen source mismatch: {name}')
    for name, expected in manifest['input_hashes'].items():
        if corridor.digest(root / name) != expected:
            raise RuntimeError(f'Frozen input mismatch: {name}')
    base = corridor.load_suite(root / 'inputs/base_suite.json')
    study = json.loads((root / 'inputs/study.json').read_text())
    suite = constraints.validate_study(study, base)
    config = load_config(root / 'inputs/experiment.yaml')
    sensor = get_lidar_profile(config.lidar_profile, config)
    started = time.perf_counter()
    try:
        manifest.update(status='running', stage='source_library')
        corridor.save(root / 'manifest.json', manifest)
        source, _, _ = corridor.source_library(suite, sensor, root / 'library')
        with np.load(root / 'library/library.npz', allow_pickle=False) as data:
            weights = suite['requirements'].get('weights', {})
            coefficient = weights.get('coverage_weight', 1)+weights.get('importance_weight', 0)+weights.get('overlap_penalty', 0)
            scores = data['scores']
            ids, traces = response_source_alternatives(source, data['masks'], scores,
                suite['alternatives_per_site'], intersection_weight=coefficient)
        corridor.save(root / 'source_selection.json', dict(candidate_ids=ids, traces=traces,
            coefficient=coefficient, committed_at=corridor.now()))
        manifest.update(source_selection_sha256=corridor.digest(root / 'source_selection.json'),
            source_library_sha256=corridor.digest(root / 'library/library.npz'))
        profiles = {p['name']: p for p in study['profiles']}
        for job in manifest['jobs']:
            job['status'] = 'running'
            manifest.update(stage='online_then_reference', current_job=job['directory'])
            corridor.save(root / 'manifest.json', manifest)
            print(f"Start {job['directory']}: {job['profile']} {job.get('target_name', job['angle_deg'])}", flush=True)
            run_case(root, job, suite, profiles[job['profile']], sensor, source, scores, ids,
                manifest['plan']['lower_extra_queries'])
            job['status'] = 'complete'
            corridor.save(root / 'manifest.json', manifest)
            summarize(root, manifest['jobs'])
            print(f"Complete {sum(j['status'] == 'complete' for j in manifest['jobs'])}/{len(manifest['jobs'])}", flush=True)
        manifest.update(status='complete', stage='complete', finished_at=corridor.now())
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['worker_wall_sec'] = time.perf_counter()-started
        corridor.save(root / 'manifest.json', manifest)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--study', default=str(BASE / 'configs/connected_constraint_holdout.json'))
    parser.add_argument('--base-suite', default=str(BASE / 'configs/connected_corridor_pilot.json'))
    parser.add_argument('--config', default=str(BASE / 'configs/experiment.yaml'))
    parser.add_argument('--output', default='results/connected_hierarchy_pilot')
    parser.add_argument('--lower-extra-queries', type=int, default=16)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-root', help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return worker(Path(args.worker_root).resolve())
    if not 0 <= args.lower_extra_queries <= 600:
        raise ValueError('Lower query cap must be between 0 and 600')
    base = corridor.load_suite(args.base_suite)
    study = json.loads(Path(args.study).read_text())
    suite = constraints.validate_study(study, base)
    jobs = constraints.study_jobs(study)
    config = load_config(args.config)
    plan = dict(conditions=len(jobs), method_outputs=len(jobs)*len(methods()), methods=methods(),
        per_site_queries=suite['alternatives_per_site'], lower_extra_queries=args.lower_extra_queries,
        lower_radius_control_m=suite['local_station_radius_m'], sensor=vars(get_lidar_profile(config.lidar_profile, config)),
        scope='Observed synthetic development conditions; geometry-single is a non-transfer control', execute=args.execute)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'inputs').mkdir()
    for origin, name in [(args.study, 'study.json'), (args.base_suite, 'base_suite.json'), (args.config, 'experiment.yaml')]:
        shutil.copy2(origin, root / 'inputs' / name)
    for name in SOURCE_FILES:
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, destination)
    (root / 'packages.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
        check=True, capture_output=True, text=True).stdout)
    corridor.save(root / 'manifest.json', dict(status='prepared', created_at=corridor.now(), plan=plan, jobs=jobs,
        command=sys.argv if argv is None else argv,
        source_hashes={name: corridor.digest(root / 'source' / name) for name in SOURCE_FILES},
        input_hashes={f'inputs/{name}': corridor.digest(root / 'inputs' / name)
                     for name in ('study.json', 'base_suite.json', 'experiment.yaml')}))
    try:
        with (root / 'worker.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(root / 'source/connected_hierarchy_study.py'),
                '--worker-root', str(root), '--execute'], cwd=root / 'source', stdout=log, stderr=subprocess.STDOUT, check=True)
        audit = subprocess.run([sys.executable, str(root / 'source/tests/validate_connected_hierarchy.py'), str(root)],
            capture_output=True, text=True)
        if audit.returncode:
            (root / 'audit.log').write_text(audit.stdout+audit.stderr)
            raise RuntimeError(f'Audit failed: {root / "audit.log"}')
        corridor.save(root / 'independent_validation.json', json.loads(audit.stdout))
    except BaseException as exc:
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest.update(status='failed', error=repr(exc))
        corridor.save(root / 'manifest.json', manifest)
        raise
    print(f'Completed and audited: {root / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()

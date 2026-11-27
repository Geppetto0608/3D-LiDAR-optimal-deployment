"""Source/geometry/random prior ablation under the same conditional probe rule.

Default is a no-write plan. Development cases only, fixed K=2, no lower search.
"""
from argparse import ArgumentParser, SUPPRESS
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

import connected_constraint_study as constraints
import connected_corridor_study as corridor
import connected_hierarchy_study as hierarchy
from src.environment import get_lidar_profile, load_config
from src.methods.connected_hierarchy import BudgetedObservations, SelectionContext, select_pool, upper_select
from src.roadside.corridor_requirements import CorridorRequirements


BASE = Path(__file__).resolve().parent
SOURCE_FILES = list(dict.fromkeys(hierarchy.SOURCE_FILES + [
    'connected_prior_ablation.py', 'tests/validate_connected_priors.py',
    'tests/test_connected_priors.py', 'tests/test_connected_hierarchy.py',
]))


def validate_options(options):
    if (not isinstance(options, dict) or set(options) != {'schema', 'query_budgets', 'random_seeds'}
            or options['schema'] != 'connected_prior_ablation_v1'):
        raise ValueError('Invalid prior ablation options')
    budgets, seeds = options['query_budgets'], options['random_seeds']
    if (not isinstance(budgets, list) or not budgets or len(budgets) > 3
            or any(type(n) is not int or n not in (1, 2, 4) for n in budgets)
            or budgets != sorted(set(budgets))):
        raise ValueError('Unique sorted query budgets from 1, 2, 4 required')
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 8
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('1-8 distinct random seeds in [0, 2**32) required')
    return options


def arms(options):
    return [dict(name=f'b{b}_{kind}' + (f'_s{seed}' if seed is not None else ''),
        query_budget=b, prior=kind, random_seed=seed)
        for b in options['query_budgets']
        for kind, seed in [('source', None), ('geometry', None)] + [('random', s) for s in options['random_seeds']]]


def validate_scope(suite, sensor):
    supported = {'max_installation_cost', 'min_xy_separation_m', 'max_sensors_per_pole',
        'mounting_z_range_m', 'min_coverage_pct', 'min_overlap_ratio', 'max_overlap_ratio',
        'min_single_failure_coverage_pct'}
    if set(suite['requirements'].get('constraints', {}))-supported or sensor.h_fov_deg != 360:
        raise ValueError('This bounded audit supports listed pair constraints and 360-degree sensors only')


def case_context(suite, job, profile, sensor):
    geometry = constraints.target_geometry(suite, job)
    canonical = corridor.candidates(geometry, suite['candidate'])
    poses = geometry.poses(canonical)
    targets, coordinates = geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
    req = CorridorRequirements(suite['requirements'], targets, coordinates, canonical, poses, profile,
        length_m=geometry.length_m, interface_half_length_m=suite['interface_half_length_m'])
    # Target candidate geometry is generated directly, without reading source layouts.
    context = SelectionContext(canonical, poses, targets, np.zeros(len(poses)), req, sensor)
    modules = (canonical[:, 0] >= geometry.length_m/2).astype(int)
    return geometry, context, coordinates, modules


def run_case(root, job, suite, profile, sensor, source_scores, options):
    directory = root / job['directory']
    directory.mkdir()
    started = time.perf_counter()
    geometry, neutral, coordinates, modules = case_context(suite, job, profile, sensor)
    mesh = geometry.create_environment_mesh()
    geometry_sec = time.perf_counter()-started
    canonical, poses, targets, req = neutral.canonical, neutral.poses, neutral.targets, neutral.requirements
    np.savez_compressed(directory/'geometry.npz', vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles),
        canonical_candidates=canonical, poses=poses, targets=targets, target_coordinates=coordinates,
        module_ids=modules, allowed_candidates=neutral.allowed)
    corridor.save(directory/'geometry.json', vars(geometry))
    corridor.save(directory/'corridor_profile.json', profile)
    corridor.save(directory/'requirements.json', suite['requirements'])
    for arm in arms(options):
        decision_start = time.perf_counter()
        context = SelectionContext(canonical, poses, targets,
            np.tile(source_scores, 2) if arm['prior'] == 'source' else np.zeros(len(poses)), req, sensor)
        cache = corridor.MaskCache(mesh, targets, sensor, poses)
        observed = BudgetedObservations(context, cache, per_site=arm['query_budget'])
        selection = select_pool('probe_context', context, observed, arm['query_budget'],
            probe_prior=arm['prior'], random_seed=arm['random_seed'], fast_screen=True)
        upper_start = time.perf_counter()
        result = upper_select(selection['candidate_ids'], context, observed, modules, 'free')
        upper_sec = time.perf_counter()-upper_start
        decision_sec = time.perf_counter()-decision_start
        result.update(**arm, decision_sec=decision_sec, upper_sec=upper_sec,
            geometry_shared_sec=geometry_sec, online_component_sum_sec=geometry_sec+decision_sec,
            optical_point_tests=selection['optical_point_tests'],
            timing_semantics='Scene/prior/probes/selection timed; geometry measured once and added as a shared component. '
                'Imports, source construction/loading, serialization and final validation excluded. Not cold-process latency.')
        path = directory/arm['name']
        result = corridor.persist_method(path, result, canonical, poses, targets, coordinates, observed, mesh, sensor, req, suite)
        selection.update(observations_sha256=hierarchy.save_observations(path, observed),
            work=observed.report(), committed_at=corridor.now())
        corridor.save(path/'selection.json', selection)
        result['selection_sha256'] = corridor.digest(path/'selection.json')
        corridor.save(path/'result.json', result)
    online_hashes = {str(p.relative_to(directory)): corridor.digest(p)
        for p in sorted(directory.glob('*/*')) if p.is_file()}
    manifest = dict(status='online_committed', online_committed_at=corridor.now(), online_hashes=online_hashes,
        geometry_sha256=corridor.digest(directory/'geometry.npz'))
    corridor.save(directory/'manifest.json', manifest)
    # Fresh cache, individually eligible candidates only, after all online arms are fixed.
    manifest['reference_started_at'] = corridor.now()
    ref_start = time.perf_counter()
    cache = corridor.MaskCache(mesh, targets, sensor, poses)
    eligible = np.flatnonzero(neutral.allowed).tolist()
    masks = np.asarray([cache(i) for i in eligible], bool).reshape(len(eligible), len(targets))
    build_sec = time.perf_counter()-ref_start
    observed = BudgetedObservations(neutral, cache, initial={i: cache(i) for i in eligible})
    reference = upper_select(eligible, neutral, observed, modules, 'free')
    reference_sec = time.perf_counter()-ref_start
    reference.update(reference_build_sec=build_sec, decision_sec=reference_sec,
        geometry_shared_sec=geometry_sec, online_component_sum_sec=geometry_sec+reference_sec,
        declared_candidates=len(poses), eligible_candidates=len(eligible))
    # Direct MaskCache work includes the reference queries, not the seeded observation wrapper.
    corridor.persist_method(directory/'reference_global', reference, canonical, poses, targets, coordinates,
        cache, mesh, sensor, req, suite)
    np.savez_compressed(directory/'reference_masks.npz', candidate_ids=eligible, masks=masks)
    manifest.update(status='complete', finished_at=corridor.now(), case_wall_sec=time.perf_counter()-started,
        reference_masks_sha256=corridor.digest(directory/'reference_masks.npz'))
    corridor.save(directory/'manifest.json', manifest)


def make_tables(root, jobs, options):
    rows, paired = [], []
    for job in jobs:
        if job['status'] != 'complete':
            continue
        directory = root/job['directory']
        reference = json.loads((directory/'reference_global/result.json').read_text())['assessment']
        results = {}
        for arm in arms(options):
            result = json.loads((directory/arm['name']/'result.json').read_text())
            results[arm['name']] = result
            a, work = result['assessment'], result['work']
            gap = reference['objective_score']-a['objective_score'] if a and reference else None
            if gap is not None and gap < -1e-7:
                raise RuntimeError('Online result exceeds full eligible reference')
            rows.append(dict(case=job['directory'], profile=job['profile'], target=job.get('target_name', job['angle_deg']),
                **arm, feasible=bool(a and a['feasible']), objective_score=a['objective_score'] if a else None,
                objective_gap=gap, coverage_pct=a['coverage_pct'] if a else None,
                candidate_queries=work['unique_candidate_queries'], rays_submitted=work.get('rays_submitted', 0),
                optical_point_tests=result['optical_point_tests'], decision_sec=result['decision_sec'],
                online_component_sum_sec=result['online_component_sum_sec']))
        for budget in options['query_budgets']:
            source, geo = (results[f'b{budget}_{k}']['assessment'] for k in ('source', 'geometry'))
            random = [results[f'b{budget}_random_s{s}']['assessment'] for s in options['random_seeds']]
            values = [a['objective_score'] for a in random if a]
            paired.append(dict(case=job['directory'], query_budget=budget, source_feasible=source is not None,
                geometry_feasible=geo is not None, random_feasible_trials=len(values), random_total_trials=len(random),
                source_minus_geometry_j=source['objective_score']-geo['objective_score'] if source and geo else None,
                random_feasible_mean_j=statistics.mean(values) if values else None,
                random_feasible_sd_j=statistics.stdev(values) if len(values) >= 2 else None,
                source_minus_random_feasible_mean_j=source['objective_score']-statistics.mean(values) if source and values else None,
                source_wins=sum(source['objective_score'] > v+1e-7 for v in values) if source else 0,
                source_ties=sum(abs(source['objective_score']-v) <= 1e-7 for v in values) if source else 0,
                source_losses=sum(source['objective_score'] < v-1e-7 for v in values) if source else 0))
    groups = defaultdict(list)
    for row in rows:
        groups[f"b{row['query_budget']}_{row['prior']}"].append(row)
    aggregates = {}
    for key, values in groups.items():
        gaps = [r['objective_gap'] for r in values if r['objective_gap'] is not None]
        aggregates[key] = dict(trials=len(values), feasible_trials=sum(r['feasible'] for r in values),
            matches_reference=sum(abs(g) <= 1e-7 for g in gaps),
            feasible_gap_observations=len(gaps), mean_feasible_j_gap=statistics.mean(gaps) if gaps else None,
            mean_queries=statistics.mean(r['candidate_queries'] for r in values),
            mean_decision_sec=statistics.mean(r['decision_sec'] for r in values))
    summary = dict(conditions=sum(j['status'] == 'complete' for j in jobs), online_rows=len(rows), groups=aggregates,
        random_seed_count=len(options['random_seeds']),
        scope='Observed development geometries, fixed K=2. Random trials are repeated priors on the same maps, '
              'not independent roads. Feasible-only gaps exclude failures, never score them zero.')
    return rows, paired, summary


def summarize(root, jobs, options):
    rows, paired, summary = make_tables(root, jobs, options)
    if not rows:
        return
    for filename, values in [('comparison.csv', rows), ('paired.csv', paired)]:
        with (root/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    corridor.save(root/'summary.json', summary)
    lines = ['# Conditional Probe Prior Ablation', '', summary['scope'],
        'Same sequential rule; prior controls seeds, score ties and fallback. No lower refinement.',
        'All final feasibility uses measured masks. Geometry/random arms receive no source scores.', '',
        '| Budget/site | Prior | Feasible trials | Matches full J | Mean feasible J gap | Mean decision sec |',
        '|---|---|---:|---:|---:|---:|']
    for key, v in summary['groups'].items():
        budget, prior = key.split('_', 1)
        gap = '-' if v['mean_feasible_j_gap'] is None else f"{v['mean_feasible_j_gap']:.6f}"
        lines.append(f"| {budget[1:]} | {prior} | {v['feasible_trials']}/{v['trials']} | "
            f"{v['matches_reference']}/{v['trials']} | {gap} | {v['mean_decision_sec']:.6f} |")
    lines.extend(['', 'J gaps are not coverage percentage points. Inspect paired.csv for common-feasible differences.',
        'Random means are across all specified seeds, never best-seed selection. No new held-out or speedup claim.',
        'The full reference casts only individually eligible candidates. Timings exclude source build/load and final validation.',
        'Source prior is current single-source J ordering, not every possible library/learned-response method.', ''])
    (root/'REPORT.md').write_text('\n'.join(lines))


def worker(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        if corridor.digest(BASE/name) != expected:
            raise RuntimeError(f'Frozen source mismatch: {name}')
    for name, expected in manifest['input_hashes'].items():
        if corridor.digest(root/name) != expected:
            raise RuntimeError(f'Frozen input mismatch: {name}')
    suite = constraints.validate_study(json.loads((root/'inputs/study.json').read_text()),
        corridor.load_suite(root/'inputs/base_suite.json'))
    study = json.loads((root/'inputs/study.json').read_text())
    options = validate_options(json.loads((root/'inputs/ablation.json').read_text()))
    config = load_config(root/'inputs/experiment.yaml')
    sensor = get_lidar_profile(config.lidar_profile, config)
    validate_scope(suite, sensor)
    start = time.perf_counter()
    try:
        manifest.update(status='running', stage='source_library')
        corridor.save(root/'manifest.json', manifest)
        source, _, _ = corridor.source_library(suite, sensor, root/'library')
        with np.load(root/'library/library.npz', allow_pickle=False) as data:
            scores = data['scores']
        manifest.update(source_library_sha256=corridor.digest(root/'library/library.npz'),
            source_frozen_at=corridor.now(), source_candidate_count=len(source))
        profiles = {p['name']: p for p in study['profiles']}
        for job in manifest['jobs']:
            job['status'] = 'running'
            manifest.update(stage='online_then_reference', current_job=job['directory'])
            corridor.save(root/'manifest.json', manifest)
            print(f"Start {job['directory']}: {job['profile']} {job.get('target_name', job['angle_deg'])}", flush=True)
            run_case(root, job, suite, profiles[job['profile']], sensor, scores, options)
            job['status'] = 'complete'
            corridor.save(root/'manifest.json', manifest)
            summarize(root, manifest['jobs'], options)
            print(f"Complete {sum(j['status'] == 'complete' for j in manifest['jobs'])}/{len(manifest['jobs'])}", flush=True)
        manifest.update(status='complete', stage='complete', finished_at=corridor.now())
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['worker_wall_sec'] = time.perf_counter()-start
        corridor.save(root/'manifest.json', manifest)


def main(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument('--study', default=str(BASE/'configs/connected_constraint_holdout.json'))
    p.add_argument('--base-suite', default=str(BASE/'configs/connected_corridor_pilot.json'))
    p.add_argument('--config', default=str(BASE/'configs/experiment.yaml'))
    p.add_argument('--output', default='results/connected_prior_ablation')
    p.add_argument('--query-budgets', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--random-seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker-root', help=SUPPRESS)
    args = p.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return worker(Path(args.worker_root).resolve())
    options = validate_options(dict(schema='connected_prior_ablation_v1', query_budgets=args.query_budgets, random_seeds=args.random_seeds))
    study = json.loads(Path(args.study).read_text())
    suite = constraints.validate_study(study, corridor.load_suite(args.base_suite))
    config = load_config(args.config)
    sensor = get_lidar_profile(config.lidar_profile, config)
    validate_scope(suite, sensor)
    jobs = constraints.study_jobs(study)
    plan = dict(conditions=len(jobs), arms=arms(options), online_outputs=len(jobs)*len(arms(options)),
        references=len(jobs), lower_refinement=False, fast_prediction_screen=True, sensor=vars(sensor),
        evaluation_scope='Already observed development geometries; not new held-out evidence', execute=args.execute)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    for origin, name in [(args.study, 'study.json'), (args.base_suite, 'base_suite.json'), (args.config, 'experiment.yaml')]:
        shutil.copy2(origin, root/'inputs'/name)
    corridor.save(root/'inputs/ablation.json', options)
    for name in SOURCE_FILES:
        destination = root/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/name, destination)
    (root/'packages.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
        check=True, capture_output=True, text=True).stdout)
    corridor.save(root/'manifest.json', dict(status='prepared', created_at=corridor.now(), plan=plan, jobs=jobs,
        command=sys.argv if argv is None else argv,
        source_hashes={name: corridor.digest(root/'source'/name) for name in SOURCE_FILES},
        input_hashes={f'inputs/{name}': corridor.digest(root/'inputs'/name)
            for name in ('study.json', 'base_suite.json', 'experiment.yaml', 'ablation.json')}))
    try:
        with (root/'worker.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(root/'source/connected_prior_ablation.py'),
                '--worker-root', str(root), '--execute'], cwd=root/'source', stdout=log, stderr=subprocess.STDOUT, check=True)
        audit = subprocess.run([sys.executable, str(root/'source/tests/validate_connected_priors.py'), str(root)],
            capture_output=True, text=True)
        if audit.returncode:
            (root/'audit.log').write_text(audit.stdout+audit.stderr)
            raise RuntimeError(f'Audit failed: {root/"audit.log"}')
        corridor.save(root/'independent_validation.json', json.loads(audit.stdout))
    except BaseException as exc:
        manifest = json.loads((root/'manifest.json').read_text())
        manifest.update(status='failed', error=repr(exc))
        corridor.save(root/'manifest.json', manifest)
        raise
    print(f'Completed and audited: {root/"REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()

"""Matched source diversity, connection redundancy and mounting-exclusion study."""
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
from src.core.coverage_requirements import CoverageRequirements
from src.environment import ConnectedCorridorGeometry, get_lidar_profile, load_config
from src.methods.connected_pairs import assess_ids, canonical_local_search, exact_pairs
from src.methods.source_modes import diverse_source_alternatives, response_source_alternatives
from src.roadside.corridor_requirements import CorridorRequirements, validate_profile


BASE = Path(__file__).resolve().parent
METHODS = ('independent_transfer', 'top_joint', 'diverse_joint', 'top_joint_ls',
           'diverse_joint_ls', 'reference_one_per_module', 'reference_global')
SOURCE_FILES = corridor.SOURCE_FILES + [
    'connected_constraint_study.py', 'src/methods/source_modes.py',
    'src/roadside/__init__.py', 'src/roadside/corridor_requirements.py',
    'tests/validate_connected_corridor.py', 'tests/validate_connected_constraints.py',
]


def study_methods(response_pool=False):
    return METHODS[:5] + (('response_joint', 'response_joint_ls') if response_pool else ()) + METHODS[5:]


def validate_study(study, base):
    if not isinstance(study, dict):
        raise ValueError('Invalid connected constraint study schema/keys')
    held_out = study.get('schema') == 'connected_constraint_holdout_v1'
    fields = ({'targets'} if held_out else {'angles_deg', 'bend_length_m'})
    if (set(study) != {'schema', 'alternatives_per_site', 'profiles'} | fields
            or study['schema'] not in ('connected_constraint_study_v1', 'connected_constraint_holdout_v1')):
        raise ValueError('Invalid connected constraint study schema/keys')
    suite = deepcopy(base)
    suite['alternatives_per_site'] = study['alternatives_per_site']
    if not held_out:
        suite['geometry']['bend_length_m'] = study['bend_length_m']
        suite['angles_deg'] = study['angles_deg']
    corridor.validate_suite(suite)
    if held_out:
        targets = study['targets']
        if not isinstance(targets, list) or not targets or len(targets) > 24:
            raise ValueError('Nonempty held-out targets required')
        names, shapes = set(), set()
        for target in targets:
            if (not isinstance(target, dict) or set(target) != {'name', 'road_width_m', 'turn_deg', 'bend_length_m'}
                    or not isinstance(target['name'], str) or not target['name'].isascii()
                    or not target['name'].isidentifier() or target['name'] in names):
                raise ValueError('Invalid or duplicate held-out target name/keys')
            shape = tuple(target[k] for k in ('road_width_m', 'turn_deg', 'bend_length_m'))
            if shape in shapes:
                raise ValueError('Duplicate held-out geometry')
            names.add(target['name'])
            shapes.add(shape)
            target_suite = deepcopy(suite)
            target_suite['geometry'].update(road_width_m=target['road_width_m'], bend_length_m=target['bend_length_m'])
            target_suite['angles_deg'] = [target['turn_deg']]
            corridor.validate_suite(target_suite)
    modes = len(base['candidate']['heights_m']) * len(base['candidate']['relative_yaws_deg'])
    if study['alternatives_per_site'] > modes:
        raise ValueError('Diverse pool budget exceeds available height/yaw modes')
    profiles = study['profiles']
    count = len(study['targets']) if held_out else len(study['angles_deg'])
    if not isinstance(profiles, list) or not profiles or len(profiles)*count > 24:
        raise ValueError('Bounded study requires 1-24 conditions')
    names = []
    for profile in profiles:
        validate_profile(profile, base['geometry']['length_m'])
        names.append(profile['name'])
    if len(set(names)) != len(names):
        raise ValueError('Profile names must be unique')
    return suite


def study_jobs(study):
    if study['schema'] == 'connected_constraint_holdout_v1':
        return [dict(profile=p['name'], angle_deg=t['turn_deg'], target_name=t['name'],
            road_width_m=t['road_width_m'], bend_length_m=t['bend_length_m'],
            directory=f'case_{pi:02d}_{ti:02d}', status='pending')
            for pi, p in enumerate(study['profiles']) for ti, t in enumerate(study['targets'])]
    return [dict(profile=p['name'], angle_deg=a, directory=f'case_{pi:02d}_{ai:02d}', status='pending')
            for pi, p in enumerate(study['profiles']) for ai, a in enumerate(study['angles_deg'])]


def target_geometry(suite, job):
    parameters = {**suite['geometry'], 'turn_deg': job['angle_deg']}
    for key in ('road_width_m', 'bend_length_m'):
        if key in job:
            parameters[key] = job[key]
    return ConnectedCorridorGeometry(**parameters)


def transfer_source(source, geometry):
    """Preserve station, side, height and inward-relative attitude on new mount lines."""
    transferred = source.copy()
    transferred[:, 1] = np.sign(source[:, 1])*geometry.mounting_offset_m
    shifted = transferred.copy()
    shifted[:, 0] += geometry.length_m/2
    return np.vstack([transferred, shifted])


def no_pair_result(count, quota):
    return dict(status='infeasible_in_declared_grid', selected_ids=None, assessment=None,
        exhaustive=True, fixed_sensor_count=2, candidate_count=count, combinations_considered=0,
        quota=quota, reason='Fewer than two candidates survive individual installation constraints')


def run_case(root, job, suite, profile, sensor, source, pools, source_best):
    directory = root / job['directory']
    directory.mkdir(parents=True)
    started = time.perf_counter()
    geometry = target_geometry(suite, job)
    mesh = geometry.create_environment_mesh()
    targets, coordinates = geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
    canonical = transfer_source(source, geometry)
    poses = geometry.poses(canonical)
    modules = np.repeat([0, 1], len(source))
    req = CorridorRequirements(suite['requirements'], targets, coordinates, canonical, poses, profile,
        length_m=geometry.length_m, interface_half_length_m=suite['interface_half_length_m'])
    target_pools = {name: values + [i+len(source) for i in values] for name, values in pools.items()}
    allowed = req.eligible_ids(range(len(poses)))
    methods = study_methods('response' in pools)
    np.savez_compressed(directory / 'geometry.npz', vertices=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.triangles), targets=targets, target_coordinates=coordinates,
        canonical_candidates=canonical, poses=poses, module_ids=modules,
        **{f'{kind}_pool': values for kind, values in target_pools.items()}, allowed_candidates=req.allowed)
    corridor.save(directory / 'geometry.json', vars(geometry))
    corridor.save(directory / 'requirements.json', suite['requirements'])
    corridor.save(directory / 'corridor_profile.json', profile)
    outputs = {}
    for name in methods[:-2]:
        start = time.perf_counter()
        cache = corridor.MaskCache(mesh, targets, sensor, poses)
        if name == 'independent_transfer':
            initial = [source_best, source_best+len(source)]
            assessment = assess_ids(initial, poses, cache, req)
            result = dict(status='complete' if assessment.feasible else 'infeasible_transfer',
                          selected_ids=initial, assessment=assessment.to_dict())
        else:
            kind = name.split('_')[0]
            eligible = req.eligible_ids(target_pools[kind])
            result = (exact_pairs(eligible, poses, cache, req, module_ids=modules,
                        pair_mask_filter=req.filter_visibility_pairs) if len(eligible) >= 2
                      else no_pair_result(len(eligible), 'one_per_module'))
            result['source_pool_size'] = len(target_pools[kind])
            result['eligible_pool_size'] = len(eligible)
            if name.endswith('_ls'):
                initialization = deepcopy(result)
                if result['selected_ids'] is None:
                    result = dict(status='not_run_no_feasible_library_pair', selected_ids=None, assessment=None)
                else:
                    result = canonical_local_search(result['selected_ids'], canonical, poses, cache, req,
                        station_radius_m=suite['local_station_radius_m'])
                result['initialization'] = initialization
        result['online_sec'] = time.perf_counter()-start
        outputs[name] = corridor.persist_method(directory / name, result, canonical, poses, targets,
            coordinates, cache, mesh, sensor, req, suite)
    # All online choices are committed before creating the offline full mask archive.
    reference_start = time.perf_counter()
    cache = corridor.MaskCache(mesh, targets, sensor, poses)
    np.savez_compressed(directory / 'reference_masks.npz', masks=np.asarray([cache(i) for i in range(len(poses))]))
    build_sec = time.perf_counter()-reference_start
    for name, quota in [('reference_one_per_module', modules), ('reference_global', None)]:
        result = (exact_pairs(allowed, poses, cache, req, module_ids=quota,
                    pair_mask_filter=req.filter_visibility_pairs) if len(allowed) >= 2
                  else no_pair_result(len(allowed), 'one_per_module' if quota is not None else 'none'))
        result.update(declared_candidate_count=len(poses), eligible_candidate_count=len(allowed),
            reference_build_sec=build_sec, reference_masks_sha256=corridor.digest(directory / 'reference_masks.npz'),
            omitted_candidates='Individual corridor installation exclusions; tested again by final evaluator')
        outputs[name] = corridor.persist_method(directory / name, result, canonical, poses, targets,
            coordinates, cache, mesh, sensor, req, suite)
    corridor.plot_case(directory, mesh, targets, outputs, poses, job['angle_deg'], methods=methods)
    corridor.save(directory / 'manifest.json', dict(status='complete', profile=profile['name'],
        angle_deg=job['angle_deg'], geometry_sha256=corridor.digest(directory / 'geometry.npz'),
        profile_sha256=corridor.digest(directory / 'corridor_profile.json'),
        completed_at=corridor.now(), case_wall_sec=time.perf_counter()-started))


def summarize(root, jobs, response_pool=False):
    methods = study_methods(response_pool)
    rows, pairs = [], []
    for job in jobs:
        if job['status'] != 'complete':
            continue
        directory = root / job['directory']
        results = {name: json.loads((directory / name / 'result.json').read_text()) for name in methods}
        reference = results['reference_global']['assessment']
        identity = {key: job[key] for key in ('target_name', 'road_width_m', 'bend_length_m') if key in job}
        for name, result in results.items():
            a = result['assessment']
            feasible = a is not None and a['feasible']
            gap = reference['objective_score']-a['objective_score'] if reference and feasible else None
            if gap is not None and gap < -1e-7:
                raise RuntimeError('Feasible deployment exceeds exhaustive reference')
            rows.append(dict(profile=job['profile'], angle_deg=job['angle_deg'], **identity, method=name,
                status=result['status'], feasible=feasible, violations=';'.join(a['violations']) if a else '',
                coverage_pct=a['coverage_pct'] if a else None, objective_score=a['objective_score'] if a else None,
                objective_gap=gap, interface_coverage_pct=result.get('interface_coverage_pct'),
                interface_overlap_ratio=result.get('interface_overlap_ratio'),
                candidate_queries=result['work']['unique_candidate_queries'],
                rays_submitted=result['work'].get('rays_submitted', 0), online_sec=result.get('online_sec'),
                result_json=str((directory / name / 'result.json').relative_to(root))))
        top, diverse = results['top_joint']['assessment'], results['diverse_joint']['assessment']
        global_feasible = reference is not None
        pairs.append(dict(profile=job['profile'], angle_deg=job['angle_deg'], **identity,
            global_feasible=global_feasible, top_feasible=top is not None, diverse_feasible=diverse is not None,
            top_j=top['objective_score'] if top else None, diverse_j=diverse['objective_score'] if diverse else None,
            diverse_minus_top_j=diverse['objective_score']-top['objective_score'] if top and diverse else None,
            reference_j=reference['objective_score'] if reference else None,
            top_query_count=results['top_joint']['work']['unique_candidate_queries'],
            diverse_query_count=results['diverse_joint']['work']['unique_candidate_queries']))
        if response_pool:
            response = results['response_joint']['assessment']
            pairs[-1].update(response_feasible=response is not None,
                response_j=response['objective_score'] if response else None,
                response_query_count=results['response_joint']['work']['unique_candidate_queries'])
    if not rows:
        return
    for filename, values in [('comparison.csv', rows), ('paired.csv', pairs)]:
        with (root / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    corridor.save(root / 'summary.json', dict(conditions=len(pairs), expected_conditions=len(jobs),
        method_rows=len(rows), feasible_deployments=sum(r['feasible'] for r in rows),
        saved_deployments=sum(r['coverage_pct'] is not None for r in rows),
        pairs=pairs, note='Same source count and same legal per-site query budgets; '
            + ('observed development geometries; source-response baseline' if response_pool else
               'fixed-rule held-out synthetic geometries' if pairs[0].get('target_name') else 'deterministic development cases')))
    if response_pool:
        return summarize_response(root, rows, pairs)
    lines = ['# Connection Constraints and Source Diversity', '',
        'Fixed K=2, one source, matched top-score and geometric-mode source pools.',
        'Profiles change connection-band redundancy and road-relative installation eligibility.',
        'Infeasible transfer is saved with violations; absent feasible library results are not zero scores.', '',
        '| Profile | Target | Angle | Top J | Diverse J | Difference | Global reference J | Queries top/diverse |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for pair in pairs:
        def display(key):
            return '-' if pair[key] is None else f'{pair[key]:.3f}'
        lines.append(f"| {pair['profile']} | {pair.get('target_name', 'development')} | {pair['angle_deg']:g} | {display('top_j')} | "
            f"{display('diverse_j')} | {display('diverse_minus_top_j')} | {display('reference_j')} | "
            f"{pair['top_query_count']}/{pair['diverse_query_count']} |")
    lines.extend(['', 'All numerical requirements are illustrative. No MAPPO or detection is evaluated.',
        'Reference optima apply only to the declared fixed-K, sampled targets, coarse candidate grid and static mesh.',
        'Inspect comparison.csv for coverage, interface overlap, violations and query/ray counts.', ''])
    (root / 'REPORT.md').write_text('\n'.join(lines))


def summarize_response(root, rows, pairs):
    budget = json.loads((root / 'manifest.json').read_text())['plan']['source_alternatives_per_site']
    lines = ['# Source Coverage Response Baseline', '',
        f'Observed development conditions, not a new held-out evaluation. Fixed K=2; {budget} alternatives per site.',
        'Source single-sensor J and pairwise mask intersections define marginal-J responses.',
        'Contexts: empty, or any source sensor at another station/side. No target masks enter selection.',
        'Seed: highest single J. Add the candidate minimizing worst response-envelope regret,',
        'then mean regret, then highest single J, then lowest ID. This is a greedy proxy, not a guarantee.',
        'Response selection does not encode target hard constraints or certify transfer error.', '',
        '| Method | Feasible | Matches full-grid J | Mean feasible J gap |',
        '|---|---:|---:|---:|']
    for name in study_methods(True):
        values = [r for r in rows if r['method'] == name and r['feasible']]
        gaps = [r['objective_gap'] for r in values if r['objective_gap'] is not None]
        mean = f'{np.mean(gaps):.6f}' if gaps else '-'
        lines.append(f'| {name} | {len(values)}/{len(pairs)} | '
            f'{sum(abs(g) <= 1e-7 for g in gaps)}/{len(pairs)} | {mean} |')
    lines.extend(['', 'J gaps are not coverage percentage points. Feasible-only means exclude failures.',
        'LS preserves original installation side and station anchors; its additional queries are not matched.',
        'Joint selectors must match legal candidate-query counts, not necessarily rays or wall time.',
        'All thresholds are illustrative; no detection, MAPPO, continuous optimum or real-road claim.', '',
        '| Profile | Target | Top J | Diverse J | Response J | Reference J | Queries top/diverse/response |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for pair in pairs:
        def fmt(key):
            return '-' if pair[key] is None else f'{pair[key]:.3f}'
        lines.append(f"| {pair['profile']} | {pair.get('target_name', pair['angle_deg'])} | "
            f"{fmt('top_j')} | {fmt('diverse_j')} | {fmt('response_j')} | {fmt('reference_j')} | "
            f"{pair['top_query_count']}/{pair['diverse_query_count']}/{pair['response_query_count']} |")
    (root / 'REPORT.md').write_text('\n'.join(lines)+'\n')


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
    suite = validate_study(study, base)
    cfg = load_config(root / 'inputs/experiment.yaml')
    sensor = get_lidar_profile(cfg.lidar_profile, cfg)
    response_pool = manifest['plan'].get('response_pool', False)
    started = time.perf_counter()
    try:
        manifest.update(status='running', stage='source_library', worker_started_at=corridor.now())
        corridor.save(root / 'manifest.json', manifest)
        source, top, best = corridor.source_library(suite, sensor, root / 'library')
        with np.load(root / 'library/library.npz', allow_pickle=False) as data:
            diverse = diverse_source_alternatives(source, data['scores'], study['alternatives_per_site'])
        pools = dict(top=top, diverse=diverse)
        selection = dict(pools=pools, best_source_id=best,
            alternatives_per_site=study['alternatives_per_site'],
            diversity='Best source pitch per height/yaw mode; start at best source J; normalized '
                      'height/yaw max-min selection; ties use source J then candidate ID')
        if response_pool:
            weights = suite['requirements'].get('weights', {})
            coefficient = (weights.get('coverage_weight', 1) + weights.get('importance_weight', 0)
                           + weights.get('overlap_penalty', 0))
            start = time.perf_counter()
            with np.load(root / 'library/library.npz', allow_pickle=False) as data:
                pools['response'], traces = response_source_alternatives(source, data['masks'], data['scores'],
                    study['alternatives_per_site'], intersection_weight=coefficient)
            selection.update(response_rule='source_marginal_minimax_greedy_v1',
                response_intersection_weight=coefficient, response_traces=traces,
                response_selection_sec=time.perf_counter()-start,
                response_scope='Empty or one other source-site sensor; uniform source target importance; '
                    'no hard-feasibility or target-response guarantee')
        corridor.save(root / 'source_pools.json', selection)
        manifest.update(source_pools_sha256=corridor.digest(root / 'source_pools.json'),
            source_library_sha256=corridor.digest(root / 'library/library.npz'),
            source_selection_frozen_at=corridor.now())
        corridor.save(root / 'manifest.json', manifest)
        profiles = {profile['name']: profile for profile in study['profiles']}
        manifest['target_queries_started_at'] = corridor.now()
        for job in manifest['jobs']:
            job['status'] = 'running'
            manifest.update(stage='target_case', current_job=job['directory'])
            corridor.save(root / 'manifest.json', manifest)
            print(f"Start {job['profile']} angle={job['angle_deg']:g}", flush=True)
            run_case(root, job, suite, profiles[job['profile']], sensor, source, pools, best)
            job['status'] = 'complete'
            corridor.save(root / 'manifest.json', manifest)
            summarize(root, manifest['jobs'], response_pool)
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
    parser.add_argument('--study', default=str(BASE / 'configs/connected_constraint_study.json'))
    parser.add_argument('--base-suite', default=str(BASE / 'configs/connected_corridor_pilot.json'))
    parser.add_argument('--config', default=str(BASE / 'configs/experiment.yaml'))
    parser.add_argument('--output', default='results/connected_constraint_study')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--response-pool', action='store_true',
                        help='Add matched-budget source marginal-response candidates (development baseline)')
    parser.add_argument('--worker-root', help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return worker(Path(args.worker_root).resolve())
    base = corridor.load_suite(args.base_suite)
    study = json.loads(Path(args.study).read_text())
    suite = validate_study(study, base)
    geometry = ConnectedCorridorGeometry(**{**suite['geometry'], 'length_m': suite['geometry']['length_m']/2,
                                           'bend_length_m': 0})
    source = corridor.candidates(geometry, suite['candidate'])
    config = load_config(args.config)
    sensor = get_lidar_profile(config.lidar_profile, config)
    jobs = study_jobs(study)
    methods = study_methods(args.response_pool)
    plan = dict(conditions=len(jobs), method_runs=len(jobs)*len(methods), methods=methods,
        source_builds=1, source_candidates=len(source), target_candidates=2*len(source),
        source_alternatives_per_site=study['alternatives_per_site'], profiles=study['profiles'],
        sensor=vars(sensor),
        source_geometry=vars(geometry), study_schema=study['schema'], execute=args.execute)
    if args.response_pool:
        plan.update(response_pool=True, evaluation_scope='Observed development geometries; not new held-out evidence',
                    response_rule='source_marginal_minimax_greedy_v1')
    if 'targets' in study:
        plan['targets'] = study['targets']
    else:
        plan.update(angles_deg=study['angles_deg'], bend_length_m=study['bend_length_m'])
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'inputs').mkdir()
    for origin, name in [(args.study, 'study.json'), (args.base_suite, 'base_suite.json'),
                         (args.config, 'experiment.yaml')]:
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
    with (root / 'worker.log').open('w') as log:
        completed = subprocess.run([sys.executable, '-u', str(root / 'source/connected_constraint_study.py'),
            '--worker-root', str(root), '--execute'], cwd=root / 'source', stdout=log, stderr=subprocess.STDOUT)
    try:
        if completed.returncode:
            raise RuntimeError(f'Worker failed; see {root / "worker.log"}')
        audit = subprocess.run([sys.executable, str(root / 'source/tests/validate_connected_constraints.py'),
            str(root)], capture_output=True, text=True)
        if audit.returncode:
            (root / 'audit.log').write_text(audit.stdout+audit.stderr)
            raise RuntimeError(f'Independent audit failed; see {root / "audit.log"}')
        corridor.save(root / 'independent_validation.json', json.loads(audit.stdout))
    except BaseException as exc:
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest.update(status='failed', error=repr(exc))
        corridor.save(root / 'manifest.json', manifest)
        raise
    print(f'Completed and independently audited: {root / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()

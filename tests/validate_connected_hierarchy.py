"""Read-only independent pair/metric checks plus recorded-policy replay.

Policy replay shares selector code and is not an independent proof of that rule.
Physical masks come from saved arrays; this audit performs no new raycasting.
"""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import connected_hierarchy_study as runner
import connected_constraint_study as constraints
from src.environment import get_lidar_profile, load_config
from src.methods.connected_hierarchy import BudgetedObservations, SelectionContext, refine_poses, select_pool
from src.roadside.corridor_requirements import CorridorRequirements
from validate_connected_corridor import digest, exhaustive_values, require
from validate_connected_constraints import allowed_candidates, check_value, deployment_metrics, response_pool


def read(path):
    return json.loads(path.read_text())


def observations(path, masks, expected_hash):
    require(digest(path) == expected_hash, 'Observation archive hash')
    with np.load(path, allow_pickle=False) as data:
        ids = data['candidate_ids'].tolist()
        require(ids == sorted(set(ids)) and all(isinstance(i, int) and 0 <= i < len(masks) for i in ids),
                'Invalid observation IDs')
        require(np.array_equal(data['masks'], masks[ids]), 'Measured masks differ from later offline archive')
    return {i: masks[i] for i in ids}


def audit(root):
    root = Path(root)
    manifest = read(root / 'manifest.json')
    require(manifest['status'] == 'complete', 'Study must be complete')
    require(manifest['plan']['methods'] == list(runner.methods()), 'Changed method set')
    for name, expected in manifest['source_hashes'].items():
        require(digest(root / 'source' / name) == expected, f'Frozen source: {name}')
    for name, expected in manifest['input_hashes'].items():
        require(digest(root / name) == expected, f'Frozen input: {name}')
    base, study = read(root / 'inputs/base_suite.json'), read(root / 'inputs/study.json')
    suite = constraints.validate_study(study, base)
    wanted = constraints.study_jobs(study)
    require([{k: v for k, v in j.items() if k != 'status'} for j in manifest['jobs']]
            == [{k: v for k, v in j.items() if k != 'status'} for j in wanted], 'Changed/duplicate job grid')
    sensor = get_lidar_profile(load_config(root / 'inputs/experiment.yaml').lidar_profile,
                               load_config(root / 'inputs/experiment.yaml'))
    spec = suite['requirements']
    require(digest(root / 'library/library.npz') == manifest['source_library_sha256'], 'Source library hash')
    require(digest(root / 'source_selection.json') == manifest['source_selection_sha256'], 'Source selection hash')
    with np.load(root / 'library/library.npz', allow_pickle=False) as data:
        source, scores, source_masks = data['canonical'], data['scores'], data['masks']
    meta = read(root / 'library/metadata.json')
    require(meta['geometry'] == {**suite['geometry'], 'length_m': suite['geometry']['length_m']/2,
        'turn_deg': 0, 'bend_length_m': 0}, 'Source geometry changed')
    weights, cost = spec.get('weights', {}), spec.get('cost', {})
    expected_scores = ((weights.get('coverage_weight', 1)+weights.get('importance_weight', 0))*source_masks.sum(axis=1)
        - weights.get('sensor_count_penalty', 0)-weights.get('installation_cost_weight', 0)
        * (cost.get('base_per_sensor', 0)+cost.get('per_height_m', 0)*source[:, 2]))
    require(np.allclose(scores, expected_scores, atol=1e-8, rtol=0), 'Source scores changed')
    coefficient = weights.get('coverage_weight', 1)+weights.get('importance_weight', 0)+weights.get('overlap_penalty', 0)
    ids, traces = response_pool(source, source_masks, scores, suite['alternatives_per_site'], coefficient)
    selection = read(root / 'source_selection.json')
    require(selection['candidate_ids'] == ids and selection['traces'] == traces
            and selection['coefficient'] == coefficient, 'Source response rule mismatch')
    transferred_ids = ids+[i+len(source) for i in ids]
    profiles = {p['name']: p for p in study['profiles']}
    checked, pool_checks, reference_checks, lower_checks = 0, 0, 0, 0
    all_results, case_best = {}, {}
    for job in manifest['jobs']:
        require(job['status'] == 'complete', 'Incomplete case')
        directory = root / job['directory']
        cm = read(directory / 'manifest.json')
        require(cm['status'] == 'complete' and cm['online_committed_at'] <= cm['reference_started_at'], 'Online/reference order')
        require(digest(directory / 'geometry.npz') == cm['geometry_sha256'], 'Geometry hash')
        require(digest(directory / 'reference_masks.npz') == cm['reference_masks_sha256'], 'Reference hash')
        expected_online = {f'{runner.method_name(s, a, t)}/{f}' for s in runner.SELECTORS for a in runner.ALLOCATIONS
            for t in runner.STAGES for f in ('result.json', 'observations.npz')}
        expected_online |= {f'selection_{s}/{f}' for s in runner.SELECTORS for f in ('selection.json', 'observations.npz')}
        require(expected_online <= set(cm['online_hashes']), 'Missing committed online files')
        for path, expected in cm['online_hashes'].items():
            require(digest(directory / path) == expected, f'Committed online file changed: {path}')
        profile = profiles[job['profile']]
        require(read(directory / 'corridor_profile.json') == profile, 'Profile changed')
        require(read(directory / 'requirements.json') == spec, 'Requirements changed')
        geometry = constraints.target_geometry(suite, job)
        require(read(directory / 'geometry.json') == vars(geometry), 'Geometry definition changed')
        with np.load(directory / 'geometry.npz', allow_pickle=False) as data:
            canonical, poses, modules = data['canonical_candidates'], data['poses'], data['module_ids']
            targets, coordinates = data['targets'], data['target_coordinates']
            saved_allowed = data['allowed_candidates']
        require(np.array_equal(canonical, constraints.transfer_source(source, geometry)), 'Road-relative transfer mismatch')
        require(np.array_equal(poses, geometry.poses(canonical)), 'World pose mismatch')
        actual_targets, actual_coordinates = geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
        require(np.array_equal(targets, actual_targets) and np.array_equal(coordinates, actual_coordinates), 'Target grid mismatch')
        require(np.array_equal(modules, np.repeat([0, 1], len(source))), 'Module IDs changed')
        req = CorridorRequirements(spec, targets, coordinates, canonical, poses, profile,
            length_m=geometry.length_m, interface_half_length_m=suite['interface_half_length_m'])
        context = SelectionContext(canonical, poses, targets, np.tile(scores, 2), req, sensor)
        allowed = allowed_candidates(canonical, profile)
        lo, hi = spec.get('constraints', {}).get('mounting_z_range_m', [0, float('inf')])
        allowed &= (poses[:, 2] >= lo-1e-9) & (poses[:, 2] <= hi+1e-9)
        require(np.array_equal(saved_allowed, allowed), 'Eligibility mismatch')
        with np.load(directory / 'reference_masks.npz', allow_pickle=False) as data:
            masks = data['masks']
        require(masks.dtype == bool and masks.shape == (len(poses), len(targets)), 'Full mask layout')
        interface = np.abs(coordinates[:, 0]-geometry.length_m/2) <= suite['interface_half_length_m']
        opts = dict(interface_mask=interface, interface_thresholds=profile['interface'])
        best, _ = exhaustive_values(poses, masks, modules, spec, candidate_allowed=allowed, **opts)
        case_best[job['directory']] = best
        results = {name: read(directory / name / 'result.json') for name in runner.methods()}
        for selector in runner.SELECTORS:
            sd = directory / f'selection_{selector}'
            saved = read(sd / 'selection.json')
            measured = observations(sd / 'observations.npz', masks, saved['observations_sha256'])
            require(selection['committed_at'] <= saved['committed_at'] <= cm['online_committed_at'], 'Source/selection chronology')
            # Replay cannot access unknown masks: only the saved measured dictionary is passed.
            replay_obs = BudgetedObservations(context, lambda i: measured[i], per_site=suite['alternatives_per_site'])
            replay = select_pool(selector, context, replay_obs, suite['alternatives_per_site'], static_ids=transferred_ids)
            for key in ('candidate_ids', 'events', 'decisions', 'contexts', 'kind', 'optical_candidate_evaluations', 'optical_point_tests'):
                require(saved[key] == replay[key], f'Recorded selection replay mismatch: {key}')
            require(len(measured) == saved['work']['unique_candidate_queries'], 'Selection work mismatch')
            require(saved['work'].get('direct_pose_evaluations', 0) == len(measured), 'Unmetered selection query')
            for group in context.groups:
                if group:
                    require(sum(i in measured for i in group) == suite['alternatives_per_site'], 'Per-site budget mismatch')
            known = set()
            for event in saved['events']:
                require(event['known_before'] == len(known) and event['candidate_id'] not in known, 'Invalid query chronology')
                known.add(event['candidate_id'])
            for snapshot in saved['contexts']:
                prior = {e['candidate_id'] for e in saved['events'] if e['phase'] == 'seed'
                    or e['phase'].startswith('round_') and int(e['phase'].split('_')[1]) < snapshot['round']}
                require(set(snapshot['measured_context_ids']) == prior, 'Unmeasured/same-round context used')
                for d in saved['decisions']:
                    if d['round'] == snapshot['round'] and d['measured_partner_id'] is not None:
                        require(d['measured_partner_id'] in prior, 'Unknown partner used')
            pool_ids = saved['candidate_ids']
            pool_best, _ = exhaustive_values(poses[pool_ids], masks[pool_ids], modules[pool_ids], spec, **opts)
            for allocation in runner.ALLOCATIONS:
                raw = results[runner.method_name(selector, allocation, 'raw')]
                key = 'global' if allocation == 'free' else 'one_per_module'
                check_value(raw['assessment']['objective_score'] if raw['assessment'] else None, pool_best[key], 'Pool optimum')
                pool_checks += 1
                require(raw['selected_ids'] is None or set(raw['selected_ids']) <= set(pool_ids), 'Upper escaped queried pool')
                if allocation == 'one_per_module' and raw['selected_ids'] is not None:
                    require(len(set(modules[raw['selected_ids']])) == 2, 'Upper module quota')
                for stage in runner.STAGES:
                    name = runner.method_name(selector, allocation, stage)
                    r = results[name]
                    require(r['selection_sha256'] == digest(sd / 'selection.json'), 'Selection provenance')
                    require((r['selector'], r['upper_allocation'], r['stage']) == (selector, allocation, stage), 'Method identity')
                    observed = observations(directory/name/'observations.npz', masks, r['observations_sha256'])
                    require(all(i in observed for i in measured), 'Lost initial observations')
                    if stage == 'raw':
                        require(set(observed) == set(measured) and not r['lower_events'], 'Raw used extra queries')
                    else:
                        initial = r['initialization']
                        require(initial['selected_ids'] == raw['selected_ids'] and initial['assessment'] == raw['assessment'], 'Lower initialization')
                        cap = manifest['plan']['lower_extra_queries']
                        require(r['lower_query_limit'] == cap, 'Changed lower query budget')
                        lower_obs = BudgetedObservations(context, lambda i: masks[i], max_extra=cap,
                            initial=measured, initial_work=saved['work'])
                        lower = refine_poses(initial, context, lower_obs,
                            station_radius_m=0 if stage == 'fixed_site' else suite['local_station_radius_m'])
                        for field in lower:
                            expected = json.loads(json.dumps(lower[field]))
                            require(r[field] == expected, f'Lower replay mismatch: {field}')
                        require(r['lower_events'] == lower_obs.events and set(observed) == set(lower_obs.masks), 'Lower query replay')
                        require(len(observed)-len(measured) == len(r['lower_events']) <= cap, 'Extra query cap')
                        require(r['work']['base_candidate_queries'] == len(measured)
                            and r['work']['extra_candidate_queries'] == len(r['lower_events']), 'Lower accounting')
                        if r['selected_ids'] is not None:
                            anchors = canonical[initial['selected_ids'], :2]
                            require(np.array_equal(canonical[r['selected_ids'], 1], anchors[:, 1]), 'Lower changed side')
                            require(np.all(np.abs(canonical[r['selected_ids'], 0]-anchors[:, 0]) <=
                                (0 if stage == 'fixed_site' else suite['local_station_radius_m'])+1e-9), 'Lower escaped anchor')
                            trace = r['trace']
                            require(trace[0]['selected_ids'] == initial['selected_ids']
                                and trace[-1]['selected_ids'] == r['selected_ids'], 'Lower trace endpoints')
                            for step_id, step in enumerate(trace):
                                move_ids = step['selected_ids']
                                step_best, _ = exhaustive_values(poses[move_ids], masks[move_ids], modules[move_ids], spec, **opts)
                                check_value(step['objective_score'], step_best['global'], 'Lower trace feasible J')
                                require(set(move_ids) <= set(observed), 'Lower trace used unmeasured pose')
                                if step_id:
                                    previous = trace[step_id-1]
                                    a_ids, b_ids = previous['selected_ids'], move_ids
                                    changed = [i for i in range(2) if a_ids[i] != b_ids[i]]
                                    require(len(changed) == 1, 'Lower changed multiple sensors')
                                    a, b = canonical[a_ids[changed[0]]], canonical[b_ids[changed[0]]]
                                    axes = np.flatnonzero(a != b)
                                    require(len(axes) == 1 and axes[0] in (0, 2, 3, 4), 'Lower changed side/multiple coordinates')
                                    levels = np.unique(canonical[:, axes[0]])
                                    require(abs(np.searchsorted(levels, a[axes[0]])-np.searchsorted(levels, b[axes[0]])) == 1,
                                        'Lower skipped adjacent coordinate')
                                    require(step['objective_score'] > previous['objective_score']+1e-8, 'Lower trace did not improve J')
                        lower_checks += 1
                    require(r['work']['unique_candidate_queries'] == len(observed)
                        and r['work'].get('direct_pose_evaluations', 0) == len(observed), 'Unmetered physical query')
        for name, result in results.items():
            all_results[job['directory'], name] = result
            selected = result['selected_ids']
            if name in runner.REFERENCES:
                key = name.removeprefix('reference_')
                check_value(result['assessment']['objective_score'] if result['assessment'] else None, best[key], 'Full reference optimum')
                reference_checks += 1
            if selected is None:
                require(result['assessment'] is None and not (directory/name/'deployment.npz').exists(), 'Absent result scored')
                continue
            require(len(selected) == 2 and len(set(selected)) == 2 and allowed[selected].all(), 'Invalid deployment IDs')
            require(result['validated_direct_raycast'] is True, 'Fresh-scene check missing')
            require(digest(directory/name/'deployment.npz') == result['deployment_sha256'], 'Deployment hash')
            with np.load(directory/name/'deployment.npz', allow_pickle=False) as data:
                for key, value in [('candidate_ids', selected), ('poses', poses[selected]), ('canonical', canonical[selected]),
                    ('targets', targets), ('target_coordinates', coordinates), ('sensor_masks', masks[selected]),
                    ('density', masks[selected].sum(axis=0))]:
                    require(np.array_equal(data[key], value), f'Deployment array mismatch: {key}')
            pair_best, _ = exhaustive_values(poses[selected], masks[selected], modules[selected], spec, **opts)
            require(result['assessment']['feasible'] and pair_best['global'] is not None, 'Infeasible saved placement')
            check_value(result['assessment']['objective_score'], pair_best['global'], 'Deployment J')
            check_value(result['assessment']['coverage_pct'], 100*np.mean(masks[selected].any(axis=0)), 'Coverage')
            deployment_metrics(result, poses, masks, coordinates, interface, spec, geometry.length_m/2, suite['target_step_m'])
            if name not in runner.REFERENCES:
                require(result['final_allocation'] == runner.allocation_signature(selected, canonical, modules), 'Allocation signature')
            checked += 1
    with (root/'comparison.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(all_results), 'CSV row count')
    seen = set()
    for row in rows:
        key = row['case'], row['method']
        require(key in all_results and key not in seen, 'Duplicate/unknown CSV row')
        seen.add(key)
        r, best = all_results[key], case_best[key[0]]['global']
        a = r['assessment']
        expected = dict(feasible=a is not None and a['feasible'], objective_score=a['objective_score'] if a else None,
            objective_gap=best-a['objective_score'] if best is not None and a else None,
            coverage_pct=a['coverage_pct'] if a else None, interface_overlap_ratio=r.get('interface_overlap_ratio'),
            candidate_queries=r['work']['unique_candidate_queries'], rays_submitted=r['work'].get('rays_submitted', 0),
            extra_queries=r['work']['extra_candidate_queries'] if r.get('stage') in ('fixed_site', 'radius_control') else 0,
            stopping_reason=r.get('stopping_reason', ''))
        for field, value in expected.items():
            check_value(row[field], value, f'CSV {field}')
    summary = read(root/'summary.json')
    require(summary['conditions'] == len(manifest['jobs']) and summary['rows'] == len(rows), 'Summary completeness')
    for name, value in summary['methods'].items():
        group = [r for (case, method), r in all_results.items() if method == name]
        gaps = [case_best[case]['global']-r['assessment']['objective_score'] for (case, method), r in all_results.items()
            if method == name and r['assessment'] and case_best[case]['global'] is not None]
        expected = dict(feasible=sum(r['assessment'] is not None for r in group), matches_global=sum(abs(g) <= 1e-7 for g in gaps),
            mean_feasible_j_gap=float(np.mean(gaps)) if gaps else None,
            mean_queries=float(np.mean([r['work']['unique_candidate_queries'] for r in group])),
            budget_limited=sum(r.get('stopping_reason') == 'query_budget_limited' for r in group))
        require(value == expected, 'Summary aggregate mismatch')
    return dict(conditions=len(manifest['jobs']), method_rows=len(rows), checked_deployments=checked,
        reference_optima_checked=reference_checks, queried_pool_optima_checked=pool_checks,
        lower_policy_replays=lower_checks, selection_policy_replays=3*len(manifest['jobs']),
        scope='Independent unpruned pair optima and deployment metrics; shared-code recorded policy replay. '
              'Hashes, per-site/extra budgets and chronology checked. No new raycasts or physical proof.',
        validator_sha256=digest(__file__))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(audit(parser.parse_args().root), indent=2, allow_nan=False))

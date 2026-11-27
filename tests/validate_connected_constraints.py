"""Independent full enumeration of saved corridor constraint study arrays."""
from argparse import ArgumentParser
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np

from validate_connected_corridor import digest, exhaustive_values, require


METHODS = ('independent_transfer', 'top_joint', 'diverse_joint', 'top_joint_ls',
           'diverse_joint_ls', 'reference_one_per_module', 'reference_global')


def source_pools(canonical, scores, count):
    top, diverse = [], []
    for site in np.unique(canonical[:, :2], axis=0):
        ids = np.flatnonzero(np.all(canonical[:, :2] == site, axis=1))
        order = sorted(ids.tolist(), key=lambda i: (-scores[i], i))
        top.extend(order[:count])
        by_mode = {}
        for i in order:
            by_mode.setdefault((canonical[i, 2], canonical[i, 3]), i)
        representatives = list(by_mode.values())
        chosen = [order[0]]
        height_scale = float(np.ptp(canonical[ids, 2])) or 1.
        yaw_scale = float(np.ptp(canonical[ids, 3])) or 1.
        while len(chosen) < count:
            distances = {i: min(((canonical[i, 2]-canonical[j, 2])/height_scale)**2
                               + ((canonical[i, 3]-canonical[j, 3])/yaw_scale)**2 for j in chosen)
                         for i in representatives if i not in chosen}
            winner = sorted(distances, key=lambda i: (-distances[i], -scores[i], i))[0]
            chosen.append(winner)
        diverse.extend(chosen)
    return dict(top=top, diverse=diverse)


def response_pool(canonical, masks, scores, count, coefficient):
    """Independent per-context counts and exhaustive greedy additions, no selector import."""
    chosen_ids, traces = [], []
    for site in np.unique(canonical[:, :2], axis=0):
        ids = [i for i, p in enumerate(canonical) if np.array_equal(p[:2], site)]
        contexts = [None] + [i for i, p in enumerate(canonical) if not np.array_equal(p[:2], site)]
        responses = {i: np.array([scores[i] if j is None else
            scores[i]-coefficient*np.count_nonzero(masks[i] & masks[j]) for j in contexts]) for i in ids}
        optimum = np.max(list(responses.values()), axis=0)
        chosen = [sorted(ids, key=lambda i: (-scores[i], i))[0]]
        history = []
        for step in range(count):
            if step:
                ranked = []
                for i in ids:
                    if i in chosen:
                        continue
                    envelope = np.max([responses[j] for j in [*chosen, i]], axis=0)
                    regret = np.maximum(optimum-envelope, 0)
                    ranked.append((float(regret.max()), float(regret.mean()), -float(scores[i]), i))
                chosen.append(sorted(ranked)[0][-1])
            regret = np.maximum(optimum-np.max([responses[i] for i in chosen], axis=0), 0)
            history.append(dict(selected_id=chosen[-1], worst_regret=float(regret.max()),
                                mean_regret=float(regret.mean())))
        chosen_ids.extend(chosen)
        traces.append(dict(site=site.tolist(), context_count=len(contexts), context_ids=contexts, steps=history))
    return chosen_ids, traces


def allowed_candidates(canonical, profile):
    allowed = np.ones(len(canonical), dtype=bool)
    for zone in profile['forbidden_installation']:
        low, high = zone['station_range_m']
        for i, row in enumerate(canonical):
            same_side = (zone['side'] == 'both' or zone['side'] == 'left' and row[1] > 0
                         or zone['side'] == 'right' and row[1] < 0)
            if low <= row[0] <= high and same_side:
                allowed[i] = False
    return allowed


def check_value(actual, expected, context):
    if expected is None:
        require(actual is None or actual == '', context)
    elif isinstance(expected, bool):
        require(actual == expected or actual == str(expected), context)
    elif isinstance(expected, (int, float, np.integer, np.floating)):
        require(actual is not None and actual != '' and np.isfinite(float(actual))
                and np.isclose(float(actual), expected, atol=1e-8, rtol=1e-12), context)
    else:
        require(actual == expected, context)


def check_method_contract(name, result, results, pools, modules, allowed, best_source, source_count, radius):
    """Validate provenance/domain independently of whether its J happens to be optimal."""
    ids = result['selected_ids']
    if ids is not None:
        require(isinstance(ids, list) and len(ids) == 2 and len(set(ids)) == 2
                and all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(modules) for i in ids),
                'Invalid selected candidate IDs')
    if name == 'independent_transfer':
        require(ids == [best_source, best_source+source_count], 'Independent transfer source provenance mismatch')
    elif name.endswith('_ls'):
        original = results[name.removesuffix('_ls')]
        initial = result['initialization']
        for field in ('selected_ids', 'assessment', 'status', 'quota', 'candidate_count',
                      'source_pool_size', 'eligible_pool_size', 'combinations_considered'):
            require(initial.get(field) == original.get(field), f'LS initialization provenance mismatch: {field}')
        if original['selected_ids'] is None:
            require(ids is None and result['status'] == 'not_run_no_feasible_library_pair', 'LS should not run without a pair')
        else:
            require(ids is not None and result['station_radius_m'] == radius, 'Changed LS domain/radius')
            require(result['assessment']['objective_score']+1e-8 >= original['assessment']['objective_score'],
                    'LS decreased objective')
            trace = result['trace']
            require(bool(trace), 'Missing LS trace')
            check_value(trace[0]['objective_score'], original['assessment']['objective_score'], 'LS initial trace J')
            check_value(trace[-1]['objective_score'], result['assessment']['objective_score'], 'LS final trace J')
            require(all(b['objective_score'] > a['objective_score']+1e-8
                        and b['elapsed_sec'] >= a['elapsed_sec'] for a, b in zip(trace, trace[1:])), 'Invalid LS trace')
    elif name.endswith('_joint'):
        kind = name.split('_')[0]
        if ids is not None:
            require(set(ids) <= set(pools[kind]) and bool(np.asarray(allowed)[ids].all()),
                    'Joint selection escaped its eligible source pool')
            require(modules[ids[0]] != modules[ids[1]], 'Joint selection violates one-per-module quota')
    elif name == 'reference_one_per_module' and ids is not None:
        require(modules[ids[0]] != modules[ids[1]], 'Reference violates one-per-module quota')


def deployment_metrics(result, poses, masks, coordinates, interface, spec, half_length, step):
    ids = result['selected_ids']
    selected = masks[ids]
    density = selected.sum(axis=0)
    a = result['assessment']
    cost = spec.get('cost', {})
    expected = dict(covered_count=int(np.count_nonzero(density)),
        weighted_target_coverage=float(np.count_nonzero(density)),
        overlap=float(np.maximum(density-1, 0).sum()), sensor_count=2,
        installation_cost=float(2*cost.get('base_per_sensor', 0)+cost.get('per_height_m', 0)*poses[ids, 2].sum()))
    require(set(a['components']) == set(expected), 'Unexpected objective components')
    for key, value in expected.items():
        check_value(a['components'][key], value, f'Assessment component mismatch: {key}')
    check_value(a['overlap_ratio'], np.mean(density >= 2), 'Assessment global overlap mismatch')
    band = dict(target_count=int(interface.sum()), coverage_pct=100*float(np.mean(density[interface] > 0)),
                overlap_ratio=float(np.mean(density[interface] >= 2)))
    for key, value in band.items():
        check_value(a['regions']['interface'][key], value, f'Assessment interface {key} mismatch')
    check_value(result['interface_coverage_pct'], band['coverage_pct'], 'Interface coverage mismatch')
    check_value(result['interface_overlap_ratio'], band['overlap_ratio'], 'Interface overlap mismatch')
    if 'min_single_failure_coverage_pct' in spec.get('constraints', {}):
        after_failure = 100*selected[::-1].mean(axis=1)
        check_value(a['robustness']['single_failure_coverage_pct'], after_failure.min(), 'Single-failure metric mismatch')
        require(np.allclose(a['robustness']['coverage_after_each_failure_pct'], after_failure,
                            atol=1e-8, rtol=0), 'Per-sensor failure metric mismatch')
    module_coverage = [100*float(np.mean(density[part] > 0)) for part in
                       (coordinates[:, 0] < half_length, coordinates[:, 0] >= half_length)]
    require(np.allclose(result['module_coverage_pct'], module_coverage, atol=1e-8, rtol=0), 'Module coverage mismatch')
    longest = 0
    for lateral_height in np.unique(coordinates[:, 1:], axis=0):
        indices = np.flatnonzero(np.all(coordinates[:, 1:] == lateral_height, axis=1))
        indices = indices[np.argsort(coordinates[indices, 0])]
        run = 0
        for missing in density[indices] == 0:
            run = run+1 if missing else 0
            longest = max(longest, run)
    check_value(result['max_sampled_blind_run_m'], longest*step, 'Sampled blind run mismatch')


def audit(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['status'] == 'complete', 'Study must be complete')
    for name, expected in manifest['source_hashes'].items():
        require(digest(root / 'source' / name) == expected, f'Frozen source hash: {name}')
    for name, expected in manifest['input_hashes'].items():
        require(digest(root / name) == expected, f'Frozen input hash: {name}')
    study = json.loads((root / 'inputs/study.json').read_text())
    base = json.loads((root / 'inputs/base_suite.json').read_text())
    held_out = study['schema'] == 'connected_constraint_holdout_v1'
    response_enabled = manifest['plan'].get('response_pool', False)
    methods = METHODS[:5] + (('response_joint', 'response_joint_ls') if response_enabled else ()) + METHODS[5:]
    require(list(manifest['plan']['methods']) == list(methods), 'Changed declared method set')
    spec = base['requirements']
    profiles = {p['name']: p for p in study['profiles']}
    identity_field = 'target_name' if held_out else 'angle_deg'
    targets_by_name = {t['name']: t for t in study.get('targets', [])}
    wanted = set(product(profiles, targets_by_name if held_out else study['angles_deg']))
    actual = [(j['profile'], j[identity_field]) for j in manifest['jobs']]
    require(len(actual) == len(wanted) and set(actual) == wanted, 'Incomplete or duplicate condition grid')
    metadata = json.loads((root / 'library/metadata.json').read_text())
    require(digest(root / 'library/library.npz') == metadata['source_pose_mask_hash'], 'Source library hash')
    source_geometry = {**base['geometry'], 'length_m': base['geometry']['length_m']/2,
                       'turn_deg': 0, 'bend_length_m': 0}
    require(metadata['geometry'] == source_geometry, 'Target geometry leaked into source library')
    if held_out or 'source_selection_frozen_at' in manifest:
        require(manifest['source_library_sha256'] == digest(root / 'library/library.npz'), 'Frozen library changed')
        require(manifest['source_pools_sha256'] == digest(root / 'source_pools.json'), 'Frozen source pools changed')
        require(manifest['source_selection_frozen_at'] <= manifest['target_queries_started_at'],
                'Source selection was not frozen before target queries')
    with np.load(root / 'library/library.npz', allow_pickle=False) as data:
        source, scores, masks = data['canonical'], data['scores'], data['masks']
    w, cost = spec.get('weights', {}), spec.get('cost', {})
    expected_scores = ((w.get('coverage_weight', 1)+w.get('importance_weight', 0))*masks.sum(axis=1)
        - w.get('sensor_count_penalty', 0) - w.get('installation_cost_weight', 0)
        * (cost.get('base_per_sensor', 0)+cost.get('per_height_m', 0)*source[:, 2]))
    require(np.allclose(scores, expected_scores, atol=1e-8, rtol=0), 'Source ranking J mismatch')
    pools = source_pools(source, scores, study['alternatives_per_site'])
    selections = json.loads((root / 'source_pools.json').read_text())
    if response_enabled:
        coefficient = w.get('coverage_weight', 1)+w.get('importance_weight', 0)+w.get('overlap_penalty', 0)
        require(selections['response_rule'] == 'source_marginal_minimax_greedy_v1', 'Unknown response rule')
        require(selections['response_intersection_weight'] == coefficient, 'Response coefficient mismatch')
        pools['response'], traces = response_pool(source, masks, scores, study['alternatives_per_site'], coefficient)
        require(selections['response_traces'] == traces, 'Response contexts/regret traces mismatch')
    require(selections['pools'] == pools, 'Source pools were not selected by the declared source-only rules')
    require(selections['best_source_id'] == int(np.argmax(scores)), 'Source winner changed')
    checked, feasible, reference_count, pool_references = 0, 0, 0, 0
    results_by_key, cases, geometry_truth = {}, [], {}
    for job in manifest['jobs']:
        require(job['status'] == 'complete', 'Unfinished condition')
        directory = root / job['directory']
        cm = json.loads((directory / 'manifest.json').read_text())
        profile = json.loads((directory / 'corridor_profile.json').read_text())
        require(profile == profiles[job['profile']], 'Changed profile')
        require(digest(directory / 'corridor_profile.json') == cm['profile_sha256'], 'Profile hash')
        require(digest(directory / 'geometry.npz') == cm['geometry_sha256'], 'Geometry hash')
        require(json.loads((directory / 'requirements.json').read_text()) == spec, 'Changed global requirements')
        geometry = json.loads((directory / 'geometry.json').read_text())
        expected_geometry = {**base['geometry'], 'turn_deg': job['angle_deg']}
        if held_out:
            t = targets_by_name[job['target_name']]
            require(job['angle_deg'] == t['turn_deg'] and job['road_width_m'] == t['road_width_m']
                    and job['bend_length_m'] == t['bend_length_m'], 'Changed held-out job parameters')
            expected_geometry.update(road_width_m=t['road_width_m'], bend_length_m=t['bend_length_m'])
        else:
            expected_geometry['bend_length_m'] = study['bend_length_m']
        require(geometry == expected_geometry, 'Incorrect turn geometry')
        with np.load(directory / 'geometry.npz', allow_pickle=False) as data:
            poses, canonical, modules = data['poses'], data['canonical_candidates'], data['module_ids']
            targets, coordinates = data['targets'], data['target_coordinates']
            transferred = source.copy()
            transferred[:, 1] = np.sign(source[:, 1])*(geometry['road_width_m']/2+geometry['sidewalk_width_m']/2)
            second = transferred.copy()
            second[:, 0] += geometry['length_m']/2
            require(np.array_equal(canonical, np.vstack([transferred, second])), 'Incorrect road-relative source transfer')
            require(np.array_equal(modules, np.repeat([0, 1], len(source))), 'Incorrect module ownership')
            allowed = allowed_candidates(canonical, profile)
            require(np.array_equal(allowed, data['allowed_candidates']), 'Installation mask mismatch')
            target_pools = {k: v+[i+len(source) for i in v] for k, v in pools.items()}
            for kind in pools:
                require(data[f'{kind}_pool'].tolist() == target_pools[kind], 'Transferred pool mismatch')
        with np.load(directory / 'reference_masks.npz', allow_pickle=False) as data:
            masks = data['masks']
        require(masks.dtype == bool and masks.shape == (len(poses), len(targets)), 'Reference mask layout')
        truth = [poses, canonical, targets, coordinates, masks]
        identity = job[identity_field]
        if identity in geometry_truth:
            require(all(np.array_equal(a, b) for a, b in zip(truth, geometry_truth[identity])),
                    'Constraint profile changed physical geometry/visibility')
        else:
            geometry_truth[identity] = truth
        interface = np.abs(coordinates[:, 0]-base['geometry']['length_m']/2) <= base['interface_half_length_m']
        options = dict(candidate_allowed=allowed, interface_mask=interface, interface_thresholds=profile['interface'])
        best, feasible_pairs = exhaustive_values(poses, masks, modules, spec, **options)
        results = {}
        for name in methods:
            r = json.loads((directory / name / 'result.json').read_text())
            results[name] = r
            results_by_key[job['profile'], identity, name] = r
            check_method_contract(name, r, results, target_pools, modules, allowed,
                selections['best_source_id'], len(source), base['local_station_radius_m'])
            if name.startswith('reference_'):
                require(digest(directory / 'reference_masks.npz') == r['reference_masks_sha256'], 'Reference mask hash')
                key = name.removeprefix('reference_')
                actual_j = r['assessment']['objective_score'] if r['assessment'] else None
                require((actual_j is None and best[key] is None) or (actual_j is not None and best[key] is not None
                    and abs(actual_j-best[key]) <= 1e-7), f'Independent optimum mismatch: {name}')
                n0, n1 = int(np.count_nonzero(allowed & (modules == 0))), int(np.count_nonzero(allowed & (modules == 1)))
                count = int(allowed.sum())
                expected_pairs = count*(count-1)//2 if key == 'global' else n0*n1
                require(r['combinations_considered'] == expected_pairs, 'Incorrect eligible pair domain')
                reference_count += 1
            if name in tuple(f'{kind}_joint' for kind in pools):
                kind = name.split('_')[0]
                ids = target_pools[kind]
                subset, _ = exhaustive_values(poses[ids], masks[ids], modules[ids], spec,
                    candidate_allowed=allowed[ids], interface_mask=interface, interface_thresholds=profile['interface'])
                reported = r['assessment']['objective_score'] if r['assessment'] else None
                require((reported is None and subset['one_per_module'] is None)
                    or (reported is not None and subset['one_per_module'] is not None
                        and abs(reported-subset['one_per_module']) <= 1e-7), 'Incorrect source-pool optimum')
                require(r['work']['unique_candidate_queries'] == int(allowed[ids].sum()), 'Unexpected pool query count')
                pool_references += 1
            if r['selected_ids'] is None:
                require(r['assessment'] is None and not (directory/name/'deployment.npz').exists(),
                        'Absent selection has assessment/deployment')
                continue
            archive = directory / name / 'deployment.npz'
            require(digest(archive) == r['deployment_sha256'], 'Deployment hash')
            require(r['validated_direct_raycast'] is True, 'Missing fresh-scene validation')
            ids = r['selected_ids']
            require(len(ids) == 2 and len(set(ids)) == 2, 'Not two real distinct sensors')
            with np.load(archive, allow_pickle=False) as data:
                for field, value in [('candidate_ids', ids), ('poses', poses[ids]), ('canonical', canonical[ids]),
                    ('targets', targets), ('target_coordinates', coordinates),
                    ('sensor_masks', masks[ids]), ('density', masks[ids].sum(axis=0))]:
                    require(np.array_equal(data[field], value), f'Deployment {field} mismatch')
                subset, _ = exhaustive_values(poses[ids], masks[ids], modules[ids], spec,
                    candidate_allowed=allowed[ids], interface_mask=interface, interface_thresholds=profile['interface'])
                require((subset['global'] is not None) == r['assessment']['feasible'], 'Feasibility mismatch')
                loose = {**spec, 'constraints': {}}
                raw, _ = exhaustive_values(poses[ids], masks[ids], modules[ids], loose)
                require(abs(raw['global']-r['assessment']['objective_score']) <= 1e-7, 'Objective mismatch')
                density = data['density']
                require(abs(100*np.mean(density > 0)-r['assessment']['coverage_pct']) <= 1e-9, 'Coverage mismatch')
                require(abs(np.mean(density[interface] >= 2)-r['interface_overlap_ratio']) <= 1e-9, 'Interface overlap mismatch')
            deployment_metrics(r, poses, masks, coordinates, interface, spec,
                               base['geometry']['length_m']/2, base['target_step_m'])
            if name.endswith('_ls'):
                initial = r['initialization']['selected_ids']
                require(np.all(np.abs(canonical[ids, 0]-canonical[initial, 0]) <= base['local_station_radius_m']+1e-9),
                        'Local search escaped original station anchors')
                require(np.array_equal(canonical[ids, 1], canonical[initial, 1]), 'Local search switched installation sides')
            checked += 1
            feasible += int(r['assessment']['feasible'])
        require(results['top_joint']['work']['unique_candidate_queries'] == results['diverse_joint']['work']['unique_candidate_queries'],
                'Top/diverse query budgets differ')
        if response_enabled:
            require(results['top_joint']['work']['unique_candidate_queries'] == results['response_joint']['work']['unique_candidate_queries'],
                    'Top/response query budgets differ')
        cases.append(dict(profile=job['profile'], angle_deg=job['angle_deg'], **({'target_name': identity} if held_out else {}),
            optimum=best, feasible_pairs=feasible_pairs,
            top_feasible=results['top_joint']['assessment'] is not None,
            diverse_feasible=results['diverse_joint']['assessment'] is not None))
    with (root / 'comparison.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(manifest['jobs'])*len(methods), 'Incorrect method report size')
    seen = set()
    for row in rows:
        key = (row['profile'], row['target_name'] if held_out else float(row['angle_deg']), row['method'])
        require(key not in seen and key in results_by_key, 'Duplicate/unknown comparison row')
        seen.add(key)
        r = results_by_key[key]
        a = r['assessment']
        require(row['feasible'] == str(a is not None and a['feasible']), 'Reported feasibility mismatch')
        if a:
            require(abs(float(row['objective_score'])-a['objective_score']) <= 1e-7, 'Reported objective mismatch')
        reference = results_by_key[key[0], key[1], 'reference_global']['assessment']
        expected = dict(status=r['status'], coverage_pct=a['coverage_pct'] if a else None,
            objective_score=a['objective_score'] if a else None,
            objective_gap=reference['objective_score']-a['objective_score'] if reference and a and a['feasible'] else None,
            violations=';'.join(a['violations']) if a else '', interface_coverage_pct=r.get('interface_coverage_pct'),
            interface_overlap_ratio=r.get('interface_overlap_ratio'), candidate_queries=r['work']['unique_candidate_queries'],
            rays_submitted=r['work'].get('rays_submitted', 0), online_sec=r.get('online_sec'))
        for field, value in expected.items():
            check_value(row[field], value, f'Reported {field} mismatch')
        job = next(j for j in manifest['jobs'] if (j['profile'], j[identity_field]) == key[:2])
        check_value(row['angle_deg'], job['angle_deg'], 'Reported angle mismatch')
        for field in ('road_width_m', 'bend_length_m'):
            if field in job:
                check_value(row[field], job[field], f'Reported {field} mismatch')
        require(row['result_json'] == str(Path(job['directory'])/key[2]/'result.json'), 'Reported result path mismatch')
    with (root / 'paired.csv').open(newline='') as handle:
        paired = list(csv.DictReader(handle))
    summary = json.loads((root / 'summary.json').read_text())
    require(len(paired) == len(manifest['jobs']) == len(summary['pairs']), 'Incomplete paired/summary report')
    paired_seen = set()
    for row in paired:
        key = row['profile'], row['target_name'] if held_out else float(row['angle_deg'])
        require(key not in paired_seen, 'Duplicate paired row')
        paired_seen.add(key)
        values = {kind: results_by_key[*key, f'{kind}_joint'] for kind in pools}
        reference = results_by_key[*key, 'reference_global']['assessment']
        expected = dict(global_feasible=reference is not None, reference_j=reference['objective_score'] if reference else None)
        for kind, value in values.items():
            expected.update({f'{kind}_feasible': value['assessment'] is not None,
                f'{kind}_j': value['assessment']['objective_score'] if value['assessment'] else None,
                f'{kind}_query_count': value['work']['unique_candidate_queries']})
        expected['diverse_minus_top_j'] = (expected['diverse_j']-expected['top_j']
            if expected['diverse_j'] is not None and expected['top_j'] is not None else None)
        match = [p for p in summary['pairs'] if (p['profile'], p[identity_field]) == key]
        require(len(match) == 1, 'Duplicate/unknown summary pair')
        job = next(j for j in manifest['jobs'] if (j['profile'], j[identity_field]) == key)
        expected.update({field: job[field] for field in ('angle_deg', 'target_name', 'road_width_m', 'bend_length_m') if field in job})
        for field, value in expected.items():
            check_value(row[field], value, f'Paired {field} mismatch')
            check_value(match[0][field], value, f'Summary paired {field} mismatch')
    for field, value in dict(conditions=len(manifest['jobs']), expected_conditions=len(manifest['jobs']),
                             method_rows=len(rows), feasible_deployments=feasible, saved_deployments=checked).items():
        check_value(summary[field], value, f'Summary {field} mismatch')
    return dict(conditions=len(cases), method_rows=len(rows), checked_deployments=checked,
        feasible_deployments=feasible, reference_optima_checked=reference_count,
        source_pool_optima_checked=pool_references, all_reference_optima_match=True,
        equal_legal_pool_query_counts=True, unchanged_geometry_across_profiles=True,
        fixed_source_geometry_verified=True, road_relative_transfer_verified=True,
        source_response_selection_verified=response_enabled,
        method_provenance_and_pool_membership_verified=True, detailed_deployment_metrics_verified=True,
        comparison_paired_summary_verified=True,
        cases=cases, validator_sha256=digest(__file__),
        independent_pair_validator_sha256=digest(Path(__file__).with_name('validate_connected_corridor.py')),
        manifest_sha256=digest(root / 'manifest.json'),
        scope='Saved arrays and hashes; full independent constrained pair enumeration; no new raycasts')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, allow_nan=False))

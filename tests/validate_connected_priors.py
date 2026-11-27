"""Saved-array audit: independent prior scores, pair optima and report statistics.

Recorded selection replay shares implementation code; no new physical raycasts.
"""
from argparse import ArgumentParser
from copy import deepcopy
import csv
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import connected_prior_ablation as runner
import connected_constraint_study as constraints
from src.environment import ConnectedCorridorGeometry, compute_view_angles_numpy, get_lidar_profile, load_config
from src.methods.connected_hierarchy import BudgetedObservations, SelectionContext, select_pool
from validate_connected_constraints import allowed_candidates, check_value, deployment_metrics
from validate_connected_corridor import digest, exhaustive_values, require


def read(path):
    return json.loads(path.read_text())


def optical_prior(poses, targets, sensor, allowed, spec):
    """Independent 360-degree tilted-plane/range calculation, not shared FOV code."""
    require(sensor.h_fov_deg == 360, 'Independent prior audit supports 360-degree sensors')
    counts = np.zeros(len(poses))
    for cid in np.flatnonzero(allowed):
        p = poses[cid]
        yaw, pitch = np.deg2rad(p[3:5])
        normal = np.array([-np.sin(pitch)*np.cos(yaw), -np.sin(pitch)*np.sin(yaw), np.cos(pitch)])
        vectors = targets-p[:3]
        distance = np.linalg.norm(vectors, axis=1)
        optical = ((distance > 1e-6) & (distance <= sensor.max_range_m)
            & (np.abs(vectors @ normal)/np.maximum(distance, 1e-30) <= np.sin(np.deg2rad(sensor.v_fov_deg/2))))
        counts[cid] = optical.sum()
    w, c = spec.get('weights', {}), spec.get('cost', {})
    return ((w.get('coverage_weight', 1)+w.get('importance_weight', 0))*counts
        - w.get('sensor_count_penalty', 0)-w.get('installation_cost_weight', 0)
        * (c.get('base_per_sensor', 0)+c.get('per_height_m', 0)*poses[:, 2]))


def ray_counts(poses, targets, sensor):
    """Accounting replay with shared float32 optics, not independent LOS physics."""
    rays = []
    for p in poses.astype(np.float32):
        vectors = targets.astype(np.float32)-p[:3]
        az, el, fwd, dist = compute_view_angles_numpy(*vectors.T, p[3], p[4])
        hh, hv = np.deg2rad([sensor.h_fov_deg, sensor.v_fov_deg])/2
        mask = (np.abs(az) <= hh) & (np.abs(el) <= hv) & (dist <= sensor.max_range_m) & (dist > 1e-6)
        if hh < np.pi-1e-6:
            mask &= fwd > .5
        rays.append(int(mask.sum()))
    return np.asarray(rays)


def check_deployment(path, result, poses, canonical, targets, coordinates, modules, masks, allowed, spec, opts, suite):
    ids = result['selected_ids']
    if ids is None:
        require(result['assessment'] is None and not (path/'deployment.npz').exists(), 'Absent deployment scored')
        return 0
    require(len(ids) == 2 and len(set(ids)) == 2 and all(type(i) is int and 0 <= i < len(poses) for i in ids)
        and allowed[ids].all(), 'Invalid deployment IDs')
    require(result['validated_direct_raycast'] is True, 'Missing fresh-scene validation')
    require(digest(path/'deployment.npz') == result['deployment_sha256'], 'Deployment hash')
    with np.load(path/'deployment.npz', allow_pickle=False) as data:
        for key, value in [('candidate_ids', ids), ('poses', poses[ids]), ('canonical', canonical[ids]),
            ('targets', targets), ('target_coordinates', coordinates), ('sensor_masks', masks[ids]),
            ('density', masks[ids].sum(axis=0))]:
            require(np.array_equal(data[key], value), f'Deployment array mismatch: {key}')
    best, _ = exhaustive_values(poses[ids], masks[ids], modules[ids], spec, **opts)
    require(result['assessment']['feasible'] and best['global'] is not None, 'Infeasible saved deployment')
    check_value(result['assessment']['objective_score'], best['global'], 'Deployment J')
    check_value(result['assessment']['coverage_pct'], 100*masks[ids].any(axis=0).mean(), 'Deployment coverage')
    deployment_metrics(result, poses, masks, coordinates, opts['interface_mask'], spec,
                       suite['geometry']['length_m']/2, suite['target_step_m'])
    return 1


def check_upper_metadata(result, candidate_count):
    """The certificate covers the measured pair domain, never the continuous road."""
    present = result['selected_ids'] is not None
    if candidate_count < 2:
        require(not present and result['status'] == 'infeasible_in_queried_pool'
            and result['quota'] == 'free', 'Empty-domain certificate')
        return
    require(result['exhaustive'] is True and result['fixed_sensor_count'] == 2
        and result['candidate_count'] == candidate_count
        and result['combinations_considered'] == comb(candidate_count, 2)
        and result['quota'] == 'none' and result['allocation_mode'] == 'free', 'Pair-domain certificate')
    require(result['status'] == ('optimal_in_declared_grid' if present else 'infeasible_in_declared_grid'),
        'Pair-domain status')
    for key in ('full_constraint_assessments', 'visibility_screened_pairs'):
        require(type(result[key]) is int and 0 <= result[key] <= comb(candidate_count, 2), 'Pair work counter')
    require(np.isfinite(result['elapsed_sec']) and result['elapsed_sec'] >= 0, 'Pair elapsed time')


def audit(root):
    root = Path(root)
    manifest = read(root/'manifest.json')
    require(manifest['status'] == 'complete', 'Study must be complete')
    require(set(manifest['source_hashes']) == set(runner.SOURCE_FILES), 'Incomplete frozen source inventory')
    require(set(manifest['input_hashes']) == {f'inputs/{name}' for name in
        ('study.json', 'base_suite.json', 'experiment.yaml', 'ablation.json')}, 'Incomplete frozen input inventory')
    for name, expected in manifest['source_hashes'].items():
        require(digest(root/'source'/name) == expected, f'Frozen source: {name}')
    for name, expected in manifest['input_hashes'].items():
        require(digest(root/name) == expected, f'Frozen input: {name}')
    options = runner.validate_options(read(root/'inputs/ablation.json'))
    study = read(root/'inputs/study.json')
    suite = constraints.validate_study(study, read(root/'inputs/base_suite.json'))
    wanted = constraints.study_jobs(study)
    require([{k: v for k, v in j.items() if k != 'status'} for j in manifest['jobs']]
        == [{k: v for k, v in j.items() if k != 'status'} for j in wanted], 'Changed/duplicate condition grid')
    require(manifest['plan']['arms'] == runner.arms(options) and manifest['plan']['lower_refinement'] is False,
        'Changed arm set/lower stage')
    config = load_config(root/'inputs/experiment.yaml')
    sensor = get_lidar_profile(config.lidar_profile, config)
    runner.validate_scope(suite, sensor)
    expected_plan = dict(conditions=len(wanted), arms=runner.arms(options),
        online_outputs=len(wanted)*len(runner.arms(options)), references=len(wanted),
        lower_refinement=False, fast_prediction_screen=True, sensor=vars(sensor),
        evaluation_scope='Already observed development geometries; not new held-out evidence', execute=True)
    require(manifest['plan'] == expected_plan, 'Plan counts/scope/sensor mismatch')
    require(digest(root/'library/library.npz') == manifest['source_library_sha256'], 'Source library hash')
    metadata = read(root/'library/metadata.json')
    require(metadata['geometry'] == {**suite['geometry'], 'length_m': suite['geometry']['length_m']/2,
        'turn_deg': 0, 'bend_length_m': 0}, 'Source geometry leaked target changes')
    with np.load(root/'library/library.npz', allow_pickle=False) as data:
        source, scores, source_masks = data['canonical'], data['scores'], data['masks']
        source_geometry = ConnectedCorridorGeometry(**metadata['geometry'])
        source_targets, source_coordinates = source_geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
        for key, value in [('canonical', runner.corridor.candidates(source_geometry, suite['candidate'])),
            ('poses', source_geometry.poses(source)), ('targets', source_targets), ('target_coordinates', source_coordinates)]:
            require(np.array_equal(data[key], value), f'Source geometry array: {key}')
        alternative_ids = data['alternative_ids'].tolist()
    require(source_masks.dtype == bool and source_masks.shape == (len(source), len(source_targets)), 'Source mask layout')
    require(manifest['source_candidate_count'] == metadata['source_candidate_count'] == len(source)
        and metadata['source_pose_mask_hash'] == manifest['source_library_sha256'], 'Source metadata counts/hash')
    spec, w, c = suite['requirements'], suite['requirements'].get('weights', {}), suite['requirements'].get('cost', {})
    expected = ((w.get('coverage_weight', 1)+w.get('importance_weight', 0))*source_masks.sum(axis=1)
        - w.get('sensor_count_penalty', 0)-w.get('installation_cost_weight', 0)
        * (c.get('base_per_sensor', 0)+c.get('per_height_m', 0)*source[:, 2]))
    require(np.allclose(scores, expected, atol=1e-8, rtol=0), 'Source single-J scores')
    source_spec = deepcopy(spec)
    source_spec.update(budget_k=1, constraints={})
    require(metadata['requirements'] == source_spec and metadata['best_source_id'] == int(np.argmax(expected)),
        'Source requirements/winner')
    alternatives = []
    for site in np.unique(source[:, :2], axis=0):
        group = np.flatnonzero(np.all(source[:, :2] == site, axis=1)).tolist()
        alternatives.extend(sorted(group, key=lambda i: (-expected[i], i))[:suite['alternatives_per_site']])
    require(alternative_ids == alternatives and metadata['alternative_count'] == len(alternative_ids),
        'Source alternative inventory')
    source_rays = ray_counts(source_geometry.poses(source), source_targets, sensor)
    require(metadata['work'] == dict(unique_candidate_queries=len(source), direct_pose_evaluations=len(source),
        raycast_calls=int(np.count_nonzero(source_rays)), rays_submitted=int(source_rays.sum())), 'Source physical work')
    profiles = {p['name']: p for p in study['profiles']}
    results, references, pools, deployments, policy_replays = {}, {}, 0, 0, 0
    for job in manifest['jobs']:
        require(job['status'] == 'complete', 'Incomplete case')
        directory = root/job['directory']
        cm = read(directory/'manifest.json')
        require(cm['status'] == 'complete' and cm['online_committed_at'] <= cm['reference_started_at'], 'Online/reference order')
        for file, key in [('geometry.npz', 'geometry_sha256'), ('reference_masks.npz', 'reference_masks_sha256')]:
            require(digest(directory/file) == cm[key], f'Case {file} hash')
        required = {f"{a['name']}/{f}" for a in runner.arms(options)
            for f in ('selection.json', 'observations.npz', 'result.json')}
        required.update(f"{a['name']}/deployment.npz" for a in runner.arms(options)
            if read(directory/a['name']/'result.json')['selected_ids'] is not None)
        require(required == set(cm['online_hashes']), 'Missing/unexpected online commitment')
        for path, expected_hash in cm['online_hashes'].items():
            require(digest(directory/path) == expected_hash, f'Online hash: {path}')
        profile = profiles[job['profile']]
        geometry, context, coordinates, modules = runner.case_context(suite, job, profile, sensor)
        require(read(directory/'geometry.json') == vars(geometry), 'Geometry definition')
        require(read(directory/'corridor_profile.json') == profile and read(directory/'requirements.json') == spec,
            'Changed constraint specification')
        canonical, poses, targets = context.canonical, context.poses, context.targets
        mesh = geometry.create_environment_mesh()
        require(np.array_equal(canonical, constraints.transfer_source(source, geometry)), 'Source/target grid correspondence')
        with np.load(directory/'geometry.npz', allow_pickle=False) as data:
            for name, value in [('canonical_candidates', canonical), ('poses', poses), ('targets', targets),
                ('target_coordinates', coordinates), ('module_ids', modules),
                ('vertices', np.asarray(mesh.vertices)), ('triangles', np.asarray(mesh.triangles))]:
                require(np.array_equal(data[name], value), f'Geometry array: {name}')
            saved_allowed = data['allowed_candidates']
        allowed = allowed_candidates(canonical, profile)
        lo, hi = spec.get('constraints', {}).get('mounting_z_range_m', [0, float('inf')])
        allowed &= (poses[:, 2] >= lo-1e-9) & (poses[:, 2] <= hi+1e-9)
        require(np.array_equal(saved_allowed, allowed), 'Eligibility mask')
        eligible = np.flatnonzero(allowed)
        with np.load(directory/'reference_masks.npz', allow_pickle=False) as data:
            require(data['candidate_ids'].tolist() == eligible.tolist(), 'Reference queried ineligible/omitted legal candidate')
            compact = data['masks']
        require(compact.dtype == bool and compact.shape == (len(eligible), len(targets)), 'Reference mask layout')
        # Zeros for unqueried forbidden poses are lookup placeholders, never evidence of visibility.
        masks = np.zeros((len(poses), len(targets)), bool)
        masks[eligible] = compact
        interface = np.abs(coordinates[:, 0]-geometry.length_m/2) <= suite['interface_half_length_m']
        opts = dict(interface_mask=interface, interface_thresholds=profile['interface'])
        best, _ = exhaustive_values(poses[eligible], compact, modules[eligible], spec, **opts)
        references[job['directory']] = best['global']
        reference = read(directory/'reference_global/result.json')
        check_upper_metadata(reference, len(eligible))
        require(reference['declared_candidates'] == len(poses) and reference['eligible_candidates'] == len(eligible),
            'Reference domain counts')
        check_value(reference['assessment']['objective_score'] if reference['assessment'] else None, best['global'], 'Full eligible optimum')
        rays = ray_counts(poses, targets, sensor)
        require(reference['work']['unique_candidate_queries'] == len(eligible)
            and reference['work'].get('direct_pose_evaluations', 0) == len(eligible)
            and reference['work'].get('raycast_calls', 0) == int(np.count_nonzero(rays[eligible]))
            and reference['work'].get('rays_submitted', 0) == int(rays[eligible].sum()), 'Reference eligible query work')
        for field in ('reference_build_sec', 'decision_sec', 'geometry_shared_sec', 'online_component_sum_sec'):
            require(np.isfinite(reference[field]) and reference[field] >= 0, 'Invalid reference timing')
        require(reference['decision_sec']+1e-8 >= reference['reference_build_sec']+reference.get('elapsed_sec', 0),
            'Reference timing boundary')
        check_value(reference['online_component_sum_sec'], reference['geometry_shared_sec']+reference['decision_sec'],
            'Reference timing component sum')
        deployments += check_deployment(directory/'reference_global', reference, poses, canonical, targets,
            coordinates, modules, masks, allowed, spec, opts, suite)
        geometry_scores = optical_prior(poses, targets, sensor, allowed, spec)
        prefixes = {}
        for arm in runner.arms(options):
            path = directory/arm['name']
            r, selection = read(path/'result.json'), read(path/'selection.json')
            for k, v in arm.items():
                require(r[k] == v, f'Arm identity: {k}')
            require(r['selection_sha256'] == digest(path/'selection.json'), 'Selection link')
            require(manifest['source_frozen_at'] <= selection['committed_at'] <= cm['online_committed_at'], 'Source/query chronology')
            require(selection['probe_prior'] == arm['prior'] and selection['random_seed'] == arm['random_seed']
                and selection['fast_screen'] is True, 'Changed prior or conditional rule')
            require(r['optical_point_tests'] == selection['optical_point_tests'], 'Optical work provenance')
            expected_prior = (np.tile(scores, 2) if arm['prior'] == 'source' else geometry_scores if arm['prior'] == 'geometry'
                else np.random.default_rng(arm['random_seed']).permutation(len(poses)).astype(float))
            require(np.allclose(selection['prior_values'], expected_prior, atol=1e-8, rtol=0), 'Independent prior values mismatch')
            require(digest(path/'observations.npz') == selection['observations_sha256'], 'Observation hash')
            with np.load(path/'observations.npz', allow_pickle=False) as data:
                ids = data['candidate_ids'].tolist()
                require(ids == sorted(set(ids)) and all(type(i) is int and 0 <= i < len(poses) and allowed[i] for i in ids),
                    'Invalid queried IDs')
                require(np.array_equal(data['masks'], masks[ids]), 'Measured masks disagree with later reference')
            require(selection['candidate_ids'] == ids, 'Pool vs observations')
            check_upper_metadata(r, len(ids))
            for group in context.groups:
                if group:
                    require(len(set(group) & set(ids)) == arm['query_budget'], 'Matched per-site budget')
            work = r['work']
            require(work == selection['work'] and work['unique_candidate_queries'] == len(ids)
                and work.get('direct_pose_evaluations', 0) == len(ids)
                and work.get('rays_submitted', 0) == int(rays[ids].sum())
                and work.get('raycast_calls', 0) == int(np.count_nonzero(rays[ids])), 'Unmetered query/ray/call work')
            neutral = SelectionContext(canonical, poses, targets,
                np.tile(scores, 2) if arm['prior'] == 'source' else np.zeros(len(poses)), context.requirements, sensor)
            measured = {i: masks[i] for i in ids}
            replay_obs = BudgetedObservations(neutral, lambda i: measured[i], per_site=arm['query_budget'])
            replay = select_pool('probe_context', neutral, replay_obs, arm['query_budget'],
                probe_prior=arm['prior'], random_seed=arm['random_seed'], fast_screen=True)
            for key in replay:
                if key != 'selection_sec':
                    require(selection[key] == replay[key], f'Recorded policy replay mismatch: {key}')
            known = set()
            for event in selection['events']:
                require(event['candidate_id'] not in known and event['known_before'] == len(known), 'Query chronology')
                known.add(event['candidate_id'])
            for snapshot in selection['contexts']:
                prior_ids = {e['candidate_id'] for e in selection['events'] if e['phase'] == 'seed'
                    or e['phase'].startswith('round_') and int(e['phase'][6:]) < snapshot['round']}
                require(set(snapshot['measured_context_ids']) == prior_ids, 'Unmeasured/same-round context')
                require(all(d['measured_partner_id'] in prior_ids for d in selection['decisions']
                    if d['round'] == snapshot['round'] and d['measured_partner_id'] is not None), 'Unknown measured partner')
            sequence = [e['candidate_id'] for e in selection['events']]
            key = arm['prior'], arm['random_seed']
            if key in prefixes:
                old_sequence, old_r = prefixes[key]
                require(sequence[:len(old_sequence)] == old_sequence, 'Budget ladder is not a nested policy prefix')
                if old_r['assessment']:
                    require(r['assessment'] and r['assessment']['objective_score']+1e-8 >= old_r['assessment']['objective_score'],
                        'More queried candidates lost a feasible incumbent')
            prefixes[key] = sequence, r
            pool_best, _ = exhaustive_values(poses[ids], masks[ids], modules[ids], spec, **opts)
            check_value(r['assessment']['objective_score'] if r['assessment'] else None, pool_best['global'], 'Queried pool optimum')
            require(r['selected_ids'] is None or set(r['selected_ids']) <= set(ids), 'Selection escaped measured pool')
            for field in ('decision_sec', 'upper_sec', 'geometry_shared_sec', 'online_component_sum_sec'):
                require(np.isfinite(r[field]) and r[field] >= 0, 'Invalid timing')
            check_value(r['online_component_sum_sec'], r['geometry_shared_sec']+r['decision_sec'], 'Timing component sum')
            require(r['decision_sec']+1e-8 >= selection['selection_sec']+r['upper_sec'], 'Invalid decision timing boundary')
            check_value(r['geometry_shared_sec'], reference['geometry_shared_sec'], 'Shared geometry timing')
            deployments += check_deployment(path, r, poses, canonical, targets, coordinates, modules, masks, allowed, spec, opts, suite)
            results[job['directory'], arm['name']] = r
            pools += 1
            policy_replays += 1
    with (root/'comparison.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == len(results), 'CSV size')
    seen = set()
    job_by_case = {j['directory']: j for j in manifest['jobs']}
    for row in rows:
        key = row['case'], row['name']
        require(key in results and key not in seen, 'Duplicate/unknown CSV row')
        seen.add(key)
        r = results[key]
        a, ref = r['assessment'], references[key[0]]
        job = job_by_case[key[0]]
        expected = dict(profile=job['profile'], target=job.get('target_name', job['angle_deg']),
            prior=r['prior'], random_seed=r['random_seed'], query_budget=r['query_budget'],
            feasible=a is not None, objective_score=a['objective_score'] if a else None,
            objective_gap=ref-a['objective_score'] if ref is not None and a else None,
            coverage_pct=a['coverage_pct'] if a else None, candidate_queries=r['work']['unique_candidate_queries'],
            rays_submitted=r['work'].get('rays_submitted', 0), optical_point_tests=r['optical_point_tests'],
            decision_sec=r['decision_sec'], online_component_sum_sec=r['online_component_sum_sec'])
        for k, v in expected.items():
            check_value(row[k], v, f'CSV {k}')
    with (root/'paired.csv').open(newline='') as f:
        paired = list(csv.DictReader(f))
    require(len(paired) == len(manifest['jobs'])*len(options['query_budgets']), 'Paired row count')
    paired_seen = set()
    for row in paired:
        key = row['case'], int(row['query_budget'])
        require(key not in paired_seen and key[0] in references, 'Duplicate/unknown pair')
        paired_seen.add(key)
        source, geo = (results[key[0], f'b{key[1]}_{k}']['assessment'] for k in ('source', 'geometry'))
        rand = [results[key[0], f'b{key[1]}_random_s{s}']['assessment'] for s in options['random_seeds']]
        vals = np.array([v['objective_score'] for v in rand if v])
        expected = dict(source_feasible=source is not None, geometry_feasible=geo is not None,
            random_feasible_trials=len(vals), random_total_trials=len(rand),
            source_minus_geometry_j=source['objective_score']-geo['objective_score'] if source and geo else None,
            random_feasible_mean_j=float(vals.mean()) if len(vals) else None,
            random_feasible_sd_j=float(vals.std(ddof=1)) if len(vals) > 1 else None,
            source_minus_random_feasible_mean_j=source['objective_score']-float(vals.mean()) if source and len(vals) else None,
            source_wins=int(np.count_nonzero(source['objective_score'] > vals+1e-7)) if source else 0,
            source_ties=int(np.count_nonzero(np.abs(source['objective_score']-vals) <= 1e-7)) if source else 0,
            source_losses=int(np.count_nonzero(source['objective_score'] < vals-1e-7)) if source else 0)
        for k, v in expected.items():
            check_value(row[k], v, f'Paired statistic {k}')
    summary = read(root/'summary.json')
    require(summary['conditions'] == len(manifest['jobs']) and summary['online_rows'] == len(results)
        and summary['random_seed_count'] == len(options['random_seeds']), 'Summary completeness')
    require(set(summary['groups']) == {f'b{b}_{p}' for b in options['query_budgets'] for p in ('source', 'geometry', 'random')},
        'Summary group set')
    for key, actual in summary['groups'].items():
        budget, prior = key.split('_')
        vals = [r for r in results.values() if r['query_budget'] == int(budget[1:]) and r['prior'] == prior]
        gaps = [references[case]-r['assessment']['objective_score'] for (case, name), r in results.items()
            if r['query_budget'] == int(budget[1:]) and r['prior'] == prior
            and r['assessment'] and references[case] is not None]
        expected = dict(trials=len(vals), feasible_trials=sum(r['assessment'] is not None for r in vals),
            matches_reference=sum(abs(g) <= 1e-7 for g in gaps), feasible_gap_observations=len(gaps),
            mean_feasible_j_gap=float(np.mean(gaps)) if gaps else None,
            mean_queries=float(np.mean([r['work']['unique_candidate_queries'] for r in vals])),
            mean_decision_sec=float(np.mean([r['decision_sec'] for r in vals])))
        require(set(actual) == set(expected), 'Unexpected summary fields')
        for k, v in expected.items():
            check_value(actual[k], v, f'Summary {key} {k}')
    return dict(conditions=len(manifest['jobs']), online_rows=len(results), checked_deployments=deployments,
        reference_optima_checked=len(references), queried_pool_optima_checked=pools, selection_policy_replays=policy_replays,
        certificate_records_checked=len(references)+pools, geometry_meshes_checked=len(manifest['jobs']),
        scope='Independent single-J priors, unpruned pair optima and aggregate checks; recorded policy replay shares code. '
            'Ray counters replay shared float32 optics. No new raycasts or independent LOS proof.', validator_sha256=digest(__file__))


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(json.dumps(audit(p.parse_args().root), indent=2, allow_nan=False))

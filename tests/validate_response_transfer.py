"""Frozen response-transfer audit, full/pool pair enumeration and fresh physics.

Selection and model reconstruction replay shared code; pair optima and sampled
frame/FOV/triangle calculations use separate implementations.
"""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import connected_response_transfer_study as runner
from connected_validation_audit import physical_audit
from src.methods.connected_hierarchy import BudgetedObservations, optical_predictions, select_pool
from validate_connected_constraints import check_value
from validate_connected_corridor import exhaustive_values, require
from validate_connected_priors import check_deployment, check_upper_metadata, ray_counts


def audit(root):
    root = Path(root)
    manifest = runner.read(root/'manifest.json')
    require(manifest['status'] == 'complete', 'Incomplete study')
    require(set(manifest['source_hashes']) == set(runner.SOURCE_FILES), 'Source inventory')
    require(set(manifest['input_hashes']) == {f'inputs/{n}' for n in
        ('design.json', 'study.json', 'base_suite.json', 'experiment.yaml')}, 'Input inventory')
    for prefix, hashes in [('source', manifest['source_hashes']), ('', manifest['input_hashes']), ('', manifest['library_hashes'])]:
        for name, expected in hashes.items():
            require(runner.corridor.digest(root/prefix/name) == expected, f'Frozen hash: {name}')
    design = runner.read(root/'inputs/design.json')
    study, suite = runner.design_inputs(design, runner.read(root/'inputs/study.json'), runner.read(root/'inputs/base_suite.json'))
    config = runner.load_config(root/'inputs/experiment.yaml')
    sensor = runner.get_lidar_profile(config.lidar_profile, config)
    wanted_jobs = runner.prior.constraints.study_jobs(study)
    require([{k:v for k,v in j.items() if k != 'status'} for j in manifest['jobs']]
        == [{k:v for k,v in j.items() if k != 'status'} for j in wanted_jobs], 'Condition grid')
    expected_plan = dict(sources=len(design['sources']), conditions=len(wanted_jobs), arms=runner.arms(design),
        online_outputs=len(wanted_jobs)*len(runner.arms(design)), references=len(wanted_jobs), sensor=vars(sensor),
        seed_prior='common target unoccluded single J', lower_refinement=False, execute=True)
    require(manifest['plan'] == expected_plan, 'Plan mismatch')
    expected_library = {'library/index.json'} | {f"library/{s['name']}/{n}" for s in design['sources']
        for n in ('atlas.npz', 'metadata.json')}
    require(set(manifest['library_hashes']) == expected_library, 'Library inventory')
    source_checks, physics = [], []
    for item in design['sources']:
        path = root/'library'/item['name']
        geometry = runner.geometry_for(suite, item)
        canonical = runner.corridor.candidates(geometry, suite['candidate'])
        targets, coordinates = geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
        metadata = runner.read(path/'metadata.json')
        require(metadata['geometry'] == vars(geometry), 'Source geometry')
        with np.load(path/'atlas.npz', allow_pickle=False) as data:
            saved = {k:data[k] for k in data.files}
        for key, value in [('canonical', canonical), ('poses', geometry.poses(canonical)),
            ('targets', targets), ('target_coordinates', coordinates)]:
            require(np.array_equal(saved[key], value), f'Source array: {key}')
        require(metadata['atlas_sha256'] == runner.corridor.digest(path/'atlas.npz'), 'Source atlas link')
        job = dict(angle_deg=item['turn_deg'], road_width_m=item['road_width_m'], bend_length_m=item['bend_length_m'])
        _, ctx, _, _ = runner.prior.case_context(suite, job,
            dict(name='source', interface={}, forbidden_installation=[]), sensor)
        require(np.array_equal(saved['optical'], optical_predictions(ctx)), 'Source optical masks')
        mesh = geometry.create_environment_mesh()
        cache = runner.corridor.MaskCache(mesh, targets, sensor, geometry.poses(canonical))
        fresh = np.asarray([cache(i) for i in range(len(canonical))])
        require(np.array_equal(fresh, saved['masks']) and saved['masks'].dtype == bool, 'Fresh source visibility')
        require(metadata['work'] == cache.report(), 'Source ray accounting')
        source_checks.append(dict(name=item['name'], blocked_labels=int(np.count_nonzero(saved['optical'] & ~fresh))))
        require(source_checks[-1]['blocked_labels'] == metadata['blocked_labels'], 'Source residual count')
        physics.append(physical_audit(dict(poses=saved['poses'], targets=targets, canonical_candidates=canonical,
            target_coordinates=coordinates, vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles)),
            fresh, vars(geometry), vars(sensor), suite['interface_half_length_m']))
    profiles = {p['name']:p for p in study['profiles']}
    records, full_optima, seen_geometry = {}, {}, {}
    deployments = 0
    for job in manifest['jobs']:
        directory = root/job['directory']
        cm = runner.read(directory/'manifest.json')
        require(job['status'] == cm['status'] == 'complete', 'Case completion')
        require(manifest['source_frozen_at'] <= cm['online_committed_at'] <= cm['reference_started_at'], 'Reference leakage chronology')
        for filename, field in [('geometry.npz', 'geometry_sha256'), ('reference_masks.npz', 'reference_masks_sha256')]:
            require(runner.corridor.digest(directory/filename) == cm[field], f'Case hash {filename}')
        expected_paths = {f"{a['name']}/{n}" for a in runner.arms(design)
            for n in ('result.json', 'selection.json', 'observations.npz', 'surrogate.npz')}
        expected_paths.update(f"{a['name']}/deployment.npz" for a in runner.arms(design)
            if runner.read(directory/a['name']/'result.json')['selected_ids'] is not None)
        require(set(cm['online_hashes']) == expected_paths, 'Online commitment inventory')
        for filename, expected in cm['online_hashes'].items():
            require(runner.corridor.digest(directory/filename) == expected, f'Online hash {filename}')
        profile = profiles[job['profile']]
        geometry, ctx, coordinates, modules = runner.prior.case_context(suite, job, profile, sensor)
        mesh = geometry.create_environment_mesh()
        with np.load(directory/'geometry.npz', allow_pickle=False) as data:
            for key, value in [('vertices', np.asarray(mesh.vertices)), ('triangles', np.asarray(mesh.triangles)),
                ('canonical_candidates', ctx.canonical), ('poses', ctx.poses), ('targets', ctx.targets),
                ('target_coordinates', coordinates), ('module_ids', modules), ('allowed_candidates', ctx.allowed)]:
                require(np.array_equal(data[key], value), f'Target geometry {key}')
        require(runner.read(directory/'geometry.json') == vars(geometry)
            and runner.read(directory/'corridor_profile.json') == profile
            and runner.read(directory/'requirements.json') == suite['requirements'], 'Case specification')
        geometry_key = json.dumps(vars(geometry), sort_keys=True)
        if geometry_key not in seen_geometry:
            cache = runner.corridor.MaskCache(mesh, ctx.targets, sensor, ctx.poses)
            fresh = np.asarray([cache(i) for i in range(len(ctx.poses))])
            physics.append(physical_audit(dict(poses=ctx.poses, targets=ctx.targets, canonical_candidates=ctx.canonical,
                target_coordinates=coordinates, vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles)),
                fresh, vars(geometry), vars(sensor), suite['interface_half_length_m']))
            seen_geometry[geometry_key] = fresh
        masks = seen_geometry[geometry_key]
        eligible = np.flatnonzero(ctx.allowed)
        with np.load(directory/'reference_masks.npz', allow_pickle=False) as data:
            require(np.array_equal(data['candidate_ids'], eligible) and np.array_equal(data['masks'], masks[eligible]),
                'Fresh eligible reference visibility')
        interface = np.abs(coordinates[:, 0]-geometry.length_m/2) <= suite['interface_half_length_m']
        opts = dict(interface_mask=interface, interface_thresholds=profile['interface'])
        best, _ = exhaustive_values(ctx.poses[eligible], masks[eligible], modules[eligible], suite['requirements'], **opts)
        ref = runner.read(directory/'reference_global/result.json')
        check_value(ref['assessment']['objective_score'] if ref['assessment'] else None, best['global'], 'Full eligible J')
        full_optima[job['directory']] = best['global']
        check_upper_metadata(ref, len(eligible))
        rays = ray_counts(ctx.poses, ctx.targets, sensor)
        for key, value in dict(unique_candidate_queries=len(eligible), direct_pose_evaluations=len(eligible),
            raycast_calls=int(np.count_nonzero(rays[eligible])), rays_submitted=int(rays[eligible].sum())).items():
            require(ref['work'].get(key, 0) == value, f'Reference work: {key}')
        deployments += check_deployment(directory/'reference_global', ref, ctx.poses, ctx.canonical, ctx.targets,
            coordinates, modules, masks, ctx.allowed, suite['requirements'], opts, suite)
        prefixes, controls = {}, {}
        for arm in runner.arms(design):
            path = directory/arm['name']
            result, selection = runner.read(path/'result.json'), runner.read(path/'selection.json')
            require(all(result[k] == v for k,v in arm.items()), 'Arm identity')
            require(result['selection_sha256'] == runner.corridor.digest(path/'selection.json')
                and selection['observations_sha256'] == runner.corridor.digest(path/'observations.npz')
                and selection['surrogate_sha256'] == runner.corridor.digest(path/'surrogate.npz'), 'Arm links')
            require(manifest['source_frozen_at'] <= selection['committed_at'] <= cm['online_committed_at'], 'Arm chronology')
            predicted, metadata = runner.surrogate(root, arm['mode'], job, ctx, coordinates, design, optical_predictions(ctx))
            require(result['surrogate'] == metadata, 'Source geometry-only selection/model metadata')
            with np.load(path/'surrogate.npz', allow_pickle=False) as data:
                require(np.array_equal(data['masks'], predicted), 'Reconstructed surrogate')
            with np.load(path/'observations.npz', allow_pickle=False) as data:
                ids = data['candidate_ids'].tolist()
                require(ids == sorted(set(ids)) and ctx.allowed[ids].all()
                    and np.array_equal(data['masks'], masks[ids]), 'Fresh queried observations')
            require(selection['candidate_ids'] == ids, 'Measured pool')
            observed = BudgetedObservations(ctx, lambda i: masks[i] if i in ids else (_ for _ in ()).throw(KeyError(i)),
                per_site=arm['query_budget'])
            replay = select_pool('probe_context', ctx, observed, arm['query_budget'], probe_prior='geometry',
                fast_screen=True, prediction_transform=lambda optical: predicted)
            for key in replay:
                if key != 'selection_sec':
                    require(selection[key] == replay[key], f'Policy replay {key}')
            for group in ctx.groups:
                require(not group or len(set(group) & set(ids)) == arm['query_budget'], 'Per-site budget')
            work = result['work']
            require(work == selection['work'] and work['unique_candidate_queries'] == len(ids)
                and work.get('direct_pose_evaluations', 0) == len(ids) and work.get('rays_submitted', 0) == int(rays[ids].sum())
                and work.get('raycast_calls', 0) == int(np.count_nonzero(rays[ids])), 'Online accounting')
            sequence = [e['candidate_id'] for e in selection['events']]
            mode = arm['mode']
            if mode in prefixes:
                require(sequence[:len(prefixes[mode])] == prefixes[mode], 'Nested budget prefix')
            prefixes[mode] = sequence
            controls[arm['query_budget'], mode] = selection
            if mode == 'nearest_unoccluded':
                base = controls[arm['query_budget'], 'geometry']
                require(all(selection[k] == base[k] for k in ('candidate_ids', 'events', 'decisions', 'prior_values')),
                    'Zero-residual control changed policy')
            if arm['query_budget'] == 1:
                require(sequence == [e['candidate_id'] for e in controls[1, 'geometry']['events']], 'Different seed policy')
            best_pool, _ = exhaustive_values(ctx.poses[ids], masks[ids], modules[ids], suite['requirements'], **opts)
            check_value(result['assessment']['objective_score'] if result['assessment'] else None, best_pool['global'], 'Pool J')
            require(result['selected_ids'] is None or set(result['selected_ids']) <= set(ids), 'Upper escaped measured pool')
            check_upper_metadata(result, len(ids))
            require(all(np.isfinite(result[k]) and result[k] >= 0 for k in
                ('decision_sec', 'upper_sec', 'geometry_shared_sec', 'online_component_sum_sec')), 'Timing finite')
            check_value(result['online_component_sum_sec'], result['decision_sec']+result['geometry_shared_sec'], 'Time components')
            require(result['decision_sec']+1e-8 >= selection['selection_sec']+result['upper_sec'], 'Timing boundary')
            deployments += check_deployment(path, result, ctx.poses, ctx.canonical, ctx.targets, coordinates, modules,
                masks, ctx.allowed, suite['requirements'], opts, suite)
            assessment = result['assessment']
            records[job['directory'], arm['name']] = dict(case=job['directory'], profile=job['profile'], target=job['target_name'],
                **arm, feasible=assessment is not None, objective_score=best_pool['global'],
                objective_gap=full_optima[job['directory']]-best_pool['global']
                    if full_optima[job['directory']] is not None and best_pool['global'] is not None else None,
                false_visible=int(np.count_nonzero(predicted[eligible] & ~masks[eligible])),
                false_blocked=int(np.count_nonzero(~predicted[eligible] & masks[eligible])), candidate_queries=len(ids),
                rays_submitted=int(rays[ids].sum()), decision_sec=result['decision_sec'])
    with (root/'comparison.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == len(records) and len({(r['case'],r['name']) for r in rows}) == len(records), 'CSV completeness')
    for row in rows:
        expected = records[row['case'], row['name']]
        require(set(row) == set(expected), 'CSV fields')
        for key, value in expected.items():
            check_value(row[key], value, f'CSV {key}')
    summary = runner.read(root/'summary.json')
    require(summary['conditions'] == len(wanted_jobs) and summary['online_rows'] == len(records)
        and set(summary['groups']) == {a['name'] for a in runner.arms(design)}, 'Summary inventory')
    for arm in runner.arms(design):
        values = [v for v in records.values() if v['name'] == arm['name']]
        gaps = [v['objective_gap'] for v in values if v['objective_gap'] is not None]
        expected = dict(trials=len(values), feasible=sum(v['feasible'] for v in values),
            matches_reference=sum(abs(g) < 1e-7 for g in gaps), gap_observations=len(gaps),
            mean_feasible_j_gap=float(np.mean(gaps)) if gaps else None,
            mean_decision_sec=float(np.mean([v['decision_sec'] for v in values])),
            false_visible=sum(v['false_visible'] for v in values), false_blocked=sum(v['false_blocked'] for v in values))
        require(set(summary['groups'][arm['name']]) == set(expected), 'Summary fields')
        for key, value in expected.items():
            check_value(summary['groups'][arm['name']][key], value, f'Summary {key}')
    return dict(status='passed', conditions=len(wanted_jobs), online_records=len(records), checked_deployments=deployments,
        full_optima=len(full_optima), pool_optima=len(records), policy_replays=len(records), sources=source_checks,
        fresh_geometry_count=len(physics), physical_entries=sum(p['candidate_target_entries'] for p in physics),
        independent_triangle_checks=sum(p['independent_triangle_checks'] for p in physics),
        scope='Independent pair optima and frame/FOV/triangle samples. Model/selection replay shares code; fresh full '
            'source/target visibility uses the common engine. No independent full-LOS proof or learned response.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    runner.corridor.save(args.root/'independent_validation.json', result)
    print(json.dumps(result, allow_nan=False))

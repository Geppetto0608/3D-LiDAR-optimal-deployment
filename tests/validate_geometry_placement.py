"""Source-free saved-array, finite-pair, repeat/timing and fresh physical audit.

Policy/table reconstruction shares production code. Pair objectives and sampled
road-frame/FOV/segment-triangle checks use separate implementations.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geometry_placement as runner
from connected_validation_audit import physical_audit
from src.methods.connected_hierarchy import BudgetedObservations, select_pool
from src.methods.geometry_controls import CONTROL_METHODS, CONTROL_TIMINGS, select_geometry_control
from validate_connected_constraints import check_value
from validate_connected_corridor import exhaustive_values, require
from validate_connected_priors import check_deployment, check_upper_metadata, ray_counts


def audit(root):
    root = Path(root)
    manifest = runner.read(root/'manifest.json')
    require(manifest['status'] == 'complete' and manifest['source_free'] is True, 'Incomplete/source-dependent study')
    require(set(manifest['source_hashes']) == set(runner.SOURCE_FILES), 'Frozen source inventory')
    expected_inputs = {'inputs/design.json', 'inputs/base_suite.json', 'inputs/experiment.yaml'}
    expected_inputs.update(f"inputs/jobs/{j['id']}.json" for j in manifest['jobs'])
    require(set(manifest['input_hashes']) == expected_inputs, 'Frozen input inventory')
    for prefix, hashes in [('source', manifest['source_hashes']), ('', manifest['input_hashes'])]:
        for name, expected in hashes.items():
            require(runner.corridor.digest(root/prefix/name) == expected, f'Frozen hash: {name}')
    require(not (root/'library').exists(), 'Unexpected source assets')
    design, base = runner.read(root/'inputs/design.json'), runner.read(root/'inputs/base_suite.json')
    plan = manifest['plan']
    methods = list(dict.fromkeys(a['method'] for a in plan['arms']))
    budgets = sorted({a['query_budget'] for a in plan['arms'] if a['query_budget'] is not None}) or [4]
    seeds = sorted({a['random_seed'] for a in plan['arms'] if a['method'] == 'random'}) or None
    suite, cases, arms = runner.validate_inputs(design, base, methods, budgets, plan['repeats'], seeds)
    expected_jobs = runner.worker_jobs(cases, arms, plan['repeats'])
    require(manifest['cases'] == cases and plan['arms'] == arms and plan['conditions'] == len(cases)
        and plan['fresh_workers'] == len(expected_jobs) and plan['source_assets_required'] is False
        and plan['source_free'] is True and plan['fixed_sensor_count'] == 2
        and plan['reference_requested'] == any(a['method'] == 'full' for a in arms) and plan['execute'] is True,
        'Plan/source-free domain')
    require([{k:v for k,v in j.items() if k not in ('status','command','output_hashes')} for j in manifest['jobs']]
        == [{k:v for k,v in j.items() if k != 'status'} for j in expected_jobs], 'Duplicate/changed worker grid or order')
    config = runner.load_config(root/'inputs/experiment.yaml')
    sensor = runner.get_lidar_profile(config.lidar_profile, config)
    require(plan['sensor'] == vars(sensor), 'Sensor mismatch')
    profiles = {p['name']:p for p in design['profiles']}
    case_by_id = {c['directory']:c for c in cases}
    # Only the union of genuinely queried poses is recast; defaults never bake a full cache.
    union_ids, geometry_data = {}, {}
    for job in manifest['jobs']:
        case = case_by_id[job['case']]
        geometry, context, coordinates, modules = runner.build_problem(suite, case, profiles[case['profile']], sensor)
        key = json.dumps(vars(geometry), sort_keys=True)
        geometry_data[key] = geometry, context, coordinates, modules
        with np.load(root/'runs'/job['id']/'observations.npz', allow_pickle=False) as data:
            ids = data['candidate_ids']
        require(ids.dtype.kind in 'iu' and ids.ndim == 1 and ids.tolist() == sorted(set(ids.tolist()))
            and all(0 <= i < len(context.poses) and context.allowed[i] for i in ids), 'Invalid observed IDs')
        union_ids.setdefault(key, set()).update(ids.tolist())
    fresh_by_geometry, physical = {}, []
    for key, (geometry, context, coordinates, _) in geometry_data.items():
        mesh = geometry.create_environment_mesh()
        cache = runner.corridor.MaskCache(mesh, context.targets, sensor, context.poses)
        ids = np.asarray(sorted(union_ids[key]), int)
        masks = np.asarray([cache(i) for i in ids], bool).reshape(len(ids), len(context.targets))
        fresh_by_geometry[key] = {int(i):mask for i,mask in zip(ids, masks)}
        if len(ids):
            physical.append(physical_audit(dict(poses=context.poses[ids], canonical_candidates=context.canonical[ids],
                targets=context.targets, target_coordinates=coordinates, vertices=np.asarray(mesh.vertices),
                triangles=np.asarray(mesh.triangles)), masks, vars(geometry), vars(sensor), suite['interface_half_length_m']))
    proofs, deployments, full_optima, timing_rows = {}, 0, {}, 0
    selections, selected_deployments = {}, {}
    for job in manifest['jobs']:
        require(job['status'] == 'complete', 'Incomplete worker')
        directory = root/'runs'/job['id']
        result, selection = runner.read(directory/'solution/result.json'), runner.read(directory/'selection.json')
        files = {'selection.json','observations.npz','geometry.npz','geometry.json','worker.json','timing.json','solution/result.json'}
        if result['selected_ids'] is not None:
            files.add('solution/deployment.npz')
        require(set(job['output_hashes']) == files, 'Worker output inventory')
        for name, expected in job['output_hashes'].items():
            require(runner.corridor.digest(directory/name) == expected, f'Worker hash: {name}')
        case, arm = case_by_id[job['case']], job['arm']
        expected_input = dict(suite=suite, case=case, profile=profiles[case['profile']], arm=arm)
        require(runner.read(root/'inputs/jobs'/f"{job['id']}.json") == expected_input, 'Worker input/data scope')
        geometry, context, coordinates, modules = runner.build_problem(suite, case, profiles[case['profile']], sensor)
        mesh = geometry.create_environment_mesh()
        key = json.dumps(vars(geometry), sort_keys=True)
        known = fresh_by_geometry[key]
        require(runner.read(directory/'geometry.json') == vars(geometry), 'Geometry description')
        with np.load(directory/'geometry.npz', allow_pickle=False) as data:
            for name, value in [('vertices', np.asarray(mesh.vertices)), ('triangles', np.asarray(mesh.triangles)),
                ('canonical_candidates', context.canonical), ('poses', context.poses), ('targets', context.targets),
                ('target_coordinates', coordinates), ('module_ids', modules), ('allowed_candidates', context.allowed)]:
                require(np.array_equal(data[name], value), f'Geometry array: {name}')
        require(np.all(context.source_prior == 0), 'Non-neutral source vector')
        with np.load(directory/'observations.npz', allow_pickle=False) as data:
            ids = data['candidate_ids'].tolist()
            observed_masks = data['masks']
            require(np.array_equal(observed_masks, np.asarray([known[i] for i in ids], bool).reshape(len(ids), len(context.targets))),
                'Fresh measured visibility mismatch')
        require(selection['observations_sha256'] == runner.corridor.digest(directory/'observations.npz')
            and selection['candidate_ids'] == ids and result['source_free'] is True
            and all(result[k] == v for k,v in arm.items()), 'Arm/source-free identity')
        rays = ray_counts(context.poses, context.targets, sensor)
        work = result['work']
        require(work == selection['work'] and work['unique_candidate_queries'] == len(ids)
            and work.get('direct_pose_evaluations', 0) == len(ids)
            and work.get('rays_submitted', 0) == int(rays[ids].sum())
            and work.get('raycast_calls', 0) == int(np.count_nonzero(rays[ids])), 'Physical work accounting')
        proof_key = job['case'], arm['name']
        if arm['method'] == 'full':
            require(ids == np.flatnonzero(context.allowed).tolist() and selection['optical_point_tests'] == 0, 'Full reference domain/work')
        else:
            require(all(not g or len(set(g)&set(ids)) == arm['query_budget'] for g in context.groups), 'Per-site query budget')
            optical_work = 0 if arm['method'] == 'random' else int(context.allowed.sum()*len(context.targets))
            require(selection['optical_point_tests'] == optical_work, 'Optical work')
        require(result['optical_point_tests'] == selection['optical_point_tests'], 'Optical work provenance')
        opts = dict(interface_mask=np.abs(coordinates[:,0]-geometry.length_m/2) <= suite['interface_half_length_m'],
            interface_thresholds=profiles[case['profile']]['interface'])
        if proof_key not in proofs:
            best, _ = exhaustive_values(context.poses[ids], observed_masks, modules[ids], suite['requirements'], **opts)
            proofs[proof_key] = best['global']
            if arm['method'] == 'full':
                full_optima[job['case']] = best['global']
            else:
                measured = {i:known[i] for i in ids}
                obs = BudgetedObservations(context, lambda i:measured[i], per_site=arm['query_budget'])
                adaptive = arm['method'] in ('adaptive', 'adaptive_batch')
                options = dict(probe_prior='geometry', fast_screen=True) if adaptive else {}
                if arm['method'] == 'adaptive_batch':
                    options['partner_engine'] = 'batch'
                kind = 'probe_context' if adaptive else 'geometry_single'
                if arm['method'] == 'random':
                    kind, options = 'random_pool', dict(random_seed=arm['random_seed'])
                if arm['method'] in CONTROL_METHODS:
                    replay = select_geometry_control(arm['method'], context, obs, arm['query_budget'])
                else:
                    replay = select_pool(kind, context, obs, arm['query_budget'], **options)
                for k in replay:
                    if k not in CONTROL_TIMINGS:
                        require(selection[k] == replay[k], f'Policy replay {k}')
        if arm['method'] in CONTROL_METHODS:
            require(all(np.isfinite(selection[k]) and selection[k] >= 0 for k in CONTROL_TIMINGS)
                and selection['selection_sec']+1e-8 >= selection['planning_sec']+selection['physical_query_sec'],
                'Control timing accounting')
            require([dict(candidate_id=e['candidate_id'], phase=e['phase']) for e in selection['events']]
                == selection['query_plan'], 'Frozen plan/physical event mismatch')
        check_value(result['assessment']['objective_score'] if result['assessment'] else None, proofs[proof_key], 'Queried pool optimum')
        check_upper_metadata(result, len(ids))
        require(result['selected_ids'] is None or set(result['selected_ids']) <= set(ids), 'Upper left measured pool')
        normalized = {k:v for k,v in selection.items() if k not in (*CONTROL_TIMINGS,'observations_sha256')}
        if proof_key in selections:
            require(selections[proof_key] == normalized, 'Non-deterministic repeated selection')
        selections[proof_key] = normalized
        selected_deployments[proof_key] = result['selected_ids']
        all_masks = np.zeros((len(context.poses),len(context.targets)), bool)
        all_masks[ids] = observed_masks
        deployments += check_deployment(directory/'solution', result, context.poses, context.canonical, context.targets,
            coordinates, modules, all_masks, context.allowed, suite['requirements'], opts, suite)
        worker, timing = runner.read(directory/'worker.json'), runner.read(directory/'timing.json')
        require(worker['status'] == 'complete' and worker['source_free'] is True and worker['work'] == work, 'Worker completion')
        marker = dict(event='decision_complete', completed_monotonic_ns=worker['completed_monotonic_ns'], source_free=True)
        expected = runner.cold_timing(timing['parent_launch_monotonic_ns'], timing['parent_exit_monotonic_ns'], marker)
        require(timing == expected and timing['cold_ready_sec']+1e-8 >= result['decision_sec'], 'Cold-ready timing/clock boundary')
        require(all(np.isfinite(result[k]) and result[k] >= 0 for k in ('decision_sec','geometry_sec','upper_sec')), 'Finite timing')
        require(result['decision_sec']+1e-8 >= result['geometry_sec']+selection['selection_sec']+result['upper_sec'], 'Child timing boundary')
        timing_rows += 1
    # Tables are reconstructed from independently checked quality/work and measured clocks.
    batch_parity = 0
    for (case, name), selection in selections.items():
        if not name.startswith('adaptive_batch_'):
            continue
        baseline = selections.get((case, name.replace('adaptive_batch_', 'adaptive_', 1)))
        if baseline is not None:
            require({k:v for k,v in selection.items() if k != 'partner_engine'} == baseline,
                'Scalar/batch query/decision parity')
            require(selected_deployments[(case, name)] == selected_deployments[
                (case, name.replace('adaptive_batch_', 'adaptive_', 1))], 'Scalar/batch deployment parity')
            batch_parity += 1
    raw, comparison, summary = runner.tables(root, manifest)
    for filename, expected in [('runs.csv',raw), ('comparison.csv',comparison)]:
        with (root/filename).open(newline='') as f:
            actual = list(csv.DictReader(f))
        require(len(actual) == len(expected), f'{filename} row count')
        for a,e in zip(actual,expected):
            require(set(a) == set(e), f'{filename} fields')
            for k,v in e.items():
                check_value(a[k],v,f'{filename} {k}')
    require(runner.read(root/'summary.json') == summary, 'Summary statistics or repeated-quality mismatch')
    result = dict(status='passed', source_free=True, conditions=len(cases), workers=len(manifest['jobs']),
        checked_deployments=deployments, unique_pool_optima=len(proofs), full_case_optima=len(full_optima),
        policy_replays=sum(a['method'] != 'full' for a in arms)*len(cases), timing_rows=timing_rows,
        fresh_candidate_queries=sum(len(v) for v in fresh_by_geometry.values()),
        physical_entries=sum(p['candidate_target_entries'] for p in physical),
        independent_triangle_checks=sum(p['independent_triangle_checks'] for p in physical),
        scope='Independent finite-pair optima and sampled road/FOV/triangle checks. Policy/table replay shares code. '
            'Common-engine recasts only previously queried poses. Repetitions are timing measurements, not new maps; '
            'no source assets, independent full-LOS proof or real-road acceptance.')
    if any(a['method'] == 'adaptive_batch' for a in arms):
        result['scalar_batch_policy_pairs'] = batch_parity
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    runner.corridor.save(args.root/'independent_validation.json',result)
    print(json.dumps(result,allow_nan=False))

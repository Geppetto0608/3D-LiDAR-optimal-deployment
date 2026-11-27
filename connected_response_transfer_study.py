"""Controlled source-occlusion response transfer pilot; no-write plan by default.

All controls use the same target optical seed ranking and conditional measured
partners. Only the predicted unknown pose mask changes. No LS/MAPPO/training.
"""
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

import connected_prior_ablation as prior
import connected_corridor_study as corridor
from src.environment import ConnectedCorridorGeometry, get_lidar_profile, load_config
from src.methods.connected_hierarchy import BudgetedObservations, optical_predictions, select_pool, upper_select
from src.methods.response_transfer import correspondence, nearest_source, transfer_residual

BASE = Path(__file__).resolve().parent
MODES = ('geometry', 'straight_residual', 'nearest_unoccluded', 'nearest_residual')
SOURCE_FILES = list(dict.fromkeys(prior.SOURCE_FILES + ['connected_response_transfer_study.py',
    'src/methods/response_transfer.py', 'tests/validate_response_transfer.py', 'tests/test_response_transfer.py',
    'connected_validation_audit.py', 'src/methods/connected_diagnostics.py']))


def read(path):
    return json.loads(Path(path).read_text())


def design_inputs(design, study, base):
    if (not isinstance(design, dict) or set(design) != {'schema', 'sources', 'targets', 'query_budgets'}
            or design['schema'] != 'connected_response_transfer_v1'):
        raise ValueError('Invalid response transfer design')
    prior.validate_options(dict(schema='connected_prior_ablation_v1', query_budgets=design['query_budgets'], random_seeds=[0]))
    if not isinstance(design['sources'], list) or not 1 <= len(design['sources']) <= 6:
        raise ValueError('Bounded source family required')
    source_study = {**study, 'targets': design['sources'], 'profiles': [study['profiles'][0]]}
    prior.constraints.validate_study(source_study, base)
    if len([s for s in design['sources'] if s['name'] == 'straight' and s['turn_deg'] == 0]) != 1:
        raise ValueError('One straight source control required')
    def shape(item):
        return tuple(item[k] for k in ('road_width_m', 'turn_deg', 'bend_length_m'))
    if (set(map(shape, design['sources'])) & set(map(shape, design['targets']))
            or {s['name'] for s in design['sources']} & {s['name'] for s in design['targets']}):
        raise ValueError('Source and target definitions must be disjoint')
    study = {**deepcopy(study), 'targets': design['targets']}
    suite = prior.constraints.validate_study(study, base)
    return study, suite


def arms(design):
    return [dict(name=f'b{b}_{mode}', query_budget=b, mode=mode)
        for b in design['query_budgets'] for mode in MODES]


def geometry_for(suite, item):
    return ConnectedCorridorGeometry(**{**suite['geometry'], **{k: item[k]
        for k in ('road_width_m', 'turn_deg', 'bend_length_m')}})


def build_library(root, suite, sensor, design):
    profiles = dict(name='source', interface={}, forbidden_installation=[])
    records = []
    for item in design['sources']:
        started = time.perf_counter()
        job = dict(angle_deg=item['turn_deg'], road_width_m=item['road_width_m'], bend_length_m=item['bend_length_m'])
        geometry, context, coordinates, _ = prior.case_context(suite, job, profiles, sensor)
        mesh = geometry.create_environment_mesh()
        optical = optical_predictions(context)
        cache = corridor.MaskCache(mesh, context.targets, sensor, context.poses)
        masks = np.asarray([cache(i) for i in range(len(context.poses))])
        directory = root/'library'/item['name']
        directory.mkdir(parents=True)
        np.savez_compressed(directory/'atlas.npz', canonical=context.canonical, poses=context.poses,
            targets=context.targets, target_coordinates=coordinates, optical=optical, masks=masks)
        record = dict(**item, geometry=vars(geometry), atlas_sha256=corridor.digest(directory/'atlas.npz'),
            optical_true=int(optical.sum()), blocked_labels=int(np.count_nonzero(optical & ~masks)),
            candidate_count=len(masks), target_count=len(context.targets), work=cache.report(),
            offline_sec=time.perf_counter()-started)
        corridor.save(directory/'metadata.json', record)
        records.append(record)
    corridor.save(root/'library/index.json', records)


def surrogate(root, mode, job, context, coordinates, design, optical):
    if mode not in MODES:
        raise ValueError('Unknown response transfer control')
    if mode == 'geometry':
        return optical.copy(), dict(mode=mode, source_name=None, mapped_targets=0)
    target = dict(road_width_m=job['road_width_m'], turn_deg=job['angle_deg'], bend_length_m=job['bend_length_m'])
    name, distance = ('straight', None) if mode == 'straight_residual' else nearest_source(design['sources'], target)
    item = next(s for s in design['sources'] if s['name'] == name)
    with np.load(root/'library'/name/'atlas.npz', allow_pickle=False) as data:
        candidate_ids, target_ids = correspondence(data['canonical'], data['target_coordinates'],
            context.canonical, coordinates, item['road_width_m'], target['road_width_m'])
        source_optical = data['optical']
        # This control does not read measured source masks; its residual is identically zero.
        source_masks = source_optical if mode == 'nearest_unoccluded' else data['masks']
        predicted = transfer_residual(optical, source_optical, source_masks, candidate_ids, target_ids)
    return predicted, dict(mode=mode, source_name=name, descriptor_distance=distance,
        mapped_targets=int(np.count_nonzero(target_ids >= 0)), source_atlas_sha256=corridor.digest(root/'library'/name/'atlas.npz'))


def run_case(root, job, suite, profile, sensor, design):
    directory = root/job['directory']
    directory.mkdir()
    started = time.perf_counter()
    geometry, context, coordinates, modules = prior.case_context(suite, job, profile, sensor)
    mesh = geometry.create_environment_mesh()
    geometry_sec = time.perf_counter()-started
    poses, canonical, targets, req = context.poses, context.canonical, context.targets, context.requirements
    np.savez_compressed(directory/'geometry.npz', vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles),
        canonical_candidates=canonical, poses=poses, targets=targets, target_coordinates=coordinates,
        module_ids=modules, allowed_candidates=context.allowed)
    corridor.save(directory/'geometry.json', vars(geometry))
    corridor.save(directory/'corridor_profile.json', profile)
    corridor.save(directory/'requirements.json', suite['requirements'])
    for arm in arms(design):
        decision_start = time.perf_counter()
        cache = corridor.MaskCache(mesh, targets, sensor, poses)
        observed = BudgetedObservations(context, cache, per_site=arm['query_budget'])
        captured = {}
        def transform(optical):
            predicted, metadata = surrogate(root, arm['mode'], job, context, coordinates, design, optical)
            captured.update(predicted=predicted, metadata=metadata)
            return predicted
        selection = select_pool('probe_context', context, observed, arm['query_budget'], probe_prior='geometry',
            fast_screen=True, prediction_transform=transform)
        upper_start = time.perf_counter()
        result = upper_select(selection['candidate_ids'], context, observed, modules, 'free')
        upper_sec = time.perf_counter()-upper_start
        decision_sec = time.perf_counter()-decision_start
        result.update(**arm, decision_sec=decision_sec, upper_sec=upper_sec, geometry_shared_sec=geometry_sec,
            online_component_sum_sec=geometry_sec+decision_sec, optical_point_tests=selection['optical_point_tests'],
            surrogate=captured['metadata'], timing_semantics='Target scene, optics, source selection/load/mapping, '
                'conditional probes and upper selection. Geometry shared separately. Offline source build, '
                'startup/import, serialization and fresh final validation excluded; not cold-start latency.')
        path = directory/arm['name']
        result = corridor.persist_method(path, result, canonical, poses, targets, coordinates, observed, mesh, sensor, req, suite)
        np.savez_compressed(path/'surrogate.npz', masks=captured['predicted'])
        selection.update(observations_sha256=prior.hierarchy.save_observations(path, observed), work=observed.report(),
            surrogate_sha256=corridor.digest(path/'surrogate.npz'), committed_at=corridor.now())
        corridor.save(path/'selection.json', selection)
        result['selection_sha256'] = corridor.digest(path/'selection.json')
        corridor.save(path/'result.json', result)
    cm = dict(status='online_committed', online_committed_at=corridor.now(),
        geometry_sha256=corridor.digest(directory/'geometry.npz'), online_hashes={str(p.relative_to(directory)):corridor.digest(p)
            for p in sorted(directory.glob('*/*')) if p.is_file()})
    corridor.save(directory/'manifest.json', cm)
    cm['reference_started_at'] = corridor.now()
    start = time.perf_counter()
    cache = corridor.MaskCache(mesh, targets, sensor, poses)
    eligible = np.flatnonzero(context.allowed).tolist()
    masks = np.asarray([cache(i) for i in eligible], bool).reshape(len(eligible), len(targets))
    build_sec = time.perf_counter()-start
    observed = BudgetedObservations(context, cache, initial={i:cache(i) for i in eligible})
    result = upper_select(eligible, context, observed, modules, 'free')
    result.update(reference_build_sec=build_sec, decision_sec=time.perf_counter()-start,
        geometry_shared_sec=geometry_sec, declared_candidates=len(poses), eligible_candidates=len(eligible))
    corridor.persist_method(directory/'reference_global', result, canonical, poses, targets, coordinates,
        cache, mesh, sensor, req, suite)
    np.savez_compressed(directory/'reference_masks.npz', candidate_ids=eligible, masks=masks)
    cm.update(status='complete', finished_at=corridor.now(), reference_masks_sha256=corridor.digest(directory/'reference_masks.npz'))
    corridor.save(directory/'manifest.json', cm)


def tables(root, jobs, design):
    rows = []
    for job in jobs:
        directory = root/job['directory']
        ref = read(directory/'reference_global/result.json')['assessment']
        with np.load(directory/'geometry.npz', allow_pickle=False) as data:
            allowed = data['allowed_candidates']
        with np.load(directory/'reference_masks.npz', allow_pickle=False) as data:
            truth = data['masks']
        for arm in arms(design):
            path = directory/arm['name']
            result = read(path/'result.json')
            assessment = result['assessment']
            with np.load(path/'surrogate.npz', allow_pickle=False) as data:
                predicted = data['masks'][allowed]
            rows.append(dict(case=job['directory'], profile=job['profile'], target=job['target_name'], **arm,
                feasible=assessment is not None, objective_score=assessment['objective_score'] if assessment else None,
                objective_gap=ref['objective_score']-assessment['objective_score'] if ref and assessment else None,
                false_visible=int(np.count_nonzero(predicted & ~truth)), false_blocked=int(np.count_nonzero(~predicted & truth)),
                candidate_queries=result['work']['unique_candidate_queries'], rays_submitted=result['work'].get('rays_submitted', 0),
                decision_sec=result['decision_sec']))
    groups = {}
    for arm in arms(design):
        values = [r for r in rows if r['name'] == arm['name']]
        gaps = [r['objective_gap'] for r in values if r['objective_gap'] is not None]
        groups[arm['name']] = dict(trials=len(values), feasible=sum(v['feasible'] for v in values),
            matches_reference=sum(abs(g) < 1e-7 for g in gaps), gap_observations=len(gaps),
            mean_feasible_j_gap=float(np.mean(gaps)) if gaps else None,
            mean_decision_sec=float(np.mean([v['decision_sec'] for v in values])),
            false_visible=sum(v['false_visible'] for v in values), false_blocked=sum(v['false_blocked'] for v in values))
    return rows, dict(conditions=len(jobs), online_rows=len(rows), groups=groups,
        scope='Predeclared new synthetic development geometries, fixed K=2; no MAPPO, learned policy or network claim.')


def summarize(root, jobs, design):
    rows, summary = tables(root, jobs, design)
    with (root/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    corridor.save(root/'summary.json', summary)


def worker(root):
    manifest = read(root/'manifest.json')
    for name, expected in manifest['source_hashes'].items():
        if corridor.digest(BASE/name) != expected:
            raise RuntimeError(f'Frozen source mismatch: {name}')
    for name, expected in manifest['input_hashes'].items():
        if corridor.digest(root/name) != expected:
            raise RuntimeError(f'Frozen input mismatch: {name}')
    design = read(root/'inputs/design.json')
    study, suite = design_inputs(design, read(root/'inputs/study.json'), read(root/'inputs/base_suite.json'))
    config = load_config(root/'inputs/experiment.yaml')
    sensor = get_lidar_profile(config.lidar_profile, config)
    start = time.perf_counter()
    try:
        manifest['status'] = 'running'
        corridor.save(root/'manifest.json', manifest)
        build_library(root, suite, sensor, design)
        manifest.update(source_frozen_at=corridor.now(), library_hashes={str(p.relative_to(root)):corridor.digest(p)
            for p in sorted((root/'library').rglob('*')) if p.is_file()})
        corridor.save(root/'manifest.json', manifest)
        profiles = {p['name']:p for p in study['profiles']}
        for job in manifest['jobs']:
            job['status'] = 'running'
            corridor.save(root/'manifest.json', manifest)
            run_case(root, job, suite, profiles[job['profile']], sensor, design)
            job['status'] = 'complete'
            corridor.save(root/'manifest.json', manifest)
        summarize(root, manifest['jobs'], design)
        manifest.update(status='complete', finished_at=corridor.now())
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['worker_wall_sec'] = time.perf_counter()-start
        corridor.save(root/'manifest.json', manifest)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--design', default=str(BASE/'configs/connected_response_transfer.json'))
    parser.add_argument('--study', default=str(BASE/'configs/connected_constraint_holdout.json'))
    parser.add_argument('--base-suite', default=str(BASE/'configs/connected_corridor_pilot.json'))
    parser.add_argument('--config', default=str(BASE/'configs/experiment.yaml'))
    parser.add_argument('--output', default='results/connected_response_transfer_20261023')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-root', help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return worker(Path(args.worker_root).resolve())
    design = read(args.design)
    study, suite = design_inputs(design, read(args.study), corridor.load_suite(args.base_suite))
    config = load_config(args.config)
    sensor = get_lidar_profile(config.lidar_profile, config)
    prior.validate_scope(suite, sensor)
    jobs = prior.constraints.study_jobs(study)
    plan = dict(sources=len(design['sources']), conditions=len(jobs), arms=arms(design),
        online_outputs=len(jobs)*len(arms(design)), references=len(jobs), sensor=vars(sensor),
        seed_prior='common target unoccluded single J', lower_refinement=False, execute=args.execute)
    print(json.dumps(plan), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    for name, value in [('design.json', design), ('study.json', study)]:
        corridor.save(root/'inputs'/name, value)
    for src, name in [(args.base_suite, 'base_suite.json'), (args.config, 'experiment.yaml')]:
        shutil.copy2(src, root/'inputs'/name)
    for name in SOURCE_FILES:
        destination = root/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/name, destination)
    (root/'packages.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
        check=True, capture_output=True, text=True).stdout)
    corridor.save(root/'manifest.json', dict(status='prepared', created_at=corridor.now(), plan=plan, jobs=jobs,
        source_hashes={name:corridor.digest(root/'source'/name) for name in SOURCE_FILES},
        input_hashes={str(p.relative_to(root)):corridor.digest(p) for p in sorted((root/'inputs').iterdir())},
        command=sys.argv if argv is None else argv))
    try:
        with (root/'worker.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(root/'source/connected_response_transfer_study.py'),
                '--worker-root', str(root), '--execute'], cwd=root/'source', stdout=log, stderr=subprocess.STDOUT, check=True)
        with (root/'audit.log').open('w') as log:
            subprocess.run([sys.executable, str(root/'source/tests/validate_response_transfer.py'), str(root)],
                cwd=root/'source', stdout=log, stderr=subprocess.STDOUT, check=True)
    except BaseException as exc:
        manifest = read(root/'manifest.json')
        manifest.update(status='failed', error=repr(exc))
        corridor.save(root/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()

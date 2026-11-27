"""Read-only study verification with fresh rays and sampled independent LOS checks.

Writes evidence only to a new, separate output directory. No training or new
placement search. Fresh Open3D parity is not independent proof of all LOS bits.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

import connected_corridor_study as corridor
import connected_prior_ablation as runner
from src.environment import ConnectedCorridorGeometry, get_lidar_profile, load_config

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'tests'))
from connected_validation_audit import physical_audit
from validate_connected_priors import audit, read
from validate_connected_corridor import require


def inventory(root):
    return {str(p.relative_to(root)): corridor.digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def physical_recheck(geometry, archive, sensor, interface_half, destination):
    mesh = geometry.create_environment_mesh()
    for key, value in [('vertices', np.asarray(mesh.vertices)), ('triangles', np.asarray(mesh.triangles))]:
        if key in archive:
            require(np.array_equal(archive[key], value), f'Physical mesh reconstruction: {key}')
        archive[key] = value
    cache = corridor.MaskCache(mesh, archive['targets'], sensor, archive['poses'])
    masks = np.asarray([cache(i) for i in range(len(archive['poses']))], bool)
    independent = physical_audit(archive, masks, vars(geometry), vars(sensor), interface_half)
    np.savez_compressed(destination, masks=masks)
    return masks, dict(**independent, fresh_work=cache.report(), masks_sha256=corridor.digest(destination))


def objective(result, masks, poses, spec):
    ids = result['selected_ids']
    if ids is None:
        return None
    density = masks[ids].sum(axis=0)
    w, c = spec.get('weights', {}), spec.get('cost', {})
    return float((w.get('coverage_weight', 1)+w.get('importance_weight', 0))*np.count_nonzero(density)
        - w.get('overlap_penalty', 0)*np.maximum(density-1, 0).sum()
        - w.get('sensor_count_penalty', 0)*len(ids)
        - w.get('installation_cost_weight', 0)*(c.get('base_per_sensor', 0)*len(ids)
            + c.get('per_height_m', 0)*poses[ids, 2].sum()))


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root != output and root not in output.parents, 'Verification output must be outside the original study')
    originals = inventory(root)
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', input_root=str(root), started_at=corridor.now(),
        scope='No training/new placement experiments; saved-array pair enumeration, fresh shared-engine rays, '
            'independent road frame/FOV math and stratified segment-triangle LOS samples.')
    started = time.perf_counter()
    try:
        source_files = list(dict.fromkeys(runner.SOURCE_FILES + [
            'connected_prior_verification.py', 'connected_validation_audit.py', 'src/methods/connected_diagnostics.py',
            'tests/test_connected_prior_verification.py']))
        for name in source_files:
            destination = output/'source'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE/name, destination)
        state['source_hashes'] = {name: corridor.digest(output/'source'/name) for name in source_files}
        corridor.save(output/'manifest.json', state)
        corridor.save(output/'original_hashes.json', originals)
        print('Checking frozen data, full/pool optima and recorded policy decisions', flush=True)
        checked = audit(root)
        corridor.save(output/'strengthened_validation.json', checked)
        manifest = read(root/'manifest.json')
        suite = runner.constraints.validate_study(read(root/'inputs/study.json'), read(root/'inputs/base_suite.json'))
        config = load_config(root/'inputs/experiment.yaml')
        sensor = get_lidar_profile(config.lidar_profile, config)
        physics = []
        (output/'physics').mkdir()
        with np.load(root/'library/library.npz', allow_pickle=False) as data:
            archive = {k: data[k] for k in ('poses', 'targets', 'target_coordinates')}
            archive['canonical_candidates'] = data['canonical']
            saved = data['masks']
        geometry = ConnectedCorridorGeometry(**read(root/'library/metadata.json')['geometry'])
        fresh, details = physical_recheck(geometry, archive, sensor, suite['interface_half_length_m'], output/'physics/source.npz')
        require(np.array_equal(fresh, saved), 'Fresh source masks disagree')
        physics.append(dict(kind='source', **details))
        seen, values = {}, {}
        for job in manifest['jobs']:
            directory = root/job['directory']
            geometry_spec = read(directory/'geometry.json')
            key = json.dumps(geometry_spec, sort_keys=True)
            with np.load(directory/'geometry.npz', allow_pickle=False) as data:
                archive = {k: data[k] for k in data.files}
            if key not in seen:
                geometry = ConnectedCorridorGeometry(**geometry_spec)
                name = f'target_{len(seen):02d}'
                print(f'Fresh rays and independent physical checks: {name}', flush=True)
                fresh, details = physical_recheck(geometry, archive, sensor, suite['interface_half_length_m'],
                    output/'physics'/f'{name}.npz')
                physics.append(dict(kind=name, geometry=geometry_spec, **details))
                seen[key] = fresh, archive
            fresh, first_archive = seen[key]
            for field in ('poses', 'targets', 'canonical_candidates', 'target_coordinates', 'vertices', 'triangles'):
                require(np.array_equal(archive[field], first_archive[field]), f'Profile changed physical geometry: {field}')
            with np.load(directory/'reference_masks.npz', allow_pickle=False) as data:
                ids = np.asarray(data['candidate_ids'], dtype=int)
                require(np.array_equal(fresh[ids], data['masks']), 'Fresh eligible reference masks disagree')
            for name in [a['name'] for a in runner.arms(read(root/'inputs/ablation.json'))] + ['reference_global']:
                path = directory/name
                result = read(path/'result.json')
                if result['selected_ids'] is not None:
                    with np.load(path/'deployment.npz', allow_pickle=False) as data:
                        require(np.array_equal(fresh[result['selected_ids']], data['sensor_masks']), 'Fresh deployment masks disagree')
                if name != 'reference_global':
                    with np.load(path/'observations.npz', allow_pickle=False) as data:
                        ids = np.asarray(data['candidate_ids'], dtype=int)
                        require(np.array_equal(fresh[ids], data['masks']), 'Fresh online observations disagree')
                value = objective(result, fresh, archive['poses'], suite['requirements'])
                require((value is None and result['assessment'] is None) or (value is not None
                    and abs(value-result['assessment']['objective_score']) < 1e-8), 'Fresh-mask independent objective mismatch')
                values[job['directory'], name] = value
        paired = {}
        for budget in read(root/'inputs/ablation.json')['query_budgets']:
            differences = [values[j['directory'], f'b{budget}_source']-values[j['directory'], f'b{budget}_geometry']
                for j in manifest['jobs'] if values[j['directory'], f'b{budget}_source'] is not None
                and values[j['directory'], f'b{budget}_geometry'] is not None]
            paired[str(budget)] = dict(common_feasible_conditions=len(differences), total_conditions=len(manifest['jobs']),
                source_wins=sum(d > 1e-7 for d in differences), ties=sum(abs(d) <= 1e-7 for d in differences),
                source_losses=sum(d < -1e-7 for d in differences),
                mean_source_minus_geometry_j=float(np.mean(differences)) if differences else None)
        corridor.save(output/'physical_validation.json', dict(geometries=physics,
            fresh_candidate_masks=sum(p['fresh_work']['unique_candidate_queries'] for p in physics),
            candidate_target_entries=sum(p['candidate_target_entries'] for p in physics),
            independent_triangle_checks=sum(p['independent_triangle_checks'] for p in physics),
            deployment_records_checked=len(values), source_vs_geometry=paired))
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        after = inventory(root)
        changed = sorted(k for k in originals.keys() | after.keys() if originals.get(k) != after.get(k))
        state.update(original_files_checked=len(originals), originals_unchanged=not changed, changed_original_files=changed,
            elapsed_sec=time.perf_counter()-started, finished_at=corridor.now())
        if changed:
            state['status'] = 'failed'
        corridor.save(output/'manifest.json', state)
        require(not changed, 'Original study files changed during verification')
    return state


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(root=str(args.root), output=str(args.output), execute=False,
            training=False, fresh_raycast_on_execute=True)))
        return
    print(json.dumps(run(args.root, args.output), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

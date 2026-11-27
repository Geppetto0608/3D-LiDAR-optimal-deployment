"""Independent saved-mask audit of the controlled connected-corridor sweep."""
from argparse import ArgumentParser
from itertools import product
import json
from pathlib import Path

import numpy as np

from validate_connected_corridor import METHODS, digest, exhaustive_values, require


def audit(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['status'] == 'complete', 'Sweep must be complete')
    for name, expected in manifest['source_hashes'].items():
        require(digest(root / 'source' / name) == expected, f'Frozen source hash: {name}')
    for name, expected in manifest['input_hashes'].items():
        require(digest(root / name) == expected, f'Frozen input hash: {name}')
    base = json.loads((root / 'inputs/base_suite.json').read_text())
    sweep = json.loads((root / 'inputs/sweep.json').read_text())
    expected = set(product(sweep['bend_lengths_m'], sweep['alternatives_per_site'], sweep['angles_deg']))
    actual = [(j['bend_length_m'], j['alternatives_per_site'], j['angle_deg']) for j in manifest['jobs']]
    require(len(actual) == len(expected) and set(actual) == expected, 'Incomplete/duplicate factorial conditions')
    metadata = json.loads((root / 'library/metadata.json').read_text())
    require(digest(root / 'library/library.npz') == metadata['source_pose_mask_hash'], 'Fixed source archive hash')
    with np.load(root / 'library/library.npz', allow_pickle=False) as data:
        source, scores, source_masks = data['canonical'], data['scores'], data['masks']
    require(source_masks.shape[0] == len(source), 'Source mask count')
    weights, cost = base['requirements'].get('weights', {}), base['requirements'].get('cost', {})
    expected_scores = ((weights.get('coverage_weight', 1) + weights.get('importance_weight', 0))
        * source_masks.sum(axis=1) - weights.get('sensor_count_penalty', 0)
        - weights.get('installation_cost_weight', 0)
        * (cost.get('base_per_sensor', 0) + cost.get('per_height_m', 0) * source[:, 2]))
    require(np.allclose(scores, expected_scores, atol=1e-8, rtol=0), 'Source-only ranking objective')
    best_source = int(np.argmax(scores))
    require(best_source == metadata['best_source_id'], 'Source winner changed')
    ranked_sites = []
    for site in np.unique(source[:, :2], axis=0):
        ids = np.flatnonzero(np.all(source[:, :2] == site, axis=1))
        ranked_sites.append(ids[np.lexsort((ids, -scores[ids]))])
    geometry_arrays, invariants, pool_scores = {}, {}, {}
    checked, reference_count, feasible_count = 0, 0, 0
    cases = []
    for job in manifest['jobs']:
        require(job['status'] == 'complete', 'Unfinished sweep condition')
        bend, count, angle = job['bend_length_m'], job['alternatives_per_site'], job['angle_deg']
        group = root / 'runs' / job['group']
        suite = json.loads((group / 'suite.json').read_text())
        restored = json.loads(json.dumps(suite))
        restored['geometry']['bend_length_m'] = base['geometry']['bend_length_m']
        restored['alternatives_per_site'] = base['alternatives_per_site']
        restored['angles_deg'] = base['angles_deg']
        require(restored == base, 'Uncontrolled suite change')
        require(suite['geometry']['bend_length_m'] == bend and suite['alternatives_per_site'] == count
                and suite['angles_deg'] == sweep['angles_deg'], 'Variant suite mismatch')
        selection = json.loads((group / 'source_selection.json').read_text())
        selected = [int(i) for ids in ranked_sites for i in ids[:count]]
        require(selection['alternative_ids'] == selected and selection['best_source_id'] == best_source,
                'Pool is not the prespecified source-only ranking')
        require(best_source in selected, 'Source winner omitted')
        directory = root / job['directory']
        local_manifest = json.loads((directory / 'manifest.json').read_text())
        require(local_manifest['status'] == 'complete' and local_manifest['angle_deg'] == angle, 'Angle manifest mismatch')
        require(digest(directory / 'geometry.npz') == local_manifest['geometry_sha256'], 'Geometry hash')
        require(json.loads((directory / 'requirements.json').read_text()) == base['requirements'],
                'Changed requirements')
        geometry = json.loads((directory / 'geometry.json').read_text())
        require(geometry['bend_length_m'] == bend and geometry['turn_deg'] == angle, 'Wrong executed geometry')
        for name, value in base['geometry'].items():
            if name != 'bend_length_m':
                require(geometry[name] == value, f'Changed fixed geometry: {name}')
        with np.load(directory / 'geometry.npz', allow_pickle=False) as data:
            poses, targets = data['poses'], data['targets']
            canonical, coordinates, modules = data['canonical_candidates'], data['target_coordinates'], data['module_ids']
            pool = selected + [i + len(source) for i in selected]
            require(data['library_pool'].tolist() == pool, 'Wrong target pool')
            shifted = source.copy()
            shifted[:, 0] += base['geometry']['length_m']/2
            require(np.array_equal(canonical, np.vstack([source, shifted])), 'Different transferred candidate domain')
            arrays = [data[name].copy() for name in ('vertices', 'triangles', 'poses', 'targets',
                                                    'canonical_candidates', 'target_coordinates', 'module_ids')]
        with np.load(directory / 'reference_masks.npz', allow_pickle=False) as data:
            masks = data['masks']
        require(masks.dtype == bool and masks.shape == (len(poses), len(targets)), 'Reference mask layout')
        arrays.append(masks)
        key = (bend, angle)
        if key in geometry_arrays:
            require(all(np.array_equal(a, b) for a, b in zip(geometry_arrays[key], arrays)),
                    'Geometry/visibility changed when only source pool size changed')
        else:
            geometry_arrays[key] = arrays
        best, feasible_pairs = exhaustive_values(poses, masks, modules, base['requirements'])
        results = {}
        for name in METHODS:
            result = json.loads((directory / name / 'result.json').read_text())
            results[name] = result
            if name.startswith('reference_'):
                require(digest(directory / 'reference_masks.npz') == result['reference_masks_sha256'], 'Reference hash')
                kind = name.removeprefix('reference_')
                value = result['assessment']['objective_score'] if result['assessment'] else None
                require((value is None and best[kind] is None) or (value is not None and best[kind] is not None
                    and abs(value - best[kind]) <= 1e-7), f'Independent optimum mismatch: {name}')
                pairs = len(poses)*(len(poses)-1)//2 if kind == 'global' else len(source)**2
                require(result['combinations_considered'] == pairs, 'Wrong pair search space')
                reference_count += 1
            if result['selected_ids'] is None:
                continue
            archive = directory / name / 'deployment.npz'
            require(digest(archive) == result['deployment_sha256'], 'Deployment hash')
            require(result['validated_direct_raycast'] is True, 'Missing independent scene validation')
            ids = result['selected_ids']
            require(len(ids) == 2 and len(set(ids)) == 2, 'Not two distinct sensors')
            with np.load(archive, allow_pickle=False) as data:
                require(data['candidate_ids'].tolist() == ids, 'Wrong sensor IDs')
                for field, truth in [('poses', poses[ids]), ('canonical', canonical[ids]),
                    ('targets', targets), ('target_coordinates', coordinates),
                    ('sensor_masks', masks[ids]), ('density', masks[ids].sum(axis=0))]:
                    require(np.array_equal(data[field], truth), f'Wrong deployment {field}')
                value, _ = exhaustive_values(data['poses'], data['sensor_masks'], modules[ids], base['requirements'])
                assessment = result['assessment']
                require((value['global'] is not None) == assessment['feasible'], 'Deployment feasibility mismatch')
                if assessment['feasible']:
                    require(abs(value['global'] - assessment['objective_score']) <= 1e-7, 'Deployment J mismatch')
                    require(best['global'] is not None and value['global'] <= best['global'] + 1e-7,
                            'Deployment exceeds full-grid optimum')
                    feasible_count += 1
                interface = np.abs(coordinates[:, 0] - base['geometry']['length_m']/2) <= base['interface_half_length_m']
                require(abs(100*np.mean(data['density'] > 0) - assessment['coverage_pct']) <= 1e-9, 'Coverage mismatch')
                require(abs(100*np.mean(data['density'][interface] > 0) - result['interface_coverage_pct']) <= 1e-9,
                        'Interface coverage mismatch')
                require(abs(np.mean(data['density'][interface] >= 2) - result['interface_overlap_ratio']) <= 1e-9,
                        'Interface overlap mismatch')
            checked += 1
        for name in ('independent_transfer', 'transfer_ls', 'reference_global', 'reference_one_per_module'):
            signature = (results[name]['selected_ids'], results[name]['assessment'])
            if (key, name) in invariants:
                require(invariants[key, name] == signature, 'Pool size changed a pool-independent method')
            else:
                invariants[key, name] = signature
        library = results['library_joint']['assessment']
        pool_scores.setdefault(key, {})[count] = library['objective_score'] if library else None
        cases.append(dict(bend_length_m=bend, angle_deg=angle, alternatives_per_site=count,
                          optimum=best, feasible_pairs=feasible_pairs))
    for key, values in pool_scores.items():
        previous = None
        for count, score in sorted(values.items()):
            if previous is not None:
                require(score is not None and score + 1e-7 >= previous, 'Nested pool reduced feasible best J')
            previous = score
    return dict(conditions=len(cases), unique_geometries=len(geometry_arrays), source_builds=1,
        checked_deployments=checked, feasible_deployments=feasible_count, reference_optima_checked=reference_count,
        all_reference_optima_match=True, all_pool_independent_methods_match=True, source_pools_nested=True,
        cases=cases, validator_sha256=digest(__file__),
        independent_pair_validator_sha256=digest(Path(__file__).with_name('validate_connected_corridor.py')),
        manifest_sha256=digest(root / 'manifest.json'),
        scope='Frozen inputs, saved masks, unpruned exhaustive feasibility/objective checks; no new raycasts.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, allow_nan=False))

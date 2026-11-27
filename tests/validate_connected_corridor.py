"""Read-only independent exhaustive audit of the default two-sensor pilot.

Uses saved masks, not new raycasts or the score-pruned selector. Unsupported
requirements fail closed. This checks the finite candidate reference, not physics.
"""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ('independent_transfer', 'transfer_ls', 'library_joint', 'library_joint_ls',
           'reference_one_per_module', 'reference_global')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exhaustive_values(poses, masks, modules, spec, *, candidate_allowed=None,
                      interface_mask=None, interface_thresholds=None):
    require(spec['budget_k'] == 2 and not spec.get('regions'), 'Only fixed-K=2, global constraints supported')
    require(not spec.get('cost', {}).get('regions'), 'Regional costs unsupported by independent audit')
    constraints = spec.get('constraints', {})
    supported = {'max_installation_cost', 'min_xy_separation_m', 'max_sensors_per_pole',
        'mounting_z_range_m', 'min_coverage_pct', 'min_overlap_ratio', 'max_overlap_ratio',
        'min_single_failure_coverage_pct'}
    require(not set(constraints) - supported, 'Unsupported independent-audit constraint')
    weights, cost = spec.get('weights', {}), spec.get('cost', {})
    require(not weights.get('robustness_weight'), 'Unsupported objective')
    if candidate_allowed is not None:
        require(candidate_allowed.dtype == bool and candidate_allowed.shape == (len(poses),), 'Invalid allowed mask')
    if interface_mask is not None:
        require(interface_mask.dtype == bool and interface_mask.shape == (masks.shape[1],)
                and interface_mask.any(), 'Invalid interface target mask')
        require(interface_thresholds is not None and not set(interface_thresholds) - {
            'min_coverage_pct', 'min_overlap_ratio', 'max_overlap_ratio'}, 'Unsupported interface constraint')
    else:
        require(not interface_thresholds, 'Interface thresholds require a target mask')
    individual = masks.sum(axis=1)
    costs = cost.get('base_per_sensor', 0.) + cost.get('per_height_m', 0.) * poses[:, 2]
    lo, hi = constraints.get('mounting_z_range_m', [0, float('inf')])
    best = {'global': None, 'one_per_module': None}
    feasible_counts = {'global': 0, 'one_per_module': 0}
    # Every pair receives independent counts and every enabled hard check.
    for i in range(len(poses) - 1):
        js = np.arange(i + 1, len(poses))
        covered = np.count_nonzero(masks[js] | masks[i], axis=1)
        double = np.count_nonzero(masks[js] & masks[i], axis=1)
        coverage = 100 * covered / masks.shape[1]
        overlap = double / masks.shape[1]
        surviving = 100 * np.minimum(individual[js], individual[i]) / masks.shape[1]
        distances = np.linalg.norm(poses[js, :2] - poses[i, :2], axis=1)
        total_cost = costs[js] + costs[i]
        valid = ((total_cost <= constraints.get('max_installation_cost', float('inf')) + 1e-9)
            & (distances + 1e-9 >= constraints.get('min_xy_separation_m', 0))
            & (coverage + 1e-9 >= constraints.get('min_coverage_pct', 0))
            & (overlap + 1e-9 >= constraints.get('min_overlap_ratio', 0))
            & (overlap <= constraints.get('max_overlap_ratio', 1) + 1e-9)
            & (surviving + 1e-9 >= constraints.get('min_single_failure_coverage_pct', 0))
            & (poses[i, 2] >= lo - 1e-9) & (poses[i, 2] <= hi + 1e-9)
            & (poses[js, 2] >= lo - 1e-9) & (poses[js, 2] <= hi + 1e-9)
            & np.any(poses[js] != poses[i], axis=1))
        if constraints.get('max_sensors_per_pole', 2) == 1:
            valid &= np.any(poses[js, :2] != poses[i, :2], axis=1)
        if candidate_allowed is not None:
            valid &= candidate_allowed[i] & candidate_allowed[js]
        if interface_mask is not None:
            first, others = masks[i, interface_mask], masks[js][:, interface_mask]
            band_coverage = 100*np.mean(others | first, axis=1)
            band_overlap = np.mean(others & first, axis=1)
            valid &= ((band_coverage + 1e-9 >= interface_thresholds.get('min_coverage_pct', 0))
                      & (band_overlap + 1e-9 >= interface_thresholds.get('min_overlap_ratio', 0))
                      & (band_overlap <= interface_thresholds.get('max_overlap_ratio', 1) + 1e-9))
        score = ((weights.get('coverage_weight', 1) + weights.get('importance_weight', 0)) * covered
            - weights.get('overlap_penalty', 0) * double
            - 2 * weights.get('sensor_count_penalty', 0)
            - weights.get('installation_cost_weight', 0) * total_cost)
        for name, use in [('global', valid), ('one_per_module', valid & (modules[js] != modules[i]))]:
            feasible_counts[name] += int(use.sum())
            if not use.any():
                continue
            value = float(score[use].max())
            if best[name] is None or value > best[name]:
                best[name] = value
    return best, feasible_counts


def audit(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['status'] == 'complete', 'Run must be complete')
    for name, expected in manifest['source_hashes'].items():
        require(digest(root / 'source' / name) == expected, f'Source hash: {name}')
    for name, expected in manifest['input_hashes'].items():
        require(digest(root / name) == expected, f'Input hash: {name}')
    suite = json.loads((root / 'inputs/suite.json').read_text())
    require(len(manifest['jobs']) == len(suite['angles_deg']), 'Incomplete angle grid')
    require([job['angle_deg'] for job in manifest['jobs']] == suite['angles_deg'], 'Unexpected angles')
    source_meta = json.loads((root / 'library/metadata.json').read_text())
    require(digest(root / 'library/library.npz') == source_meta['source_pose_mask_hash'], 'Source library hash')
    checked, angles = 0, []
    for job in manifest['jobs']:
        require(job['status'] == 'complete', 'Unfinished angle')
        directory = root / job['directory']
        geom_meta = json.loads((directory / 'manifest.json').read_text())
        require(digest(directory / 'geometry.npz') == geom_meta['geometry_sha256'], 'Geometry archive hash')
        require(json.loads((directory / 'requirements.json').read_text()) == suite['requirements'],
                'Requirements changed within angle')
        with np.load(directory / 'geometry.npz', allow_pickle=False) as data:
            poses, targets = data['poses'], data['targets']
            canonical, coordinates = data['canonical_candidates'], data['target_coordinates']
            modules = data['module_ids']
        with np.load(directory / 'reference_masks.npz', allow_pickle=False) as data:
            masks = data['masks']
        require(masks.dtype == bool and masks.shape == (len(poses), len(targets)), 'Invalid reference masks')
        best, feasible = exhaustive_values(poses, masks, modules, suite['requirements'])
        for name in METHODS:
            result = json.loads((directory / name / 'result.json').read_text())
            if name.startswith('reference_'):
                require(digest(directory / 'reference_masks.npz') == result['reference_masks_sha256'],
                        'Reference mask hash')
                key = name.removeprefix('reference_')
                actual = result['assessment']['objective_score'] if result['assessment'] else None
                require((actual is None and best[key] is None)
                    or (actual is not None and best[key] is not None and abs(actual - best[key]) <= 1e-7),
                    f'Independent exhaustive optimum mismatch: {name}')
                expected_pairs = (len(poses)*(len(poses)-1)//2 if key == 'global'
                                  else int(np.count_nonzero(modules == 0) * np.count_nonzero(modules == 1)))
                require(result['combinations_considered'] == expected_pairs, 'Incorrect search domain size')
            if result['selected_ids'] is None:
                continue
            path = directory / name / 'deployment.npz'
            require(digest(path) == result['deployment_sha256'], 'Deployment hash')
            require(result['validated_direct_raycast'] is True, 'Missing direct-raycast validation')
            with np.load(path, allow_pickle=False) as data:
                ids = data['candidate_ids']
                require(ids.tolist() == result['selected_ids'], 'Candidate ID mismatch')
                require(len(ids) == 2 and len(np.unique(ids)) == 2, 'Not two distinct real sensors')
                for saved, expected in [('poses', poses[ids]), ('canonical', canonical[ids]),
                    ('targets', targets), ('target_coordinates', coordinates),
                    ('sensor_masks', masks[ids]), ('density', masks[ids].sum(axis=0))]:
                    require(np.array_equal(data[saved], expected), f'{name}: {saved} mismatch')
                pair_best, _ = exhaustive_values(data['poses'], data['sensor_masks'], modules[ids],
                                                suite['requirements'])
                a = result['assessment']
                require((pair_best['global'] is not None) == a['feasible'], 'Feasibility mismatch')
                if a['feasible']:
                    require(abs(pair_best['global'] - a['objective_score']) <= 1e-7, 'Objective mismatch')
                require(abs(100*np.mean(data['density'] > 0) - a['coverage_pct']) <= 1e-9, 'Coverage mismatch')
                interface = np.abs(coordinates[:, 0] - suite['geometry']['length_m']/2) <= suite['interface_half_length_m']
                require(abs(100*np.mean(data['density'][interface] > 0) - result['interface_coverage_pct']) <= 1e-9,
                        'Interface coverage mismatch')
            checked += 1
        angles.append(dict(angle_deg=job['angle_deg'], optimum=best, feasible_pairs=feasible))
    return dict(checked_deployments=checked, checked_angles=len(angles),
        all_reference_optima_match=True, independent_full_pair_evaluation=angles,
        validator_sha256=digest(__file__), manifest_sha256=digest(root / 'manifest.json'),
        scope='Stored masks/arrays/hashes, independent unpruned enumeration. No new raycasts or training.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2, allow_nan=False))

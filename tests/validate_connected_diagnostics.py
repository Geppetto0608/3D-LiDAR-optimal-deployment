"""Independent saved-mask enumeration of diagnostic domain losses and omissions.

Uses the original run's frozen unpruned auditor, not the new pair-table code.
The component-optimum check uses a sandwich proof when LS equals the complete
anchor-domain optimum; other inputs fail closed rather than assume reachability.
"""
from argparse import ArgumentParser
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.roadside.corridor_requirements import CorridorRequirements


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    require(manifest['status'] == 'complete', 'Completed diagnostic required')
    root = Path(manifest['input_root'])
    provenance = json.loads((directory/'provenance.json').read_text())
    for name, value in provenance['input_hashes'].items():
        require(digest(root/name) == value, f'Original input changed: {name}')
    for name, value in provenance['code_hashes'].items():
        require(digest(directory/'source'/name) == value, f'Diagnostic snapshot changed: {name}')
    original = json.loads((root/'manifest.json').read_text())
    frozen = root/'source/tests/validate_connected_corridor.py'
    require(digest(frozen) == original['source_hashes']['tests/validate_connected_corridor.py'], 'Frozen auditor changed')
    module_spec = importlib.util.spec_from_file_location('frozen_corridor_auditor', frozen)
    independent = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(independent)
    base = json.loads((root/'inputs/base_suite.json').read_text())
    details = json.loads((directory/'case_diagnostics.json').read_text())
    with (directory/'summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r['profile'], r['target'], r['pool']): r for r in rows}
    require(len(lookup) == len(rows) == 2*len(original['jobs']), 'Missing/duplicate diagnostic rows')
    require([d['job'] for d in details] == original['jobs'], 'Diagnostic conditions changed')
    domain_checks, omission_checks, component_checks, witness_checks = 0, 0, 0, 0
    for case in details:
        job = case['job']
        folder = root/job['directory']
        specification = json.loads((folder/'requirements.json').read_text())
        profile = json.loads((folder/'corridor_profile.json').read_text())
        geometry = json.loads((folder/'geometry.json').read_text())
        with np.load(folder/'geometry.npz', allow_pickle=False) as archive:
            poses, canonical = archive['poses'], archive['canonical_candidates']
            coordinates, targets, modules = archive['target_coordinates'], archive['targets'], archive['module_ids']
            allowed = archive['allowed_candidates']
            pools = {kind: archive[kind+'_pool'].tolist() for kind in ('top', 'diverse')}
        with np.load(folder/'reference_masks.npz', allow_pickle=False) as archive:
            masks = archive['masks']
        interface = np.abs(coordinates[:, 0]-geometry['length_m']/2) <= base['interface_half_length_m']
        options = dict(candidate_allowed=allowed, interface_mask=interface, interface_thresholds=profile['interface'])
        best, _ = independent.exhaustive_values(poses, masks, modules, specification, **options)
        require(abs(best['global']-case['global_optimum']['score']) <= 1e-7, 'Independent global J mismatch')
        req = CorridorRequirements(specification, targets, coordinates, canonical, poses, profile,
            length_m=geometry['length_m'], interface_half_length_m=base['interface_half_length_m'])
        require(np.array_equal(req.allowed, allowed), 'Independent eligibility mismatch')
        # Check every quota-optimal pair, not just the original selected optimum.
        optimal_pairs = []
        weights, cost = specification.get('weights', {}), specification.get('cost', {})
        costs = cost.get('base_per_sensor', 0)+cost.get('per_height_m', 0)*poses[:, 2]
        for i in range(len(poses)-1):
            js = np.arange(i+1, len(poses))
            union = np.count_nonzero(masks[js] | masks[i], axis=1)
            double = np.count_nonzero(masks[js] & masks[i], axis=1)
            values = ((weights.get('coverage_weight', 1)+weights.get('importance_weight', 0))*union
                - weights.get('overlap_penalty', 0)*double-2*weights.get('sensor_count_penalty', 0)
                - weights.get('installation_cost_weight', 0)*(costs[i]+costs[js]))
            for j in js[(np.abs(values-best['one_per_module']) <= 1e-7) & (modules[js] != modules[i])]:
                ids = [i, int(j)]
                if req.evaluate(poses[ids], masks[ids].sum(axis=0), sensor_masks=masks[ids]).feasible:
                    optimal_pairs.append(ids)
        require(optimal_pairs, 'Expected a feasible quota optimum')
        for kind in ('top', 'diverse'):
            value = case['methods'][kind]
            omitted, search = value['omission'], value['search']
            key = (job['profile'], job.get('target_name', str(job['angle_deg'])), kind)
            row = lookup[key]
            require(row['status'] == 'complete', 'This validator expects feasible inputs')
            pool = pools[kind]
            pool_best, _ = independent.exhaustive_values(poses[pool], masks[pool], modules[pool], specification,
                candidate_allowed=allowed[pool], interface_mask=interface, interface_thresholds=profile['interface'])
            require(abs(pool_best['one_per_module']-omitted['restricted']['score']) <= 1e-7, 'Independent pool J mismatch')
            missing = [sum(i not in pool for i in pair) for pair in optimal_pairs]
            require(len(optimal_pairs) == omitted['optimal_pair_count'], 'Optimal tie count mismatch')
            require(min(missing) == omitted['minimum_target_candidate_additions'], 'Minimum missing candidate count mismatch')
            require(omitted['witness_ids'] in optimal_pairs and sum(i not in pool for i in omitted['witness_ids']) == min(missing),
                    'Wrong minimum-addition witness')
            omission_checks += 1
            saved = json.loads((folder/(kind+'_joint_ls')/'result.json').read_text())
            initial, final = saved['initialization']['selected_ids'], saved['selected_ids']
            current_j = saved['assessment']['objective_score']
            side_domains = [np.flatnonzero(canonical[:, 1] == canonical[cid, 1]) for cid in initial]
            anchor_domains = [ids[np.abs(canonical[ids, 0]-canonical[cid, 0]) <= base['local_station_radius_m']+1e-9]
                              for ids, cid in zip(side_domains, initial)]
            verified = []
            for name, domains in [('fixed_side_optimum', side_domains), ('anchor_optimum', anchor_domains)]:
                ids = np.concatenate(domains)
                labels = np.r_[np.zeros(len(domains[0]), dtype=int), np.ones(len(domains[1]), dtype=int)]
                result, _ = independent.exhaustive_values(poses[ids], masks[ids], labels, specification,
                    candidate_allowed=allowed[ids], interface_mask=interface, interface_thresholds=profile['interface'])
                require(abs(result['one_per_module']-search[name]['score']) <= 1e-7, f'Independent {name} mismatch')
                verified.append(result['one_per_module'])
                domain_checks += 1
            # Any feasible component is a subset of the full anchor domain and
            # contains the feasible saved endpoint. Equal bounds certify its J.
            require(abs(verified[-1]-current_j) <= 1e-7,
                'Sandwich certificate unavailable: independent component traversal is required for this input')
            component = search['feasible_component_optimum']
            require(abs(component['score']-current_j) <= 1e-7 and abs(component['minimum_objective_dip']) <= 1e-7,
                    'Component optimum/dip mismatch')
            require(abs(search['one_agent_relocation_optimum']['score']-current_j) <= 1e-7,
                    'One-agent relocation exceeded its enclosing anchor domain')
            require(component['witness_path'] == [final], 'Expected a trivial optimal-endpoint witness path')
            actual = req.evaluate(poses[final], masks[final].sum(axis=0), sensor_masks=masks[final])
            require(actual.feasible and abs(actual.objective_score-current_j) <= 1e-7, 'Endpoint feasibility mismatch')
            component_checks += 1
            losses = [best['global']-verified[0], verified[0]-verified[1], 0., 0.]
            for name, expected in zip(('fixed_side_loss_j', 'anchor_loss_j', 'disconnected_loss_j', 'local_search_loss_j'), losses):
                require(abs(float(row[name])-expected) <= 1e-7, f'CSV loss mismatch: {name}')
            require(abs(sum(losses)-float(row['ls_gap_j'])) <= 1e-7, 'CSV total loss mismatch')
            for pair in [omitted['witness_ids'], search['fixed_side_optimum']['selected_ids'], search['anchor_optimum']['selected_ids']]:
                check = req.evaluate(poses[pair], masks[pair].sum(axis=0), sensor_masks=masks[pair])
                require(check.feasible, 'Infeasible counterfactual witness')
                witness_checks += 1
    return dict(status='passed', conditions=len(details), diagnostic_rows=len(rows),
        independent_domain_optima=domain_checks, all_optimal_pair_omission_checks=omission_checks,
        component_optimum_sandwich_checks=component_checks, feasible_witness_checks=witness_checks,
        original_inputs_unchanged=True, validator_sha256=digest(__file__),
        original_frozen_enumerator_sha256=digest(frozen),
        diagnostic_provenance_sha256=digest(directory/'provenance.json'),
        scope='Stored arrays only; original frozen exhaustive enumerator plus shared-evaluator witnesses. '
              'Component J certified by equal anchor-domain upper and feasible endpoint lower bounds; '
              'reachable-state counts are not independently audited.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    if args.output:
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))

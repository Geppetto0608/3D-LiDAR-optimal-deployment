"""Saved-result audit: source compression, active constraints, LS and physical checks.

Counterfactuals use the complete saved target masks. They are diagnostics, NOT
new online algorithms or matched-cost performance measurements.
"""
from argparse import ArgumentParser
import csv
from itertools import combinations
import json
from math import comb
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

import connected_corridor_study as corridor
from src.methods.connected_diagnostics import best_assignment, pair_table
from src.methods.connected_pairs import canonical_local_search
from src.roadside.corridor_requirements import CorridorRequirements

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'tests'))
from validate_connected_constraints import audit, require


def compression_audit(canonical, masks, scores, pools, count, coefficient, *, preserve_seed=True):
    integer = masks.astype(np.int64)
    responses = np.column_stack([scores, scores[:, None]-coefficient*(integer @ integer.T)])
    exact_pool, rows = [], []
    for site in np.unique(canonical[:, :2], axis=0):
        same = np.all(canonical[:, :2] == site, axis=1)
        ids = np.flatnonzero(same)
        require(1 <= count <= len(ids) and comb(len(ids), count) <= 10000, 'Compression diagnostic allocation limit')
        contexts = np.r_[0, np.flatnonzero(~same)+1]
        values = responses[np.ix_(ids, contexts)]
        best = values.max(axis=0)
        seed = int(np.argmax(scores[ids]))

        def loss(indices):
            regret = np.maximum(best-values[list(indices)].max(axis=0), 0.)
            return float(regret.max()), float(regret.mean())

        options = []
        for subset in combinations(range(len(ids)), count):
            if not preserve_seed or seed in subset:
                worst, mean = loss(subset)
                options.append((round(worst, 9), round(mean, 9), tuple(map(int, ids[list(subset)])), worst, mean))
        optimum = min(options)
        optimal_subsets = [list(option[2]) for option in options
            if abs(option[3]-optimum[3]) <= 1e-8 and abs(option[4]-optimum[4]) <= 1e-8]
        exact_pool.extend(optimum[2])
        row = dict(site=site.tolist(), candidates=len(ids), contexts=len(contexts),
            subsets_checked=len(options), seed_id=int(ids[seed]), exact_ids=list(optimum[2]),
            exact_worst_regret=optimum[3], exact_mean_regret=optimum[4], optimal_subsets=optimal_subsets,
            preserve_seed=preserve_seed)
        for kind, pool in pools.items():
            selected = [int(np.flatnonzero(ids == i)[0]) for i in pool if same[i]]
            require(len(selected) == count, 'Unequal source site budget')
            worst, mean = loss(selected)
            row.update({f'{kind}_worst_regret': worst, f'{kind}_mean_regret': mean})
        rows.append(row)
    return exact_pool, rows


def compression_family_domain(canonical, rows):
    """Optimistic target oracle over ALL tied source-optimal site subsets.

    The two target modules share one source library. If both sensors use the
    same source site, a SINGLE optimal subset must contain both source IDs.
    """
    union, compatibility = [], np.ones((2*len(canonical), 2*len(canonical)), bool)
    for row in rows:
        families = [set(s) for s in row['optimal_subsets']]
        source_ids = np.flatnonzero(np.all(canonical[:, :2] == row['site'], axis=1))
        ids = np.r_[source_ids, source_ids+len(canonical)]
        present = set.union(*families)
        union.extend(sorted(present))
        for i in ids:
            for j in ids:
                required = {int(i % len(canonical)), int(j % len(canonical))}
                compatibility[i, j] = any(required <= subset for subset in families)
    return union+[i+len(canonical) for i in union], compatibility


def constraint_audit(poses, masks, allowed, interface, spec, thresholds, table):
    """Count constraint-specific exclusions and remove-one checks on the FULL domain."""
    c = spec.get('constraints', {})
    n, target_count = masks.shape
    domain = np.triu(np.ones((n, n), bool), 1) & np.any(poses[:, None] != poses[None, :], axis=2)
    checks = {'installation_eligibility': allowed[:, None] & allowed[None, :]}
    if 'max_installation_cost' in c:
        checks['installation_cost'] = table.costs[:, None]+table.costs[None, :] <= c['max_installation_cost']+1e-9
    if 'min_xy_separation_m' in c:
        checks['min_xy_separation'] = np.linalg.norm(poses[:, None, :2]-poses[None, :, :2], axis=2)+1e-9 >= c['min_xy_separation_m']
    if 'max_sensors_per_pole' in c:
        checks['pole_capacity'] = (np.any(poses[:, None, :2] != poses[None, :, :2], axis=2)
                                   if c['max_sensors_per_pole'] == 1 else np.ones((n, n), bool))
    if 'mounting_z_range_m' in c:
        lo, hi = c['mounting_z_range_m']
        valid = (poses[:, 2]+1e-9 >= lo) & (poses[:, 2] <= hi+1e-9)
        checks['mounting_height'] = valid[:, None] & valid[None, :]
    for key, array, minimum in [('min_coverage_pct', 100*table.covered/target_count, True),
        ('min_overlap_ratio', table.double/target_count, True), ('max_overlap_ratio', table.double/target_count, False)]:
        if key in c:
            checks[key] = array+1e-9 >= c[key] if minimum else array <= c[key]+1e-9
    if 'min_single_failure_coverage_pct' in c:
        individual = 100*masks.mean(axis=1)
        checks['single_failure_coverage'] = np.minimum(individual[:, None], individual[None, :])+1e-9 >= c['min_single_failure_coverage_pct']
    band = masks[:, interface].astype(np.int64)
    double = band @ band.T
    union = band.sum(axis=1)[:, None]+band.sum(axis=1)[None, :]-double
    for key, array, minimum in [('min_coverage_pct', 100*union/band.shape[1], True),
        ('min_overlap_ratio', double/band.shape[1], True), ('max_overlap_ratio', double/band.shape[1], False)]:
        if key in thresholds:
            checks['interface_'+key] = array+1e-9 >= thresholds[key] if minimum else array <= thresholds[key]+1e-9
    full = domain & np.logical_and.reduce(list(checks.values()))
    require(np.array_equal(full, domain & table.feasible), 'Independent per-constraint decomposition mismatch')
    best = float(table.scores[full].max()) if full.any() else None
    rows = []
    for name, valid in checks.items():
        others = [value for key, value in checks.items() if key != name]
        relaxed = domain & np.logical_and.reduce(others) if others else domain
        relaxed_best = float(table.scores[relaxed].max()) if relaxed.any() else None
        rows.append(dict(constraint=name, rejected_without_other_constraints=int((domain & ~valid).sum()),
            feasible_pairs=int(full.sum()), additional_feasible_pairs_if_removed=int((relaxed & ~full).sum()),
            reference_j=best, j_if_removed=relaxed_best,
            j_increase_if_removed=relaxed_best-best if best is not None and relaxed_best is not None else None))
    return rows


def optical_mask(pose, targets, profile, *, tolerance=0.):
    """Independent 360-degree tilted-plane dot product, no shared angle helper."""
    require(profile['h_fov_deg'] == 360, 'Physical audit currently supports 360-degree profiles only')
    yaw, pitch = np.deg2rad(pose[3:5])
    normal = np.array([-np.sin(pitch)*np.cos(yaw), -np.sin(pitch)*np.sin(yaw), np.cos(pitch)])
    vector = targets.astype(float)-pose[:3]
    distance = np.linalg.norm(vector, axis=1)
    elevation_sine = np.abs(vector @ normal)/np.maximum(distance, 1e-30)
    return ((distance > 1e-6) & (distance <= profile['max_range_m']+tolerance)
            & (elevation_sine <= np.sin(np.deg2rad(profile['v_fov_deg']/2))+tolerance))


def segment_blocked(origin, target, vertices, triangles):
    """Independent two-sided segment/triangle intersection; no Open3D calls."""
    delta = np.asarray(target, float)-origin
    distance = np.linalg.norm(delta)
    direction = delta/distance
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    e1, e2 = b-a, c-a
    p = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, p)
    valid = np.abs(det) > 1e-10
    inverse = np.divide(1., det, out=np.zeros_like(det), where=valid)
    offset = origin-a
    u = np.einsum('ij,ij->i', offset, p)*inverse
    q = np.cross(offset, e1)
    v = (q @ direction)*inverse
    t = np.einsum('ij,ij->i', e2, q)*inverse
    valid &= (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (t > 1e-7) & (t < distance-1e-3)
    return bool(valid.any())


def independent_road_frame(coordinates, geometry):
    """Integrate the tangent of the stated straight/arc/straight road independently."""
    angle = np.deg2rad(geometry['turn_deg'])
    length, bend = geometry['length_m'], geometry['bend_length_m']
    start, end = (length-bend)/2, (length+bend)/2
    centers, headings = [], []
    for station in coordinates[:, 0]:
        if not angle or station <= start:
            x, y, heading = station, 0., 0.
        elif station <= end:
            radius = bend/angle
            heading = (station-start)/radius
            x, y = start+radius*np.sin(heading), radius*(1-np.cos(heading))
        else:
            radius = bend/angle
            x = start+radius*np.sin(angle)+(station-end)*np.cos(angle)
            y = radius*(1-np.cos(angle))+(station-end)*np.sin(angle)
            heading = angle
        centers.append([x, y])
        headings.append(heading)
    headings = np.asarray(headings)
    normal = np.column_stack([-np.sin(headings), np.cos(headings)])
    return np.column_stack([np.asarray(centers)+coordinates[:, 1, None]*normal, coordinates[:, 2]]), headings


def physical_audit(archive, masks, geometry, profile, interface_half):
    poses, targets = archive['poses'], archive['targets']
    canonical, coordinates = archive['canonical_candidates'], archive['target_coordinates']
    expected, heading = independent_road_frame(canonical[:, :3], geometry)
    require(np.allclose(expected, poses[:, :3], atol=1e-8, rtol=0), 'Independent canonical/world position mismatch')
    yaw = np.rad2deg(heading)+np.where(canonical[:, 1] > 0, -90, 90)+canonical[:, 3]
    require(np.all(np.abs((poses[:, 3]-yaw+180)%360-180) < 1e-8)
            and np.array_equal(poses[:, 4], canonical[:, 4]), 'Independent attitude transfer mismatch')
    expected_targets, _ = independent_road_frame(coordinates, geometry)
    require(np.allclose(expected_targets, targets, atol=1e-5, rtol=0), 'Independent target coordinate mismatch')
    optical = np.asarray([optical_mask(p, targets, profile) for p in poses])
    loose = np.asarray([optical_mask(p, targets, profile, tolerance=1e-6) for p in poses])
    require(not (masks & ~loose).any(), 'Stored visibility violates independent range/FOV')
    # Each candidate class gets deterministic visible, occluded and out-of-FOV samples.
    rays = 0
    for cid in np.unique(np.linspace(0, len(poses)-1, min(8, len(poses)), dtype=int)):
        for category in (masks[cid], optical[cid] & ~masks[cid], ~loose[cid]):
            candidates = np.flatnonzero(category)
            for tid in candidates[np.unique(np.linspace(0, len(candidates)-1, min(8, len(candidates)), dtype=int))] if len(candidates) else []:
                blocked = segment_blocked(poses[cid, :3], targets[tid], archive['vertices'], archive['triangles'])
                visible = optical[cid, tid] and not blocked
                require(visible == masks[cid, tid], f'Independent triangle visibility mismatch: {cid}/{tid}')
                rays += 1
    half = geometry['length_m']/2
    other_module = (canonical[:, 0, None] < half) != (coordinates[None, :, 0] < half)
    cross = masks & other_module
    distance = np.abs(coordinates[:, 0]-half)
    curvature = np.deg2rad(geometry['turn_deg'])/geometry['bend_length_m'] if geometry['bend_length_m'] else 0.
    inside = distance <= geometry['bend_length_m']/2
    area = np.where(inside, 1-curvature*coordinates[:, 1], 1.)
    return dict(candidate_target_entries=int(masks.size), independent_triangle_checks=rays,
        optically_possible_but_occluded=int((optical & ~masks).sum()),
        cross_module_visible_entries=int(cross.sum()),
        cross_module_entries_outside_service_band=int((cross & (distance > interface_half)).sum()),
        farthest_cross_module_target_station_offset_m=float(distance[np.any(cross, axis=0)].max()) if cross.any() else None,
        area_jacobian_min=float(area.min()), area_jacobian_max=float(area.max()))


def csv_save(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def run(root, output):
    manifest = json.loads((root/'manifest.json').read_text())
    originals = {str(p.relative_to(root)): corridor.digest(p) for p in root.rglob('*') if p.is_file()}
    output.mkdir(parents=True, exist_ok=False)
    source_files = list(dict.fromkeys(corridor.SOURCE_FILES + ['connected_validation_audit.py',
        'src/methods/connected_diagnostics.py', 'src/roadside/__init__.py', 'src/roadside/corridor_requirements.py',
        'tests/validate_connected_constraints.py', 'tests/validate_connected_corridor.py']))
    for name in source_files:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/name, destination)
    started = time.perf_counter()
    state = dict(status='running', input_root=str(root), started_at=corridor.now())
    corridor.save(output/'manifest.json', state)
    try:
        checked = audit(root)
        corridor.save(output/'strengthened_validation.json', checked)
        base = json.loads((root/'inputs/base_suite.json').read_text())
        cfg = yaml.safe_load((root/'inputs/experiment.yaml').read_text())
        profile = cfg['lidar_profiles'][cfg['lidar_profile']]
        pools = json.loads((root/'source_pools.json').read_text())['pools']
        with np.load(root/'library/library.npz', allow_pickle=False) as data:
            source, masks, scores = data['canonical'], data['masks'], data['scores']
        weights = base['requirements'].get('weights', {})
        coefficient = weights.get('coverage_weight', 1)+weights.get('importance_weight', 0)+weights.get('overlap_penalty', 0)
        exact, compression = compression_audit(source, masks, scores, pools,
            manifest['plan']['source_alternatives_per_site'], coefficient)
        free_exact, free_compression = compression_audit(source, masks, scores, pools,
            manifest['plan']['source_alternatives_per_site'], coefficient, preserve_seed=False)
        family_ids, family_compatibility = compression_family_domain(source, compression)
        free_family_ids, free_family_compatibility = compression_family_domain(source, free_compression)
        corridor.save(output/'source_compression.json', dict(rows=compression, exact_seeded_pool=exact,
            unseeded_rows=free_compression, exact_unseeded_pool=free_exact,
            tie_tolerance_decimal_places=9, scope='Source-only exact compression; seeded/unseeded diagnostics, no target inputs'))
        rows, constraints, physics, losses = [], [], [], []
        seen = set()
        replayed = 0
        for job in manifest['jobs']:
            directory = root/job['directory']
            g = np.load(directory/'geometry.npz', allow_pickle=False)
            with np.load(directory/'reference_masks.npz', allow_pickle=False) as data:
                masks = data['masks']
            corridor_profile = json.loads((directory/'corridor_profile.json').read_text())
            geometry = json.loads((directory/'geometry.json').read_text())
            poses, canonical, coordinates, modules = g['poses'], g['canonical_candidates'], g['target_coordinates'], g['module_ids']
            interface = np.abs(coordinates[:, 0]-geometry['length_m']/2) <= base['interface_half_length_m']
            req = CorridorRequirements(base['requirements'], g['targets'], coordinates, canonical, poses,
                corridor_profile, length_m=geometry['length_m'], interface_half_length_m=base['interface_half_length_m'])
            table = pair_table(poses, masks, base['requirements'], g['allowed_candidates'], interface, corridor_profile['interface'])
            identity = dict(profile=job['profile'], target=job.get('target_name', str(job['angle_deg'])))
            for row in constraint_audit(poses, masks, g['allowed_candidates'], interface,
                                       base['requirements'], corridor_profile['interface'], table):
                constraints.append({**identity, **row})
            ids = exact+[i+len(source) for i in exact]
            quota = modules[:, None] != modules[None, :]
            counterfactual = best_assignment(table, ids, ids, quota)
            family = best_assignment(table, family_ids, family_ids, quota & family_compatibility)
            free_ids = free_exact+[i+len(source) for i in free_exact]
            free_counterfactual = best_assignment(table, free_ids, free_ids, quota)
            free_family = best_assignment(table, free_family_ids, free_family_ids, quota & free_family_compatibility)
            raw = json.loads((directory/'response_joint/result.json').read_text())
            reference = json.loads((directory/'reference_global/result.json').read_text())
            rows.append(dict(**identity, recorded_response_j=raw['assessment']['objective_score'] if raw['assessment'] else None,
                exact_source_replay_j=counterfactual['score'], selected_ids=counterfactual['selected_ids'],
                best_source_optimal_family_j=family['score'],
                unseeded_exact_source_replay_j=free_counterfactual['score'],
                best_unseeded_source_optimal_family_j=free_family['score'],
                reference_j=reference['assessment']['objective_score'] if reference['assessment'] else None,
                scope='Offline counterfactual with saved full masks; not online benchmark'))
            for kind, pool in pools.items():
                saved = json.loads((directory/f'{kind}_joint_ls/result.json').read_text())
                initial = saved['initialization']['selected_ids']
                if initial is None:
                    continue
                visited = set(req.eligible_ids(pool+[i+len(source) for i in pool]))
                def mask_for(i):
                    visited.add(i)
                    return masks[i]
                replay = canonical_local_search(initial, canonical, poses, mask_for, req,
                    station_radius_m=base['local_station_radius_m'])
                require(replay['selected_ids'] == saved['selected_ids']
                        and json.loads(json.dumps(replay['assessment'])) == saved['assessment'], 'LS replay differs')
                require(len(visited) == saved['work']['unique_candidate_queries'], 'LS query replay differs')
                for field in ('neighbor_evaluations', 'sweeps', 'stopping_reason'):
                    require(replay[field] == saved[field], f'LS replay {field} differs')
                sides = [np.flatnonzero(canonical[:, 1] == canonical[i, 1]) for i in initial]
                anchors = [group[np.abs(canonical[group, 0]-canonical[i, 0]) <= base['local_station_radius_m']+1e-9]
                           for group, i in zip(sides, initial)]
                same_side, same_anchor = best_assignment(table, *sides), best_assignment(table, *anchors)
                losses.append(dict(**identity, pool=kind, global_j=reference['assessment']['objective_score'],
                    fixed_side_j=same_side['score'], fixed_anchor_j=same_anchor['score'], saved_ls_j=saved['assessment']['objective_score']))
                replayed += 1
            key = json.dumps(geometry, sort_keys=True)
            if key not in seen:
                physics.append(dict(**identity, **physical_audit(g, masks, geometry, profile, base['interface_half_length_m'])))
                seen.add(key)
            g.close()
        for name, values in [('counterfactual.csv', rows), ('constraint_activity.csv', constraints),
                             ('physical_checks.csv', physics), ('ls_domains.csv', losses)]:
            csv_save(output/name, values)
        require(all(corridor.digest(root/name) == expected for name, expected in originals.items()), 'Original study was modified')
        summary = dict(conditions=len(rows), source_sites=len(compression),
            source_greedy_suboptimal_sites=sum(r['response_worst_regret'] > r['exact_worst_regret']+1e-8 for r in compression),
            exact_source_counterfactual_improvements=sum(r['exact_source_replay_j'] > r['recorded_response_j']+1e-8 for r in rows
                if r['recorded_response_j'] is not None and r['exact_source_replay_j'] is not None),
            exact_source_counterfactual_ties=sum(abs(r['exact_source_replay_j']-r['recorded_response_j']) <= 1e-8 for r in rows
                if r['recorded_response_j'] is not None and r['exact_source_replay_j'] is not None),
            source_optimal_family_misses=sum(r['best_source_optimal_family_j'] < r['reference_j']-1e-8 for r in rows
                if r['best_source_optimal_family_j'] is not None and r['reference_j'] is not None),
            unseeded_source_optimal_family_misses=sum(r['best_unseeded_source_optimal_family_j'] < r['reference_j']-1e-8 for r in rows
                if r['best_unseeded_source_optimal_family_j'] is not None and r['reference_j'] is not None),
            ls_replays=replayed, ls_with_anchor_value_gap=sum(r['fixed_anchor_j'] > r['saved_ls_j']+1e-8 for r in losses),
            unique_geometries=len(physics), independent_triangle_checks=sum(r['independent_triangle_checks'] for r in physics),
            frozen_inputs_unchanged=len(originals), scope='Diagnostic only, no new training or target optimization benchmark')
        corridor.save(output/'summary.json', summary)
        lines = ['# Extended Validation Audit', '', 'Diagnostic on observed stored results, not a new algorithm benchmark.', '',
            f"- Strengthened saved-result audit: {checked['checked_deployments']} deployments passed.",
            f"- Exact seeded source compression: greedy misses {summary['source_greedy_suboptimal_sites']}/{len(compression)} sites.",
            f"- Exact-source offline target replay: {summary['exact_source_counterfactual_improvements']} improvements, {summary['exact_source_counterfactual_ties']} ties.",
            f"- Even a target oracle over tied source-optimal subsets misses {summary['source_optimal_family_misses']} full-grid optima.",
            f"- Without preserving the solo-best seed, that oracle still misses {summary['unseeded_source_optimal_family_misses']} full-grid optima.",
            f"- LS replay: {replayed} outputs and query counts agree; {summary['ls_with_anchor_value_gap']} have an anchor-domain value gap.",
            f"- Independent range/FOV plus sampled segment-triangle checks: {len(physics)} geometries, {summary['independent_triangle_checks']} segments.", '',
            'Constraint activity measures added feasible pairs and optimum change when ONE constraint is removed.',
            'A redundant height constraint may simply restate the candidate domain. Do not count every configured',
            'constraint as an independent source of optimization difficulty.', '',
            'The interface band is a service ROI, not a certified interaction boundary. Inspect physical_checks.csv',
            'for cross-module visibility beyond that band. Segment checks use the saved mesh, not real-road truth.',
            'Source context error, compression approximation and restricted refinement are distinct mechanisms.',
            'No original result files are changed. See CSV/JSON diagnostics and frozen audit source.', '']
        (output/'REPORT.md').write_text('\n'.join(lines))
        corridor.save(output/'provenance.json', dict(input_hashes=originals,
            source_hashes={name: corridor.digest(output/'source'/name) for name in source_files}))
        state.update(status='complete', finished_at=corridor.now(), wall_sec=time.perf_counter()-started)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), wall_sec=time.perf_counter()-started)
        raise
    finally:
        corridor.save(output/'manifest.json', state)
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE/'results/connected_response_study_20261022')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root/'validation_audit_20261022'
    require(output != root and output not in root.parents, 'Output must not replace input root or an ancestor')
    manifest = json.loads((root/'manifest.json').read_text())
    require(manifest['status'] == 'complete' and manifest['plan'].get('response_pool'), 'Complete response study required')
    print(json.dumps(dict(conditions=len(manifest['jobs']), execute=args.execute, output=str(output),
        scope='Saved arrays plus bounded independent geometry/segment checks; no training or visibility cache build'), indent=2))
    if args.execute:
        run(root, output)


if __name__ == '__main__':
    main()

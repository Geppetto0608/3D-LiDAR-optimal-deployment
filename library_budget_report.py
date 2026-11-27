"""Read-only audit and descriptive report for a completed low-K cost study.

Imports frozen geometry/optics, never solves or trains. --direct reraycasts every
target candidate. All outputs must live outside the original experiment root.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import csv
import hashlib
import json
import math
import sys
import time
import traceback

import numpy as np
import pandas as pd


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def same(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key, value in expected.items():
            same(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for a, b in zip(actual, expected):
            same(a, b, label)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(actual is not None and np.isfinite(actual) and
                math.isclose(actual, expected, abs_tol=1e-7, rel_tol=1e-10), label + ': number')
    else:
        require(actual == expected, label + ': value')


def metrics(poses, masks, critical, ids, k, req, costs=None, limit=None, floor=0.):
    """Independent evaluator: no call to the experiment's assess/solver helpers."""
    require(isinstance(ids, list) and all(type(i) is int and 0 <= i < len(poses) for i in ids), 'Invalid IDs')
    density = np.sum(masks[ids], axis=0, dtype=np.int64)
    n, nc = len(density), int(critical.sum())
    covered, double = int((density > 0).sum()), int((density > 1).sum())
    cc, cd = int((density[critical] > 0).sum()), int((density[critical] > 1).sum())
    violations = []
    if len(ids) > k or len(ids) != len(set(ids)):
        violations.append('budget_or_duplicate')
    xy = poses[ids, :2]
    if len(ids) > 1:
        distances = [np.linalg.norm(a - b) for i, a in enumerate(xy) for b in xy[i + 1:]]
        if min(distances) < req['min_separation_m'] - 1e-7:
            violations.append('minimum_separation')
        if len({tuple(p) for p in xy}) != len(ids):
            violations.append('one_sensor_per_pole')
    if double < math.ceil(n * req['overlap_ratio'] - 1e-9):
        violations.append('global_overlap')
    if cd < math.ceil(nc * req['critical_overlap_ratio'] - 1e-9):
        violations.append('critical_overlap')
    if cc < math.ceil(nc * floor - 1e-9):
        violations.append('critical_coverage')
    cost = float(np.sum(costs[ids])) if costs is not None else None
    if limit is not None and (cost is None or cost > limit + 1e-7):
        violations.append('installation_cost')
    result = dict(num_selected=len(ids), covered_count=covered, coverage_pct=100. * covered / n,
        overlap_count=double, overlap_pct=100. * double / n,
        overlap_of_covered_pct=100. * double / covered if covered else None,
        critical_overlap_count=cd, critical_overlap_pct=100. * cd / nc if nc else None,
        mean_density=float(density.mean()), feasible=not violations, violations=violations)
    if costs is not None or floor:
        result.update(installation_cost=cost, cost_budget=limit, critical_covered_count=cc,
                      critical_coverage_pct=100. * cc / nc if nc else None)
    return result


def audit_frontier(f, poses, masks, critical, ks, req, costs=None, profile=None, gap=.01):
    require([r['budget_k'] for r in f['frontier']] == ks, 'Frontier K completeness')
    require(f['requirements'] == req, 'Frontier requirements mismatch')
    require(f['candidate_count'] == len(poses) and f['target_count'] == len(critical), 'Frontier shape')
    floor = profile['critical_coverage_ratio'] if profile else 0.
    limits = {str(k): profile['cost_per_k'] * k for k in ks} if profile and profile['cost_per_k'] is not None else None
    if profile:
        same(f['cost_limits'], limits, 'Cost limits')
        same(f['critical_coverage_ratio'], floor, 'Critical floor')
    prev = -1
    count = 0
    for row in f['frontier']:
        k, ids = row['budget_k'], row['selected_ids']
        limit = limits[str(k)] if limits is not None else None
        if ids is None:
            require(row['metrics'] is None and row['status'] in ('infeasible', 'no_feasible_found'), 'Missing solution semantics')
            if row['status'] == 'infeasible':
                require(row['solver_status'] == 3 and row['upper_bound_count'] is None, 'False infeasibility label')
            require(not row['discrete_optimal'] and not row['near_optimal_certified'] and row['certified_gap'] is None, 'Missing solution certificate')
            require(not row['alternatives'], 'Missing solution has alternatives')
        else:
            actual = metrics(poses, masks, critical, ids, k, req, costs, limit, floor)
            require(actual['feasible'] and row['status'] == 'feasible', 'Invalid feasible deployment')
            same(row['metrics'], actual, 'Saved metrics')
            lo, hi = actual['covered_count'], row['upper_bound_count']
            require(hi is not None and np.isfinite(hi) and lo <= hi + 1e-5 and hi <= masks.any(axis=0).sum() + 1e-5, 'Invalid bound')
            raw = row['solver_best_bound']
            expected_hi = min(float(masks.any(axis=0).sum()), raw) if raw is not None else float(masks.any(axis=0).sum())
            same(hi, max(float(lo), expected_hi), 'Reported bound conversion')
            expected_gap = (hi - lo) / lo if lo else (0. if hi == 0 else None)
            same(row['certified_gap'], expected_gap, 'Gap')
            require(row['discrete_optimal'] == (hi - lo <= 1e-5), 'Exact flag')
            require(row['near_optimal_certified'] == (expected_gap is not None and expected_gap <= gap), 'Near-optimal flag')
            require(lo >= prev, 'Decreasing feasible K frontier')
            prev = lo
            for alt in row['alternatives']:
                v = metrics(poses, masks, critical, alt['selected_ids'], k, req, costs, limit, floor)
                same({key: value for key, value in alt.items() if key != 'selected_ids'}, v, 'Alternative')
                require(v['feasible'] and v['covered_count'] <= lo, 'Invalid alternative ranking')
                count += 1
        trace = row['quality_time_trace']
        require(trace and all(np.isfinite(t['solver_sec']) and t['solver_sec'] >= 0 for t in trace), 'Trace times')
        require(all(a['solver_sec'] <= b['solver_sec'] + 1e-5 for a, b in zip(trace, trace[1:])), 'Trace chronology')
        if ids is not None:
            lows = [t['lower'] for t in trace if t['lower'] is not None]
            require(lows and all(a <= b for a, b in zip(lows, lows[1:])), 'Trace incumbent decreased')
            same(trace[-1]['lower'], row['metrics']['covered_count'], 'Final trace incumbent')
            same(trace[-1]['upper'], row['upper_bound_count'], 'Final trace upper')
        if row['termination_reason'] == 'gap_and_stagnation':
            require(row['near_optimal_certified'], 'Plateau incorrectly certified')
        for key in ('solver_sec', 'heuristic_sec', 'model_build_sec', 'selection_sec'):
            require(np.isfinite(row[key]) and row[key] >= 0, 'Invalid runtime')
    require(f['wall_sec'] + 1e-6 >= f['preparation_sec'] + sum(r['selection_sec'] for r in f['frontier']), 'Frontier runtime accounting')
    return count


def pareto_cost_count(options):
    best, result = -1, []
    for cost, count in sorted(options):
        if count > best:
            result.append((int(cost), int(count)))
            best = count
    return result


def disjoint_critical_bound(masks, groups, costs, k, budget, required):
    """Exact enumeration of a relaxation with one pose per disjoint critical group.

    First exclude multi-group poses only when a singleton-sum upper bound proves
    they cannot participate in ANY floor-satisfying solution. Remaining poses see
    at most one group, and attaining the floor needs every group with K equal to
    their count. Ignoring site conflicts can only enlarge the feasible set.
    """
    require(len(groups) == k and k > 0, 'Bound requires K critical groups')
    membership = np.stack(groups)
    require(np.all(membership.sum(axis=0) <= 1), 'Critical groups overlap')
    missing_group_upper = sum(sorted(membership.sum(axis=1).tolist())[-(k - 1):]) if k > 1 else 0
    require(missing_group_upper < required, 'Not all critical groups are necessary')
    counts = np.column_stack([masks[:, g].sum(axis=1) for g in groups])
    cross = (counts > 0).sum(axis=1) > 1
    cross_upper = None
    if cross.any():
        totals = counts.sum(axis=1)
        other_upper = sum(sorted(totals.tolist())[-(k - 1):]) if k > 1 else 0
        cross_upper = int(totals[cross].max()) + other_upper
        require(cross_upper < required, 'Pose sees multiple groups and cannot be excluded by singleton bound')
    costs, counts = costs[~cross], counts[~cross]
    integer_costs = np.rint(costs * 1_000_000).astype(np.int64)
    require(np.allclose(integer_costs / 1_000_000, costs, atol=1e-10, rtol=0), 'Costs not quantized')
    cap = math.floor(budget * 1_000_000 + .1)
    states = [(0, 0)]
    sizes = []
    for g in range(k):
        options = pareto_cost_count(zip(integer_costs[counts[:, g] > 0], counts[counts[:, g] > 0, g]))
        sizes.append(len(options))
        states = pareto_cost_count((a + b, x + y) for a, x in states for b, y in options if a + b <= cap)
    upper = max((count for cost, count in states), default=0)
    global_upper = max(upper, missing_group_upper, cross_upper or 0)
    return dict(critical_upper_count=upper, required_count=required, budget=budget,
                global_critical_upper_count=global_upper, missing_group_upper_count=missing_group_upper,
                infeasible_proved=global_upper < required, pareto_options=sizes,
                excluded_multi_group_poses=int(cross.sum()), multi_group_solution_upper=cross_upper,
                relaxation='One sensor per necessary disjoint critical group; site conflicts relaxed')


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def check_csv(row, expected):
    for key, value in expected.items():
        require(key in row, 'Missing CSV column: ' + key)
        if value is None:
            require(row[key] == '', 'Nonempty missing CSV value: ' + key)
        elif isinstance(value, bool):
            require(row[key] == str(value), 'CSV boolean: ' + key)
        elif isinstance(value, (float, int, np.number)):
            same(float(row[key]), float(value), 'CSV ' + key)
        else:
            require(row[key] == value, 'CSV text: ' + key)


def audit(root, output, direct):
    started = time.perf_counter()
    manifest, spec = read(root / 'manifest.json'), read(root / 'spec.json')
    require(manifest['status'] == 'complete', 'Experiment is not complete')
    require(sha(root / 'spec.json') == manifest['spec_sha256'], 'Spec hash mismatch')
    for name, expected in manifest['source_hashes'].items():
        require(sha(root / 'source' / name) == expected, 'Frozen source hash: ' + name)
    inputs = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()
              and p.suffix in ('.json', '.jsonl', '.npz', '.npy', '.csv', '.py')}
    require(not any(x == 'src' or x.startswith('src.') for x in sys.modules), 'Use a fresh process')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / 'source'))
    import library_budget_study as study
    from src.core.library_transfer_evaluation import Work
    from src.core.visibility import MeshVisibility
    from src.roadside.rsu_stress import outside_footprints
    require(Path(study.__file__).resolve() == root / 'source/library_budget_study.py', 'Not frozen import')
    study.verify_library(root)
    library_spec = read(root / 'library_spec.json')
    counts, reference_rows, timing_rows, audit_rows = Counter(), [], [], []
    proof_rows, scene_plans, archive_digests, portfolio_changes = [], {}, {}, []
    root_csv = read_csv(root / 'comparison.csv')
    joined = []
    blocked = RuntimeError('Optimization/training is forbidden in this audit')
    with patch.object(study, 'solve_frontier', side_effect=blocked), patch.object(study, 'optimize', side_effect=blocked), \
            patch.object(study.core, 'optimize', side_effect=blocked), patch.object(study.core, 'solve_frontier', side_effect=blocked):
        for d in library_spec['sources']:
            folder = root / 'library' / d['name']
            data = read(folder / 'layouts.json')
            with np.load(folder / 'candidates.npz', allow_pickle=False) as z:
                masks = np.unpackbits(z['packed_masks'], axis=1, count=len(z['targets'])).astype(bool)
                for name, f in data['profiles'].items():
                    ks = library_spec['source_k'] if name == 'union' else library_spec['source_redundancy_k']
                    counts['source_alternatives'] += audit_frontier(f, z['poses'], masks, z['critical'], ks,
                        library_spec['requirements'][name], gap=library_spec['optimization']['gap'])
                    counts['source_frontier_rows'] += len(ks)
            counts['source_archives'] += 1
        expected_cases = {d['name'] + '_' + level for d in spec['targets'] for level in spec['scene_levels']}
        require(len(manifest['jobs']) == len(expected_cases), 'Job count')
        for job in manifest['jobs']:
            d, level = job['descriptor'], job['scene']
            case = d['name'] + '_' + level
            require(case in expected_cases and job['status'] == 'complete' and job['returncode'] == 0, 'Incomplete/duplicate case')
            expected_cases.remove(case)
            folder = root / 'cases' / case
            result = read(folder / 'result.json')
            status = read(root / 'job_status' / (job['id'] + '.json'))
            same({key: status[key] for key in result}, result, 'Job status')
            require(result['validated_direct'] and result['status'] == 'complete', 'Missing run validation')
            problem = study.domain(d, spec['sampling'])
            with np.load(folder / 'candidates.npz', allow_pickle=False) as z:
                archive = {key: z[key] for key in z.files}
            for key in ('poses', 'targets', 'critical'):
                require(np.array_equal(problem[key], archive[key]), 'Geometry replay: ' + key)
            poses, critical = archive['poses'], archive['critical']
            n = len(critical)
            packed = archive['packed_masks']
            masks = np.unpackbits(packed, axis=1, count=n).astype(bool)
            require(np.array_equal(np.packbits(masks, axis=1), packed), 'Nonzero mask padding')
            costs = np.load(folder / 'installation_costs.npy', allow_pickle=False)
            expected_costs, info = study.installation_costs(problem, spec['cost_model'])
            require(np.array_equal(costs, expected_costs), 'Cost geometry replay')
            same(read(folder / 'cost_model.json'), info, 'Cost metadata')
            hubs, cm = np.array(info['hubs_xy']), spec['cost_model']
            independently_costed = np.array([round(cm['base'] + cm['per_height_m'] * max(p[2] - cm['reference_height_m'], 0.)
                + cm['per_service_distance_m'] * min(math.hypot(p[0] - h[0], p[1] - h[1]) for h in hubs), 6) for p in poses])
            require(np.allclose(costs, independently_costed, atol=1e-10, rtol=0), 'Independent cost formula')
            pool = study.core.library_pool(root, dict(spec, sources=library_spec['sources']), problem)
            saved_pool = read(folder / 'transfer.json')
            for key in ('order', 'sources'):
                same(saved_pool[key], pool[key], 'Transfer replay')
            counts['transfer_replays'] += 1
            q = min(spec['query_budget'], len(pool['order']))
            optical = study.core._optical_masks(poses.astype(np.float32), problem['targets'].astype(np.float32),
                                               study.core._profile_values(problem['profile']))
            union = study.core.shortlist(optical, poses, q)
            diverse = [pool['order'][i] for i in study.core.shortlist(optical[pool['order']], poses[pool['order']], q, double=True)]
            ks = spec['target_k'][d['name']]
            plans = read(folder / 'query_plans.json')
            require(sha(folder / 'query_plans.json') == result['plans_sha256'], 'Query plan mutated')
            require(len(plans) == len(spec['profiles']) * (4 + len(spec['random_seeds'])), 'Plan count')
            expected_plans = set()
            refs, portfolios = {}, {}
            for name, profile in spec['profiles'].items():
                surrogate = read(folder / ('surrogate_' + name) / 'frontier.json')
                counts['surrogate_alternatives'] += audit_frontier(surrogate, poses, optical, critical, ks,
                    spec['requirements'], costs, profile, spec['optimization']['gap'])
                candidates = [i for rank in range(6) for row in surrogate['frontier']
                              if rank < len(row['alternatives']) for i in row['alternatives'][rank]['selected_ids']]
                portfolios[name] = list(dict.fromkeys(candidates + union))[:q]
                f = read(folder / ('reference_' + name) / 'frontier.json')
                counts['reference_alternatives'] += audit_frontier(f, poses, masks, critical, ks,
                    spec['requirements'], costs, profile, spec['optimization']['gap'])
                refs[name] = f
                for row in f['frontier']:
                    reference_rows.append(dict(case=case, geometry=d['name'], track=d['track'], scene=level,
                        profile=name, budget_k=row['budget_k'], **{k: row[k] for k in (
                            'status', 'solver_status', 'upper_bound_count', 'certified_gap', 'discrete_optimal',
                            'near_optimal_certified', 'solver_sec', 'termination_reason')},
                        **{k: v for k, v in (row['metrics'] or {}).items() if k != 'violations'}))
                    if row['status'] == 'infeasible' and d['track'] == 'B':
                        targets, coords = problem['geometry'].target_samples(spec['sampling']['target_b_step_m'], spec['sampling']['target_heights_m'])
                        keep = outside_footprints(targets, problem['boxes'])
                        require(np.array_equal(targets[keep], problem['targets']), 'Bound target ordering')
                        stations = np.linspace(0., d['length_m'], d['modules'] + 1)[1:-1]
                        groups = [np.abs(coords[keep, 0] - s) <= 12. for s in stations]
                        require(np.array_equal(np.any(groups, axis=0), critical), 'Bound critical support')
                        proof = disjoint_critical_bound(masks, groups, costs, row['budget_k'],
                            profile['cost_per_k'] * row['budget_k'], math.ceil(profile['critical_coverage_ratio'] * int(critical.sum()) - 1e-9))
                        proof_rows.append(dict(case=case, profile=name, budget_k=row['budget_k'],
                                              critical_targets=int(critical.sum()), **proof))
                for method in ('library_prefix', 'library_diverse', 'geometry_union', 'geometry_portfolio'):
                    expected_plans.add((name, method, None))
                expected_plans.update((name, 'random', seed) for seed in spec['random_seeds'])
            del optical
            case_csv = read_csv(folder / 'comparison.csv')
            lookup = {(r['profile'], r['method'], r['seed'], int(r['budget_k'])): r for r in case_csv}
            require(len(lookup) == len(case_csv) == result['rows'], 'Duplicate case rows')
            for p in plans:
                key = p['profile'], p['method'], p['seed']
                require(key in expected_plans, 'Duplicate/missing plan')
                expected_plans.remove(key)
                name = p['profile']
                expected_ids = {'library_prefix': pool['order'][:q], 'library_diverse': diverse,
                    'geometry_union': union, 'geometry_portfolio': portfolios[name]}.get(p['method'])
                if p['method'] == 'random':
                    expected_ids = np.random.default_rng(p['seed']).permutation(len(poses))[:q].tolist()
                require(p['selected_ids'] == sorted(expected_ids) and len(set(expected_ids)) == q, 'Shortlist replay')
                require(p['actual_q'] == q and p['requested_q'] == spec['query_budget'] and p['pool_count'] == len(pool['order']), 'Q fairness')
                signature = d['name'], name, p['method'], p['seed']
                if signature in scene_plans and scene_plans[signature] != p['selected_ids']:
                    require(p['method'] == 'geometry_portfolio', 'Deterministic shortlist changed across occlusion')
                    portfolio_changes.append(dict(case=case, profile=name, reason='Time-limited optical portfolio differs across independent workers'))
                scene_plans[signature] = p['selected_ids']
                arm = read(folder / p['arm_id'] / 'result.json')
                same(arm['plan'], p, 'Arm plan')
                f = arm['frontier']
                saved = read(folder / p['arm_id'] / 'frontier.json')
                same(dict(f, frontier=[{k: v for k, v in r.items() if k != 'full_candidate_ids'} for r in f['frontier']]), saved, 'Duplicate frontier artifacts')
                ids = p['selected_ids']
                with np.load(folder / p['arm_id'] / 'queried.npz', allow_pickle=False) as z:
                    require(np.array_equal(z['candidate_ids'], ids) and np.array_equal(z['packed_masks'], packed[ids]), 'Query masks differ')
                counts['online_alternatives'] += audit_frontier(f, poses[ids], masks[ids], critical, ks,
                    spec['requirements'], costs[ids], spec['profiles'][name], spec['optimization']['gap'])
                preparation = float(lookup[(name, p['method'], '' if p['seed'] is None else str(p['seed']), ks[0])]['preparation_sec'])
                same(arm['online_all_k_sec'], preparation + p['planning_sec'] + arm['visibility_sec'] + f['wall_sec'], 'Online time boundary')
                cumulative = f['preparation_sec']
                for row, ref in zip(f['frontier'], refs[name]['frontier']):
                    full_ids = None if row['selected_ids'] is None else [ids[i] for i in row['selected_ids']]
                    same(row['full_candidate_ids'], full_ids, 'Full candidate mapping')
                    values = row['metrics']
                    cumulative += row['selection_sec']
                    lo = hi = None
                    if values:
                        require(ref['upper_bound_count'] is not None and ref['upper_bound_count'] + 1e-5 >= values['covered_count'], 'Reference contradicts online solution')
                        lo = max(0., ref['metrics']['coverage_pct'] - values['coverage_pct']) if ref['metrics'] else None
                        hi = max(0., 100. * ref['upper_bound_count'] / n - values['coverage_pct'])
                    expected = dict(case=case, track=d['track'], scene=level, profile=name, method=p['method'], seed=p['seed'],
                        budget_k=row['budget_k'], requested_q=p['requested_q'], actual_q=q, pool_count=p['pool_count'],
                        candidate_count=len(poses), target_count=n, critical_target_count=int(critical.sum()), status=row['status'],
                        **{k: v for k, v in (values or {}).items() if k != 'violations'},
                        cost_limit=spec['profiles'][name]['cost_per_k'] * row['budget_k'] if spec['profiles'][name]['cost_per_k'] is not None else None,
                        critical_floor=spec['profiles'][name]['critical_coverage_ratio'], queried_pool_gap=row['certified_gap'],
                        reference_status=ref['status'], reference_coverage_pct=ref['metrics']['coverage_pct'] if ref['metrics'] else None,
                        reference_gap=ref['certified_gap'], loss_lower_pp=lo, loss_upper_pp=hi,
                        preparation_sec=preparation, planning_sec=p['planning_sec'], visibility_sec=arm['visibility_sec'],
                        solver_sec=row['solver_sec'], k_selection_sec=row['selection_sec'],
                        online_through_k_sec=preparation + p['planning_sec'] + arm['visibility_sec'] + cumulative,
                        online_all_k_sec=arm['online_all_k_sec'], rays_submitted=arm['work'].get('rays_submitted', 0),
                        reference_all_k_sec=preparation + float(lookup[(name, p['method'], '' if p['seed'] is None else str(p['seed']), row['budget_k'])]['reference_visibility_sec']) + refs[name]['wall_sec'])
                    csv_row = lookup[(name, p['method'], '' if p['seed'] is None else str(p['seed']), row['budget_k'])]
                    check_csv(csv_row, expected)
                    if values is None:
                        for field in ('coverage_pct', 'covered_count', 'installation_cost', 'critical_coverage_pct', 'feasible'):
                            require(csv_row[field] == '', 'Failed solution is zero-filled')
                    counts['online_rows'] += 1
                timing_rows.append(dict(case=case, geometry=d['name'], track=d['track'], scene=level, profile=name,
                    method=p['method'], seed=p['seed'], k_count=len(ks), online_all_k_sec=arm['online_all_k_sec'],
                    planning_sec=p['planning_sec'], visibility_sec=arm['visibility_sec'], rays_submitted=arm['work'].get('rays_submitted', 0)))
                counts['shortlist_replays'] += 1
            require(not expected_plans, 'Missing plans')
            joined.extend(case_csv)
            audit_record = dict(case=case, candidates=len(poses), targets=n, direct=direct)
            if direct:
                before = time.perf_counter()
                work = Work()
                provider = MeshVisibility(study.scene(problem, level), problem['targets'], problem['profile'], work=work)
                h = hashlib.sha256()
                for cid, pose in enumerate(poses):
                    fresh = np.packbits(provider.mask(pose))
                    require(np.array_equal(fresh, packed[cid]), f'Direct raycast mismatch: {case}/{cid}')
                    h.update(fresh.tobytes())
                audit_record.update(direct_sec=time.perf_counter() - before, rays=work.counts.get('rays_submitted', 0), fresh_sha256=h.hexdigest())
                counts['direct_candidate_masks'] += len(poses)
                counts['direct_rays'] += audit_record['rays']
                del provider
            counts['candidate_masks_checked'] += len(poses)
            if d['name'] in archive_digests:
                old = archive_digests[d['name']]
                require(np.array_equal(old['poses'], poses) and np.array_equal(old['targets'], archive['targets']) and
                        np.array_equal(old['critical'], critical) and np.array_equal(old['costs'], costs), 'Scene support/cost changes')
                mixed, dense = (old['packed'], packed) if level == 'dense' else (packed, old['packed'])
                require(not np.any(dense & ~mixed), 'Adding obstacles increases physical visibility')
                counts['occlusion_monotonicity_pairs'] += 1
                del archive_digests[d['name']]
            else:
                archive_digests[d['name']] = dict(poses=poses, targets=archive['targets'], critical=critical, costs=costs, packed=packed)
            audit_rows.append(audit_record)
            counts['cases'] += 1
            save(output / 'progress.json', dict(status='running', counts=dict(counts)))
        require(not expected_cases, 'Missing cases')
    require(sorted(map(lambda r: json.dumps(r, sort_keys=True), joined)) ==
            sorted(map(lambda r: json.dumps(r, sort_keys=True), root_csv)), 'Root/case CSV mismatch')
    require(len(root_csv) == manifest['plan']['expected_rows'] and len(reference_rows) == manifest['plan']['reference_rows'], 'Expected study size')
    for rel, expected in inputs.items():
        require(sha(root / rel) == expected, 'Input changed during audit: ' + rel)
    data = pd.read_csv(root / 'comparison.csv')
    case_to_geometry = {j['descriptor']['name'] + '_' + j['scene']: j['descriptor']['name'] for j in manifest['jobs']}
    data['geometry'] = data['case'].map(case_to_geometry)
    summaries = summarize(data, pd.DataFrame(timing_rows), pd.DataFrame(reference_rows), output)
    pd.DataFrame(audit_rows).to_csv(output / 'case_validation.csv', index=False)
    pd.DataFrame(proof_rows).to_csv(output / 'infeasibility_proofs.csv', index=False)
    pd.DataFrame(portfolio_changes).to_csv(output / 'portfolio_scene_variation.csv', index=False)
    report = dict(status='passed', verification_status='VERIFIED' if direct else 'ANALYZED',
        root=str(root), created_at=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
        direct_raycast=direct, rerun_optimization=False, raw_inputs_unchanged=True,
        wall_sec=time.perf_counter() - started, script_sha256=sha(Path(__file__)), input_hashes=inputs,
        frozen_source_hashes=manifest['source_hashes'], summary=summaries, infeasibility_proofs=proof_rows,
        portfolio_scene_variations=portfolio_changes,
        limitations=['Same Open3D/optical model; physical model accuracy not established.',
            'No independent optimality proof for feasible large MILPs; bound/flag consistency checked.',
            'Source archives hash-checked and saved solutions recomputed, not reraycast in this audit.',
            'Time-limited optical portfolio reconstructed from saved solutions, not reoptimized.',
            'Eight reused development geometries; descriptive statistics, no p-values or generalization claim.',
            'Shared hardware, non-randomized method order, no repeated cold timing benchmark.'])
    save(output / 'validation_summary.json', report)
    write_report(report, manifest, output)
    save(output / 'progress.json', dict(status='passed', counts=dict(counts)))
    return report


def summarize(data, times, refs, output):
    keys = ['case', 'geometry', 'track', 'scene', 'profile', 'budget_k']
    require(not data.duplicated(['case', 'profile', 'method', 'seed', 'budget_k']).any(), 'Duplicate comparison key')
    quantities = ['coverage_pct', 'critical_coverage_pct', 'overlap_pct', 'installation_cost', 'num_selected',
                  'loss_lower_pp', 'loss_upper_pp', 'online_through_k_sec']
    cells = []
    for key, group in data.groupby(keys + ['method'], dropna=False):
        all_feasible = (group.status == 'feasible').all()
        expected = 3 if key[-1] == 'random' else 1
        require(len(group) == expected, 'Incomplete seed group')
        cells.append(dict(zip(keys + ['method'], key), all_feasible=bool(all_feasible), trials=len(group),
                          **{q: float(group[q].mean()) if all_feasible else None for q in quantities}))
    cells = pd.DataFrame(cells)
    common = cells.groupby(keys).all_feasible.all()
    valid = cells.merge(common.rename('common_feasible').reset_index(), on=keys)
    valid = valid[valid.common_feasible].copy()
    require(valid.groupby(keys).size().eq(5).all(), 'Incomplete paired methods')
    # A K frontier is timed once, not once for every K row that repeats its cost.
    timed = times.groupby(['case', 'geometry', 'track', 'scene', 'profile', 'method'], as_index=False)[
        ['online_all_k_sec', 'planning_sec', 'visibility_sec', 'rays_submitted']].mean()
    tables = dict(condition_means=cells, timing_conditions=timed, references=refs,
        infeasible_rows=data[data.status != 'feasible'],
        method_summary=valid.groupby('method', as_index=False)[quantities].mean(),
        profile_summary=valid.groupby(['profile', 'method'], as_index=False)[quantities].mean(),
        track_summary=valid.groupby(['track', 'method'], as_index=False)[quantities].mean(),
        scene_summary=valid.groupby(['scene', 'method'], as_index=False)[quantities].mean(),
        geometry_summary=valid.groupby(['geometry', 'method'], as_index=False)[quantities].mean(),
        timing_summary=timed.groupby('method', as_index=False)[['online_all_k_sec', 'planning_sec', 'visibility_sec', 'rays_submitted']].mean())
    tables['map_balanced_summary'] = tables['geometry_summary'].groupby('method', as_index=False)[quantities].mean()
    pairs = []
    for key, group in valid.groupby(keys):
        base = group[group.method == 'library_diverse'].iloc[0]
        for _, row in group[group.method != 'library_diverse'].iterrows():
            pairs.append(dict(zip(keys, key), comparator=row.method,
                              difference_pp=float(base.coverage_pct - row.coverage_pct)))
    paired = pd.DataFrame(pairs)
    stats = []
    for comparator, group in paired.groupby('comparator'):
        diff = group.difference_pp
        stats.append(dict(comparator=comparator, n=len(diff), mean_difference_pp=diff.mean(),
                          wins=int((diff > 1e-7).sum()), ties=int((diff.abs() <= 1e-7).sum()), losses=int((diff < -1e-7).sum()),
                          min_difference_pp=diff.min(), max_difference_pp=diff.max()))
    tables['paired_conditions'] = paired
    tables['paired_summary'] = pd.DataFrame(stats)
    tables['paired_by_geometry'] = paired.groupby(['geometry', 'comparator'], as_index=False).difference_pp.mean()
    tables['paired_by_profile'] = paired.groupby(['profile', 'comparator'], as_index=False).difference_pp.mean()
    tables['feasibility'] = data.groupby(['profile', 'method', 'status']).size().rename('rows').reset_index()
    match = ['case', 'method', 'seed', 'budget_k']
    if {'cost_limit', 'critical_floor', 'critical_target_count', 'critical_covered_count'}.issubset(data.columns):
        baseline = data[data.profile == 'base'][match + ['installation_cost', 'critical_covered_count']].rename(
            columns={'installation_cost': 'base_cost', 'critical_covered_count': 'base_critical_count'})
        activity = data[data.profile != 'base'].merge(baseline, on=match, validate='many_to_one')
        activity['base_cost_violation'] = activity.base_cost > activity.cost_limit + 1e-7
        activity['base_critical_violation'] = activity.base_critical_count < np.ceil(activity.critical_floor * activity.critical_target_count - 1e-9)
        activity['base_any_violation'] = activity.base_cost_violation | activity.base_critical_violation
        tables['constraint_activity'] = activity[match + ['profile', 'base_cost_violation', 'base_critical_violation', 'base_any_violation']]
        tables['constraint_activity_summary'] = activity.groupby(['profile', 'method'], as_index=False)[
            ['base_cost_violation', 'base_critical_violation', 'base_any_violation']].mean()
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False)
    feasible_refs = refs[refs.status == 'feasible']
    return dict(online_status=data.status.value_counts().to_dict(), reference_status=refs.status.value_counts().to_dict(),
        common_feasible_conditions=int(common.sum()), total_conditions=len(common),
        methods=tables['method_summary'].to_dict('records'), timings=tables['timing_summary'].to_dict('records'),
        paired=stats, profiles=tables['profile_summary'].to_dict('records'),
        reference_solver_status={str(k): int(v) for k, v in refs.solver_status.value_counts().items()},
        reference_exact=int(refs.discrete_optimal.sum()), reference_within_one_percent=int(refs.near_optimal_certified.sum()),
        maximum_reference_gap=float(feasible_refs.certified_gap.max()),
        online_above_reference_incumbent=int((data.coverage_pct > data.reference_coverage_pct + 1e-7).sum()),
        actual_q=data.groupby('case').actual_q.first().to_dict())


def write_report(report, manifest, output):
    s = report['summary']
    times = {r['method']: r for r in s['timings']}
    checked_date = datetime.fromisoformat(report['created_at']).astimezone(timezone(timedelta(hours=9))).date()
    lines = ['# Low-K Cost Study: Results and Audit', '', '## Material Passport',
        f"- Verification: {report['verification_status']}. Date: {checked_date} (KST).",
        '- Original experiment preserved; no optimization/training rerun.',
        '- Scope: synthetic geometric coverage, relative installation costs, eight development maps.', '',
        '## Validation', f"- Checks: `{json.dumps(report['counts'], sort_keys=True)}`.",
        f"- Online status: `{s['online_status']}`; references: `{s['reference_status']}`.",
        '- Every geometry, cost vector, transfer and shortlist reconstructed; every stored feasible deployment independently recomputed.',
        '- Root/case CSVs, time boundaries, source/code/input hashes, solver flags and bound arithmetic checked.',
        f"- Fresh target candidate raycasting: {report['direct_raycast']}. Raw inputs unchanged: True.", '',
        '## Paired Quality and Time',
        f"Quality means use {s['common_feasible_conditions']}/{s['total_conditions']} common feasible map/scene/profile/K cells.",
        'Random seeds are averaged inside each cell. Two globally infeasible conditions are excluded from quality, never zero-filled.',
        'Time means use all 64 map/scene/profile frontiers once each, averaging random seeds first; each frontier contains 2 or 3 K levels.', '',
        '| Method | Coverage % | Critical % | Overlap % | Relative cost | All-K online seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for row in s['methods']:
        lines.append(f"| {row['method']} | {row['coverage_pct']:.3f} | {row['critical_coverage_pct']:.3f} | {row['overlap_pct']:.3f} | {row['installation_cost']:.3f} | {times[row['method']]['online_all_k_sec']:.2f} |")
    lines += ['', 'Library diverse minus each comparator:', '', '| Comparator | Mean pp | Win / tie / loss |', '|---|---:|---|']
    for p in s['paired']:
        lines.append(f"| {p['comparator']} | {p['mean_difference_pp']:+.3f} | {p['wins']} / {p['ties']} / {p['losses']} |")
    lines += ['', '## Certification Limits',
        f"- Full reference: {s['reference_exact']} exact objective certificates; {s['reference_within_one_percent']} within 1% including exact, out of 182 feasible rows.",
        f"- Maximum reported relative gap: {100*s['maximum_reference_gap']:.3f}%. Solver statuses: `{s['reference_solver_status']}`.",
        f"- {s['online_above_reference_incumbent']} online rows exceed the saved full-domain incumbent, not its upper bound. This is not beating the optimum.",
        '- Null certificates and infeasibility are not gap zero. Stagnation is not an optimality proof.', '', '## Infeasibility Recheck']
    for p in report['infeasibility_proofs']:
        lines.append(f"- {p['case']}, {p['profile']}, K={p['budget_k']}: global critical upper {p['global_critical_upper_count']}/{p['critical_targets']}, required {p['required_count']}; independent finite-bound proof={p['infeasible_proved']}.")
        lines.append(f"  Three exhaustive cases: a multi-group pose is selected (upper {p['multi_group_solution_upper']}); no multi-group pose and a group is missed (upper {p['missing_group_upper_count']}); exactly one pose per group under the cost cap (relaxed upper {p['critical_upper_count']}). The global upper is the maximum of these, not the one-per-group bound alone.")
    lines += ['', 'The proof applies only to this declared candidate grid/cost model. It does not prove real-road installation impossible.',
        '', '## Interpretation and Limitations',
        '- Library benefit here concerns coverage under the same hard constraints, not guaranteed lower cost or universal superiority.',
        '- Cost is a cap, not a minimized objective. Required critical coverage is a floor, not the maximized metric.',
        '- A actual Q=139..162 includes the complete transferred pool: prefix/diverse select the same IDs. B actual Q=256 can select different subsets.',
        f"- {len(report['portfolio_scene_variations'])} optical portfolio plans differ across independent scene workers; saved surrogate solutions were validated, not reoptimized. Timing-dependent search can change these plans.",
        '- B480 has two K values, other maps three; map-balanced and per-geometry summaries are provided alongside cell means.',
        '- Runtime measurements are shared-server, single-run, fixed method order. Do not claim controlled speedup or compare directly with the previous high-K experiment.',
        '- No new overlap minimum in this phase; no detector, dynamic traffic, RL or common multi-scene robust placement.',
        '', '## Statistical Fallacy Check (11/11)',
        '1. Simpson: track/scene/profile/geometry summaries provided; overall average is not uniform superiority.',
        '2. Ecological: eight geometries, not 1,288 independent roads.',
        '3. Berkson/selection: common feasibility is explicit, with all failures separately tabulated.',
        '4. Collider: no causal regression; selection of feasible outcomes limits the quality estimand.',
        '5. Base rates: report 182/184 conditions; no detection accuracy or diagnostic predictive value.',
        '6. Regression to mean: no pre/post improvement claim from changing K and constraints.',
        '7. Survivorship: all infeasible/unknown rows retained; no zero imputation.',
        '8. Look-elsewhere: all methods/profiles reported; no significance test or cherry-picked p-values.',
        '9. Forking paths: frozen protocol retained; this post hoc report is descriptive, not preregistered inference.',
        '10. Causation: controlled code comparison on synthetic maps, not proof of real deployment impact.',
        '11. Reverse causality: library precedes target queries; no learned causal claim.',
        '', '## Reproduction', '```bash',
        f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B library_budget_report.py --root {report['root']} --output results/library_lowk_cost_review_repeat --direct", '```',
        'Existing output directories are refused. Results are ignored artifacts and not included in a repository clone.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direct', action='store_true')
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    require(not output.exists() and not output.is_relative_to(root), 'Use a new output outside the experiment')
    output.mkdir(parents=True)
    try:
        report = audit(root, output, args.direct)
        print(json.dumps(dict(status=report['status'], counts=report['counts'], output=str(output)), indent=2))
    except BaseException as exc:
        save(output / 'failure.json', dict(status='failed', error=repr(exc), traceback=traceback.format_exc()))
        save(output / 'progress.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()

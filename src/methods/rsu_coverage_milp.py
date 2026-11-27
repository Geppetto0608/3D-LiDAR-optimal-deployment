"""Opt-in finite union optimization with real double-coverage constraints.

Identical visibility columns are aggregated losslessly, separately for critical
and ordinary targets. No legacy solver or physics defaults are changed.
"""
import math
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def assess(poses, masks, selected, critical, requirements, k, *,
           candidate_costs=None, cost_budget=None, critical_coverage_ratio=0.):
    ids = list(selected)
    density = masks[ids].sum(axis=0, dtype=np.int32)
    covered = int(np.count_nonzero(density))
    double = int(np.count_nonzero(density >= 2))
    critical_double = int(np.count_nonzero((density >= 2) & critical))
    violations = []
    if len(ids) > k or len(set(ids)) != len(ids):
        violations.append('budget_or_duplicate')
    if len(ids) > 1:
        xy = poses[ids, :2]
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        distances = distances[np.triu_indices(len(ids), 1)]
        if np.any(distances < requirements['min_separation_m'] - 1e-7):
            violations.append('minimum_separation')
        if len(np.unique(xy, axis=0)) != len(ids):
            violations.append('one_sensor_per_pole')
    minimum = math.ceil(requirements['overlap_ratio'] * len(density) - 1e-9)
    critical_min = math.ceil(requirements['critical_overlap_ratio'] * int(critical.sum()) - 1e-9)
    if double < minimum:
        violations.append('global_overlap')
    if critical_double < critical_min:
        violations.append('critical_overlap')
    critical_covered = int(np.count_nonzero((density > 0) & critical))
    if critical_covered < math.ceil(critical_coverage_ratio * int(critical.sum()) - 1e-9):
        violations.append('critical_coverage')
    cost = float(np.asarray(candidate_costs)[ids].sum()) if candidate_costs is not None else None
    if cost_budget is not None and (cost is None or cost > cost_budget + 1e-7):
        violations.append('installation_cost')
    value = dict(num_selected=len(ids), covered_count=covered,
        coverage_pct=100. * covered / len(density), overlap_count=double,
        overlap_pct=100. * double / len(density),
        overlap_of_covered_pct=100. * double / covered if covered else None,
        critical_overlap_count=critical_double,
        critical_overlap_pct=100. * critical_double / int(critical.sum()) if critical.any() else None,
        mean_density=float(density.mean()), feasible=not violations, violations=violations)
    if candidate_costs is not None or critical_coverage_ratio:
        value.update(installation_cost=cost, cost_budget=cost_budget,
            critical_covered_count=critical_covered,
            critical_coverage_pct=100. * critical_covered / int(critical.sum()) if critical.any() else None)
    return value


def compress(masks, critical):
    # A critical flag is part of the signature, so regional requirements remain exact.
    signatures = np.column_stack([critical.astype(np.uint8), np.packbits(masks.T, axis=1)])
    unique, counts = np.unique(signatures, axis=0, return_counts=True)
    matrix = sparse.csr_matrix(np.unpackbits(unique[:, 1:], axis=1, count=len(masks)), dtype=float)
    return matrix, counts.astype(float), unique[:, 0].astype(bool)


def _site_constraints(poses, separation):
    sites, pole_ids = np.unique(poses[:, :2], axis=0, return_inverse=True)
    rows, cols = [], []
    members = [np.flatnonzero(pole_ids == i).tolist() for i in range(len(sites))]
    groups = list(members)
    if separation > 0:
        for a, b in sorted(cKDTree(sites).query_pairs(max(0., separation - 1e-7))):
            groups.append(members[a] + members[b])
    for row, ids in enumerate(groups):
        rows.extend([row] * len(ids))
        cols.extend(ids)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(groups), len(poses)))


def solve_frontier(poses, masks, critical, k_values, requirements, *, seconds=600.,
                   alternatives=12, threads=1, soft_mem_gb=12., stop_gap=.01,
                   stop_min_sec=300., stop_stagnation_sec=120., progress_path=None,
                   candidate_costs=None, cost_limits=None, critical_coverage_ratio=0.):
    import gurobipy as gp
    from gurobipy import GRB

    begin = time.perf_counter()
    poses, masks, critical = np.asarray(poses), np.asarray(masks), np.asarray(critical)
    ks = list(k_values)
    if (poses.ndim != 2 or poses.shape[1] != 5 or not np.isfinite(poses).all()
            or masks.dtype != bool or masks.ndim != 2 or masks.shape[0] != len(poses)
            or masks.shape[1] == 0 or critical.dtype != bool or critical.shape != (masks.shape[1],)
            or len(np.unique(poses, axis=0)) != len(poses)
            or not ks or ks != sorted(set(ks)) or any(type(k) is not int or not 1 <= k <= 64 for k in ks)
            or set(requirements) != {'overlap_ratio', 'critical_overlap_ratio', 'min_separation_m'}
            or not np.isfinite(list(requirements.values())).all()
            or not 0 <= requirements['overlap_ratio'] <= 1
            or not 0 <= requirements['critical_overlap_ratio'] <= 1
            or requirements['min_separation_m'] < 0
            or (requirements['critical_overlap_ratio'] > 0 and not critical.any())
            or not np.isfinite([seconds, soft_mem_gb, stop_gap, stop_min_sec, stop_stagnation_sec]).all()
            or seconds <= 0 or soft_mem_gb <= 0 or not 0 <= stop_gap <= 1
            or min(stop_min_sec, stop_stagnation_sec) < 0 or threads < 1 or not 1 <= alternatives <= 64):
        raise ValueError('Invalid constrained coverage problem')
    if (not np.isfinite(critical_coverage_ratio) or not 0 <= critical_coverage_ratio <= 1
            or (critical_coverage_ratio > 0 and not critical.any())):
        raise ValueError('Invalid critical coverage floor')
    if candidate_costs is not None:
        candidate_costs = np.asarray(candidate_costs, dtype=float)
        if (candidate_costs.shape != (len(poses),) or not np.isfinite(candidate_costs).all()
                or np.any(candidate_costs < 0)):
            raise ValueError('Invalid per-candidate installation costs')
    if cost_limits is not None:
        if (candidate_costs is None or set(cost_limits) != set(ks)
                or not np.isfinite(list(cost_limits.values())).all()
                or min(cost_limits.values()) < 0
                or any(cost_limits[b] < cost_limits[a] for a, b in zip(ks, ks[1:]))):
            raise ValueError('Cost limits must cover every K and be nondecreasing')
    if masks.size > 1_500_000_000:
        raise ValueError('Explicit dense working-mask safety limit exceeded')
    n, m = masks.shape[1], len(poses)
    matrix, weights, is_critical = compress(masks, critical)
    if matrix.nnz > 80_000_000:
        raise ValueError('Exact sparse incidence exceeds explicit 80M-edge guard')
    transpose = matrix.T.tocsr()
    ng = len(weights)
    regional = weights * is_critical
    global_min = math.ceil(requirements['overlap_ratio'] * n - 1e-9)
    regional_min = math.ceil(requirements['critical_overlap_ratio'] * int(critical.sum()) - 1e-9)
    critical_min = math.ceil(critical_coverage_ratio * int(critical.sum()) - 1e-9)
    site_matrix = _site_constraints(poses, requirements['min_separation_m'])
    ceiling = int(weights[np.asarray(matrix.sum(axis=1)).ravel() > 0].sum())
    preparation_sec = time.perf_counter() - begin
    previous, rows = [], []
    for k in ks:
        start = time.perf_counter()
        cost_budget = cost_limits[k] if cost_limits is not None else None
        candidates = {}
        trace = []
        def retain(ids):
            ids = sorted(ids)
            value = assess(poses, masks, ids, critical, requirements, k,
                candidate_costs=candidate_costs, cost_budget=cost_budget,
                critical_coverage_ratio=critical_coverage_ratio)
            if value['feasible']:
                candidates[tuple(ids)] = value
            return value
        retain(previous)
        # Feasible heuristic starts use the same physical masks and constraints as the solve.
        singleton = np.asarray(transpose @ weights).ravel()
        ranking = np.lexsort((np.arange(m), -singleton))
        seeds = [None] + [int(ranking[i]) for i in np.linspace(0, max(0, m-1), min(m, 4), dtype=int)]
        for seed in seeds:
            ids, available = [], np.ones(m, dtype=bool)
            density = np.zeros(ng)
            for iteration in range(min(k, m)):
                if cost_budget is not None:
                    available &= candidate_costs <= cost_budget - candidate_costs[ids].sum() + 1e-7
                contribution = weights * (density == 0)
                if regional[density > 0].sum() < critical_min:
                    contribution += 4 * regional * (density == 0)
                if weights[density >= 2].sum() < global_min:
                    contribution += 2 * weights * (density == 1)
                if regional[density >= 2].sum() < regional_min:
                    contribution += 4 * regional * (density < 2)
                gain = np.asarray(transpose @ contribution).ravel()
                gain[~available] = -np.inf
                if not np.isfinite(gain).any():
                    break
                cid = seed if iteration == 0 and seed is not None and available[seed] else int(np.argmax(gain))
                ids.append(cid)
                density += matrix[:, cid].toarray().ravel()
                distance = np.linalg.norm(poses[:, :2] - poses[cid, :2], axis=1)
                available &= (distance >= requirements['min_separation_m'] - 1e-7) & (distance > 1e-8)
                retain(ids)
        warm = sorted(candidates, key=lambda ids: (-candidates[ids]['covered_count'], len(ids), ids))
        heuristic_sec = time.perf_counter() - start
        model = gp.Model('rsu_union_with_redundancy')
        state = dict(best=candidates[warm[0]]['covered_count'] if warm else None,
                     last_improvement=0., last_sample=-1., stopped=False, error=None)
        try:
            model.Params.OutputFlag = 0
            model.Params.Threads = threads
            model.Params.SoftMemLimit = soft_mem_gb
            model.Params.TimeLimit = seconds
            model.Params.MIPGap = 0.
            if cost_budget is not None:
                model.Params.FeasibilityTol = 1e-8
                model.Params.IntFeasTol = 1e-8
            model.Params.PoolSolutions = alternatives
            model.Params.PoolSearchMode = 1 if alternatives > 1 else 0
            x = model.addMVar(m, vtype=GRB.BINARY, name='sensor')
            y = model.addMVar(ng, vtype=GRB.BINARY, name='covered')
            z = model.addMVar(ng, vtype=GRB.BINARY, name='double') if global_min or regional_min else None
            model.addConstr(x.sum() <= k)
            if cost_budget is not None:
                model.addConstr(candidate_costs @ x <= cost_budget)
            if site_matrix.shape[0]:
                model.addConstr(site_matrix @ x <= 1.)
            d = matrix @ x
            model.addConstr(d >= y)
            model.addConstr(d <= k * y)
            if critical_min:
                model.addConstr(regional @ y >= critical_min)
            if z is not None:
                model.addConstr(d >= 2 * z)
                model.addConstr(d <= 1 + (k - 1) * z)
                model.addConstr(weights @ z >= global_min)
                model.addConstr(regional @ z >= regional_min)
            model.setObjective(weights @ y, GRB.MAXIMIZE)
            if warm:
                initial = np.zeros(m)
                initial[list(warm[0])] = 1.
                density = matrix @ initial
                x.Start, y.Start = initial, (density >= 1).astype(float)
                if z is not None:
                    z.Start = (density >= 2).astype(float)
            model_build_sec = time.perf_counter() - start - heuristic_sec
            def callback(mod, where):
                if where not in (GRB.Callback.MIP, GRB.Callback.MIPSOL):
                    return
                try:
                    elapsed = float(mod.cbGet(GRB.Callback.RUNTIME))
                    if where == GRB.Callback.MIPSOL:
                        ids = np.flatnonzero(np.asarray(mod.cbGetSolution(x)) > .5).tolist()
                        actual = retain(ids)
                        if not actual['feasible']:
                            raise RuntimeError('Solver incumbent violates direct constraints')
                        objective = float(mod.cbGet(GRB.Callback.MIPSOL_OBJ))
                        if abs(objective-actual['covered_count']) > 1e-4:
                            raise RuntimeError('Solver objective disagrees with direct coverage')
                        if state['best'] is None or actual['covered_count'] > state['best']:
                            state.update(best=actual['covered_count'], last_improvement=elapsed)
                        bound = float(mod.cbGet(GRB.Callback.MIPSOL_OBJBND))
                    else:
                        bound = float(mod.cbGet(GRB.Callback.MIP_OBJBND))
                    bound_available = np.isfinite(bound) and abs(bound) < GRB.INFINITY
                    upper = min(float(ceiling), bound) if bound_available else float(ceiling)
                    lower = state['best']
                    bound_source = 'solver_callback' if bound_available else 'target_union_fallback'
                    if lower is not None and upper + 1e-5 < lower:
                        # Pool-search callbacks can expose a search bound below a
                        # known incumbent. Never certify from that intermediate value.
                        upper = float(ceiling)
                        bound_source = 'target_union_fallback'
                    if lower is not None:
                        upper = max(upper, float(lower))
                    gap = (upper-lower)/lower if lower else (0. if lower == 0 and upper == 0 else None)
                    if elapsed-state['last_sample'] >= 30 or where == GRB.Callback.MIPSOL:
                        point = dict(solver_sec=elapsed, lower=lower, upper=upper, relative_gap=gap,
                                     bound_source=bound_source, solver_raw_bound=bound if bound_available else None)
                        trace.append(point)
                        state['last_sample'] = elapsed
                        if progress_path is not None:
                            import json
                            with open(progress_path, 'a') as stream:
                                stream.write(json.dumps(dict(k=k, **point), allow_nan=False) + '\n')
                    if (gap is not None and gap <= stop_gap and elapsed >= stop_min_sec
                            and elapsed-state['last_improvement'] >= stop_stagnation_sec):
                        state['stopped'] = True
                        mod.terminate()
                except Exception as exc:
                    state['error'] = repr(exc)
                    mod.terminate()
            model.optimize(callback)
            if state['error']:
                raise RuntimeError(state['error'])
            for number in range(model.SolCount):
                model.Params.SolutionNumber = number
                value = retain(np.flatnonzero(np.asarray(x.Xn) > .5).tolist())
                if not value['feasible']:
                    raise RuntimeError('Solution pool violates constraints')
            ordered = sorted(candidates, key=lambda ids: (-candidates[ids]['covered_count'], len(ids), ids))
            selected = list(ordered[0]) if ordered else None
            value = candidates[ordered[0]] if ordered else None
            raw_bound = float(model.ObjBound)
            bound_available = np.isfinite(raw_bound) and abs(raw_bound) < GRB.INFINITY
            upper = min(float(ceiling), raw_bound) if bound_available else float(ceiling)
            status = int(model.Status)
            if status == GRB.INFEASIBLE and value is not None:
                raise RuntimeError('Solver infeasible contradicts feasible warm start')
            lower = value['covered_count'] if value else None
            if lower is not None and upper + 1e-5 < lower:
                raise RuntimeError('Final bound below feasible objective')
            if lower is not None:
                upper = max(upper, float(lower))
            gap = (upper-lower)/lower if lower else (0. if lower == 0 and upper == 0 else None)
            trace.append(dict(solver_sec=float(model.Runtime), lower=lower, upper=upper, relative_gap=gap,
                              bound_source='final_solver_and_union' if bound_available else 'target_union_fallback'))
            rows.append(dict(budget_k=k, selected_ids=selected, metrics=value,
                status='feasible' if value else ('infeasible' if status == GRB.INFEASIBLE else 'no_feasible_found'),
                solver_status=status, upper_bound_count=None if status == GRB.INFEASIBLE else upper,
                solver_best_bound=raw_bound if bound_available else None,
                certified_gap=gap, discrete_optimal=lower is not None and upper-lower <= 1e-5,
                near_optimal_certified=gap is not None and gap <= stop_gap,
                alternatives=[dict(selected_ids=list(ids), **candidates[ids]) for ids in ordered[:alternatives]],
                solver_sec=float(model.Runtime), heuristic_sec=heuristic_sec, model_build_sec=model_build_sec,
                selection_sec=time.perf_counter()-start, quality_time_trace=trace,
                termination_reason='gap_and_stagnation' if state['stopped'] else f'status_{status}'))
            if selected is not None:
                previous = selected
        finally:
            model.dispose()
    result = dict(schema='rsu_constrained_frontier_v1', requirements=requirements, frontier=rows,
        candidate_count=m, target_count=n, compressed_target_count=ng, incidence_edges=matrix.nnz,
        preparation_sec=preparation_sec, wall_sec=time.perf_counter()-begin,
        scope='Exact formulation on supplied candidates/targets; time-limited certificates are explicit.')
    if candidate_costs is not None or critical_coverage_ratio:
        result.update(schema='rsu_cost_critical_frontier_v1', cost_limits=cost_limits,
                      critical_coverage_ratio=critical_coverage_ratio)
    return result

r"""Exact lazy-cut MILP for unweighted at-most-K binary union coverage.

For any S, f(T) <= f(S) + sum(i in T\S) |A_i\union(S)|. Thus
theta <= f(S) + gains(S) @ x is valid, and tight at x=1_S.
Separating every violated integer incumbent yields the original discrete problem.
No candidate pruning, target aggregation, physics, or default solver changes.
"""
import time

import numpy as np


_POPCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def certified_stagnation_stop(elapsed, last_improvement, lower, upper, policy):
    """A flat incumbent alone is never an optimality certificate."""
    gap = (upper - lower) / lower if lower > 0 else (0. if upper == 0 else float('inf'))
    return (elapsed >= policy['minimum_seconds']
            and elapsed - last_improvement >= policy['stagnation_seconds']
            and np.isfinite(gap) and 0 <= gap <= policy['relative_gap'])


class PackedCoverage:
    def __init__(self, masks, targets):
        self.masks = np.asarray(masks)
        self.targets = int(targets)
        if self.targets <= 0 or self.masks.dtype != np.uint8 or self.masks.ndim != 2 or not len(self.masks):
            raise ValueError('Nonempty uint8 packed masks and positive target count required')
        if self.masks.shape[1] != (self.targets+7)//8:
            raise ValueError('Packed mask shape mismatch')
        self.valid = np.packbits(np.ones(self.targets, dtype=bool))
        if np.any(self.masks[:, -1] & np.bitwise_not(self.valid[-1])):
            raise ValueError('Nonzero padding bits in archive')
        self.queries = 0

    def union(self, ids):
        return (np.bitwise_or.reduce(self.masks[list(ids)], axis=0) if len(ids)
                else np.zeros(self.masks.shape[1], dtype=np.uint8))

    def count(self, ids):
        return int(_POPCOUNT[self.union(ids)].sum())

    def gains(self, union):
        self.queries += 1
        uncovered = self.valid & np.bitwise_not(union)
        result = np.empty(len(self.masks), dtype=np.int64)
        for first in range(0, len(result), 2048):
            block = self.masks[first:first+2048] & uncovered
            result[first:first+len(block)] = _POPCOUNT[block].sum(axis=1)
        return result


def swap_start(coverage, k, start_ids, seconds=30.):
    """Independent discrete candidate swaps, no online/transfer poses consumed."""
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError('Nonnegative finite swap time required')
    start = time.perf_counter()
    selected = sorted(set(start_ids))
    best = coverage.count(selected)
    sweeps = 0
    reason = 'local_optimum'
    while time.perf_counter()-start < seconds:
        next_ids, next_count = selected, best
        # One insertion or replacement; each trial scans all archived candidates.
        for drop in ([None] if len(selected) < k else []) + list(selected):
            if time.perf_counter()-start >= seconds:
                reason = 'time_limit'
                break
            retained = [i for i in selected if i != drop]
            union = coverage.union(retained)
            gains = coverage.gains(union)
            gains[retained] = -1
            added = int(np.argmax(gains))
            value = int(_POPCOUNT[union].sum()) + int(gains[added])
            if gains[added] > 0 and value > next_count:
                next_ids, next_count = sorted([*retained, added]), value
        sweeps += 1
        if next_count <= best:
            break
        selected, best = next_ids, next_count
    if time.perf_counter()-start >= seconds:
        reason = 'time_limit'
    return selected, dict(coverage_count=best, sweeps=sweeps, stop_reason=reason,
                         elapsed_sec=time.perf_counter()-start, time_limit_sec=seconds)


def solve_cut_milp(masks, num_targets, k, start_ids, *, seconds=300., swap_seconds=30.,
                   log_file=None, threads=1, soft_mem_gb=90., stop_policy=None):
    import gurobipy as gp
    from gurobipy import GRB
    if type(k) is not int or k < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive integer K and finite solver time required')
    if stop_policy is not None:
        required = {'minimum_seconds', 'stagnation_seconds', 'relative_gap', 'interval_seconds'}
        if (not isinstance(stop_policy, dict) or set(stop_policy) != required
                or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0
                       for v in stop_policy.values())
                or stop_policy['minimum_seconds'] > seconds
                or stop_policy['interval_seconds'] <= 0
                or stop_policy['relative_gap'] > 1):
            raise ValueError('Invalid certified-gap/stagnation stop policy')
        stop_policy = dict(stop_policy)
    coverage = PackedCoverage(masks, num_targets)
    if len(set(start_ids)) != len(start_ids) or len(start_ids) > k or any(i<0 or i>=len(masks) for i in start_ids):
        raise ValueError('Invalid warm start IDs')
    begin = time.perf_counter()
    original_count = coverage.count(start_ids)
    best_ids, swap = swap_start(coverage, k, start_ids, swap_seconds)
    best_count = coverage.count(best_ids)
    model = gp.Model('union_coverage_lazy_cuts')
    state = dict(ids=best_ids, count=best_count, cuts=0, callbacks=0, error=None, trace=[],
                 quality_time_trace=[], last_improvement=0., last_sample=-float('inf'),
                 termination_reason=None)
    try:
        model.Params.OutputFlag = int(log_file is not None)
        model.Params.LogToConsole = 0
        if log_file:
            model.Params.LogFile = str(log_file)
        model.Params.Threads = threads
        model.Params.TimeLimit = seconds
        model.Params.MIPGap = 0.
        model.Params.SoftMemLimit = soft_mem_gb
        model.Params.LazyConstraints = 1
        # This model has only the budget and coverage cuts, not the huge incidence matrix.
        x = model.addVars(len(masks), vtype=GRB.BINARY, name='x')
        variables = [x[i] for i in range(len(masks))]
        theta = model.addVar(lb=0., ub=num_targets, name='covered_count')
        model.addConstr(gp.quicksum(variables) <= k)
        model.setObjective(theta, GRB.MAXIMIZE)
        def cut(ids):
            union = coverage.union(ids)
            gains = coverage.gains(union)
            nz = np.flatnonzero(gains)
            return int(_POPCOUNT[union].sum()) + gp.LinExpr(gains[nz].tolist(), [variables[i] for i in nz])
        model.addConstr(theta <= cut([]), name='singleton_upper_bound')
        model.addConstr(theta <= cut(best_ids), name='warm_start_cut')
        start_set = set(best_ids)
        model.setAttr('Start', variables, [float(i in start_set) for i in range(len(masks))])
        theta.Start = best_count
        def observe(elapsed, bound, event, force=False):
            if stop_policy is None:
                return
            if not force and elapsed - state['last_sample'] < stop_policy['interval_seconds']:
                return
            bound = min(float(num_targets), float(bound)) if np.isfinite(bound) else float(num_targets)
            if bound + 1e-5 < state['count']:
                raise RuntimeError('Callback bound below a physically feasible incumbent')
            bound = max(bound, float(state['count']))
            gap = ((bound - state['count']) / state['count'] if state['count']
                   else (0. if bound == 0 else None))
            state['quality_time_trace'].append(dict(solver_elapsed_sec=float(elapsed),
                covered_count=state['count'], coverage_pct=100. * state['count'] / num_targets,
                upper_bound_count=bound, upper_bound_coverage_pct=100. * bound / num_targets,
                certified_gap=gap, event=event))
            state['last_sample'] = elapsed
        observe(0., num_targets, 'warm_start', force=True)
        def callback(m, where):
            if where != GRB.Callback.MIPSOL and (stop_policy is None or where != GRB.Callback.MIP):
                return
            try:
                elapsed = float(m.cbGet(GRB.Callback.RUNTIME))
                if where == GRB.Callback.MIP:
                    if stop_policy is None:
                        return
                    bound = min(float(num_targets), float(m.cbGet(GRB.Callback.MIP_OBJBND)))
                    observe(elapsed, bound, 'progress')
                    if certified_stagnation_stop(elapsed, state['last_improvement'],
                                                  state['count'], bound, stop_policy):
                        state['termination_reason'] = 'certified_gap_and_stagnation'
                        observe(elapsed, bound, 'certified_early_stop', force=True)
                        m.terminate()
                    return
                state['callbacks'] += 1
                values = m.cbGetSolution(variables)
                ids = [i for i, value in enumerate(values) if value > .5]
                actual = coverage.count(ids)
                if len(ids) > k:
                    raise RuntimeError('Solver incumbent violates budget')
                if actual > state['count']:
                    state.update(ids=ids, count=actual)
                    state['last_improvement'] = elapsed
                    state['trace'].append(dict(elapsed_sec=time.perf_counter()-begin, covered_count=actual))
                    observe(elapsed, m.cbGet(GRB.Callback.MIPSOL_OBJBND), 'improvement', force=True)
                if m.cbGetSolution(theta) > actual + 1e-5:
                    m.cbLazy(theta <= cut(ids))
                    state['cuts'] += 1
            except Exception as exc:
                state['error'] = repr(exc)
                m.terminate()
        model.optimize(callback)
        if state['error']:
            raise RuntimeError(f'Lazy constraint callback failed: {state["error"]}')
        raw_bound = float(model.ObjBound)
        bound = min(float(num_targets), raw_bound) if np.isfinite(raw_bound) else float(num_targets)
        if bound + 1e-5 < state['count']:
            raise RuntimeError('Bound is below directly evaluated feasible solution')
        incumbent = float(model.ObjVal) if model.SolCount else None
        if model.SolCount:
            returned = [i for i, value in enumerate(model.getAttr('X', variables)) if value > .5]
            if abs(coverage.count(returned)-incumbent) > 1e-5:
                raise RuntimeError('Solver objective differs from exact union coverage')
        observe(float(model.Runtime), bound, 'finished', force=True)
        return dict(selected_ids=state['ids'], objective=state['count'],
            original_start_objective=original_count, swap=swap,
            solver_incumbent_objective=incumbent, solver_status=int(model.Status),
            solver_best_bound=raw_bound if np.isfinite(raw_bound) else None,
            certified_upper_bound=bound, bound_source='min(solver_bound, target_count)' if np.isfinite(raw_bound) else 'target_count_only',
            certified_gap=(bound-state['count'])/state['count'] if state['count'] else (0. if bound==0 else None),
            discrete_optimal=abs(bound-state['count']) <= 1e-5,
            solver_sec=float(model.Runtime), total_sec=time.perf_counter()-begin,
            cuts=state['cuts'], incumbent_callbacks=state['callbacks'], trace=state['trace'],
            quality_time_trace=state['quality_time_trace'],
            termination_reason=state['termination_reason'] or f'solver_status_{int(model.Status)}',
            marginal_scans=coverage.queries, candidate_count=len(masks), target_count=num_targets,
            parameters=dict(seconds=seconds, swap_seconds=swap_seconds, threads=threads, soft_mem_gb=soft_mem_gb,
                            mip_gap=0., lazy_constraints=1, stop_policy=stop_policy),
            semantics='Same archived discrete candidates, unweighted union, at-most-K. Not cold-map runtime.')
    finally:
        model.dispose()

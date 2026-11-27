"""Opt-in bounded coverage optimization for expanded K<=8 libraries.

Reuses the existing packed swap/cut-MILP. Alternative layouts are feasible
multi-start heuristics, not certified top-N solutions. No legacy defaults change.
"""
import time

import numpy as np

from src.methods.coverage_cut_milp import PackedCoverage, solve_cut_milp, swap_start
from src.methods.library_query_controls import _optical_masks, _profile_values, _union_order


def coverage_metrics(masks, ids):
    density = masks[list(ids)].sum(axis=0, dtype=np.int64)
    n = masks.shape[1]
    covered = int((density > 0).sum())
    double = int((density >= 2).sum())
    return dict(num_selected=len(ids), covered_count=covered, coverage_pct=100 * covered / n,
        overlap_count=double, overlap_pct=100 * double / n,
        overlap_of_covered_pct=100 * double / covered if covered else None,
        mean_density=float(density.mean()), excess_observations=int(np.maximum(density - 1, 0).sum()))


def _greedy(coverage, k, first=None):
    selected = [] if first is None else [int(first)]
    single = coverage.gains(coverage.union([]))
    while len(selected) < min(k, len(coverage.masks)):
        gains = coverage.gains(coverage.union(selected))
        gains[selected] = -1
        best = int(np.lexsort((np.arange(len(gains)), -single, -gains))[0])
        if gains[best] <= 0:
            break
        selected.append(best)
    return sorted(selected)


def optimize_frontier(poses, masks, targets, k_values, *, seconds=3., alternatives=1,
                      swap_seconds=.05, stop_policy=None, certificate_gap=.01):
    """Per-K feasible incumbent and bound; preserve the previous feasible budget.

    The time setting applies to each MILP call, not to the whole pipeline. One
    alternative uses no multi-start overhead, equally for all evaluation arms.
    """
    started = time.perf_counter()
    poses, targets, masks = np.asarray(poses), np.asarray(targets), np.asarray(masks)
    ks = list(k_values)
    if (poses.ndim != 2 or poses.shape[1:] != (5,) or targets.ndim != 2
            or targets.shape[1:] != (3,) or not len(targets) or masks.dtype != bool
            or masks.shape != (len(poses), len(targets))
            or not np.isfinite(poses).all() or not np.isfinite(targets).all()
            or len(np.unique(poses, axis=0)) != len(poses)
            or not ks or ks != sorted(set(ks))
            or any(type(k) is not int or not 1 <= k <= 8 for k in ks)
            or type(alternatives) is not int or not 1 <= alternatives <= 64
            or not np.isfinite([seconds, swap_seconds, certificate_gap]).all()
            or seconds <= 0 or swap_seconds < 0 or not 0 <= certificate_gap <= 1):
        raise ValueError('Invalid expanded finite coverage problem or solver limits')
    n, m = len(targets), len(poses)
    packed = PackedCoverage(np.packbits(masks, axis=1), n) if m else None
    ceiling = int(np.any(masks, axis=0).sum())
    single = masks.sum(axis=1)
    seed_order = np.lexsort((np.arange(m), -single))
    # Different first sites, including lower-ranked locations, preserve geometric modes.
    seed_indices = np.linspace(0, max(m - 1, 0), min(m, max(0, alternatives - 1)), dtype=int)
    seeds = seed_order[seed_indices]
    rows, previous = [], []
    for k in ks:
        begin = time.perf_counter()
        pool = {tuple(previous)}
        if m:
            pool.add(tuple(_greedy(packed, k)))
            for first in seeds:
                ids = _greedy(packed, k, first)
                ids, _ = swap_start(packed, k, ids, seconds=swap_seconds)
                pool.add(tuple(ids))
        key = lambda ids: (-coverage_metrics(masks, ids)['covered_count'], len(ids), ids)
        warm = sorted(pool, key=key)[0]
        lower = coverage_metrics(masks, warm)['covered_count']
        upper = min(ceiling, int(np.sort(single)[::-1][:k].sum()))
        solver = dict(solver_status='analytic_bound', solver_sec=0., certified_upper_bound=upper)
        if lower < upper:
            solver = solve_cut_milp(packed.masks, n, k, list(warm), seconds=seconds,
                swap_seconds=swap_seconds, threads=1, soft_mem_gb=8., stop_policy=stop_policy)
            pool.add(tuple(sorted(solver['selected_ids'])))
            upper = min(float(upper), solver['certified_upper_bound'])
        ordered = sorted(pool, key=key)
        best = ordered[0]
        metrics = coverage_metrics(masks, best)
        if upper + 1e-5 < metrics['covered_count']:
            raise RuntimeError('Bound below feasible union')
        upper = max(float(upper), float(metrics['covered_count']))
        solutions = [dict(selected_ids=list(ids), **coverage_metrics(masks, ids),
            incumbent_loss_count=metrics['covered_count'] - coverage_metrics(masks, ids)['covered_count'])
            for ids in ordered[:alternatives]]
        for solution in solutions:
            value = solution['covered_count']
            solution['upper_bound_count'] = upper
            solution['certified_gap'] = (upper - value) / value if value else (0. if upper == 0 else None)
            solution['near_optimal_certified'] = upper - value <= certificate_gap * value + 1e-5
        rows.append(dict(budget_k=k, selected_ids=list(best), **metrics, upper_bound_count=upper,
            certified_gap=(upper - metrics['covered_count']) / metrics['covered_count']
                if metrics['covered_count'] else (0. if upper == 0 else None),
            discrete_optimal=abs(upper - metrics['covered_count']) <= 1e-5,
            near_optimal_certified=upper - metrics['covered_count'] <= certificate_gap * metrics['covered_count'] + 1e-5,
            certificate_gap_threshold=certificate_gap,
            alternatives=solutions, solver=solver, selection_sec=time.perf_counter() - begin))
        previous = list(best)
    return dict(schema='expanded_coverage_frontier_v1', candidate_count=m, target_count=n,
        budget_mode='at_most_k', k_values=ks, frontier=rows, coverage_ceiling_count=ceiling,
        phase_wall_sec=dict(all_k_selection=time.perf_counter() - started),
        alternatives_semantics='Verified feasible multi-start/swap alternatives, not exact top-N',
        certificate_scope='Only the supplied finite candidate/target mask problem')


def geometry_plan(poses, targets, profile, q, k_values, method, *, seconds=1., seed=None):
    """Source-free plans; no physical visibility input and no raycaster call."""
    begin = time.perf_counter()
    m = len(poses)
    if type(q) is not int or not 0 <= q <= m:
        raise ValueError('Invalid query budget')
    surrogate = None
    if method == 'random':
        order = np.random.default_rng(seed).choice(m, q, replace=False).tolist()
    elif method in ('geometry_single', 'geometry_union', 'geometry_portfolio_milp'):
        masks = _optical_masks(np.asarray(poses, np.float32), np.asarray(targets, np.float32),
                               _profile_values(profile))
        if method == 'geometry_single':
            order = np.lexsort((np.arange(m), -masks.sum(axis=1)))[:q].tolist()
        elif method == 'geometry_union':
            order = _union_order(masks, q)
        else:
            surrogate = optimize_frontier(poses, masks, targets, k_values,
                seconds=seconds, alternatives=8, swap_seconds=.01)
            order, used = [], set()
            for rank in range(8):
                for row in surrogate['frontier']:
                    if rank >= len(row['alternatives']):
                        continue
                    for cid in row['alternatives'][rank]['selected_ids']:
                        if cid not in used:
                            order.append(cid)
                            used.add(cid)
            if len(order) < q:
                for cid in _union_order(masks, m):
                    if len(order) >= q:
                        break
                    if cid not in used:
                        order.append(cid)
                        used.add(cid)
            order = order[:q]
    else:
        raise ValueError('Unknown source-free method')
    return dict(method=method, seed=seed, query_order=order, selected_ids=sorted(order),
        planning_sec=time.perf_counter() - begin, surrogate_frontier=surrogate,
        optical_candidate_evaluations=m if method != 'random' else 0,
        semantics='Unoccluded distance/FOV surrogate only; planning CPU is included in time')

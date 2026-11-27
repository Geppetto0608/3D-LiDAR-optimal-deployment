"""Source-only, unweighted coverage frontiers on an explicit finite domain.

This first library contract has no cost, spacing, quota or service constraints.
Eligibility of individual poses belongs to the geometry-specific builder.
"""
from hashlib import sha256
from itertools import combinations
import json
from math import comb
import time

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def _relative_gap(upper, lower):
    return (upper - lower) / lower if lower else (0.0 if upper == 0 else None)


def build_frontier(poses, masks, targets, k_values, *, alternatives_per_k=3,
                   enumeration_limit=100000, proposals=(), context=None):
    """Keep best verified layouts for each at-most-K budget without regression.

    Exact enumeration is all-or-nothing under a precomputed subset-count guard.
    Otherwise retain greedy prefixes and supplied feasible layouts, with a valid
    analytic upper bound. Alternatives include the incumbent as their first row.
    Equal masks at different poses are intentionally not deduplicated.

    Proposals may contain selected_ids, budget_k, origin and covered_count only.
    Claimed source solver certificates are not imported into a different domain.
    Validation here uses the supplied masks, NOT independent physical raycasting.
    """
    started = time.perf_counter()
    poses = np.asarray(poses, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    masks = np.asarray(masks)
    if poses.ndim != 2 or poses.shape[1:] != (5,) or not np.isfinite(poses).all():
        raise ValueError('Expected finite poses[M,5]')
    if (targets.ndim != 2 or targets.shape[1:] != (3,) or not len(targets)
            or not np.isfinite(targets).all()):
        raise ValueError('Expected nonempty finite targets[N,3]')
    m, n = len(poses), len(targets)
    if masks.dtype != bool or masks.shape != (m, n):
        raise ValueError('Expected boolean masks[M,N]')
    if len(np.unique(poses, axis=0)) != m:
        raise ValueError('Duplicate physical pose rows')
    if len(np.unique(targets, axis=0)) != n:
        raise ValueError('Duplicate target rows')
    ks = [_integer(k, 'K', 1) for k in k_values]
    if not ks or ks != sorted(set(ks)):
        raise ValueError('K values must be nonempty, increasing and unique')
    limit = _integer(enumeration_limit, 'enumeration_limit')
    keep = _integer(alternatives_per_k, 'alternatives_per_k', 1)
    if context is not None and not isinstance(context, dict):
        raise ValueError('Context must be a dictionary')
    context = json.loads(json.dumps(context or {}, allow_nan=False, sort_keys=True))

    fingerprint = sha256()
    for array in (poses, targets, np.packbits(masks, axis=1)):
        fingerprint.update(str(array.shape).encode('ascii'))
        fingerprint.update(np.ascontiguousarray(array).tobytes())
    fingerprint.update(json.dumps(context, sort_keys=True).encode('utf-8'))

    pool = {}

    def add(ids, origin):
        key = tuple(sorted(ids))
        if key not in pool:
            count = int(np.any(masks[list(key)], axis=0).sum()) if key else 0
            pool[key] = (count, origin)
        return pool[key][0]

    add((), 'empty')
    proposal_count = 0
    for proposal in proposals:
        if (not isinstance(proposal, dict) or 'selected_ids' not in proposal
                or set(proposal) - {'selected_ids', 'budget_k', 'origin', 'covered_count'}):
            raise ValueError('Unsupported proposal fields; bounds require separate domain verification')
        ids = [_integer(i, 'candidate ID') for i in proposal['selected_ids']]
        if len(set(ids)) != len(ids) or any(i >= m for i in ids):
            raise ValueError('Proposal has duplicate or out-of-domain candidate IDs')
        if len(ids) > _integer(proposal.get('budget_k', len(ids)), 'proposal budget'):
            raise ValueError('Proposal exceeds its declared budget')
        origin = proposal.get('origin', 'provided')
        if not isinstance(origin, str) or not origin:
            raise ValueError('Proposal origin must be a nonempty string')
        count = add(ids, origin)
        if ('covered_count' in proposal
                and _integer(proposal['covered_count'], 'covered_count') != count):
            raise ValueError('Proposal coverage differs from supplied masks')
        proposal_count += 1

    selected, covered = [], np.zeros(n, dtype=bool)
    single_counts = masks.sum(axis=1, dtype=np.int64)
    for _ in range(min(max(ks), m)):
        gains = (masks & ~covered).sum(axis=1, dtype=np.int64)
        # Match the existing greedy tie preference: single coverage, then ID.
        order = np.lexsort((np.arange(m), -single_counts, -gains))
        best = int(order[0])
        if gains[best] == 0:
            break
        selected.append(best)
        covered |= masks[best]
        add(selected, 'greedy')

    max_count = min(max(ks), m)
    subset_count = sum(comb(m, r) for r in range(max_count + 1))
    enumerated = subset_count <= limit
    if enumerated:
        for size in range(1, max_count + 1):
            for ids in combinations(range(m), size):
                add(ids, 'enumeration')

    union_count = int(np.any(masks, axis=0).sum())
    top_counts = np.sort(single_counts)[::-1]

    def layout(ids):
        count, origin = pool[ids]
        return dict(selected_ids=list(ids), poses=poses[list(ids)].tolist(),
                    num_selected=len(ids), covered_count=count,
                    coverage_pct=100.0 * count / n, origin=origin)

    rows, previous = [], 0
    for k in ks:
        eligible = sorted((ids for ids in pool if len(ids) <= k),
                          key=lambda ids: (-pool[ids][0], len(ids), ids))
        incumbent = layout(eligible[0])
        lower = incumbent['covered_count']
        upper = lower if enumerated else min(union_count, int(top_counts[:k].sum()))
        if upper < lower or lower < previous:
            raise RuntimeError('Invalid bound or decreasing at-most-K incumbent')
        alternatives = []
        for ids in eligible[:keep]:
            alternative = layout(ids)
            alternative['incumbent_loss_count'] = lower - alternative['covered_count']
            alternatives.append(alternative)
        rows.append(dict(**incumbent, budget_k=k, upper_bound_count=upper,
                         certified_gap=_relative_gap(upper, lower),
                         certification='discrete_optimal' if upper == lower else 'best_known',
                         bound_source='exhaustive_subsets' if enumerated else 'union_and_single_counts',
                         marginal_covered_count=lower - previous,
                         previous_requested_budget=rows[-1]['budget_k'] if rows else 0,
                         alternatives=alternatives))
        previous = lower
    report = dict(schema='deployment_library_frontier_v1', context=context,
                domain_sha256=fingerprint.hexdigest(), candidate_count=m, target_count=n,
                coverage_ceiling_count=union_count, coverage_ceiling_pct=100.0 * union_count / n,
                ceiling_scope='union of all eligible candidates; no joint installation constraints',
                budget_mode='at_most_k', objective='unweighted_union_count', k_values=ks,
                frontier=rows, enumeration=dict(completed=enumerated, subset_count=subset_count,
                    limit=limit, reason='complete' if enumerated else 'subset_limit'),
                proposed_layout_count=proposal_count, retained_pool_size=len(pool),
                validation_scope='supplied candidate masks; physical verification belongs to builder',
                frontier_work_sec=time.perf_counter() - started)
    report['minimum_to_candidate_ceiling'] = minimum_sensor_bracket(report, union_count)
    report['minimum_to_all_targets'] = minimum_sensor_bracket(report, n)
    return report


def minimum_sensor_bracket(report, required_covered_count):
    """Bracket minimum count for an integer target requirement in this domain.

    A feasible m-sensor witness bounds the minimum from above. A valid at-most-K
    coverage bound below the threshold excludes ALL budgets <= K. Missing K rows
    and equal incumbent scores alone cannot establish the minimum count.
    """
    required = _integer(required_covered_count, 'required_covered_count')
    if report.get('schema') != 'deployment_library_frontier_v1':
        raise ValueError('Expected a validated deployment library frontier')
    if required > report['target_count']:
        raise ValueError('Required count exceeds target count')
    if required > report['coverage_ceiling_count']:
        return dict(required_covered_count=required, lower_bound_sensors=None,
                    upper_bound_sensors=None, status='unattainable_in_declared_domain',
                    domain_sha256=report['domain_sha256'],
                    largest_evaluated_budget=max(report['k_values']))
    if required == 0:
        lower, upper = 0, 0
    else:
        lower, upper = 1, None
        for row in report['frontier']:
            if row['upper_bound_count'] < required:
                lower = max(lower, row['budget_k'] + 1)
            for solution in [row, *row['alternatives']]:
                if solution['covered_count'] >= required:
                    count = solution['num_selected']
                    upper = count if upper is None else min(upper, count)
        if upper is not None and lower > upper:
            raise ValueError('Inconsistent bounds and feasible witness')
    return dict(required_covered_count=required, lower_bound_sensors=lower,
                upper_bound_sensors=upper,
                status='certified_minimum' if lower == upper else
                       ('bounded' if upper is not None else 'not_reached_in_saved_solutions'),
                domain_sha256=report['domain_sha256'],
                largest_evaluated_budget=max(report['k_values']))

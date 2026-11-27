"""Source-free, equal-query planners over the full target candidate domain.

Geometry methods optimize an unoccluded optical surrogate on the CPU. They do
not observe physical coverage or certify deployment quality. Planning never
constructs a scene, casts rays, writes files, or consumes source-library data.
"""
from math import comb
import time

import numpy as np

from src.core.deployment_library import build_frontier
from src.environment import angular_visibility_mask, compute_view_angles_numpy


_POLICIES = {
    'geometry_single': 'descending unoccluded single count, then full candidate ID',
    'geometry_union': 'greedy marginal unoccluded union; ties by single count then ID; '
                      'fill zero-gain slots without repeats',
    'geometry_portfolio': 'exhaustive at-most-K unoccluded frontier; alternative rank '
                          'then increasing K then sorted candidate IDs; fill with '
                          'the standalone geometry_union order, skipping selected IDs',
    'random': 'sorted numpy.default_rng(seed).choice(M, q, replace=False)',
}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def _rows(values, width, name):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'Expected finite {name}[N,{width}]') from exc
    if array.shape == (0,):
        array = array.reshape(0, width)
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise ValueError(f'Expected finite {name}[N,{width}]')
    if len(np.unique(array, axis=0)) != len(array):
        raise ValueError(f'Duplicate {name} rows')
    with np.errstate(over='ignore'):
        optical = array.astype(np.float32)
    if not np.isfinite(optical).all():
        raise ValueError(f'{name} must be representable as finite float32')
    return array, optical


def _profile_values(profile):
    try:
        h, v, distance = (float(getattr(profile, key)) for key in
                          ('h_fov_deg', 'v_fov_deg', 'max_range_m'))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Expected profile with finite h_fov_deg, v_fov_deg, max_range_m') from exc
    if (not np.isfinite([h, v, distance]).all() or not 0 < h <= 360
            or not 0 < v <= 180 or distance <= 0):
        raise ValueError('Profile requires 0 < h_fov_deg <= 360, 0 < v_fov_deg <= 180, '
                         'and finite positive max_range_m')
    return h, v, distance


def _optical_masks(poses, targets, profile_values):
    h, v, max_range = profile_values
    half_h, half_v = np.deg2rad([h, v]) / 2
    masks = np.zeros((len(poses), len(targets)), dtype=bool)
    for cid, pose in enumerate(poses):
        vectors = targets - pose[:3]
        az, el, _, distance = compute_view_angles_numpy(*vectors.T, pose[3], pose[4])
        masks[cid] = angular_visibility_mask(az, el, distance, half_h, half_v, max_range)
    return masks


def _union_order(masks, count):
    single = masks.sum(axis=1, dtype=np.int64)
    covered = np.zeros(masks.shape[1], dtype=bool)
    used = np.zeros(len(masks), dtype=bool)
    order = []
    for _ in range(count):
        gains = (masks & ~covered).sum(axis=1, dtype=np.int64)
        candidates = np.flatnonzero(~used)
        ranked = np.lexsort((candidates, -single[candidates], -gains[candidates]))
        cid = int(candidates[ranked[0]])
        order.append(cid)
        used[cid] = True
        covered |= masks[cid]
    return order


def plan_control(poses, targets, profile, query_budget, k_values, method, *,
                 seed=None, enumeration_limit=100000) -> dict:
    """Return a JSON-native plan without making any physical candidate queries.

    Poses[M,5] and targets[N,3] are finite, distinct rows; IDs always index the
    original pose array. M <= 2000 and M*N <= 4,000,000. K values are nonempty,
    strictly increasing integers in 1..64. Targets may be empty only for q=0.
    Budgets are strict integers with 0 <= q <= M, never silently clamped.
    The profile exposes h_fov_deg, v_fov_deg, and max_range_m.

    Random requires an explicit nonnegative integer seed and uses a local RNG.
    Optional seeds on deterministic methods are validated and recorded only.
    Geometry work is M optical candidate evaluations and M*N point tests;
    random and q=0 perform none. planning_sec includes validation, surrogate
    computation, and selection. It is the only nondeterministic result field.

    Portfolio uses build_frontier with q alternatives per K, requires completed
    exhaustive surrogate enumeration, and rejects an insufficient limit. Its
    fallback is standalone greedy-union order, not greedy conditional on the
    portfolio prefix. No surrogate result is a physical coverage certificate.
    """
    started = time.perf_counter()
    if not isinstance(method, str) or method not in _POLICIES:
        raise ValueError(f'Unknown query control method: {method!r}')
    q = _integer(query_budget, 'query_budget')
    limit = _integer(enumeration_limit, 'enumeration_limit')
    if seed is not None:
        seed = _integer(seed, 'seed')
    elif method == 'random':
        raise ValueError('random requires an explicit nonnegative integer seed')
    try:
        ks = [_integer(k, 'K', 1) for k in k_values]
    except TypeError as exc:
        raise ValueError('K values must be an iterable of integers') from exc
    if not ks or ks != sorted(set(ks)) or max(ks) > 64:
        raise ValueError('K values must be nonempty, increasing, unique integers in 1..64')
    poses, optical_poses = _rows(poses, 5, 'poses')
    targets, optical_targets = _rows(targets, 3, 'targets')
    m, n = len(poses), len(targets)
    if m > 2000 or m * n > 4_000_000:
        raise ValueError('Planning caps require M <= 2000 and M*N <= 4000000')
    if q > m:
        raise ValueError('query_budget must not exceed candidate count')
    if q and not n:
        raise ValueError('Nonzero query_budget requires nonempty targets')
    profile_values = _profile_values(profile)
    order, evaluations, point_tests = [], 0, 0
    enumeration_completed = None
    subset_count = None
    if q and method == 'random':
        order = sorted(np.random.default_rng(seed).choice(m, q, replace=False).tolist())
    elif q:
        if method == 'geometry_portfolio':
            subset_count = sum(comb(m, size) for size in range(min(max(ks), m) + 1))
            if subset_count > limit:
                raise ValueError('Portfolio requires completed surrogate enumeration; '
                                 f'{subset_count} subsets exceed enumeration_limit={limit}')
        masks = _optical_masks(optical_poses, optical_targets, profile_values)
        evaluations, point_tests = m, m * n
        if method == 'geometry_single':
            counts = masks.sum(axis=1, dtype=np.int64)
            order = np.lexsort((np.arange(m), -counts))[:q].tolist()
        elif method == 'geometry_union':
            order = _union_order(masks, q)
        else:
            frontier = build_frontier(poses, masks, targets, ks,
                                      alternatives_per_k=q, enumeration_limit=limit)
            enumeration_completed = bool(frontier['enumeration']['completed'])
            if not enumeration_completed:
                raise ValueError('Portfolio requires completed surrogate enumeration')
            rows = sorted(frontier['frontier'], key=lambda row: row['budget_k'])
            seen = set()
            for rank in range(q):
                for row in rows:
                    if rank < len(row['alternatives']):
                        for cid in sorted(row['alternatives'][rank]['selected_ids']):
                            if cid not in seen and len(order) < q:
                                seen.add(cid)
                                order.append(int(cid))
                if len(order) == q:
                    break
            if len(order) < q:
                # The first q greedy IDs contain enough IDs outside any prefix.
                for cid in _union_order(masks, q):
                    if cid not in seen:
                        order.append(cid)
                        seen.add(cid)
                    if len(order) == q:
                        break
    return dict(method=method, seed=seed, selected_ids=sorted(order), query_order=order,
                query_budget=q, candidate_count=m, target_count=n, k_values=ks,
                planning_sec=time.perf_counter() - started,
                optical_candidate_evaluations=evaluations, optical_point_tests=point_tests,
                surrogate_enumeration_completed=enumeration_completed,
                surrogate_subset_count=subset_count,
                semantics=dict(policy=_POLICIES[method], candidate_ids='full input pose row indices',
                               information='poses, target coordinates, and optical profile only',
                               surrogate='unoccluded float32 optical gates; occlusion is ignored',
                               optimization='CPU-counted approximate target optimization, '
                                            'not a free physical coverage oracle',
                               physical_queries=0, physical_coverage_certified=False))

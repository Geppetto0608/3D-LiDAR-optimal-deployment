"""Offline, unpruned pair tables and labelled action-graph diagnostics.

Only the declared two-sensor corridor constraints are supported. This module
uses stored incidence arrays, not visibility queries or an online repair policy.
"""
from collections import deque
from dataclasses import dataclass
import heapq

import numpy as np


TOL = 1e-8


@dataclass
class PairTable:
    scores: np.ndarray
    feasible: np.ndarray
    covered: np.ndarray
    double: np.ndarray
    costs: np.ndarray


def pair_table(poses, masks, specification, allowed, interface, thresholds):
    poses, masks = np.asarray(poses), np.asarray(masks)
    n = len(poses)
    if (poses.shape != (n, 5) or not np.isfinite(poses).all() or masks.ndim != 2
            or masks.shape[0] != n or masks.shape[1] == 0 or masks.dtype != bool
            or np.asarray(allowed).shape != (n,) or np.asarray(allowed).dtype != bool
            or np.asarray(interface).shape != (masks.shape[1],) or np.asarray(interface).dtype != bool
            or not np.any(interface)):
        raise ValueError('Expected finite poses, boolean masks, eligibility and a nonempty interface')
    if specification['budget_k'] != 2 or specification.get('regions'):
        raise ValueError('Only fixed K=2 without world-axis regions is supported')
    c, w, cost = (specification.get(k, {}) for k in ('constraints', 'weights', 'cost'))
    supported = {'max_installation_cost', 'min_xy_separation_m', 'max_sensors_per_pole',
        'mounting_z_range_m', 'min_coverage_pct', 'min_overlap_ratio', 'max_overlap_ratio',
        'min_single_failure_coverage_pct'}
    if (set(c)-supported or set(w)-{'coverage_weight', 'importance_weight', 'overlap_penalty',
            'sensor_count_penalty', 'installation_cost_weight'}
            or set(cost)-{'base_per_sensor', 'per_height_m'}
            or set(thresholds)-{'min_coverage_pct', 'min_overlap_ratio', 'max_overlap_ratio'}):
        raise ValueError('Unsupported independent diagnostic requirement')
    costs = cost.get('base_per_sensor', 0)+cost.get('per_height_m', 0)*poses[:, 2]
    individual = masks.sum(axis=1)
    zlo, zhi = c.get('mounting_z_range_m', [0, np.inf])
    covered, double = (np.zeros((n, n), dtype=np.int32) for _ in range(2))
    scores, feasible = np.full((n, n), -np.inf), np.zeros((n, n), dtype=bool)
    for i in range(n-1):
        js = np.arange(i+1, n)
        union = np.count_nonzero(masks[js] | masks[i], axis=1)
        overlap = np.count_nonzero(masks[js] & masks[i], axis=1)
        total_cost = costs[i]+costs[js]
        band_union = 100*np.mean(masks[js][:, interface] | masks[i, interface], axis=1)
        band_double = np.mean(masks[js][:, interface] & masks[i, interface], axis=1)
        valid = (allowed[i] & allowed[js]
            & (total_cost <= c.get('max_installation_cost', np.inf)+1e-9)
            & (np.linalg.norm(poses[js, :2]-poses[i, :2], axis=1)+1e-9 >= c.get('min_xy_separation_m', 0))
            & (100*union/masks.shape[1]+1e-9 >= c.get('min_coverage_pct', 0))
            & (overlap/masks.shape[1]+1e-9 >= c.get('min_overlap_ratio', 0))
            & (overlap/masks.shape[1] <= c.get('max_overlap_ratio', 1)+1e-9)
            & (100*np.minimum(individual[i], individual[js])/masks.shape[1]+1e-9
               >= c.get('min_single_failure_coverage_pct', 0))
            & (poses[i, 2]+1e-9 >= zlo) & (poses[i, 2] <= zhi+1e-9)
            & (poses[js, 2]+1e-9 >= zlo) & (poses[js, 2] <= zhi+1e-9)
            & (band_union+1e-9 >= thresholds.get('min_coverage_pct', 0))
            & (band_double+1e-9 >= thresholds.get('min_overlap_ratio', 0))
            & (band_double <= thresholds.get('max_overlap_ratio', 1)+1e-9)
            & np.any(poses[js] != poses[i], axis=1))
        if c.get('max_sensors_per_pole', 2) == 1:
            valid &= np.any(poses[js, :2] != poses[i, :2], axis=1)
        values = ((w.get('coverage_weight', 1)+w.get('importance_weight', 0))*union
            - w.get('overlap_penalty', 0)*overlap - 2*w.get('sensor_count_penalty', 0)
            - w.get('installation_cost_weight', 0)*total_cost)
        for array, values in ((scores, values), (feasible, valid), (covered, union), (double, overlap)):
            array[i, js], array[js, i] = values, values
    return PairTable(scores, feasible, covered, double, costs)


def best_assignment(table, first, second, extra=None):
    first, second = np.asarray(sorted(set(first)), dtype=int), np.asarray(sorted(set(second)), dtype=int)
    use = table.feasible[np.ix_(first, second)].copy()
    if extra is not None:
        use &= extra[np.ix_(first, second)]
    if not use.any():
        return dict(score=None, selected_ids=None, feasible_assignments=0)
    values = table.scores[np.ix_(first, second)]
    best = float(values[use].max())
    a, b = np.argwhere(use & (values >= best-TOL))[0]
    return dict(score=best, selected_ids=[int(first[a]), int(second[b])],
                feasible_assignments=int(use.sum()))


def omission_witness(table, pool, modules):
    quota = modules[:, None] != modules[None, :]
    ids = np.arange(len(modules))
    reference = best_assignment(table, ids, ids, quota)
    restricted = best_assignment(table, pool, pool, quota)
    if reference['score'] is None:
        return dict(reference=reference, restricted=restricted, optimal_pair_count=0, witness_ids=None)
    optima = np.argwhere(np.triu(table.feasible & quota & (table.scores >= reference['score']-TOL), 1))
    present = np.zeros(len(modules), dtype=bool)
    present[list(pool)] = True
    additions = np.count_nonzero(~present[optima], axis=1)
    witness = optima[int(np.argmin(additions))]
    essential = set(optima[0])
    for pair in optima[1:]:
        essential.intersection_update(pair)
    return dict(reference=reference, restricted=restricted, optimal_pair_count=len(optima),
        optimal_pairs_inside_pool=int(np.count_nonzero(additions == 0)), witness_ids=witness.tolist(),
        missing_ids=witness[~present[witness]].tolist(), minimum_target_candidate_additions=int(additions.min()),
        candidates_in_any_optimum=np.unique(optima).tolist(), essential_candidate_ids=sorted(map(int, essential)))


def coordinate_neighbors(canonical):
    canonical = np.asarray(canonical)
    lookup = {tuple(row): i for i, row in enumerate(canonical)}
    if len(lookup) != len(canonical):
        raise ValueError('Duplicate canonical candidates')
    levels = {axis: np.unique(canonical[:, axis]) for axis in (0, 2, 3, 4)}
    neighbors = []
    for row in canonical:
        adjacent = []
        for axis, values in levels.items():
            index = int(np.searchsorted(values, row[axis]))
            for step in (-1, 1):
                if 0 <= index+step < len(values):
                    trial = row.copy()
                    trial[axis] = values[index+step]
                    if tuple(trial) in lookup:
                        adjacent.append(lookup[tuple(trial)])
        neighbors.append(tuple(adjacent))
    return neighbors


def labelled_reachability(table, first, second, neighbors, start, required_state=None):
    """Exact feasible component and minimum necessary J dip to its best state."""
    first, second = tuple(sorted(first)), tuple(sorted(second))
    domain = [set(first), set(second)]
    start = tuple(map(int, start))
    if start[0] not in domain[0] or start[1] not in domain[1] or not table.feasible[start]:
        raise ValueError('Start must be a feasible labelled assignment inside the fixed domains')

    def adjacent(state):
        for agent in (0, 1):
            for replacement in neighbors[state[agent]]:
                if replacement not in domain[agent]:
                    continue
                candidate = (replacement, state[1]) if agent == 0 else (state[0], replacement)
                if table.feasible[candidate]:
                    yield candidate

    component, queue = {start}, deque([start])
    while queue:
        for trial in adjacent(queue.popleft()):
            if trial not in component:
                component.add(trial)
                queue.append(trial)
    if required_state is not None and tuple(required_state) not in component:
        raise ValueError('Initial and final deployments are in different feasible action components')
    best = max(float(table.scores[state]) for state in component)
    best_ids = min(state for state in component if table.scores[state] >= best-TOL)
    start_score = float(table.scores[start])
    capacity, parents, heap = {start: start_score}, {}, [(-start_score, start)]
    while heap:
        negative, state = heapq.heappop(heap)
        bound = -negative
        if bound < capacity[state]-TOL:
            continue
        if table.scores[state] >= best-TOL:
            goal, bottleneck = state, bound
            break
        for trial in adjacent(state):
            proposed = min(bound, float(table.scores[trial]))
            if proposed > capacity.get(trial, -np.inf)+TOL:
                capacity[trial], parents[trial] = proposed, state
                heapq.heappush(heap, (-proposed, trial))
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return dict(score=best, selected_ids=list(best_ids), reachable_states=len(component),
        minimum_objective_dip=max(0., start_score-bottleneck),
        bottleneck_score=bottleneck, witness_path=[list(p) for p in path],
        witness_path_scores=[float(table.scores[p]) for p in path])


def search_diagnostic(table, canonical, initial, final, radius):
    if not np.isfinite(radius) or radius < 0 or len(initial) != 2 or len(final) != 2:
        raise ValueError('Expected two labelled poses and a nonnegative finite radius')
    neighbors = coordinate_neighbors(canonical)
    fixed_sides = [np.flatnonzero(canonical[:, 1] == canonical[i, 1]).tolist() for i in initial]
    anchored = [np.flatnonzero((canonical[:, 1] == canonical[i, 1])
                & (np.abs(canonical[:, 0]-canonical[i, 0]) <= radius+1e-9)).tolist() for i in initial]
    if any(final[k] not in anchored[k] for k in (0, 1)) or not table.feasible[tuple(final)]:
        raise ValueError('Final deployment escaped the labelled anchor domain or is infeasible')
    side_best = best_assignment(table, *fixed_sides)
    anchor_best = best_assignment(table, *anchored)
    reachable = labelled_reachability(table, *anchored, neighbors, final, required_state=initial)
    one_agent = [best_assignment(table, anchored[0], [final[1]]),
                 best_assignment(table, [final[0]], anchored[1])]
    one_best = max(one_agent, key=lambda value: value['score'])
    coordinate = {tuple(final)}
    rejected = []
    for agent in (0, 1):
        for replacement in neighbors[final[agent]]:
            if replacement not in anchored[agent] or replacement in final:
                continue
            trial = list(final)
            trial[agent] = replacement
            if table.feasible[tuple(trial)]:
                coordinate.add(tuple(trial))
            else:
                rejected.append(trial)
    local_score = max(float(table.scores[state]) for state in coordinate)
    return dict(fixed_side_optimum=side_best, anchor_optimum=anchor_best,
        feasible_component_optimum=reachable, one_agent_relocation_optimum=one_best,
        one_coordinate_best_j=local_score, rejected_coordinate_moves=rejected,
        anchored_candidates_per_agent=list(map(len, anchored)),
        labelled_anchors=[float(canonical[i, 0]) for i in initial],
        fixed_sides=[float(canonical[i, 1]) for i in initial])

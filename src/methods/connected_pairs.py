"""Small fixed-two-sensor reference and opt-in road-coordinate local search."""
import time

import numpy as np


def assess_ids(ids, poses, mask_for, requirements):
    ids = list(ids)
    masks = np.asarray([mask_for(i) for i in ids], dtype=bool)
    return requirements.evaluate(poses[ids], masks.sum(axis=0), sensor_masks=masks)


def exact_pairs(ids, poses, mask_for, requirements, *, module_ids=None, pair_mask_filter=None):
    """Enumerate every distinct pair; optionally require different modules.

    Batched union scores prune only pairs that cannot beat the incumbent.
    All prospective improvements use the full shared constraint evaluator.
    """
    started = time.perf_counter()
    ids = np.asarray(sorted(set(int(i) for i in ids)), dtype=int)
    if len(ids) < 2 or ids.min() < 0 or ids.max() >= len(poses):
        raise ValueError('At least two valid unique candidate IDs required')
    if requirements.spec['budget_k'] != 2:
        raise ValueError('This exhaustive reference supports fixed K=2 only')
    masks = np.asarray([mask_for(i) for i in ids], dtype=bool)
    w = requirements.requirements.weights
    rewards = w.coverage_weight + w.importance_weight * requirements.importance + w.overlap_penalty
    penalties = (w.sensor_count_penalty
        + w.installation_cost_weight * requirements.installation_costs(poses[ids])
        + w.overlap_penalty * masks.sum(axis=1))
    best_score, selected, best = -np.inf, None, None
    considered, assessed, screened = 0, 0, 0
    for i in range(len(ids) - 1):
        js = np.arange(i + 1, len(ids))
        if module_ids is not None:
            js = js[np.asarray(module_ids)[ids[js]] != np.asarray(module_ids)[ids[i]]]
        considered += len(js)
        if pair_mask_filter is not None and len(js):
            allowed = np.asarray(pair_mask_filter(masks[i], masks[js]))
            if allowed.dtype != bool or allowed.shape != (len(js),):
                raise ValueError('Pair visibility filter must return one boolean per pair')
            screened += int(np.count_nonzero(~allowed))
            js = js[allowed]
        scores = (masks[js] | masks[i]) @ rewards - penalties[i] - penalties[js]
        for index in np.argsort(-scores, kind='stable'):
            if scores[index] < best_score - 1e-8:
                break
            j = js[index]
            pair = [int(ids[i]), int(ids[j])]
            if scores[index] <= best_score + 1e-8 and selected is not None and pair >= selected:
                continue
            assessment = requirements.evaluate(poses[pair],
                masks[i].astype(np.int64) + masks[j], sensor_masks=masks[[i, j]])
            assessed += 1
            if not np.isclose(assessment.objective_score, scores[index], atol=1e-8, rtol=1e-12):
                raise RuntimeError('Pair score differs from shared evaluator')
            if assessment.feasible and (assessment.objective_score > best_score + 1e-8
                    or (abs(assessment.objective_score - best_score) <= 1e-8
                        and (selected is None or pair < selected))):
                selected, best, best_score = pair, assessment, assessment.objective_score
    return dict(selected_ids=selected, assessment=None if best is None else best.to_dict(),
                exhaustive=True, fixed_sensor_count=2, candidate_count=len(ids),
                combinations_considered=considered, full_constraint_assessments=assessed,
                visibility_screened_pairs=screened,
                quota='one_per_module' if module_ids is not None else 'none',
                status='optimal_in_declared_grid' if selected is not None else 'infeasible_in_declared_grid',
                elapsed_sec=time.perf_counter() - started)


def canonical_local_search(initial_ids, canonical, poses, mask_for, requirements,
                           *, station_radius_m=8., max_sweeps=100):
    """One sensor/one neighboring coordinate; no side jump or change in K.

    The station limit is measured from the initial road-coordinate anchor.
    This is not the legacy Cartesian 11-action LS/MAPPO environment.
    """
    if not np.isfinite(station_radius_m) or station_radius_m < 0 or max_sweeps < 1:
        raise ValueError('Invalid local search budget')
    selected = list(initial_ids)
    current = assess_ids(selected, poses, mask_for, requirements)
    if not current.feasible:
        return dict(status='not_run_initial_infeasible', selected_ids=None, assessment=None)
    anchors = canonical[selected, 0].copy()
    lookup = {tuple(row): i for i, row in enumerate(canonical)}
    if len(lookup) != len(canonical):
        raise ValueError('Duplicate canonical candidates')
    levels = {axis: np.unique(canonical[:, axis]) for axis in (0, 2, 3, 4)}
    started = time.perf_counter()
    evaluated, history = 0, [dict(elapsed_sec=0., objective_score=current.objective_score)]
    for sweep in range(max_sweeps):
        best_ids, best = selected, current
        for agent, cid in enumerate(selected):
            for axis, values in levels.items():
                index = int(np.searchsorted(values, canonical[cid, axis]))
                for offset in (-1, 1):
                    if not 0 <= index + offset < len(values):
                        continue
                    proposal = canonical[cid].copy()
                    proposal[axis] = values[index + offset]
                    replacement = lookup.get(tuple(proposal))
                    if (replacement is None or replacement in selected
                            or abs(proposal[0] - anchors[agent]) > station_radius_m + 1e-9):
                        continue
                    trial = selected.copy()
                    trial[agent] = replacement
                    assessment = assess_ids(trial, poses, mask_for, requirements)
                    evaluated += 1
                    if assessment.feasible and assessment.objective_score > best.objective_score + 1e-8:
                        best_ids, best = trial, assessment
        if best_ids == selected:
            reason = 'no_improving_one_coordinate_move'
            break
        selected, current = best_ids, best
        history.append(dict(elapsed_sec=time.perf_counter() - started,
                            objective_score=current.objective_score))
    else:
        reason = 'max_sweeps'
    return dict(status='complete', selected_ids=selected, assessment=current.to_dict(),
                neighbor_evaluations=evaluated, sweeps=sweep + 1, stopping_reason=reason,
                station_radius_m=station_radius_m, elapsed_sec=time.perf_counter() - started,
                trace=history)

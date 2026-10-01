import numpy as np
from src.core.models import PlacementProblem, PlacementResult


def evaluate_selection(problem: PlacementProblem, selected_ids, metadata=None):
    ids = list(selected_ids)
    if len(set(ids)) != len(ids) or len(ids) > problem.budget_k:
        raise ValueError('Selection violates unique candidate or budget constraint')
    density = np.zeros(len(problem.importance), dtype=np.int64)
    for i in ids:
        if i < 0 or i >= len(problem.candidates.candidates):
            raise ValueError('Invalid candidate ID')
        density += problem.candidates.candidates[i].covered
    covered = density > 0
    components = dict(covered_count=int(covered.sum()),
                      weighted_target_coverage=float(problem.importance[covered].sum()),
                      overlap=float(np.maximum(density - 1, 0).sum()),
                      sensor_count=len(ids), installation_cost=float(problem.costs[ids].sum()))
    w = problem.objective
    score = (w.coverage_weight * components['covered_count']
             + w.importance_weight * components['weighted_target_coverage']
             - w.overlap_penalty * components['overlap']
             - w.sensor_count_penalty * len(ids)
             - w.installation_cost_weight * components['installation_cost'])
    poses = problem.candidates.positions_array(ids).reshape(-1, 5)
    return PlacementResult(ids, poses, float(score), 100 * covered.sum() / max(len(covered), 1),
                           components, metadata or {})

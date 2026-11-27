import numpy as np
from src.core.models import PlacementProblem, PlacementResult


def objective_from_density(density, importance, costs, weights):
    """Shared union objective; overlap counts excess observations, not targets."""
    covered = np.asarray(density) > 0
    components = dict(covered_count=int(covered.sum()),
                      weighted_target_coverage=float(np.asarray(importance)[covered].sum()),
                      overlap=float(np.maximum(np.asarray(density) - 1, 0).sum()),
                      sensor_count=len(costs), installation_cost=float(np.asarray(costs).sum()))
    score = (weights.coverage_weight * components['covered_count']
             + weights.importance_weight * components['weighted_target_coverage']
             - weights.overlap_penalty * components['overlap']
             - weights.sensor_count_penalty * components['sensor_count']
             - weights.installation_cost_weight * components['installation_cost'])
    return float(score), components


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
    score, components = objective_from_density(density, problem.importance, problem.costs[ids], problem.objective)
    poses = problem.candidates.positions_array(ids).reshape(-1, 5)
    return PlacementResult(ids, poses, float(score), 100 * covered.sum() / max(len(covered), 1),
                           components, metadata or {})

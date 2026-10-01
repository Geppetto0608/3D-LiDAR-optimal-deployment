from src.core.models import PlacementProblem
from src.core.objective import ObjectiveWeights
from src.roadside.candidate_generator import build_oriented_candidates


def build_problem(scenario, profile, config, budget_k):
    objective = ObjectiveWeights(**config.raw.get('objective', {}))
    if int(config.target.get('road_detection', 1)) != 1:
        raise NotImplementedError('Common objective currently supports single-observation coverage')
    # Retain all orientations for weighted runs: legacy top-M/dedup can discard
    # useful complements or lower-cost candidates under a different objective.
    kwargs = {} if objective.legacy else dict(max_orientations_per_position=0, dedup_exact_cover=False)
    candidates = build_oriented_candidates(scenario, profile, config, **kwargs)
    return PlacementProblem(candidates, budget_k, objective,
                            importance=config.raw.get('target_importance'),
                            costs=config.raw.get('installation_costs'))

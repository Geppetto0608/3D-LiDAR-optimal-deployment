"""Count-based objective; coverage percentages remain separate report metrics."""
from dataclasses import dataclass, fields
import numpy as np


@dataclass(frozen=True)
class ObjectiveWeights:
    coverage_weight: float = 1.0
    importance_weight: float = 0.0
    overlap_penalty: float = 0.0
    sensor_count_penalty: float = 0.0
    installation_cost_weight: float = 0.0
    robustness_weight: float = 0.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{field.name} must be finite and nonnegative')
        if self.robustness_weight:
            raise NotImplementedError('Robustness requires an explicit failure model')

    @property
    def legacy(self):
        return self == ObjectiveWeights()

    def coefficients(self, problem):
        # excess observations = sum(density) - number of covered targets.
        weights = self.coverage_weight + self.importance_weight * problem.importance
        rewards = weights + self.overlap_penalty
        costs = np.array([
            self.sensor_count_penalty + self.installation_cost_weight * problem.costs[i]
            + self.overlap_penalty * candidate.covered.sum()
            for i, candidate in enumerate(problem.candidates.candidates)
        ], dtype=float)
        return rewards, costs

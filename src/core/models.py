from dataclasses import dataclass, field
import numpy as np
from src.core.candidate import CandidateSet
from src.core.objective import ObjectiveWeights


@dataclass(frozen=True)
class SensorPose:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float = 0.0

    def to_legacy_pose5(self):
        if self.roll != 0:
            raise ValueError('Legacy pose format cannot represent nonzero roll')
        return [self.x, self.y, self.z, self.yaw, self.pitch]


@dataclass
class PlacementProblem:
    candidates: CandidateSet
    budget_k: int
    objective: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    importance: np.ndarray | None = None
    costs: np.ndarray | None = None

    def __post_init__(self):
        if isinstance(self.budget_k, bool) or not isinstance(self.budget_k, (int, np.integer)) or self.budget_k < 0:
            raise ValueError('budget_k must be a nonnegative integer')
        n = len(self.candidates.target_points)
        m = len(self.candidates.candidates)
        self.importance = np.ones(n) if self.importance is None else np.asarray(self.importance, dtype=float)
        self.costs = np.zeros(m) if self.costs is None else np.asarray(self.costs, dtype=float)
        for name, values, shape in [('importance', self.importance, (n,)), ('costs', self.costs, (m,))]:
            if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Invalid {name}: expected nonnegative finite {shape}')
        for i, c in enumerate(self.candidates.candidates):
            if c.candidate_id != i or c.covered.shape != (n,) or c.covered.dtype != bool:
                raise ValueError('Candidates must have contiguous IDs and boolean target masks')


@dataclass
class PlacementResult:
    selected_ids: list[int]
    poses: np.ndarray
    objective_score: float
    coverage_pct: float
    components: dict
    metadata: dict = field(default_factory=dict)

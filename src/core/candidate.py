from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

@dataclass
class Candidate:
    candidate_id: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    covered: np.ndarray

    @property
    def pose(self) -> list[float]:
        return [self.x, self.y, self.z, self.yaw, self.pitch]


@dataclass
class CandidateSet:
    candidates: list[Candidate]
    target_points: np.ndarray
    cache_key: str
    config_hash: str
    stats: dict[str, Any]

    def positions_array(self, candidate_ids: Sequence[int]) -> np.ndarray:
        lookup = {c.candidate_id: c for c in self.candidates}
        return np.asarray([lookup[int(cid)].pose for cid in candidate_ids], dtype=float)

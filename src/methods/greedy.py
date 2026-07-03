"""GREEDY method using the MILP candidate builder and shared environment coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.environment import ExperimentConfig, Scenario, get_lidar_profile
from src.methods.milp import CandidateSet, build_oriented_candidates, planned_candidate_signature
from src.outputs import build_result_record, save_positions_npy, save_result_json


@dataclass
class GreedySelection:
    selected_ids: list[int]
    marginal_gains: list[int]
    covered_mask: np.ndarray


def greedy_select(candidate_set: CandidateSet, budget_k: int) -> GreedySelection:
    covered = np.zeros(len(candidate_set.target_points), dtype=bool)
    selected: list[int] = []
    gains: list[int] = []
    available = set(range(len(candidate_set.candidates)))
    for _ in range(int(budget_k)):
        best_id = None
        best_gain = -1
        best_total = -1
        for cid in available:
            cand = candidate_set.candidates[cid]
            gain = int(np.logical_and(cand.covered, ~covered).sum())
            total = int(cand.covered.sum())
            if (gain, total, -cid) > (best_gain, best_total, -(best_id if best_id is not None else 10**12)):
                best_id, best_gain, best_total = cid, gain, total
        if best_id is None or best_gain <= 0:
            break
        selected.append(best_id)
        gains.append(best_gain)
        covered |= candidate_set.candidates[best_id].covered
        available.remove(best_id)
    return GreedySelection(selected, gains, covered)


def run_greedy_budget(
    scenario: Scenario,
    lidar_profile: str,
    config: ExperimentConfig,
    k_values: Sequence[int],
    *,
    output_tag: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if dry_run:
        signature = planned_candidate_signature(scenario, lidar_profile, config)
        return [{**signature, "method": "GREEDY", "budget_k": int(k), "note": "dry_run_no_candidate_build"} for k in k_values]
    candidate_set = build_oriented_candidates(scenario, lidar_profile, config)
    profile = get_lidar_profile(lidar_profile, config)
    records: list[dict[str, Any]] = []
    for k in sorted(int(v) for v in k_values):
        selection = greedy_select(candidate_set, k)
        positions = candidate_set.positions_array(selection.selected_ids)
        coverage_ratio = float(selection.covered_mask.sum() / max(len(candidate_set.target_points), 1))
        result_base = config.results_dir / "greedy" / f"greedy_{lidar_profile.lower()}_{output_tag}_id{scenario.scenario_id}_k{k}"
        npy_path = save_positions_npy(result_base.with_suffix(".npy"), positions)
        record = build_result_record(
            method="GREEDY",
            scenario_id=scenario.scenario_id,
            topology=scenario.topology,
            lidar_profile=lidar_profile,
            lidar_model=lidar_profile,
            max_range_m=profile.max_range_m,
            budget_k=k,
            budget_mode="at_most",
            candidate_mode=f"topm{config.max_orientations_per_position}",
            action_space="[x,y,z,yaw,pitch]",
            num_selected=len(selection.selected_ids),
            positions_npy=str(npy_path),
            coverage_pct=coverage_ratio * 100.0,
            coverage_ratio=coverage_ratio,
            covered_count=int(selection.covered_mask.sum()),
            num_targets=int(len(candidate_set.target_points)),
            episodes=None,
            seed=None,
            trial=None,
            mappo_tag=None,
            config_hash=candidate_set.config_hash,
            cache_key=candidate_set.cache_key,
            extra={"selected_candidate_ids": selection.selected_ids, "marginal_gains": selection.marginal_gains},
        )
        save_result_json(result_base.with_suffix(".json"), record)
        records.append(record)
    return records


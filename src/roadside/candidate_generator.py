"""Legacy-order roadside candidates; shared by all discrete methods."""
from __future__ import annotations
from typing import Any, Callable, Hashable, Iterable
import numpy as np
from src.core.candidate import Candidate, CandidateSet
from src.roadside.visibility_adapter import RoadsideVisibility
from src.environment import (
    ExperimentConfig,
    MapGeometry,
    Scenario,
    allowed_mount_yaws_deg,
    bake_visibility_map,
    compute_config_hash,
    coverage_mask_for_pose,
    get_lidar_profile,
    installation_base_positions,
    visibility_cache_path,
)

def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = float(start)
    while value <= float(stop) + 1e-9:
        yield round(value, 10)
        value += float(step)


def _base_positions(geometry: MapGeometry, config: ExperimentConfig) -> list[tuple[float, float, float]]:
    return installation_base_positions(geometry, config)


def planned_candidate_signature(scenario: Scenario, lidar_profile: str, config: ExperimentConfig) -> dict[str, Any]:
    profile = get_lidar_profile(lidar_profile, config)
    geometry = MapGeometry(scenario)
    base_positions = _base_positions(geometry, config)
    pitch_count = len(list(_frange(config.pitch_min_deg, config.pitch_max_deg, config.pitch_step_deg)))
    top_m = int(config.max_orientations_per_position)
    orientation_count = 0
    for x, y, _z in base_positions:
        yaw_count = len(allowed_mount_yaws_deg(x, y, geometry, config.yaw_step_deg, config.yaw_axis_min_deg, config.yaw_axis_max_deg))
        raw = yaw_count * pitch_count
        orientation_count += min(raw, top_m) if top_m > 0 else raw
    config_hash = compute_config_hash(config, scenario, profile)
    cache_key, cache_path = visibility_cache_path(config, scenario, profile)
    return {
        "scenario_id": int(scenario.scenario_id),
        "topology": scenario.topology,
        "lidar_profile": profile.name,
        "topM": top_m,
        "xy_domain": config.xy_domain,
        "num_base_positions": int(len(base_positions)),
        "num_candidates_planned": int(orientation_count),
        "candidate_builder": "src.methods.milp.build_oriented_candidates",
        "coverage_eval": "src.environment.evaluate_coverage",
        "visibility_cache_path": str(cache_path),
        "config_hash": config_hash,
        "cache_key": cache_key,
    }


def build_oriented_candidates(scenario: Scenario, lidar_profile: str, config: ExperimentConfig, *, max_orientations_per_position: int | None = None, dedup_exact_cover: bool = True,
                              position_filter: Callable[[float, float, float], bool] | None = None,
                              dedup_group: Callable[[float, float, float], Hashable] | None = None,
                              work=None) -> CandidateSet:
    profile = get_lidar_profile(lidar_profile, config)
    visibility = (bake_visibility_map(config, scenario, profile) if work is None else
                  bake_visibility_map(config, scenario, profile, work=work))
    target_points = np.asarray(visibility["target_points"], dtype=float)
    geometry = MapGeometry(scenario)
    provider = RoadsideVisibility(visibility, geometry, profile, config)
    top_m = config.max_orientations_per_position if max_orientations_per_position is None else int(max_orientations_per_position)
    candidates: list[Candidate] = []
    seen_covers: set[Any] = set()
    before_pruning = after_pruning = skipped_dedup = 0
    pitches = list(_frange(config.pitch_min_deg, config.pitch_max_deg, config.pitch_step_deg))
    base_positions = _base_positions(geometry, config)
    if position_filter is not None:
        base_positions = [p for p in base_positions if position_filter(*p)]
    for x, y, z in base_positions:
        yaws = allowed_mount_yaws_deg(x, y, geometry, config.yaw_step_deg, config.yaw_axis_min_deg, config.yaw_axis_max_deg)
        local: list[tuple[int, float, float, np.ndarray]] = []
        for yaw in yaws:
            for pitch in pitches:
                before_pruning += 1
                if work is not None:
                    work.add('candidate_pose_evaluations')
                covered = provider.mask([x, y, z, yaw, pitch])
                local.append((int(covered.sum()), float(yaw), float(pitch), covered))
        local.sort(key=lambda item: (-item[0], item[1], item[2]))
        if top_m > 0:
            local = local[:top_m]
        after_pruning += len(local)
        for _, yaw, pitch, covered in local:
            if dedup_exact_cover:
                key = np.packbits(covered).tobytes()
                if dedup_group is not None:
                    key = (dedup_group(x, y, z), key)
                if key in seen_covers:
                    skipped_dedup += 1
                    continue
                seen_covers.add(key)
            candidates.append(Candidate(len(candidates), x, y, z, yaw, pitch, covered))
    coverage_edges = int(sum(int(c.covered.sum()) for c in candidates))
    return CandidateSet(
        candidates=candidates,
        target_points=target_points,
        cache_key=str(visibility["cache_key"]),
        config_hash=str(visibility.get("meta", {}).get("config_hash", "")),
        stats={
            "xy_domain": config.xy_domain,
            "num_targets": int(len(target_points)),
            "num_base_positions": int(len(base_positions)),
            "num_candidates": int(len(candidates)),
            "orientation_candidates_before_pruning": int(before_pruning),
            "orientation_candidates_after_pruning": int(after_pruning),
            "skipped_dedup": int(skipped_dedup),
            "num_coverage_edges": coverage_edges,
        },
    )

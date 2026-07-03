"""MILP method: candidate builder, at-most-K solver, and budget frontier runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from src.environment import (
    ExperimentConfig,
    MapGeometry,
    Scenario,
    allowed_mount_yaws_deg,
    bake_visibility_map,
    compute_config_hash,
    coverage_mask_for_pose,
    get_lidar_profile,
    visibility_cache_path,
)
from src.outputs import build_result_record, save_positions_npy, save_result_json


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


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = float(start)
    while value <= float(stop) + 1e-9:
        yield round(value, 10)
        value += float(step)


def _base_positions(geometry: MapGeometry, config: ExperimentConfig) -> list[tuple[float, float, float]]:
    zs = list(_frange(config.z_min, config.z_max, config.z_step))
    positions: list[tuple[float, float, float]] = []
    cols = int(geometry.map_len / geometry.grid_size)
    rows = int(geometry.map_width / geometry.grid_size)
    seen_bins: set[tuple[int, int]] = set()
    step_m = max(float(config.xy_step_m), geometry.grid_size)
    for gx in range(cols):
        for gy in range(rows):
            x = (gx + 0.5) * geometry.grid_size
            y = (gy + 0.5) * geometry.grid_size
            if not geometry.is_installable(float(x), float(y)):
                continue
            key = (int(np.floor(x / step_m)), int(np.floor(y / step_m)))
            if key in seen_bins:
                continue
            seen_bins.add(key)
            for z in zs:
                positions.append((float(x), float(y), float(z)))
    return positions


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
        "num_base_positions": int(len(base_positions)),
        "num_candidates_planned": int(orientation_count),
        "candidate_builder": "src.methods.milp.build_oriented_candidates",
        "coverage_eval": "src.environment.evaluate_coverage",
        "visibility_cache_path": str(cache_path),
        "config_hash": config_hash,
        "cache_key": cache_key,
    }


def build_oriented_candidates(scenario: Scenario, lidar_profile: str, config: ExperimentConfig, *, max_orientations_per_position: int | None = None, dedup_exact_cover: bool = True) -> CandidateSet:
    profile = get_lidar_profile(lidar_profile, config)
    visibility = bake_visibility_map(config, scenario, profile)
    target_points = np.asarray(visibility["target_points"], dtype=float)
    geometry = MapGeometry(scenario)
    top_m = config.max_orientations_per_position if max_orientations_per_position is None else int(max_orientations_per_position)
    candidates: list[Candidate] = []
    seen_covers: set[bytes] = set()
    before_pruning = after_pruning = skipped_dedup = 0
    pitches = list(_frange(config.pitch_min_deg, config.pitch_max_deg, config.pitch_step_deg))
    base_positions = _base_positions(geometry, config)
    for x, y, z in base_positions:
        yaws = allowed_mount_yaws_deg(x, y, geometry, config.yaw_step_deg, config.yaw_axis_min_deg, config.yaw_axis_max_deg)
        local: list[tuple[int, float, float, np.ndarray]] = []
        for yaw in yaws:
            for pitch in pitches:
                before_pruning += 1
                covered = coverage_mask_for_pose([x, y, z, yaw, pitch], visibility, geometry, profile, config)
                local.append((int(covered.sum()), float(yaw), float(pitch), covered))
        local.sort(key=lambda item: (-item[0], item[1], item[2]))
        if top_m > 0:
            local = local[:top_m]
        after_pruning += len(local)
        for _, yaw, pitch, covered in local:
            if dedup_exact_cover:
                key = np.packbits(covered).tobytes()
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
            "num_targets": int(len(target_points)),
            "num_base_positions": int(len(base_positions)),
            "num_candidates": int(len(candidates)),
            "orientation_candidates_before_pruning": int(before_pruning),
            "orientation_candidates_after_pruning": int(after_pruning),
            "skipped_dedup": int(skipped_dedup),
            "num_coverage_edges": coverage_edges,
        },
    )


def milp_dry_run_signature(scenario: Scenario, lidar_profile: str, config: ExperimentConfig, k_values: Sequence[int]) -> list[dict[str, Any]]:
    signature = planned_candidate_signature(scenario, lidar_profile, config)
    return [{**signature, "method": "MILP_BUDGET", "budget_k": int(k), "note": "dry_run_no_solve_no_candidate_build"} for k in k_values]


def _coverage_for_selected(candidate_set: CandidateSet, selected: Sequence[int]) -> tuple[int, float]:
    covered = np.zeros(len(candidate_set.target_points), dtype=bool)
    for cid in selected:
        covered |= candidate_set.candidates[int(cid)].covered
    count = int(covered.sum())
    return count, float(count / max(len(candidate_set.target_points), 1))


def _solve_with_pulp(candidate_set: CandidateSet, budget_k: int, config: ExperimentConfig, start_ids: Sequence[int] | None = None) -> dict[str, Any]:
    import time
    import pulp

    t0 = time.time()
    prob = pulp.LpProblem("lidar_budget_coverage", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(candidate_set.candidates))]
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(len(candidate_set.target_points))]
    target_to_candidates: list[list[int]] = [[] for _ in range(len(candidate_set.target_points))]
    for cand in candidate_set.candidates:
        for target_idx in np.flatnonzero(cand.covered):
            target_to_candidates[int(target_idx)].append(cand.candidate_id)
    for target_idx, cids in enumerate(target_to_candidates):
        prob += y[target_idx] <= pulp.lpSum(x[cid] for cid in cids) if cids else y[target_idx] == 0
    prob += pulp.lpSum(x) <= int(budget_k)
    prob += pulp.lpSum(y)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(config.time_limit_sec), gapRel=float(config.mip_gap))
    prob.solve(solver)
    selected = [i for i, var in enumerate(x) if float(var.value() or 0.0) > 0.5]
    return {"selected_ids": selected, "solver_status": pulp.LpStatus.get(prob.status, str(prob.status)), "solver_status_code": prob.status, "objective": float(pulp.value(prob.objective) or 0.0), "best_bound": None, "mip_gap": None, "solve_time_sec": time.time() - t0}


def _solve_with_gurobi(candidate_set: CandidateSet, budget_k: int, config: ExperimentConfig, start_ids: Sequence[int] | None = None) -> dict[str, Any]:
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("lidar_budget_coverage")
    model.Params.OutputFlag = 0
    model.Params.Threads = int(config.threads)
    model.Params.TimeLimit = float(config.time_limit_sec)
    model.Params.MIPGap = float(config.mip_gap)

    def set_optional_param(name: str, value: Any) -> None:
        try:
            setattr(model.Params, name, value)
        except gp.GurobiError as exc:
            print(f"[WARN] Gurobi parameter {name}={value} was ignored: {exc}")

    nodefile_dir = config.solver.get("nodefile_dir")
    if nodefile_dir:
        from pathlib import Path

        Path(str(nodefile_dir)).mkdir(parents=True, exist_ok=True)
        set_optional_param("NodefileDir", str(nodefile_dir))
    if config.solver.get("nodefile_start_gb") is not None:
        set_optional_param("NodefileStart", float(config.solver.get("nodefile_start_gb")))
    if config.solver.get("soft_mem_limit_gb") is not None:
        set_optional_param("SoftMemLimit", float(config.solver.get("soft_mem_limit_gb")))
    x = model.addVars(len(candidate_set.candidates), vtype=GRB.BINARY, name="x")
    y = model.addVars(len(candidate_set.target_points), vtype=GRB.BINARY, name="y")
    start_set = {int(v) for v in (start_ids or [])}
    for i in range(len(candidate_set.candidates)):
        if start_set:
            x[i].Start = 1.0 if i in start_set else 0.0
    target_to_candidates: list[list[int]] = [[] for _ in range(len(candidate_set.target_points))]
    for cand in candidate_set.candidates:
        for target_idx in np.flatnonzero(cand.covered):
            target_to_candidates[int(target_idx)].append(cand.candidate_id)
    for target_idx, cids in enumerate(target_to_candidates):
        model.addConstr(y[target_idx] <= gp.quicksum(x[cid] for cid in cids) if cids else y[target_idx] == 0)
    model.addConstr(gp.quicksum(x[i] for i in range(len(candidate_set.candidates))) <= int(budget_k))
    model.setObjective(gp.quicksum(y[j] for j in range(len(candidate_set.target_points))), GRB.MAXIMIZE)
    model.optimize()
    selected = [i for i in range(len(candidate_set.candidates)) if model.SolCount > 0 and float(x[i].X) > 0.5]
    if model.SolCount == 0 and start_ids:
        selected = [int(v) for v in start_ids]
    return {"selected_ids": selected, "solver_status": str(model.Status), "solver_status_code": int(model.Status), "objective": float(model.ObjVal) if model.SolCount > 0 else None, "best_bound": float(model.ObjBound) if model.SolCount > 0 else None, "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None, "solve_time_sec": float(model.Runtime)}


def solve_budget_frontier(scenario: Scenario, lidar_profile: str, config: ExperimentConfig, k_values: Sequence[int], *, output_tag: str) -> list[dict[str, Any]]:
    from src.methods.greedy import greedy_select

    candidate_set = build_oriented_candidates(scenario, lidar_profile, config)
    profile = get_lidar_profile(lidar_profile, config)
    records: list[dict[str, Any]] = []
    frontier_best_count = -1
    frontier_best_ratio = 0.0
    frontier_best_selected: list[int] = []
    frontier_source_k: int | None = None
    frontier_source_method = "none"
    for k in sorted(int(v) for v in k_values):
        greedy_start = greedy_select(candidate_set, k).selected_ids
        greedy_count, greedy_ratio = _coverage_for_selected(candidate_set, greedy_start)
        if greedy_count > frontier_best_count:
            frontier_best_count, frontier_best_ratio, frontier_best_selected = greedy_count, greedy_ratio, list(greedy_start)
            frontier_source_k, frontier_source_method = k, "greedy_warm_start"
        start_ids = frontier_best_selected if frontier_best_selected and len(frontier_best_selected) <= int(k) else None
        if config.solver_name.lower() == "gurobi":
            try:
                solve_result = _solve_with_gurobi(candidate_set, k, config, start_ids=start_ids)
                solve_result["solver_backend"] = "gurobi"
            except Exception as exc:
                if "gurobi" not in exc.__class__.__module__.lower() and "gurobi" not in exc.__class__.__name__.lower():
                    raise
                print(f"[WARN] Gurobi failed ({exc}); falling back to PuLP/CBC for this MILP run.")
                solve_result = _solve_with_pulp(candidate_set, k, config, start_ids=start_ids)
                solve_result["solver_backend"] = "pulp_fallback_from_gurobi"
                solve_result["gurobi_error"] = str(exc)
        else:
            solve_result = _solve_with_pulp(candidate_set, k, config, start_ids=start_ids)
            solve_result["solver_backend"] = "pulp"
        selected = [int(v) for v in solve_result.get("selected_ids", [])]
        raw_count, raw_ratio = _coverage_for_selected(candidate_set, selected)
        if raw_count >= frontier_best_count:
            frontier_best_count, frontier_best_ratio, frontier_best_selected = raw_count, raw_ratio, list(selected)
            frontier_source_k, frontier_source_method = k, "milp_raw_solution"
        result_base = config.results_dir / "milp" / f"milp_{lidar_profile.lower()}_{output_tag}_id{scenario.scenario_id}_k{k}"
        positions = candidate_set.positions_array(frontier_best_selected)
        npy_path = save_positions_npy(result_base.with_suffix(".npy"), positions)
        record = build_result_record(
            method="MILP_BUDGET",
            scenario_id=scenario.scenario_id,
            topology=scenario.topology,
            lidar_profile=lidar_profile,
            lidar_model=lidar_profile,
            max_range_m=profile.max_range_m,
            budget_k=k,
            budget_mode="at_most",
            candidate_mode=f"topm{config.max_orientations_per_position}",
            action_space="discrete_topm_candidate_set",
            num_selected=len(frontier_best_selected),
            positions_npy=str(npy_path),
            coverage_pct=frontier_best_ratio * 100.0,
            coverage_ratio=frontier_best_ratio,
            covered_count=int(frontier_best_count),
            num_targets=int(len(candidate_set.target_points)),
            episodes=None,
            seed=None,
            trial=None,
            mappo_tag=None,
            config_hash=candidate_set.config_hash,
            cache_key=candidate_set.cache_key,
            raw_coverage_pct=raw_ratio * 100.0,
            frontier_coverage_pct=frontier_best_ratio * 100.0,
            frontier_source_budget_k=frontier_source_k,
            frontier_source_method=frontier_source_method,
            solver_status=solve_result.get("solver_status"),
            solver_backend=solve_result.get("solver_backend"),
            solver_error=solve_result.get("gurobi_error"),
            mip_gap=solve_result.get("mip_gap"),
            extra={"selected_candidate_ids": selected, "frontier_selected_candidate_ids": frontier_best_selected},
        )
        save_result_json(result_base.with_suffix(".json"), record)
        records.append(record)
    return records

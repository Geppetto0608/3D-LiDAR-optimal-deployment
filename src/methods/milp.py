"""MILP method: candidate builder, at-most-K solver, and budget frontier runner."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.environment import (
    ExperimentConfig,
    Scenario,
    get_lidar_profile,
)
from src.outputs import build_result_record, save_positions_npy, save_result_json


from src.core.candidate import Candidate, CandidateSet
from src.roadside.candidate_generator import (
    _frange, _base_positions, planned_candidate_signature, build_oriented_candidates,
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


def _solve_with_pulp(candidate_set: CandidateSet, budget_k: int, config: ExperimentConfig, start_ids: Sequence[int] | None = None, *, coefficients=None) -> dict[str, Any]:
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
    if coefficients is None:
        prob += pulp.lpSum(y)
    else:
        rewards, costs = coefficients
        prob += pulp.lpSum(float(rewards[j]) * y[j] for j in range(len(y))) - pulp.lpSum(float(costs[i]) * x[i] for i in range(len(x)))
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(config.time_limit_sec), gapRel=float(config.mip_gap))
    prob.solve(solver)
    if coefficients is not None and prob.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        raise RuntimeError('CBC produced no feasible integer solution')
    selected = [i for i, var in enumerate(x) if float(var.value() or 0.0) > 0.5]
    return {"selected_ids": selected, "solver_status": pulp.LpStatus.get(prob.status, str(prob.status)), "solver_status_code": prob.status, "objective": float(pulp.value(prob.objective) or 0.0), "best_bound": None, "mip_gap": None, "solve_time_sec": time.time() - t0}


def _solve_with_gurobi(candidate_set: CandidateSet, budget_k: int, config: ExperimentConfig, start_ids: Sequence[int] | None = None, *, coefficients=None) -> dict[str, Any]:
    import gurobipy as gp
    import scipy.sparse as sp
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

    num_candidates = len(candidate_set.candidates)
    num_targets = len(candidate_set.target_points)
    z = model.addMVar(num_candidates + num_targets, vtype=GRB.BINARY, name="z")
    x = z[:num_candidates]
    y = z[num_candidates:]
    start_set = {int(v) for v in (start_ids or [])}
    for i in range(num_candidates):
        if start_set:
            z[i].Start = 1.0 if i in start_set else 0.0

    # Build target coverage constraints as one sparse matrix operation:
    # y_j - sum_i a_ij x_i <= 0, plus sum_i x_i <= K.
    target_rows: list[np.ndarray] = []
    candidate_cols: list[np.ndarray] = []
    for cand in candidate_set.candidates:
        covered_idx = np.flatnonzero(cand.covered).astype(np.int32, copy=False)
        if covered_idx.size == 0:
            continue
        target_rows.append(covered_idx)
        candidate_cols.append(np.full(covered_idx.shape, cand.candidate_id, dtype=np.int32))
    if target_rows:
        rows = np.concatenate(target_rows)
        cols = np.concatenate(candidate_cols)
        data = np.ones(rows.shape[0], dtype=np.float64)
        coverage = sp.coo_matrix((data, (rows, cols)), shape=(num_targets, num_candidates)).tocsr()
    else:
        coverage = sp.csr_matrix((num_targets, num_candidates), dtype=np.float64)
    cover_constraints = sp.hstack([-coverage, sp.eye(num_targets, format="csr")], format="csr")
    budget_constraints = sp.hstack(
        [sp.csr_matrix(np.ones((1, num_candidates), dtype=np.float64)), sp.csr_matrix((1, num_targets), dtype=np.float64)],
        format="csr",
    )
    constraints = sp.vstack([cover_constraints, budget_constraints], format="csr")
    rhs = np.concatenate([np.zeros(num_targets, dtype=np.float64), np.asarray([float(budget_k)], dtype=np.float64)])
    model.addMConstr(constraints, z, "<", rhs, name="coverage_budget")
    if coefficients is None:
        model.setObjective(y.sum(), GRB.MAXIMIZE)
    else:
        rewards, costs = coefficients
        model.setObjective(rewards @ y - costs @ x, GRB.MAXIMIZE)
    model.optimize()
    solution = z.X[:num_candidates] if model.SolCount > 0 else []
    selected = [i for i, value in enumerate(solution) if float(value) > 0.5]
    if model.SolCount == 0 and start_ids:
        selected = [int(v) for v in start_ids]
    return {"selected_ids": selected, "solver_status": str(model.Status), "solver_status_code": int(model.Status), "objective": float(model.ObjVal) if model.SolCount > 0 else None, "best_bound": float(model.ObjBound) if model.SolCount > 0 else None, "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None, "solve_time_sec": float(model.Runtime)}


def solve_problem(problem, config: ExperimentConfig):
    from src.core.evaluation import evaluate_selection
    from src.methods.greedy import solve_problem as greedy_problem

    warm = greedy_problem(problem)
    if not problem.candidates.candidates or problem.budget_k == 0:
        return warm
    backend = config.solver_name.lower()
    if backend == 'gurobi':
        solver = _solve_with_gurobi
    elif backend in ('pulp', 'cbc'):
        solver = _solve_with_pulp
    else:
        raise ValueError(f'Unsupported solver: {backend}')
    result = solver(problem.candidates, problem.budget_k, config, warm.selected_ids,
                    coefficients=problem.objective.coefficients(problem))
    solved = evaluate_selection(problem, result['selected_ids'], result)
    if solved.objective_score < warm.objective_score:
        warm.metadata = {**result, 'selection_source': 'greedy_warm_start'}
        return warm
    solved.metadata['selection_source'] = 'solver'
    return solved


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
        if k == 1:
            solve_result = {
                "selected_ids": greedy_start,
                "solver_status": "SKIPPED_EXACT_K1",
                "solver_status_code": None,
                "objective": float(greedy_count),
                "best_bound": float(greedy_count),
                "mip_gap": 0.0,
                "solve_time_sec": 0.0,
                "solver_backend": "analytic_skip",
            }
        elif frontier_best_count >= len(candidate_set.target_points):
            solve_result = {
                "selected_ids": frontier_best_selected,
                "solver_status": "SKIPPED_PROVEN_FULL_COVERAGE",
                "solver_status_code": None,
                "objective": float(frontier_best_count),
                "best_bound": float(frontier_best_count),
                "mip_gap": 0.0,
                "solve_time_sec": 0.0,
                "solver_backend": "analytic_skip",
            }
        elif config.solver_name.lower() == "gurobi":
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
            extra={"selected_candidate_ids": selected, "frontier_selected_candidate_ids": frontier_best_selected,
                   "solver_objective": solve_result.get("objective"),
                   "solver_best_bound": solve_result.get("best_bound"),
                   "solve_time_sec": solve_result.get("solve_time_sec"),
                   "candidate_count": len(candidate_set.candidates)},
        )
        save_result_json(result_base.with_suffix(".json"), record)
        records.append(record)
    return records

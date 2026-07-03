#!/usr/bin/env python3
"""Experiment runner for MILP, GREEDY, and MAPPO.

Defaults are intentionally non-destructive: commands print plans/dry-runs unless
--execute is supplied.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from src.environment import ExperimentConfig, load_config, load_scenarios, select_scenarios
from src.methods.greedy import run_greedy_budget
from src.methods.mappo import MappoPlanRow, plan_mappo_runs, run_mappo
from src.methods.milp import milp_dry_run_signature, solve_budget_frontier


def _parse_ints(value: str | None, fallback: Sequence[int]) -> list[int]:
    if value is None or value.strip() == "":
        return [int(v) for v in fallback]
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _load_common(args) -> tuple[ExperimentConfig, list, list[int], list[int], str, str]:
    config = load_config(args.config)
    if args.output_root:
        config.output["results_dir"] = args.output_root
    scenario_ids = _parse_ints(args.scenario_ids, config.scenario_ids)
    k_values = _parse_ints(args.k_values, config.k_values)
    profile = args.profile or config.lidar_profile
    tag = args.tag or config.experiment_tag
    scenarios = select_scenarios(load_scenarios(), scenario_ids)
    if not scenarios:
        raise RuntimeError(f"No scenarios selected: {scenario_ids}")
    return config, scenarios, scenario_ids, k_values, profile, tag


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _print_plan(args) -> int:
    config, scenarios, scenario_ids, k_values, profile, tag = _load_common(args)
    milp_rows = []
    greedy_rows = []
    for scenario in scenarios:
        milp_rows.extend(milp_dry_run_signature(scenario, profile, config, k_values))
        for row in milp_dry_run_signature(scenario, profile, config, k_values):
            greedy_rows.append({**row, "method": "GREEDY", "note": "shares_milp_candidate_builder"})
    mappo_rows = plan_mappo_runs(
        scenarios,
        profile,
        k_values,
        config,
        tag=tag,
        episodes=args.episodes,
        trials=args.trials,
        seed_base=args.seed_base,
    )
    _print_json(
        {
            "mode": "PLAN",
            "config": args.config,
            "results_dir": str(config.results_dir),
            "scenario_ids": scenario_ids,
            "k_values": k_values,
            "profile": profile,
            "tag": tag,
            "execute": False,
            "milp_jobs": len(milp_rows),
            "greedy_jobs": len(greedy_rows),
            "mappo_jobs": len(mappo_rows),
            "mappo_episodes": int(args.episodes if args.episodes is not None else config.episodes),
            "mappo_trials": int(args.trials if args.trials is not None else config.trials),
            "sample_milp": milp_rows[:2],
            "sample_mappo": [asdict(row) for row in mappo_rows[:2]],
        }
    )
    return 0


def _run_milp(args) -> int:
    config, scenarios, _scenario_ids, k_values, profile, tag = _load_common(args)
    if not args.execute:
        for scenario in scenarios:
            for row in milp_dry_run_signature(scenario, profile, config, k_values):
                _print_json({**row, "runner": "test.py milp", "execute": False})
        return 0
    for scenario in scenarios:
        records = solve_budget_frontier(scenario, profile, config, k_values, output_tag=tag)
        for record in records:
            _print_json({"runner": "test.py milp", "execute": True, "result": record})
    return 0


def _run_greedy(args) -> int:
    config, scenarios, _scenario_ids, k_values, profile, tag = _load_common(args)
    for scenario in scenarios:
        rows = run_greedy_budget(
            scenario,
            profile,
            config,
            k_values,
            output_tag=tag,
            dry_run=not args.execute,
        )
        for row in rows:
            _print_json({**row, "runner": "test.py greedy", "execute": bool(args.execute)})
    return 0


def _mappo_rows(args) -> tuple[ExperimentConfig, list[MappoPlanRow]]:
    config, scenarios, _scenario_ids, k_values, profile, tag = _load_common(args)
    rows = plan_mappo_runs(
        scenarios,
        profile,
        k_values,
        config,
        tag=tag,
        episodes=args.episodes,
        trials=args.trials,
        seed_base=args.seed_base,
    )
    return config, rows


def _run_mappo_cmd(args) -> int:
    config, rows = _mappo_rows(args)
    if not args.execute:
        for row in rows:
            _print_json({**asdict(row), "method": "MAPPO", "runner": "test.py mappo", "execute": False})
        return 0
    result = run_mappo(rows, config)
    _print_json({"runner": "test.py mappo", "execute": True, "records": len(result.get("records", []))})
    return 0


def _run_all(args) -> int:
    _print_plan(args)
    if not args.execute:
        return 0
    if args.include_greedy:
        _run_greedy(args)
    if args.include_milp:
        _run_milp(args)
    if args.include_mappo:
        _run_mappo_cmd(args)
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--scenario-ids", default=None, help="Comma-separated scenario ids. Default: config scenario_ids.")
    p.add_argument("--k-values", default=None, help="Comma-separated K values. Default: config k_values.")
    p.add_argument("--profile", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--output-root", default="results/active")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=None)
    p.add_argument("--execute", action="store_true", help="Actually run the selected method. Without this, only dry-run/plan.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment runner. Safe by default; use --execute to run.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="Show planned MILP/GREEDY/MAPPO jobs.")
    _add_common(p)
    p.set_defaults(func=_print_plan)

    p = sub.add_parser("milp", help="Run or dry-run MILP budget frontier.")
    _add_common(p)
    p.set_defaults(func=_run_milp)

    p = sub.add_parser("greedy", help="Run or dry-run GREEDY baseline.")
    _add_common(p)
    p.set_defaults(func=_run_greedy)

    p = sub.add_parser("mappo", help="Run or dry-run MAPPO.")
    _add_common(p)
    p.set_defaults(func=_run_mappo_cmd)

    p = sub.add_parser("all", help="Plan all methods; optionally execute selected methods.")
    _add_common(p)
    p.add_argument("--include-greedy", action="store_true")
    p.add_argument("--include-milp", action="store_true")
    p.add_argument("--include-mappo", action="store_true")
    p.set_defaults(func=_run_all)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

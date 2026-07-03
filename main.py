#!/usr/bin/env python3
"""Unified CLI for MILP, GREEDY, MAPPO, compare, visualize, and full scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.environment import load_config, load_scenarios, select_scenarios
from src.methods.greedy import run_greedy_budget
from src.methods.mappo import plan_mappo_runs, run_mappo
from src.methods.milp import milp_dry_run_signature, solve_budget_frontier
from src.outputs import build_comparison
from src.visualize import export_result_figures, find_visualization_inputs, visualize_dry_run


def _parse_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_methods(value: str | None) -> list[str]:
    if not value:
        return []
    aliases = {"milp": "MILP_BUDGET", "greedy": "GREEDY", "mappo": "MAPPO", "ppo": "MAPPO"}
    return [aliases.get(v.strip().lower(), v.strip().upper()) for v in value.split(",") if v.strip()]


def _common_config(args):
    config = load_config(args.config)
    if getattr(args, "output_root", None):
        config.output["results_dir"] = args.output_root
    return config


def cmd_milp(args) -> int:
    config = _common_config(args)
    scenario_ids = _parse_ints(args.scenario_ids) or config.scenario_ids
    k_values = _parse_ints(args.k_values) or config.k_values
    profile = args.profile or config.lidar_profile
    tag = args.tag or config.experiment_tag
    scenarios = select_scenarios(load_scenarios(), scenario_ids)
    if args.dry_run:
        for scenario in scenarios:
            for row in milp_dry_run_signature(scenario, profile, config, k_values):
                row.update({"output_root": str(config.results_dir), "time_limit_sec": config.time_limit_sec, "mip_gap": config.mip_gap, "threads": config.threads})
                print(row)
        return 0
    for scenario in scenarios:
        solve_budget_frontier(scenario, profile, config, k_values, output_tag=tag)
    return 0


def cmd_greedy(args) -> int:
    config = _common_config(args)
    scenario_ids = _parse_ints(args.scenario_ids) or config.scenario_ids
    k_values = _parse_ints(args.k_values) or config.k_values
    profile = args.profile or config.lidar_profile
    tag = args.tag or config.experiment_tag
    for scenario in select_scenarios(load_scenarios(), scenario_ids):
        rows = run_greedy_budget(scenario, profile, config, k_values, output_tag=tag, dry_run=args.dry_run)
        for row in rows:
            row["output_root"] = str(config.results_dir)
            print(row)
    return 0


def cmd_mappo(args) -> int:
    config = _common_config(args)
    scenario_ids = _parse_ints(args.scenario_ids) or config.scenario_ids
    k_values = _parse_ints(args.k_values) or config.k_values
    profile = args.profile or config.lidar_profile
    rows = plan_mappo_runs(select_scenarios(load_scenarios(), scenario_ids), profile, k_values, config, tag=args.tag, episodes=args.episodes, trials=args.trials)
    if args.dry_run:
        for row in rows:
            print({**row.__dict__, "method": "MAPPO", "trainer": "mappo", "output_root": str(config.results_dir), "note": "dry_run_no_training_no_file_written"})
        return 0
    run_mappo(rows, config)
    return 0


def cmd_compare(args) -> int:
    output = build_comparison(
        args.config,
        scenario_ids=_parse_ints(args.scenario_ids),
        k_values=_parse_ints(args.k_values),
        mappo_tag=args.mappo_tag,
        output_csv=args.output_csv,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    if output is not None:
        print(output)
    return 0


def cmd_visualize(args) -> int:
    scenario_ids = _parse_ints(args.scenario_ids)
    k_values = _parse_ints(args.k_values)
    methods = _parse_methods(args.methods) or ["MILP_BUDGET", "GREEDY", "MAPPO"]
    if args.dry_run:
        visualize_dry_run(
            args.config,
            result_json=args.result_json,
            output_dir=args.output_dir,
            output_3d_dir=args.output_3d_dir,
            results_root=args.results_root,
            scenario_ids=scenario_ids,
            k_values=k_values,
            methods=methods,
            mappo_tag=args.mappo_tag,
            save_beams=args.save_beams,
        )
        return 0
    result_jsons = [Path(args.result_json)] if args.result_json else find_visualization_inputs(args.results_root, methods, scenario_ids, k_values, args.mappo_tag)
    if not result_jsons:
        print("No matching result JSON files found.")
        return 0
    for result_json in result_jsons:
        for path in export_result_figures(result_json, args.config, args.output_dir, save_3d=args.save_3d, save_beams=args.save_beams, train_log=args.train_log, output_3d_dir=args.output_3d_dir):
            print(path)
    return 0


def cmd_full(args) -> int:
    print("Full pipeline scaffold only. Run subcommands explicitly: milp, greedy, mappo, compare, visualize.")
    print(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LiDAR placement experiment CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, include_tag: bool = True):
        p.add_argument("--config", default="configs/experiment.yaml")
        p.add_argument("--scenario-ids", default=None)
        p.add_argument("--k-values", default=None)
        p.add_argument("--profile", default=None)
        if include_tag:
            p.add_argument("--tag", default=None)
        p.add_argument("--output-root", default="results/active")
        p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("milp")
    common(p)
    p.set_defaults(func=cmd_milp)

    p = sub.add_parser("greedy")
    common(p)
    p.set_defaults(func=cmd_greedy)

    p = sub.add_parser("mappo")
    common(p, include_tag=False)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--tag", required=True)
    p.set_defaults(func=cmd_mappo)

    p = sub.add_parser("compare")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--scenario-ids", default=None)
    p.add_argument("--k-values", default=None)
    p.add_argument("--mappo-tag", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--output-root", default="results/active")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("visualize")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--result-json", default=None)
    p.add_argument("--scenario-ids", default=None)
    p.add_argument("--k-values", default=None)
    p.add_argument("--methods", default=None)
    p.add_argument("--mappo-tag", default=None)
    p.add_argument("--results-root", default="results/active")
    p.add_argument("--output-dir", default="results/active/figures")
    p.add_argument("--output-3d-dir", default="results/active/3d")
    p.add_argument("--save-3d", action="store_true")
    p.add_argument("--save-beams", action="store_true")
    p.add_argument("--train-log", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_visualize)

    p = sub.add_parser("full")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--run-milp", action="store_true")
    p.add_argument("--run-greedy", action="store_true")
    p.add_argument("--run-mappo", action="store_true")
    p.set_defaults(func=cmd_full)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

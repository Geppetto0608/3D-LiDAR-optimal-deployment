"""Result writing, loading, comparison, and lightweight plotting/export utilities."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from src.environment import ExperimentConfig, load_config


REQUIRED_FIELDS = [
    "method",
    "scenario_id",
    "topology",
    "lidar_profile",
    "lidar_model",
    "max_range_m",
    "budget_k",
    "budget_mode",
    "candidate_mode",
    "action_space",
    "num_selected",
    "positions_npy",
    "coverage_pct",
    "coverage_ratio",
    "covered_count",
    "num_targets",
    "episodes",
    "seed",
    "trial",
    "mappo_tag",
    "config_hash",
    "cache_key",
    "created_at",
]


METHOD_DIRS = {
    "MILP": "milp",
    "MILP_BUDGET": "milp",
    "GREEDY": "greedy",
    "MAPPO": "mappo",
    "PPO": "mappo",
}


def build_result_record(**kwargs: Any) -> dict[str, Any]:
    record = {field: kwargs.get(field) for field in REQUIRED_FIELDS}
    record["created_at"] = record["created_at"] or datetime.now().isoformat(timespec="seconds")
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing schema fields: {missing}")
    record.update({k: v for k, v in kwargs.items() if k not in record})
    return record


def validate_result_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Result record missing required fields: {missing}")


def save_positions_npy(path: str | Path, positions: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(positions, dtype=np.float32))
    return path


def save_result_json(path: str | Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_result_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def method_result_dir(results_dir: str | Path, method: str) -> Path:
    return Path(results_dir) / METHOD_DIRS.get(str(method).upper(), str(method).lower())


def find_results(
    results_dir: str | Path,
    method: str | None = None,
    tag: str | None = None,
    profile: str | None = None,
    scenario_id: int | None = None,
    k: int | None = None,
) -> list[Path]:
    base = Path(results_dir)
    search_dirs = [method_result_dir(base, method)] if method else [base / name for name in sorted(set(METHOD_DIRS.values()))]
    out: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                data = load_result_json(path)
            except Exception:
                continue
            if method and str(data.get("method", "")).upper() != method.upper():
                continue
            if tag and str(data.get("mappo_tag", data.get("experiment_tag", ""))) != str(tag):
                continue
            if profile and str(data.get("lidar_profile", data.get("lidar_model", ""))).upper() != profile.upper():
                continue
            if scenario_id is not None and int(data.get("scenario_id", -1)) != int(scenario_id):
                continue
            if k is not None and int(data.get("budget_k", -1)) != int(k):
                continue
            out.append(path)
    return sorted(out)


def _comparison_row(record: dict[str, Any]) -> dict[str, Any]:
    method = record.get("method")
    frontier = record.get("frontier_coverage_pct", record.get("coverage_pct"))
    raw = record.get("raw_coverage_pct", record.get("coverage_pct"))
    return {
        "scenario_id": record.get("scenario_id"),
        "topology": record.get("topology"),
        "profile": record.get("lidar_profile"),
        "method": method,
        "K": record.get("budget_k"),
        "coverage_pct": frontier if method == "MILP_BUDGET" else record.get("coverage_pct"),
        "raw_coverage_pct": raw,
        "frontier_coverage_pct": frontier,
        "trial": record.get("trial"),
        "seed": record.get("seed"),
        "mappo_tag": record.get("mappo_tag"),
        "result_file": record.get("positions_npy"),
        "solver_status": record.get("solver_status"),
        "mip_gap": record.get("mip_gap"),
    }


def collect_comparison_rows(config: ExperimentConfig, *, mappo_tag: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ["MILP_BUDGET", "GREEDY", "MAPPO"]:
        for path in find_results(config.results_dir, method=method):
            record = load_result_json(path)
            if method == "MAPPO" and mappo_tag and record.get("mappo_tag") != mappo_tag:
                continue
            rows.append(_comparison_row(record))
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["method"] == "MAPPO":
            grouped[(row["scenario_id"], row["profile"], row["K"])].append(row)
    for key, group in grouped.items():
        covs = [float(g["coverage_pct"]) for g in group if g.get("coverage_pct") not in (None, "")]
        if covs:
            rows.append(
                {
                    "scenario_id": key[0],
                    "topology": group[0].get("topology"),
                    "profile": key[1],
                    "method": "MAPPO_MEAN",
                    "K": key[2],
                    "coverage_pct": mean(covs),
                    "frontier_coverage_pct": "",
                    "trial": "",
                    "seed": "",
                    "mappo_tag": mappo_tag or group[0].get("mappo_tag"),
                    "result_file": "",
                    "std_coverage_pct": pstdev(covs) if len(covs) > 1 else 0.0,
                }
            )
    return rows


def build_comparison(
    config_path: str | Path,
    *,
    scenario_ids: list[int] | None = None,
    k_values: list[int] | None = None,
    mappo_tag: str | None = None,
    output_csv: str | Path | None = None,
    output_root: str | Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    config = load_config(config_path)
    if output_root is not None:
        config.output["results_dir"] = str(output_root)
    scenario_ids = scenario_ids or config.scenario_ids
    k_values = k_values or config.k_values
    output = Path(output_csv) if output_csv else config.results_dir / "tables" / "compare_methods.csv"
    if dry_run:
        print(
            json.dumps(
                {
                    "method": "COMPARE",
                    "config": str(config_path),
                    "lidar_profile": config.lidar_profile,
                    "scenario_ids": scenario_ids,
                    "k_values": k_values,
                    "mappo_tag": mappo_tag,
                    "output_csv": str(output),
                    "results_root": str(config.results_dir),
                    "expected_methods": ["MILP_BUDGET", "GREEDY", "MAPPO"],
                    "note": "dry_run_no_file_written",
                },
                indent=2,
            )
        )
        return None
    rows = collect_comparison_rows(config, mappo_tag=mappo_tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id",
        "topology",
        "profile",
        "method",
        "K",
        "coverage_pct",
        "frontier_coverage_pct",
        "trial",
        "seed",
        "mappo_tag",
        "std_coverage_pct",
        "raw_coverage_pct",
        "solver_status",
        "mip_gap",
        "result_file",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output


def plot_frontier(csv_path: str | Path, output_png: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(df["scenario_id"].dropna().unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 3.4 * max(len(scenarios), 1)), dpi=160, squeeze=False)
    for ax, sid in zip(axes[:, 0], scenarios):
        sub = df[df["scenario_id"] == sid]
        for method, group in sub.groupby("method"):
            plot_group = group.sort_values("K")
            y_col = "frontier_coverage_pct" if method == "MILP_BUDGET" else "coverage_pct"
            ax.plot(plot_group["K"], plot_group[y_col], marker="o", label=method)
        ax.set_title(f"Scenario {sid}")
        ax.set_xlabel("Budget K")
        ax.set_ylabel("Coverage [%]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


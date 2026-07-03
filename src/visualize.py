"""Visualization exports built on src.environment and src.outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.environment import MapGeometry, evaluate_coverage, load_config, load_positions, load_scenarios
from src.outputs import find_results, load_result_json


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def save_topview(path: str | Path, geometry: MapGeometry, positions: np.ndarray, evaluation, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = evaluation.target_points
    covered = evaluation.covered_mask
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    if len(pts):
        ax.scatter(pts[~covered, 0], pts[~covered, 1], s=3, c="#d9d9d9", label="uncovered")
        ax.scatter(pts[covered, 0], pts[covered, 1], s=3, c="#1f77b4", label="covered")
    if len(positions):
        ax.scatter(positions[:, 0], positions[:, 1], s=60, c="#e66101", edgecolors="black", label="LiDAR")
        for idx, row in enumerate(positions):
            yaw = np.deg2rad(row[3] if len(row) >= 4 else 0.0)
            ax.arrow(row[0], row[1], np.cos(yaw) * 3.0, np.sin(yaw) * 3.0, color="#e66101", head_width=0.7)
            pitch = row[4] if len(row) >= 5 else 0.0
            ax.text(row[0], row[1], f"{idx}: z={row[2]:.1f}, p={pitch:.1f}", fontsize=7)
    ax.set_xlim(0.0, geometry.map_len)
    ax.set_ylim(0.0, geometry.map_width)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_coverage_heatmap(path: str | Path, evaluation, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = evaluation.target_points
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    if len(pts):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=evaluation.covered_mask.astype(float), s=4, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, label="covered")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _sample_rows(points: np.ndarray, max_count: int) -> np.ndarray:
    if len(points) <= max_count:
        return points
    idx = np.linspace(0, len(points) - 1, int(max_count), dtype=np.int32)
    return points[idx]


def _box_vertices(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> list[list[tuple[float, float, float]]]:
    return [
        [(x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1)],
        [(x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2)],
        [(x1, y1, z1), (x2, y1, z1), (x2, y1, z2), (x1, y1, z2)],
        [(x1, y2, z1), (x2, y2, z1), (x2, y2, z2), (x1, y2, z2)],
        [(x1, y1, z1), (x1, y2, z1), (x1, y2, z2), (x1, y1, z2)],
        [(x2, y1, z1), (x2, y2, z1), (x2, y2, z2), (x2, y1, z2)],
    ]


def _add_box(ax, x1: float, y1: float, x2: float, y2: float, height: float, color: str, *, z_base: float = 0.0, alpha: float = 0.8) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = _box_vertices(float(x1), float(y1), float(z_base), float(x2), float(y2), float(z_base + height))
    poly = Poly3DCollection(faces, facecolors=color, edgecolors="#333333", linewidths=0.2, alpha=alpha)
    ax.add_collection3d(poly)


def _draw_infrastructure_3d(ax, geometry: MapGeometry) -> None:
    road_color = "#2f2f2f"
    median_color = "#c98200"
    lane_color = "#8a8a8a"
    sidewalk_color = "#f0d800"
    lane_width = (geometry.scenario.road_width - geometry.median_width) / geometry.num_lanes if geometry.num_lanes > 0 else 3.5

    def draw_segment(center: float, horizontal: bool) -> None:
        length = geometry.map_len if horizontal else geometry.map_width
        if (not horizontal) and geometry.scenario.topology == "T_JUNCTION":
            length = 50.0 + geometry.half_road

        if horizontal:
            _add_box(ax, 0.0, center - geometry.half_road, length, center + geometry.half_road, 0.08, road_color, z_base=0.02, alpha=0.95)
            _add_box(ax, 0.0, center - geometry.median_width / 2.0, length, center + geometry.median_width / 2.0, 0.12, median_color, z_base=0.05)
        else:
            _add_box(ax, center - geometry.half_road, 0.0, center + geometry.half_road, length, 0.08, road_color, z_base=0.02, alpha=0.95)
            _add_box(ax, center - geometry.median_width / 2.0, 0.0, center + geometry.median_width / 2.0, length, 0.12, median_color, z_base=0.05)

        for lane_idx in range(1, int(geometry.num_lanes)):
            dist = geometry.median_width / 2.0 + lane_idx * lane_width
            if dist >= geometry.half_road:
                continue
            for sign in (-1.0, 1.0):
                lane_pos = center + sign * dist
                if horizontal:
                    _add_box(ax, 0.0, lane_pos - 0.075, length, lane_pos + 0.075, 0.02, lane_color, z_base=0.07)
                else:
                    _add_box(ax, lane_pos - 0.075, 0.0, lane_pos + 0.075, length, 0.02, lane_color, z_base=0.07)

        for sign in (-1.0, 1.0):
            sidewalk_center = center + sign * (geometry.half_road + geometry.sidewalk_width / 2.0)
            if horizontal:
                if geometry.scenario.topology == "T_JUNCTION" and sign == 1.0:
                    segments = [(0.0, geometry.map_len)]
                elif geometry.road_centers_x:
                    segments = [(0.0, 50.0 - geometry.half_road), (50.0 + geometry.half_road, geometry.map_len)]
                else:
                    segments = [(0.0, geometry.map_len)]
                for start, end in segments:
                    if end > start:
                        _add_box(ax, start, sidewalk_center - geometry.sidewalk_width / 2.0, end, sidewalk_center + geometry.sidewalk_width / 2.0, 0.15, sidewalk_color, z_base=0.08)
            else:
                vertical_limit = 50.0 + geometry.half_road if geometry.scenario.topology == "T_JUNCTION" else geometry.map_width
                for start, end in [(0.0, 50.0 - geometry.half_road), (50.0 + geometry.half_road, vertical_limit)]:
                    if end > start:
                        _add_box(ax, sidewalk_center - geometry.sidewalk_width / 2.0, start, sidewalk_center + geometry.sidewalk_width / 2.0, end, 0.15, sidewalk_color, z_base=0.08)

    _add_box(ax, 0.0, 0.0, geometry.map_len, geometry.map_width, 0.04, "#242424", z_base=0.0, alpha=0.35)
    for center in geometry.road_centers_y:
        draw_segment(float(center), horizontal=True)
    for center in geometry.road_centers_x:
        draw_segment(float(center), horizontal=False)
    for x1, y1, x2, y2 in geometry.negative_zones:
        _add_box(ax, x1, y1, x2, y2, geometry.building_height, "#5a5a63", z_base=0.1, alpha=0.65)


def save_episode_graph(train_log_npz: str | Path, output_png: str | Path, *, title: str = "Training curve") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(train_log_npz)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    plotted = False
    for key in ["coverage_pct", "coverage", "eval_coverage_pct", "eval_coverage", "reward", "score"]:
        if key in data:
            arr = np.asarray(data[key], dtype=float).reshape(-1)
            ax.plot(np.arange(len(arr)), arr, label=key)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No recognized episode metrics in npz", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Episode / evaluation index")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    return output_png


def save_3d_snapshot(path: str | Path, positions: np.ndarray, evaluation, *, title: str, **_kwargs) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = _kwargs["geometry"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    _draw_infrastructure_3d(ax, geometry)

    pts = np.asarray(evaluation.target_points, dtype=np.float32)
    if len(pts):
        uncovered = _sample_rows(pts[~evaluation.covered_mask], 6000)
        covered = _sample_rows(pts[evaluation.covered_mask], 6000)
        if len(uncovered):
            ax.scatter(uncovered[:, 0], uncovered[:, 1], uncovered[:, 2], s=2, c="#c7c7c7", alpha=0.12, label="uncovered")
        if len(covered):
            ax.scatter(covered[:, 0], covered[:, 1], covered[:, 2], s=3, c="#2ca02c", alpha=0.45, label="covered")

    if len(positions):
        pos = np.atleast_2d(positions)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=60, c="#d62728", depthshade=True, label="LiDAR")
        for idx, row in enumerate(pos):
            yaw = np.deg2rad(float(row[3]))
            pitch = np.deg2rad(float(row[4]))
            dx = np.cos(pitch) * np.cos(yaw) * 5.0
            dy = np.cos(pitch) * np.sin(yaw) * 5.0
            dz = np.sin(pitch) * 5.0
            ax.quiver(row[0], row[1], row[2], dx, dy, dz, color="#ffbf00", linewidth=1.8, arrow_length_ratio=0.25)
            ax.text(row[0], row[1], row[2] + 1.0, f"A{idx}", color="black", fontsize=8)

    ax.set_xlim(0.0, geometry.map_len)
    ax.set_ylim(0.0, geometry.map_width)
    ax.set_zlim(0.0, max(geometry.building_height, float(np.max(positions[:, 2]) + 2.0 if len(positions) else 8.0)))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.view_init(elev=38, azim=-55)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def export_result_figures(
    result_json: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    save_3d: bool = False,
    save_beams: bool = False,
    train_log: str | Path | None = None,
    output_3d_dir: str | Path | None = None,
) -> list[Path]:
    record = load_result_json(result_json)
    config = load_config(config_path)
    scenarios = {s.scenario_id: s for s in load_scenarios()}
    scenario = scenarios[int(record["scenario_id"])]
    profile = str(record["lidar_profile"])
    positions = load_positions(record["positions_npy"])
    evaluation = evaluate_coverage(positions, scenario, profile, config)
    geometry = MapGeometry(scenario)
    output_dir = Path(output_dir)
    suffix = f"_{str(record['mappo_tag']).lower()}" if record.get("mappo_tag") else ""
    if record.get("trial") not in (None, ""):
        suffix += f"_t{record['trial']}"
    prefix = f"{record['method'].lower()}_{profile.lower()}_s{scenario.scenario_id}_k{record['budget_k']}{suffix}"
    title = f"{record['method']} / {profile} / scenario {scenario.scenario_id} {scenario.topology} / K={record['budget_k']} / coverage={evaluation.coverage_pct:.2f}%"
    paths = [
        save_topview(output_dir / f"{prefix}_topview.png", geometry, positions, evaluation, title=title),
        save_coverage_heatmap(output_dir / f"{prefix}_heatmap.png", evaluation, title=title),
    ]
    if save_3d:
        paths.append(save_3d_snapshot((Path(output_3d_dir) if output_3d_dir else output_dir) / f"{prefix}_3d.png", positions, evaluation, title=title, save_beams=save_beams, geometry=geometry))
    log_path = train_log or record.get("train_log_npz")
    if log_path:
        paths.append(save_episode_graph(log_path, output_dir / f"{prefix}_episode_graph.png", title=title))
    return paths


def visualize_dry_run(
    config_path: str | Path,
    *,
    result_json: str | None,
    output_dir: str,
    output_3d_dir: str,
    results_root: str,
    scenario_ids: list[int],
    k_values: list[int],
    methods: list[str],
    mappo_tag: str | None,
    save_beams: bool,
) -> None:
    config = load_config(config_path)
    print(
        json.dumps(
            {
                "method": "VISUALIZE",
                "config": str(config_path),
                "result_json": result_json,
                "output_dir": output_dir,
                "output_3d_dir": output_3d_dir,
                "supported_outputs": ["topview", "heatmap", "3d_matplotlib_snapshot"],
                "results_root": results_root or str(config.results_dir),
                "scenario_ids": scenario_ids,
                "k_values": k_values,
                "methods": methods,
                "mappo_tag": mappo_tag,
                "save_beams": bool(save_beams),
                "note": "dry_run_no_file_written",
            },
            indent=2,
        )
    )


def find_visualization_inputs(results_root: str | Path, methods: list[str], scenario_ids: list[int], k_values: list[int], mappo_tag: str | None) -> list[Path]:
    result_jsons: list[Path] = []
    for method in methods:
        for scenario_id in scenario_ids or [None]:
            for k in k_values or [None]:
                tag = mappo_tag if method == "MAPPO" else None
                result_jsons.extend(find_results(results_root, method=method, tag=tag, scenario_id=scenario_id, k=k))
    return sorted(set(result_jsons))

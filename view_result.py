#!/usr/bin/env python3
"""Interactive 3D result viewer ported from the legacy copy viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.environment import (
    MapGeometry,
    compute_view_angles_numpy,
    evaluate_coverage,
    get_lidar_profile,
    load_config,
    load_positions,
    load_scenarios,
)
from src.outputs import find_results, load_result_json
from src.visualize import _draw_infrastructure_3d


METHOD_ALIASES = {
    "milp": "MILP_BUDGET",
    "milp_budget": "MILP_BUDGET",
    "greedy": "GREEDY",
    "mappo": "MAPPO",
    "ppo": "MAPPO",
}


def _method_name(value: str | None) -> str | None:
    if not value:
        return None
    return METHOD_ALIASES.get(value.strip().lower(), value.strip().upper())


def _load_scenario(scenario_id: int):
    scenarios = {int(s.scenario_id): s for s in load_scenarios()}
    if int(scenario_id) not in scenarios:
        raise KeyError(f"Scenario id not found: {scenario_id}")
    return scenarios[int(scenario_id)]


def _resolve_result_entries(args) -> list[Path]:
    if args.result_json:
        return [Path(args.result_json)]
    method = _method_name(args.method)
    entries = find_results(
        args.results_root,
        method=method,
        tag=args.mappo_tag if method == "MAPPO" else None,
        scenario_id=args.scenario_id,
        k=args.k,
    )
    return entries


def _choose_result(entries: list[Path]) -> Path:
    if not entries:
        raise FileNotFoundError("No matching result JSON files found.")
    if len(entries) == 1:
        return entries[0]
    print("\nAvailable result files")
    print("-" * 100)
    for idx, path in enumerate(entries, start=1):
        try:
            record = load_result_json(path)
            label = (
                f"{record.get('method')} | s={record.get('scenario_id')} | "
                f"k={record.get('budget_k')} | trial={record.get('trial')} | "
                f"tag={record.get('mappo_tag')} | cov={float(record.get('coverage_pct', 0.0)):.2f}%"
            )
        except Exception:
            label = path.name
        print(f"{idx:3d}. {label}\n     {path}")
    while True:
        value = input("\nSelect result number to view, or q to quit: ").strip()
        if value.lower() == "q":
            raise SystemExit(0)
        try:
            index = int(value)
        except ValueError:
            print("Enter a number.")
            continue
        if 1 <= index <= len(entries):
            return entries[index - 1]
        print("Out of range.")


def _coverage_mask_for_pose(positions: np.ndarray, target_points: np.ndarray, profile) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if len(target_points) == 0:
        return [np.zeros(0, dtype=bool) for _ in positions]
    half_h = np.deg2rad(float(profile.h_fov_deg)) / 2.0
    half_v = np.deg2rad(float(profile.v_fov_deg)) / 2.0
    for row in np.atleast_2d(positions):
        x, y, z, yaw, pitch = map(float, row[:5])
        dx = target_points[:, 0] - x
        dy = target_points[:, 1] - y
        dz = target_points[:, 2] - z
        azimuth, elevation, _forward, dist = compute_view_angles_numpy(dx, dy, dz, yaw, pitch)
        front = np.ones_like(dist, dtype=bool) if half_h >= np.pi - 1e-6 else np.abs(azimuth) <= half_h
        mask = (dist <= float(profile.max_range_m)) & front & (np.abs(elevation) <= half_v)
        masks.append(mask)
    return masks


def _sample_rows(points: np.ndarray, max_count: int) -> np.ndarray:
    if len(points) <= max_count:
        return points
    idx = np.linspace(0, len(points) - 1, int(max_count), dtype=np.int32)
    return points[idx]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _load_training_log(record: dict) -> dict[str, np.ndarray] | None:
    path = record.get("train_log_npz") or record.get("extra", {}).get("train_log_npz")
    if not path or not Path(path).exists():
        return None
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    for key in data.files:
        out[key] = np.asarray(data[key])
    if "episode" not in out:
        length = 0
        for key in ("coverage_pct", "coverage", "eval_coverage", "reward", "score"):
            if key in out:
                length = len(np.asarray(out[key]).reshape(-1))
                break
        out["episode"] = np.arange(length, dtype=np.int32)
    return out


def _coverage_density(positions: np.ndarray, evaluation, profile) -> tuple[np.ndarray, list[np.ndarray]]:
    masks = _coverage_mask_for_pose(positions, evaluation.target_points, profile)
    density = np.zeros(len(evaluation.target_points), dtype=np.float32)
    for mask in masks:
        density[mask] += 1.0
    return density, masks


def _summarize_agents(positions: np.ndarray, masks: list[np.ndarray], density: np.ndarray, road_detection: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for idx, (pos, mask) in enumerate(zip(np.atleast_2d(positions), masks)):
        visible = int(np.count_nonzero(mask))
        local_density = density[mask] if visible else np.asarray([], dtype=np.float32)
        unique = int(np.count_nonzero(local_density <= float(road_detection))) if visible else 0
        overlap = int(np.count_nonzero(local_density >= float(road_detection + 1))) if visible else 0
        rows.append(
            {
                "agent_id": idx,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "yaw": float(pos[3]),
                "pitch": float(pos[4]),
                "visible_count": visible,
                "unique_count": unique,
                "overlap_count": overlap,
            }
        )
    return rows


def _show_heatmap(record: dict, positions: np.ndarray, evaluation, agent_rows: list[dict[str, float | int]], train_log: dict[str, np.ndarray] | None) -> None:
    import matplotlib.pyplot as plt

    pts = evaluation.target_points
    covered = evaluation.covered_mask
    fig = plt.figure(figsize=(18, 10), dpi=120)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[:, 0])
    ax_st = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[1, 1])
    if len(pts):
        ax.scatter(pts[~covered, 0], pts[~covered, 1], c="lightgray", s=3, alpha=0.3, label="uncovered")
        ax.scatter(pts[covered, 0], pts[covered, 1], c="tab:green", s=3, alpha=0.8, label="covered")
    for idx, row in enumerate(np.atleast_2d(positions)):
        ax.scatter(row[0], row[1], c="red", s=80, marker="x")
        yaw = np.deg2rad(float(row[3]))
        ax.arrow(row[0], row[1], np.cos(yaw) * 4.0, np.sin(yaw) * 4.0, color="red", head_width=1.0)
        ax.text(row[0] + 0.6, row[1] + 0.6, f"A{idx}", color="black", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title(
        f"{record.get('method')} | s={record.get('scenario_id')} | k={record.get('budget_k')} | "
        f"coverage={evaluation.coverage_pct:.2f}%"
    )

    if train_log:
        episode = np.asarray(train_log.get("episode", []), dtype=np.float32).reshape(-1)
        reward = np.asarray(train_log.get("reward", []), dtype=np.float32).reshape(-1)
        coverage = np.asarray(train_log.get("coverage_pct", train_log.get("coverage", [])), dtype=np.float32).reshape(-1)
        if len(reward) and len(episode) == len(reward):
            ax_st.plot(episode, reward, color="tab:orange", alpha=0.35, label="reward")
            if len(reward) >= 100:
                ax_st.plot(episode[99:], _moving_average(reward, 100), color="tab:orange", linewidth=2.0, label="reward MA(100)")
        ax_cov = ax_st.twinx()
        if len(coverage) and len(episode) == len(coverage):
            ax_cov.plot(episode, coverage, color="tab:blue", alpha=0.45, label="coverage")
            if len(coverage) >= 100:
                ax_cov.plot(episode[99:], _moving_average(coverage, 100), color="tab:blue", linewidth=2.0, label="coverage MA(100)")
        ax_st.set_title("Training Stability")
        ax_st.set_xlabel("Episode")
        ax_st.set_ylabel("Reward", color="tab:orange")
        ax_cov.set_ylabel("Coverage (%)", color="tab:blue")
        ax_st.grid(True, alpha=0.25)
        ax_st.legend(loc="upper left", fontsize=8)
        ax_cov.legend(loc="upper right", fontsize=8)
    else:
        ax_st.text(0.5, 0.5, "No training log found", ha="center", va="center", transform=ax_st.transAxes)
        ax_st.set_axis_off()

    ax_tbl.set_axis_off()
    overlap_count = int(np.count_nonzero(evaluation.density_map >= 2.0))
    overlap_pct = 100.0 * overlap_count / max(int(evaluation.covered_count), 1)
    summary = [
        f"Global Coverage: {evaluation.coverage_pct:.2f}%",
        f"Covered Targets: {evaluation.covered_count} / {evaluation.num_targets}",
        f"Overlap Targets: {overlap_count}",
        f"Overlap / Covered: {overlap_pct:.2f}%",
        f"LiDAR: {record.get('lidar_profile')}",
        f"Range: {float(record.get('max_range_m', 0.0)):.1f} m",
    ]
    ax_tbl.text(0.0, 1.05, "\n".join(summary), transform=ax_tbl.transAxes, va="top", ha="left", fontsize=10, family="monospace")
    table_rows = [
        [
            str(row["agent_id"]),
            f"{float(row['x']):.1f}",
            f"{float(row['y']):.1f}",
            f"{float(row['z']):.1f}",
            f"{float(row['yaw']):.1f}",
            f"{float(row['pitch']):.1f}",
            str(row["visible_count"]),
            str(row["unique_count"]),
            str(row["overlap_count"]),
        ]
        for row in agent_rows
    ]
    if table_rows:
        table = ax_tbl.table(
            cellText=table_rows,
            colLabels=["Ag", "X", "Y", "Z", "Yaw", "Pitch", "Vis", "Unique", "Overlap"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.0, 0.0, 1.0, 0.78],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.18)
    ax_tbl.set_title("Per-Agent Summary", loc="left", fontsize=11)
    fig.tight_layout()
    plt.show(block=False)


def _box_mesh(x1: float, y1: float, x2: float, y2: float, z: float, color: list[float], *, z_base: float = 0.0):
    import open3d as o3d

    width = max(float(x2 - x1), 1e-6)
    depth = max(float(y2 - y1), 1e-6)
    height = max(float(z), 1e-6)
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=depth, depth=height)
    mesh.translate([float(x1), float(y1), float(z_base)])
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _add_legacy_infrastructure(vis, geometry: MapGeometry) -> None:
    deep_yellow = [0.8, 0.5, 0.0]
    road_color = [0.15, 0.15, 0.15]
    lane_color = [0.5, 0.5, 0.5]
    sidewalk_color = [1.0, 0.9, 0.0]
    lane_width = (
        (float(geometry.scenario.road_width) - float(geometry.median_width)) / float(geometry.num_lanes)
        if int(geometry.num_lanes) > 0
        else 3.5
    )

    def add_infrastructure(center: float, horizontal: bool) -> None:
        length = float(geometry.map_len if horizontal else geometry.map_width)
        if not horizontal and geometry.scenario.topology == "T_JUNCTION":
            length = 50.0 + float(geometry.half_road)

        if horizontal:
            vis.add_geometry(_box_mesh(0.0, center - geometry.half_road, length, center + geometry.half_road, 0.08, road_color, z_base=0.02))
            vis.add_geometry(_box_mesh(0.0, center - geometry.median_width / 2.0, length, center + geometry.median_width / 2.0, 0.12, deep_yellow, z_base=0.05))
        else:
            vis.add_geometry(_box_mesh(center - geometry.half_road, 0.0, center + geometry.half_road, length, 0.08, road_color, z_base=0.02))
            vis.add_geometry(_box_mesh(center - geometry.median_width / 2.0, 0.0, center + geometry.median_width / 2.0, length, 0.12, deep_yellow, z_base=0.05))

        for i in range(1, int(geometry.num_lanes)):
            dist = geometry.median_width / 2.0 + i * lane_width
            if dist >= geometry.half_road:
                continue
            for sign in (-1.0, 1.0):
                lp = center + sign * dist
                if horizontal:
                    vis.add_geometry(_box_mesh(0.0, lp - 0.075, length, lp + 0.075, 0.02, lane_color, z_base=0.07))
                else:
                    vis.add_geometry(_box_mesh(lp - 0.075, 0.0, lp + 0.075, length, 0.02, lane_color, z_base=0.07))

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
                    if end <= start:
                        continue
                    vis.add_geometry(
                        _box_mesh(
                            start,
                            sidewalk_center - geometry.sidewalk_width / 2.0,
                            end,
                            sidewalk_center + geometry.sidewalk_width / 2.0,
                            0.15,
                            sidewalk_color,
                            z_base=0.08,
                        )
                    )
            else:
                vertical_limit = 50.0 + geometry.half_road if geometry.scenario.topology == "T_JUNCTION" else geometry.map_width
                segments = [(0.0, 50.0 - geometry.half_road), (50.0 + geometry.half_road, vertical_limit)]
                for start, end in segments:
                    if end <= start:
                        continue
                    vis.add_geometry(
                        _box_mesh(
                            sidewalk_center - geometry.sidewalk_width / 2.0,
                            start,
                            sidewalk_center + geometry.sidewalk_width / 2.0,
                            end,
                            0.15,
                            sidewalk_color,
                            z_base=0.08,
                        )
                    )

    for center in geometry.road_centers_y:
        add_infrastructure(float(center), horizontal=True)
    for center in geometry.road_centers_x:
        add_infrastructure(float(center), horizontal=False)


def _point_cloud(points: np.ndarray, color: list[float]):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def _installable_points(geometry: MapGeometry, step_m: float) -> np.ndarray:
    step = max(float(step_m), 0.25)
    points: list[list[float]] = []
    xs = np.arange(0.0, geometry.map_len + 1e-9, step, dtype=float)
    ys = np.arange(0.0, geometry.map_width + 1e-9, step, dtype=float)
    tolerance = step / 2.0
    for x in xs:
        for y in ys:
            if geometry.is_installable(float(x), float(y), tolerance_m=tolerance):
                points.append([float(x), float(y), 0.32])
    return np.asarray(points, dtype=np.float32)


def _line_set(points: list[list[float]], lines: list[list[int]], color: list[float]):
    import open3d as o3d

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.paint_uniform_color(color)
    return line_set


def _rotation_from_z_to_vector(vector: np.ndarray) -> np.ndarray:
    import open3d as o3d

    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        return np.eye(3)
    target = vector / norm
    source = np.array([0.0, 0.0, 1.0])
    axis = np.cross(source, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-9:
        return np.eye(3) if target[2] >= 0 else o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0.0, 0.0])
    axis = axis / axis_norm
    angle = float(np.arccos(np.clip(np.dot(source, target), -1.0, 1.0)))
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def _add_lidar_markers(vis, positions: np.ndarray) -> None:
    import open3d as o3d

    for idx, row in enumerate(np.atleast_2d(positions)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.65)
        sphere.paint_uniform_color([1.0, 0.05, 0.02])
        sphere.translate(row[:3])
        vis.add_geometry(sphere)

        yaw = np.deg2rad(float(row[3]))
        pitch = np.deg2rad(float(row[4]))
        direction = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.12,
            cone_radius=0.35,
            cylinder_height=3.0,
            cone_height=0.9,
        )
        arrow.paint_uniform_color([1.0, 0.85, 0.05])
        arrow.rotate(_rotation_from_z_to_vector(direction), center=[0.0, 0.0, 0.0])
        arrow.translate(row[:3])
        vis.add_geometry(arrow)

        label_anchor = o3d.geometry.TriangleMesh.create_sphere(radius=0.18)
        label_anchor.paint_uniform_color([1.0, 1.0, 1.0])
        label_anchor.translate([row[0], row[1], row[2] + 1.2 + 0.2 * idx])
        vis.add_geometry(label_anchor)


def _show_open3d(record: dict, positions: np.ndarray, scenario, evaluation, config, *, show_beams: bool, max_points: int) -> None:
    import matplotlib.pyplot as plt
    import open3d as o3d

    geometry = MapGeometry(scenario)
    profile = get_lidar_profile(str(record["lidar_profile"]), config)
    covered_points = evaluation.target_points[evaluation.covered_mask]
    uncovered_points = evaluation.target_points[~evaluation.covered_mask]
    covered_points = _sample_rows(covered_points, max_points)
    uncovered_points = _sample_rows(uncovered_points, max_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Result Visualization", width=1280, height=820)
    opt = vis.get_render_option()
    opt.background_color = [0.08, 0.08, 0.08]
    opt.point_size = 4.0

    ground = _box_mesh(0.0, 0.0, geometry.map_len, geometry.map_width, 0.04, [0.16, 0.16, 0.16])
    vis.add_geometry(ground)
    _add_legacy_infrastructure(vis, geometry)
    for zone in geometry.negative_zones:
        x1, y1, x2, y2 = map(float, zone)
        vis.add_geometry(_box_mesh(x1, y1, x2, y2, geometry.building_height, [0.35, 0.35, 0.38], z_base=0.1))

    if len(uncovered_points):
        vis.add_geometry(_point_cloud(uncovered_points, [0.75, 0.15, 0.12]))
    if len(covered_points):
        vis.add_geometry(_point_cloud(covered_points, [0.15, 0.85, 0.20]))

    installable = _installable_points(geometry, config.xy_step_m)
    if len(installable):
        installable_pcd = _point_cloud(installable, [0.05, 0.95, 1.0])
        vis.add_geometry(installable_pcd)

    if show_beams and len(evaluation.target_points):
        masks = _coverage_mask_for_pose(positions, evaluation.target_points, profile)
        line_points: list[list[float]] = []
        lines: list[list[int]] = []
        for sensor_idx, (row, mask) in enumerate(zip(np.atleast_2d(positions), masks)):
            hits = _sample_rows(evaluation.target_points[mask], max(8, max_points // max(len(positions), 1) // 20))
            for hit in hits:
                start = len(line_points)
                line_points.append([float(row[0]), float(row[1]), float(row[2])])
                line_points.append([float(hit[0]), float(hit[1]), float(hit[2])])
                lines.append([start, start + 1])
        if line_points:
            vis.add_geometry(_line_set(line_points, lines, [1.0, 0.72, 0.08]))

    _add_lidar_markers(vis, positions)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0.0, 0.0, 0.0]))

    ctr = vis.get_view_control()
    ctr.set_lookat([geometry.map_len / 2.0, geometry.map_width / 2.0, 3.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_front([0.0, -0.65, 0.75])
    ctr.set_zoom(0.72)

    print("3D viewer opened. Close the Open3D window to return.")
    try:
        while True:
            keep_running = vis.poll_events()
            vis.update_renderer()
            plt.pause(0.01)
            if not keep_running:
                break
    finally:
        vis.destroy_window()


def _show_matplotlib_3d(record: dict, positions: np.ndarray, scenario, evaluation, config, *, show_beams: bool, max_points: int) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    geometry = MapGeometry(scenario)
    profile = get_lidar_profile(str(record["lidar_profile"]), config)

    fig = plt.figure(figsize=(12, 9), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    _draw_infrastructure_3d(ax, geometry)

    covered_points = _sample_rows(evaluation.target_points[evaluation.covered_mask], max_points)
    uncovered_points = _sample_rows(evaluation.target_points[~evaluation.covered_mask], max_points)
    if len(uncovered_points):
        ax.scatter(uncovered_points[:, 0], uncovered_points[:, 1], uncovered_points[:, 2], s=3, c="#b0b0b0", alpha=0.12, label="uncovered")
    if len(covered_points):
        ax.scatter(covered_points[:, 0], covered_points[:, 1], covered_points[:, 2], s=4, c="#2ca02c", alpha=0.45, label="covered")

    installable = _installable_points(geometry, config.xy_step_m)
    if len(installable):
        installable = _sample_rows(installable, 8000)
        ax.scatter(installable[:, 0], installable[:, 1], installable[:, 2], s=5, c="#00d7ff", alpha=0.35, label="installable")

    pos = np.atleast_2d(positions)
    if len(pos):
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=90, c="#d62728", depthshade=True, label="LiDAR")
        for idx, row in enumerate(pos):
            yaw = np.deg2rad(float(row[3]))
            pitch = np.deg2rad(float(row[4]))
            direction = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=float)
            ax.quiver(row[0], row[1], row[2], direction[0] * 5.0, direction[1] * 5.0, direction[2] * 5.0, color="#ffbf00", linewidth=1.8, arrow_length_ratio=0.25)
            ax.text(row[0], row[1], row[2] + 1.0, f"A{idx}", color="black", fontsize=9)

    if show_beams and len(evaluation.target_points) and len(pos):
        masks = _coverage_mask_for_pose(pos, evaluation.target_points, profile)
        segments: list[list[list[float]]] = []
        beam_budget = max(8, max_points // max(len(pos), 1) // 25)
        for row, mask in zip(pos, masks):
            hits = _sample_rows(evaluation.target_points[mask], beam_budget)
            for hit in hits:
                segments.append([[float(row[0]), float(row[1]), float(row[2])], [float(hit[0]), float(hit[1]), float(hit[2])]])
        if segments:
            ax.add_collection3d(Line3DCollection(segments, colors="#ffbf00", linewidths=0.35, alpha=0.18))

    ax.set_xlim(0.0, geometry.map_len)
    ax.set_ylim(0.0, geometry.map_width)
    ax.set_zlim(0.0, max(geometry.building_height, float(np.max(pos[:, 2]) + 3.0 if len(pos) else 8.0)))
    ax.set_box_aspect((geometry.map_len, geometry.map_width, max(geometry.building_height, 12.0)))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(
        f"{record.get('method')} | scenario={scenario.scenario_id} | K={record.get('budget_k')} | "
        f"coverage={evaluation.coverage_pct:.2f}%"
    )
    ax.view_init(elev=38, azim=-55)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    print("Matplotlib 3D viewer opened. Use the toolbar/mouse to rotate, pan, and zoom; close the window to return.")
    plt.show()


def view_result(result_json: str | Path, args) -> None:
    record = load_result_json(result_json)
    config = load_config(args.config)
    scenario = _load_scenario(int(record["scenario_id"]))
    positions = load_positions(record["positions_npy"])
    evaluation = evaluate_coverage(positions, scenario, str(record["lidar_profile"]), config)
    profile = get_lidar_profile(str(record["lidar_profile"]), config)
    density, masks = _coverage_density(positions, evaluation, profile)
    evaluation.density_map[:] = density
    road_detection = int(config.target.get("road_detection", 1))
    agent_rows = _summarize_agents(positions, masks, density, road_detection)
    train_log = _load_training_log(record)

    print(
        f"{record.get('method')} | scenario={scenario.scenario_id} | k={record.get('budget_k')} | "
        f"selected={len(positions)} | coverage={evaluation.coverage_pct:.2f}% "
        f"({evaluation.covered_count}/{evaluation.num_targets})"
    )
    for row in agent_rows:
        print(
            f"A{int(row['agent_id']):02d}: "
            f"x={float(row['x']):.2f}, y={float(row['y']):.2f}, z={float(row['z']):.2f}, "
            f"yaw={float(row['yaw']):.2f}, pitch={float(row['pitch']):.2f}, "
            f"vis={int(row['visible_count'])}, unique={int(row['unique_count'])}, overlap={int(row['overlap_count'])}"
        )

    if not args.no_heatmap:
        _show_heatmap(record, positions, evaluation, agent_rows, train_log)
    if args.viewer in {"matplotlib", "both"}:
        _show_matplotlib_3d(
            record,
            positions,
            scenario,
            evaluation,
            config,
            show_beams=not args.no_beams,
            max_points=args.max_points,
        )
    if args.viewer in {"open3d", "both"}:
        _show_open3d(
            record,
            positions,
            scenario,
            evaluation,
            config,
            show_beams=not args.no_beams,
            max_points=args.max_points,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive 3D result viewer.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--results-root", default="results/active")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--method", default=None, help="milp, greedy, or mappo")
    parser.add_argument("--scenario-id", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--mappo-tag", default=None)
    parser.add_argument("--list", action="store_true", help="List matching result JSON files and exit.")
    parser.add_argument("--viewer", choices=["matplotlib", "open3d", "both"], default="matplotlib")
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--no-beams", action="store_true")
    parser.add_argument("--max-points", type=int, default=50000)
    args = parser.parse_args()

    entries = _resolve_result_entries(args)
    if args.list:
        for path in entries:
            print(path)
        return 0
    selected = _choose_result(entries)
    view_result(selected, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

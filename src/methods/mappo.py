"""Native MAPPO port from the legacy copy, adapted to the shared environment."""

from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.environment import (
    ExperimentConfig,
    MapGeometry,
    Scenario,
    allowed_mount_yaws_deg,
    bake_visibility_map,
    compute_config_hash,
    compute_view_angles_numpy,
    evaluate_coverage,
    get_lidar_profile,
    get_scenario,
    unpack_visible_indices,
)
from src.outputs import build_result_record, save_positions_npy, save_result_json


HINT_DIM = 7
ACTION_DIM = 11


@dataclass
class MappoPlanRow:
    scenario_id: int
    topology: str
    lidar_profile: str
    budget_k: int
    trial: int
    seed: int
    episodes: int
    output_tag: str


@dataclass
class LegacyMAPPOParams:
    num_agents: int
    episodes: int
    xy_step_m: float
    z_min: float
    z_max: float
    z_step: float
    yaw_step_deg: float
    yaw_axis_min_deg: float
    yaw_axis_max_deg: float
    pitch_step_deg: float
    min_pitch: float
    max_pitch: float
    local_map_size: int = 7
    score_base_voxel: float = 0.01
    road_detection: int = 1
    max_voxel_est: float = 3000.0
    warm_start_enabled: bool = True
    spawn_top_k: int = 200
    w_abs_coverage: float = 0.1
    w_overlap_pen: float = 0.0
    w_far_dist: float = 1.0
    w_uncovered_move: float = 1.5
    dist_reward_scale: float = 20.0
    w_marginal_gain: float = 1.0
    w_outward_pen: float = 2.0
    edge_margin_m: float = 5.0
    ppo_lr: float = 1e-5
    ppo_gamma: float = 0.999
    ppo_eps_clip: float = 0.15
    ppo_k_epochs: int = 5
    ppo_update_timestep: int = 512
    ppo_entropy_coef: float = 0.003
    ppo_gae_lambda: float = 0.95
    ppo_value_clip: float = 0.2
    ppo_minibatch_size: int = 128
    ppo_entropy_min: float = 0.0
    ppo_entropy_decay: float = 0.997
    ppo_random_action_prob: float = 0.005
    eval_interval: int = 100
    eval_steps: int = 5
    plateau_patience: int = 200
    plateau_entropy_scale: float = 0.3
    plateau_uncovered_scale: float = 0.3
    print_interval: int = 20
    uncovered_centroid: np.ndarray | None = None

    @property
    def z_values(self) -> np.ndarray:
        return np.arange(self.z_min, self.z_max + 0.1, self.z_step, dtype=np.float32)

    @property
    def z_idx_max(self) -> int:
        vals = self.z_values
        return max(int(len(vals) - 1), 0)


def _mappo_value(config: ExperimentConfig, key: str, default: Any) -> Any:
    return config.mappo.get(key, default)


def _legacy_params(row: MappoPlanRow, config: ExperimentConfig) -> LegacyMAPPOParams:
    return LegacyMAPPOParams(
        num_agents=int(row.budget_k),
        episodes=int(row.episodes),
        xy_step_m=float(config.xy_step_m),
        z_min=float(config.z_min),
        z_max=float(config.z_max),
        z_step=float(config.z_step),
        yaw_step_deg=float(config.yaw_step_deg),
        yaw_axis_min_deg=float(config.yaw_axis_min_deg),
        yaw_axis_max_deg=float(config.yaw_axis_max_deg),
        pitch_step_deg=float(config.pitch_step_deg),
        min_pitch=float(config.pitch_min_deg),
        max_pitch=float(config.pitch_max_deg),
        local_map_size=int(_mappo_value(config, "local_map_size", 7)),
        score_base_voxel=float(_mappo_value(config, "score_base_voxel", 0.01)),
        road_detection=int(config.target.get("road_detection", 1)),
        max_voxel_est=float(_mappo_value(config, "max_voxel_est", 3000.0)),
        warm_start_enabled=bool(_mappo_value(config, "warm_start_enabled", True)),
        spawn_top_k=int(_mappo_value(config, "spawn_top_k", 200)),
        w_abs_coverage=float(_mappo_value(config, "w_abs_coverage", 0.1)),
        w_overlap_pen=float(_mappo_value(config, "w_overlap_pen", 0.0)),
        w_far_dist=float(_mappo_value(config, "w_far_dist", 1.0)),
        w_uncovered_move=float(_mappo_value(config, "w_uncovered_move", 1.5)),
        dist_reward_scale=float(_mappo_value(config, "dist_reward_scale", 20.0)),
        w_marginal_gain=float(_mappo_value(config, "w_marginal_gain", 1.0)),
        w_outward_pen=float(_mappo_value(config, "w_outward_pen", 2.0)),
        edge_margin_m=float(_mappo_value(config, "edge_margin_m", 5.0)),
        ppo_lr=float(_mappo_value(config, "ppo_lr", 1e-5)),
        ppo_gamma=float(_mappo_value(config, "ppo_gamma", 0.999)),
        ppo_eps_clip=float(_mappo_value(config, "ppo_eps_clip", 0.15)),
        ppo_k_epochs=int(_mappo_value(config, "ppo_k_epochs", 5)),
        ppo_update_timestep=int(_mappo_value(config, "ppo_update_timestep", 512)),
        ppo_entropy_coef=float(_mappo_value(config, "ppo_entropy_coef", 0.003)),
        ppo_gae_lambda=float(_mappo_value(config, "ppo_gae_lambda", 0.95)),
        ppo_value_clip=float(_mappo_value(config, "ppo_value_clip", 0.2)),
        ppo_minibatch_size=int(_mappo_value(config, "ppo_minibatch_size", 128)),
        ppo_entropy_min=float(_mappo_value(config, "ppo_entropy_min", 0.0)),
        ppo_entropy_decay=float(_mappo_value(config, "ppo_entropy_decay", 0.997)),
        ppo_random_action_prob=float(_mappo_value(config, "ppo_random_action_prob", 0.005)),
        eval_interval=int(_mappo_value(config, "eval_interval", 100)),
        eval_steps=int(_mappo_value(config, "eval_steps", 5)),
        plateau_patience=int(_mappo_value(config, "plateau_patience", 200)),
        plateau_entropy_scale=float(_mappo_value(config, "plateau_entropy_scale", 0.3)),
        plateau_uncovered_scale=float(_mappo_value(config, "plateau_uncovered_scale", 0.3)),
        print_interval=int(_mappo_value(config, "print_interval", 20)),
    )


def plan_mappo_runs(
    scenarios: Sequence[Scenario],
    lidar_profile: str,
    k_values: Sequence[int],
    config: ExperimentConfig,
    *,
    tag: str,
    episodes: int | None = None,
    trials: int | None = None,
    seed_base: int | None = None,
) -> list[MappoPlanRow]:
    rows: list[MappoPlanRow] = []
    ep = int(episodes if episodes is not None else config.episodes)
    tr = int(trials if trials is not None else config.trials)
    base = int(seed_base if seed_base is not None else config.seed_base)
    for scenario in scenarios:
        for k in k_values:
            for trial in range(tr):
                rows.append(MappoPlanRow(scenario.scenario_id, scenario.topology, lidar_profile, int(k), trial, base + trial, ep, tag))
    return rows


def _heading_vector_xy(yaw_deg: float) -> np.ndarray:
    yr = np.deg2rad(float(yaw_deg))
    return np.array([np.cos(yr), np.sin(yr)], dtype=np.float32)


def _relative_yaw(yaw_deg: float, base_yaw: float) -> float:
    return (float(yaw_deg) - float(base_yaw) + 180.0) % 360.0 - 180.0


def _compute_view_angles_torch(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor, yaw_deg: float, pitch_deg: float):
    pitch_rad = float(np.deg2rad(float(pitch_deg)))
    yaw_rad = float(np.deg2rad(float(yaw_deg)))
    dist = torch.sqrt(dx * dx + dy * dy + dz * dz)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    dx_y = dx * cy + dy * sy
    dy_y = -dx * sy + dy * cy
    dx_f = dx_y * cp + dz * sp
    dz_f = -dx_y * sp + dz * cp
    return torch.atan2(dy_y, dx_f), torch.atan2(dz_f, torch.sqrt(dx_f * dx_f + dy_y * dy_y)), dx_f, dist


def calculate_gdop_score(pos_i: np.ndarray, pos_j: np.ndarray) -> float:
    target = (pos_i + pos_j) / 2.0
    vi, vj = target - pos_i, target - pos_j
    ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
    if ni == 0.0 or nj == 0.0:
        return 0.0
    cos_theta = float(np.dot(vi, vj) / (ni * nj))
    return float(np.sqrt(1.0 - min(cos_theta**2, 1.0)))


def _ensure_positions_array(positions: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(positions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 5:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 5 - arr.shape[1]), dtype=np.float32)], axis=1)
    return arr[:, :5]


def _select_spread_spawns(valid_spawns: list[tuple[int, int]], num_agents: int) -> list[tuple[int, int]]:
    if num_agents <= 0 or not valid_spawns:
        return []
    pts = np.asarray(valid_spawns, dtype=np.float32)
    n = len(pts)
    if num_agents >= n:
        return list(valid_spawns)
    first_idx = random.randrange(n)
    chosen = [first_idx]
    chosen_mask = np.zeros(n, dtype=bool)
    chosen_mask[first_idx] = True
    diff = pts - pts[first_idx]
    min_dist = np.sum(diff * diff, axis=1)
    min_dist[chosen_mask] = -1.0
    for _ in range(1, num_agents):
        idx = int(np.argmax(min_dist))
        chosen.append(idx)
        chosen_mask[idx] = True
        diff = pts - pts[idx]
        dist = np.sum(diff * diff, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen_mask] = -1.0
    return [valid_spawns[i] for i in chosen]


def _subsample_spawns_by_spacing(valid_spawns: list[tuple[int, int]], geometry: MapGeometry, step_m: float) -> list[tuple[int, int]]:
    if not valid_spawns:
        return []
    step_m = float(step_m)
    if step_m <= geometry.grid_size + 1e-9:
        return list(valid_spawns)
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for gx, gy in valid_spawns:
        wx = (gx + 0.5) * geometry.grid_size
        wy = (gy + 0.5) * geometry.grid_size
        key = (int(np.floor(wx / step_m)), int(np.floor(wy / step_m)))
        if key in seen:
            continue
        seen.add(key)
        selected.append((gx, gy))
    return selected


class LegacyMAPPOEnv:
    def __init__(self, scenario: Scenario, lidar_profile: str, config: ExperimentConfig, params: LegacyMAPPOParams):
        self.scenario = scenario
        self.config = config
        self.params = params
        self.profile = get_lidar_profile(lidar_profile, config)
        self.geometry = MapGeometry(scenario)
        self.cols = int(self.geometry.map_len / self.geometry.grid_size)
        self.rows = int(self.geometry.map_width / self.geometry.grid_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = bake_visibility_map(config, scenario, self.profile)
        self.target_points_np = np.asarray(self.data["target_points"], dtype=np.float32)
        self.target_weights_np = np.asarray(
            self.data.get("target_weights", np.ones(len(self.target_points_np), dtype=np.float32) * float(params.score_base_voxel)),
            dtype=np.float32,
        )
        self.target_points = torch.tensor(self.target_points_np, dtype=torch.float32, device=self.device)
        self.target_weights = torch.tensor(self.target_weights_np, dtype=torch.float32, device=self.device)
        self.total_voxels = len(self.target_points_np)
        self.total_target_voxels = self.total_voxels
        self.zs = np.asarray(self.data.get("zs", params.z_values), dtype=np.float32)
        self.initial_z_idx = len(self.zs) // 2 if len(self.zs) else 0
        self.initial_z = float(self.zs[self.initial_z_idx]) if len(self.zs) else float(params.z_min)
        self.quadrants = self._get_separated_spawn_points()
        self.counts: dict[tuple[int, int, int], int] = dict(self.data.get("counts", {}))
        self._visible_idx_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self.max_voxel_est = max(float(self.data.get("max_voxel_est", params.max_voxel_est)), 1.0)
        params.max_voxel_est = self.max_voxel_est
        self.prev_score = 0.0
        self.best_cov = 0.0
        self._prev_uncovered_dists: torch.Tensor | None = None
        self.h_fov_rad = np.deg2rad(self.profile.h_fov_deg) / 2.0
        self.v_fov_rad = np.deg2rad(self.profile.v_fov_deg) / 2.0
        self.min_dist = 0.5
        centers = [(x, y) for x in self.geometry.road_x_centers for y in self.geometry.road_y_centers]
        self.align_centers = np.asarray(centers, dtype=np.float32) if centers else np.empty((0, 2), dtype=np.float32)
        self.road_x_centers = np.asarray(getattr(self.geometry, "road_x_centers", []), dtype=np.float32)
        self.road_y_centers = np.asarray(getattr(self.geometry, "road_y_centers", []), dtype=np.float32)
        if self.total_voxels > 0:
            params.uncovered_centroid = np.mean(self.target_points_np[:, :2], axis=0)
        else:
            params.uncovered_centroid = None

    def _get_separated_spawn_points(self) -> list[list[tuple[int, int]]]:
        quadrants: list[list[tuple[int, int]]] = [[], [], [], []]
        cx = self.cols // 2
        cy = self.rows // 2
        for x in range(self.cols):
            for y in range(self.rows):
                wx = (x + 0.5) * self.geometry.grid_size
                wy = (y + 0.5) * self.geometry.grid_size
                if self.geometry.is_installable(wx, wy):
                    if x >= cx and y >= cy:
                        quadrants[0].append((x, y))
                    elif x < cx and y >= cy:
                        quadrants[1].append((x, y))
                    elif x < cx and y < cy:
                        quadrants[2].append((x, y))
                    else:
                        quadrants[3].append((x, y))
        return quadrants

    def _nearest_z_index(self, z: float) -> int | None:
        if self.zs.size == 0:
            return None
        return int(np.argmin(np.abs(self.zs - float(z))))

    def visible_indices(self, gx: int, gy: int, z_idx: int) -> np.ndarray:
        key = (int(gx), int(gy), int(z_idx))
        cached = self._visible_idx_cache.get(key)
        if cached is not None:
            return cached
        if self.total_voxels == 0 or z_idx < 0 or z_idx >= len(self.zs):
            out = np.array([], dtype=np.int64)
            self._visible_idx_cache[key] = out
            self.counts[key] = 0
            return out
        idx = unpack_visible_indices(self.data, key)
        self._visible_idx_cache[key] = idx
        self.counts[key] = int(self.counts.get(key, len(idx)))
        return idx

    def rank_spawns_by_visibility(self, valid_spawns: list[tuple[int, int]], top_k: int) -> list[tuple[int, int]]:
        if not valid_spawns:
            return []
        counts_xy: dict[tuple[int, int], int] = {}
        for gx, gy in valid_spawns:
            best = 0
            for z_idx in range(len(self.zs)):
                best = max(best, int(len(self.visible_indices(gx, gy, z_idx))))
            counts_xy[(gx, gy)] = best
        ranked = sorted(valid_spawns, key=lambda p: counts_xy.get(p, 0), reverse=True)
        if top_k and top_k > 0:
            ranked = ranked[: min(top_k, len(ranked))]
        return ranked if ranked else list(valid_spawns)

    def evaluate_positions(self, positions: np.ndarray) -> tuple[float, float]:
        positions = _ensure_positions_array(positions)
        if self.total_voxels == 0 or positions.size == 0:
            return 0.0, 0.0
        density_map = np.zeros(self.total_voxels, dtype=np.float32)
        for pos in positions:
            x, y, z, yaw, pitch = map(float, pos[:5])
            gx = int(np.floor(x / self.geometry.grid_size))
            gy = int(np.floor(y / self.geometry.grid_size))
            z_idx = self._nearest_z_index(z)
            if z_idx is None:
                continue
            idx = self.visible_indices(gx, gy, z_idx)
            if idx.size == 0:
                continue
            pts = self.target_points_np[idx]
            dx = pts[:, 0] - x
            dy = pts[:, 1] - y
            dz = pts[:, 2] - z
            azimuth, elevation, forward, dist = compute_view_angles_numpy(dx, dy, dz, yaw, pitch)
            if self.h_fov_rad >= (np.pi - 1e-6):
                front_mask = np.ones_like(dist, dtype=bool)
            else:
                front_mask = forward > self.min_dist
            fov_mask = (np.abs(azimuth) <= self.h_fov_rad) & (np.abs(elevation) <= self.v_fov_rad) & front_mask & (dist <= self.profile.max_range_m)
            valid_idx = idx[fov_mask]
            if len(valid_idx) > 0:
                np.add.at(density_map, valid_idx, 1.0)
        cur_score = float(np.sum(np.minimum(density_map / float(self.params.road_detection), 1.0) * self.target_weights_np))
        cov = (np.count_nonzero(density_map >= float(self.params.road_detection)) / self.total_voxels) * 100.0
        return float(cov), cur_score

    def calculate_reward(self, agents: Sequence["PPOAgent"]) -> tuple[float, np.ndarray, dict[str, float]]:
        num_agents = len(agents)
        if len(self._visible_idx_cache) > 200000:
            self._visible_idx_cache.clear()
        agent_pos_list = [[(ag.x + 0.5) * self.geometry.grid_size, (ag.y + 0.5) * self.geometry.grid_size, ag.z] for ag in agents]
        with torch.no_grad():
            agent_pos_tensor = torch.tensor(agent_pos_list, dtype=torch.float32, device=self.device)
            agent_xy_tensor = agent_pos_tensor[:, :2]
            density_map = torch.zeros(self.total_voxels, dtype=torch.float32, device=self.device)
            agent_view_masks: list[torch.Tensor | None] = []
            for i, ag in enumerate(agents):
                cached = self.visible_indices(ag.x, ag.y, ag.z_idx)
                if cached.size == 0:
                    agent_view_masks.append(None)
                    continue
                idx_tensor = torch.from_numpy(cached).to(self.device, non_blocking=True)
                points_k = self.target_points[idx_tensor]
                d_vec = points_k - agent_pos_tensor[i]
                azimuth, elevation, forward, dist = _compute_view_angles_torch(d_vec[:, 0], d_vec[:, 1], d_vec[:, 2], ag.yaw, ag.pitch)
                if self.h_fov_rad >= (np.pi - 1e-6):
                    front_mask = torch.ones_like(dist, dtype=torch.bool)
                else:
                    front_mask = forward > self.min_dist
                fov_mask = (torch.abs(azimuth) <= self.h_fov_rad) & (torch.abs(elevation) <= self.v_fov_rad) & front_mask & (dist <= self.profile.max_range_m)
                final_indices = idx_tensor[fov_mask]
                if final_indices.numel() > 0:
                    density_map.index_add_(0, final_indices, torch.ones_like(final_indices, dtype=torch.float32))
                    agent_view_masks.append(final_indices)
                else:
                    agent_view_masks.append(None)
            score_map = torch.clamp(density_map / float(self.params.road_detection), max=1.0)
            cur_score = torch.sum(score_map * self.target_weights).item()
            overlap_map = torch.clamp(density_map - float(self.params.road_detection), min=0.0)
            overlap_pen = torch.sum(overlap_map * self.target_weights).item()
            delta_score = cur_score - self.prev_score
            delta_score_reward = delta_score * 25.0
            abs_cov_reward = cur_score * self.params.w_abs_coverage
            overlap_penalty = overlap_pen * self.params.w_overlap_pen
            team_reward = delta_score_reward + abs_cov_reward - overlap_penalty
            self.prev_score = cur_score
            covered_mask = density_map >= float(self.params.road_detection)
            covered_voxels = covered_mask.sum().item()
            cov = (covered_voxels / max(self.total_target_voxels, 1)) * 100.0
            indiv_rewards = np.zeros(num_agents, dtype=np.float32)
            scale = float(self.params.dist_reward_scale) or 1.0
            dist_bonus = 0.0
            dist_penalty = 0.0
            uncovered_bonus = 0.0
            outward_penalty = 0.0
            marginal_gain = 0.0
            if num_agents > 1:
                dists = torch.cdist(agent_xy_tensor, agent_xy_tensor, p=2)
                eye_mask = torch.eye(num_agents, device=self.device, dtype=torch.bool)
                min_dists = dists.masked_fill(eye_mask, float("inf")).min(dim=1).values
                far_bonus = torch.clamp(min_dists / scale, min=0.0, max=1.0) * self.params.w_far_dist
                indiv_rewards += far_bonus.detach().cpu().numpy()
                dist_bonus = float(far_bonus.sum().item())
                too_close = min_dists < 5.0
                if too_close.any():
                    penalty = -5.0 * self.params.w_far_dist
                    close_mask = too_close.detach().cpu().numpy()
                    indiv_rewards[close_mask] += penalty
                    dist_penalty = float(penalty * np.count_nonzero(close_mask))
            if (~covered_mask).any().item():
                uncovered_points = self.target_points[~covered_mask][:, :2]
                min_uncovered = torch.cdist(agent_xy_tensor, uncovered_points, p=2).min(dim=1).values
                prev = self._prev_uncovered_dists
                if prev is None or prev.numel() != num_agents:
                    self._prev_uncovered_dists = min_uncovered.detach()
                else:
                    delta = (prev - min_uncovered).clamp(min=-scale, max=scale)
                    bonus = delta * self.params.w_uncovered_move
                    indiv_rewards += bonus.detach().cpu().numpy()
                    uncovered_bonus = float(bonus.sum().item())
                    self._prev_uncovered_dists = min_uncovered.detach()
                if uncovered_points.numel() > 0:
                    self.params.uncovered_centroid = torch.mean(uncovered_points, dim=0).detach().cpu().numpy()
            else:
                self.params.uncovered_centroid = None
            if self.params.edge_margin_m > 0.0 and self.params.w_outward_pen > 0.0:
                center = np.array([self.geometry.map_len * 0.5, self.geometry.map_width * 0.5], dtype=np.float32)
                edge_margin_safe = max(float(self.params.edge_margin_m), 1e-6)
                for i, ag in enumerate(agents):
                    wx = (ag.x + 0.5) * self.geometry.grid_size
                    wy = (ag.y + 0.5) * self.geometry.grid_size
                    dist_edge = min(wx, wy, self.geometry.map_len - wx, self.geometry.map_width - wy)
                    if dist_edge > self.params.edge_margin_m:
                        continue
                    vec = center - np.array([wx, wy], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm < 1e-6:
                        continue
                    vec /= norm
                    heading = _heading_vector_xy(ag.yaw)
                    dot = float(np.dot(heading, vec))
                    if dot >= 0.0:
                        continue
                    edge_factor = (self.params.edge_margin_m - dist_edge) / edge_margin_safe
                    penalty = (-dot) * edge_factor * self.params.w_outward_pen
                    if penalty > 0.0:
                        indiv_rewards[i] -= penalty
                        outward_penalty -= penalty
            if agent_view_masks and self.params.w_marginal_gain > 0.0:
                unique_flags = density_map == float(self.params.road_detection)
                for i, idx_tensor in enumerate(agent_view_masks):
                    if idx_tensor is None or idx_tensor.numel() == 0:
                        continue
                    unique_idx = idx_tensor[unique_flags[idx_tensor]]
                    if unique_idx.numel() == 0:
                        continue
                    unique_score = torch.sum(self.target_weights[unique_idx]).item()
                    bonus = unique_score * self.params.w_marginal_gain
                    indiv_rewards[i] += bonus
                    marginal_gain += bonus
        return float(team_reward), indiv_rewards, {
            "score": float(cur_score),
            "cov": float(cov),
            "overlap_q": 0.0,
            "over_pen": float(overlap_pen),
            "team_reward": float(team_reward),
            "delta_score": float(delta_score),
            "delta_score_reward": float(delta_score_reward),
            "abs_cov_reward": float(abs_cov_reward),
            "overlap_penalty": float(overlap_penalty),
            "dist_bonus": float(dist_bonus),
            "dist_penalty": float(dist_penalty),
            "uncovered_bonus": float(uncovered_bonus),
            "outward_penalty": float(outward_penalty),
            "marginal_gain": float(marginal_gain),
        }


class MAPPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, global_dim: int, local_map_size: int, pose_dim: int = 5, hint_dim: int = HINT_DIM):
        super().__init__()
        self.pose_dim = pose_dim
        self.hint_dim = hint_dim
        self.local_map_size = local_map_size
        self.local_flat = local_map_size * local_map_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        fusion_dim = (32 * local_map_size * local_map_size) + pose_dim + hint_dim
        self.actor_body = nn.Sequential(nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.actor_head = nn.Linear(128, action_dim)
        self.critic = nn.Sequential(nn.Linear(global_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def _encode_actor(self, state: torch.Tensor) -> torch.Tensor:
        pose = state[:, : self.pose_dim]
        local_map = state[:, self.pose_dim : self.pose_dim + self.local_flat]
        local_map = local_map.view(-1, 1, self.local_map_size, self.local_map_size)
        hint = state[:, self.pose_dim + self.local_flat :]
        return self.actor_body(torch.cat([pose, hint, self.cnn(local_map)], dim=1))

    def act(self, state: torch.Tensor, mask: torch.Tensor) -> tuple[int, torch.Tensor]:
        logits = self.actor_head(self._encode_actor(state)).masked_fill(mask == 0, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def evaluate(self, state: torch.Tensor, global_state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_head(self._encode_actor(state)).masked_fill(mask == 0, -1e9)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), self.critic(global_state), dist.entropy()


class SharedMAPPO:
    def __init__(self, state_dim: int, action_dim: int, global_dim: int, local_map_size: int, params: LegacyMAPPOParams, device: torch.device):
        self.params = params
        self.device = device
        self.policy = MAPPOActorCritic(state_dim, action_dim, global_dim, local_map_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=params.ppo_lr)
        self.policy_old = MAPPOActorCritic(state_dim, action_dim, global_dim, local_map_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer: dict[str, list[Any]] = {k: [] for k in ["s", "g", "a", "m", "lp", "r", "t"]}
        self.entropy_coef = params.ppo_entropy_coef

    def act(self, state: torch.Tensor, mask: torch.Tensor, deterministic: bool = False) -> tuple[int, torch.Tensor]:
        if deterministic:
            with torch.no_grad():
                logits = self.policy_old.actor_head(self.policy_old._encode_actor(state.unsqueeze(0))).squeeze(0)
                logits = logits.masked_fill(mask == 0, -1e9)
                action = int(torch.argmax(logits).item())
            return action, torch.tensor(0.0, device=self.device).reshape(1)
        with torch.no_grad():
            action, logp = self.policy_old.act(state.unsqueeze(0), mask.unsqueeze(0))
        if self.params.ppo_random_action_prob > 0 and random.random() < self.params.ppo_random_action_prob:
            valid_actions = torch.nonzero(mask > 0, as_tuple=False).flatten()
            if valid_actions.numel() > 0:
                action = int(valid_actions[torch.randint(len(valid_actions), (1,), device=self.device)].item())
                with torch.no_grad():
                    logits = self.policy_old.actor_head(self.policy_old._encode_actor(state.unsqueeze(0))).squeeze(0)
                    logits = logits.masked_fill(mask == 0, -1e9)
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(torch.tensor(action, device=self.device))
        return action, logp.reshape(1)

    def store(self, state: torch.Tensor, mask: torch.Tensor, action: int, logp: torch.Tensor, reward: float, global_state: torch.Tensor, done: bool = False) -> None:
        scaled_reward = reward * 0.1
        if not np.isfinite(scaled_reward):
            print("[WARN] Non-finite reward detected, skip step")
            return
        scaled_reward = float(np.clip(scaled_reward, -10.0, 10.0))
        self.buffer["s"].append(state.detach())
        self.buffer["g"].append(global_state.detach().to(self.device))
        self.buffer["a"].append(torch.tensor(action, device=self.device))
        self.buffer["m"].append(mask.detach())
        self.buffer["lp"].append(logp.detach())
        self.buffer["r"].append(scaled_reward)
        self.buffer["t"].append(done)
        if len(self.buffer["r"]) >= self.params.ppo_update_timestep:
            self.train_net()

    def train_net(self) -> None:
        if not self.buffer["r"]:
            return
        rewards = torch.tensor(self.buffer["r"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer["t"], dtype=torch.float32, device=self.device)
        if not torch.isfinite(rewards).all():
            print("[WARN] Non-finite rewards detected, skip update")
            self.clear()
            return
        old_s = torch.stack(self.buffer["s"]).detach()
        old_g = torch.stack(self.buffer["g"]).detach()
        old_a = torch.stack(self.buffer["a"]).detach()
        old_m = torch.stack(self.buffer["m"]).detach()
        old_lp = torch.stack(self.buffer["lp"]).detach()
        if not (torch.isfinite(old_s).all() and torch.isfinite(old_g).all()):
            print("[WARN] Non-finite states detected, skip update")
            self.clear()
            return
        with torch.no_grad():
            old_values = self.policy_old.critic(old_g).squeeze()
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + (self.params.ppo_gamma * next_value * mask) - old_values[t]
            gae = delta + (self.params.ppo_gamma * self.params.ppo_gae_lambda * mask * gae)
            advantages[t] = gae
            next_value = old_values[t]
        returns = advantages + old_values
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        batch_size = len(rewards)
        mini_batch = min(self.params.ppo_minibatch_size, batch_size)
        for _ in range(self.params.ppo_k_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch):
                mb = indices[start : start + mini_batch]
                logp, val, ent = self.policy.evaluate(old_s[mb], old_g[mb], old_a[mb], old_m[mb])
                if not (torch.isfinite(logp).all() and torch.isfinite(val).all() and torch.isfinite(ent).all()):
                    print("[WARN] Non-finite policy outputs, skip minibatch")
                    continue
                if not torch.isfinite(old_lp[mb]).all():
                    print("[WARN] Non-finite old log-probs, skip minibatch")
                    continue
                logp_diff = (logp - old_lp[mb]).clamp(-20.0, 20.0)
                ratio = torch.exp(logp_diff)
                adv = advantages[mb]
                if not torch.isfinite(adv).all():
                    print("[WARN] Non-finite advantages, skip minibatch")
                    continue
                policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - self.params.ppo_eps_clip, 1 + self.params.ppo_eps_clip) * adv).mean()
                val = val.squeeze()
                if self.params.ppo_value_clip > 0:
                    v_old = old_values[mb]
                    v_clip = v_old + torch.clamp(val - v_old, -self.params.ppo_value_clip, self.params.ppo_value_clip)
                    value_loss = 0.5 * torch.max((val - returns[mb]) ** 2, (v_clip - returns[mb]) ** 2).mean()
                else:
                    value_loss = 0.5 * nn.MSELoss()(val, returns[mb])
                loss = policy_loss + value_loss - (self.entropy_coef * ent.mean())
                if not torch.isfinite(loss):
                    print("[WARN] Non-finite loss, skip minibatch")
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()
        self.entropy_coef = max(self.params.ppo_entropy_min, self.entropy_coef * self.params.ppo_entropy_decay)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear()

    def clear(self) -> None:
        for key in self.buffer:
            self.buffer[key] = []


class PPOAgent:
    def __init__(self, env: LegacyMAPPOEnv, valid_spawns: list[tuple[int, int]], shared: SharedMAPPO):
        self.env = env
        self.params = env.params
        self.geometry = env.geometry
        self.x, self.y = random.choice(valid_spawns)
        self.z_idx = env.initial_z_idx
        self.z = env.initial_z
        self.yaw = self._sample_allowed_yaw()
        self.pitch = self._sample_allowed_pitch()
        self.map_w = env.cols
        self.map_h = env.rows
        self.shared = shared
        self.device = shared.device
        self.action_dim = ACTION_DIM
        self._align_centers = env.align_centers
        self._road_x_centers = env.road_x_centers
        self._road_y_centers = env.road_y_centers
        self._max_map_dim = max(env.geometry.map_len, env.geometry.map_width)
        self._clamp_yaw_to_target()
        self._clamp_pitch_to_range()
        self._clamp_z_to_range()
        self.temp_s: torch.Tensor | None = None
        self.temp_m: torch.Tensor | None = None
        self.temp_a: int | None = None
        self.temp_lp: torch.Tensor | None = None

    def _xy_step_cells(self) -> int:
        step_m = max(float(self.params.xy_step_m), self.geometry.grid_size)
        return max(1, int(round(step_m / self.geometry.grid_size)))

    def _relative_yaw_range(self) -> tuple[float, float]:
        lo, hi = sorted([float(self.params.yaw_axis_min_deg), float(self.params.yaw_axis_max_deg)])
        return lo - 90.0, hi - 90.0

    def _world_xy(self) -> tuple[float, float]:
        return (self.x + 0.5) * self.geometry.grid_size, (self.y + 0.5) * self.geometry.grid_size

    def _get_road_normal_yaw(self) -> float:
        wx, wy = self._world_xy()
        return float(self.geometry.road_normal_yaw_deg(wx, wy))

    def _sample_allowed_yaw(self) -> float:
        wx, wy = self._world_xy()
        yaws = np.asarray(
            allowed_mount_yaws_deg(wx, wy, self.geometry, self.params.yaw_step_deg, self.params.yaw_axis_min_deg, self.params.yaw_axis_max_deg),
            dtype=np.float32,
        )
        return float(random.choice(yaws)) if yaws.size else float(self._get_road_normal_yaw())

    def _sample_allowed_pitch(self) -> float:
        pmin, pmax = sorted([float(self.params.min_pitch), float(self.params.max_pitch)])
        step = float(self.params.pitch_step_deg)
        if step > 0.0 and (pmax - pmin) > 1e-9:
            vals = np.arange(pmin, pmax + 1e-6, step, dtype=np.float32)
            if vals.size > 0:
                return float(random.choice(vals))
        return float(0.5 * (pmin + pmax))

    def _compute_target_vector(self, wx: float, wy: float) -> tuple[np.ndarray, float]:
        if self._road_x_centers.size > 0 and self._road_y_centers.size > 0:
            dx = np.abs(self._road_x_centers - wx)
            dy = np.abs(self._road_y_centers - wy)
            ix = int(np.argmin(dx))
            iy = int(np.argmin(dy))
            vec_x = np.array([self._road_x_centers[ix] - wx, 0.0], dtype=np.float32)
            vec_y = np.array([0.0, self._road_y_centers[iy] - wy], dtype=np.float32)
            eps = 1e-6
            wx_w = 1.0 / (dx[ix] + eps)
            wy_w = 1.0 / (dy[iy] + eps)
            target_vec = (vec_x * (wx_w / (wx_w + wy_w))) + (vec_y * (wy_w / (wx_w + wy_w)))
            return target_vec, float(np.linalg.norm(target_vec))
        if self._align_centers.size > 0:
            vecs = self._align_centers - np.array([wx, wy], dtype=np.float32)
            dists = np.linalg.norm(vecs, axis=1)
            idx = int(np.argmin(dists))
            return vecs[idx], float(dists[idx])
        return np.zeros(2, dtype=np.float32), 0.0

    def _clamp_yaw_to_target(self) -> None:
        base_yaw = self._get_road_normal_yaw()
        rel = _relative_yaw(self.yaw, base_yaw)
        min_rel, max_rel = self._relative_yaw_range()
        rel = float(np.clip(rel, min_rel, max_rel))
        step = float(self.params.yaw_step_deg)
        if step > 0.0:
            rel = round(rel / step) * step
            rel = float(np.clip(rel, min_rel, max_rel))
        self.yaw = (base_yaw + rel) % 360.0

    def _clamp_pitch_to_range(self) -> None:
        pmin, pmax = sorted([float(self.params.min_pitch), float(self.params.max_pitch)])
        self.pitch = float(np.clip(self.pitch, pmin, pmax))

    def _clamp_z_to_range(self) -> None:
        zs = self.params.z_values
        if zs.size == 0:
            return
        self.z_idx = int(np.clip(self.z_idx, 0, len(zs) - 1))
        self.z = float(zs[self.z_idx])

    def get_state_and_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_yaw = self._get_road_normal_yaw()
        relative_yaw = _relative_yaw(self.yaw, base_yaw)
        z_norm = self.z_idx / max(self.params.z_idx_max, 1)
        pitch_norm = (self.pitch - self.params.min_pitch) / (self.params.max_pitch - self.params.min_pitch + 1e-5)
        pos = [self.x / self.map_w, self.y / self.map_h, z_norm, relative_yaw / 180.0, pitch_norm]
        local_map: list[float] = []
        max_voxel_est = max(float(self.env.max_voxel_est), 1.0)
        half = self.params.local_map_size // 2
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                key = (self.x + dx * 2, self.y + dy * 2, self.z_idx)
                if 0 <= key[0] < self.map_w and 0 <= key[1] < self.map_h:
                    val = len(self.env.visible_indices(*key)) / max_voxel_est
                else:
                    val = 0.0
                local_map.append(float(val))
        wx, wy = self._world_xy()
        target_vec, dist = self._compute_target_vector(wx, wy)
        if dist > 1e-6:
            to_target = target_vec / dist
        else:
            dist = 0.0
            to_target = np.zeros(2, dtype=np.float32)
        dist_norm = dist / self._max_map_dim if self._max_map_dim > 0 else 0.0
        heading = _heading_vector_xy(self.yaw)
        align = float(np.dot(to_target, heading)) if np.linalg.norm(to_target) > 0 else 0.0
        unc_centroid = self.params.uncovered_centroid
        if unc_centroid is not None:
            dx_u = float(unc_centroid[0] - wx)
            dy_u = float(unc_centroid[1] - wy)
            dist_u = float(np.hypot(dx_u, dy_u))
            to_uncovered = [dx_u / dist_u, dy_u / dist_u] if dist_u > 1e-6 else [0.0, 0.0]
        else:
            dist_u = 0.0
            to_uncovered = [0.0, 0.0]
        dist_u_norm = dist_u / self._max_map_dim if self._max_map_dim > 0 else 0.0
        global_hint = [float(to_target[0]), float(to_target[1]), float(dist_norm), align, float(to_uncovered[0]), float(to_uncovered[1]), float(dist_u_norm)]
        state = torch.tensor(pos + local_map + global_hint, dtype=torch.float32, device=self.device)
        mask = torch.ones(self.action_dim, dtype=torch.float32, device=self.device)
        step_cells = self._xy_step_cells()
        moves = [(step_cells, 0), (-step_cells, 0), (0, step_cells), (0, -step_cells)]
        for i, (dx, dy) in enumerate(moves):
            nx, ny = self.x + dx, self.y + dy
            wx_n = (nx + 0.5) * self.geometry.grid_size
            wy_n = (ny + 0.5) * self.geometry.grid_size
            if not (0 <= nx < self.map_w and 0 <= ny < self.map_h) or not self.geometry.is_installable(wx_n, wy_n, self.z):
                mask[i + 1] = 0
        rel = _relative_yaw(self.yaw, self._get_road_normal_yaw())
        min_rel, max_rel = self._relative_yaw_range()
        if rel + self.params.yaw_step_deg > max_rel + 1e-6:
            mask[5] = 0
        if rel - self.params.yaw_step_deg < min_rel - 1e-6:
            mask[6] = 0
        if self.params.pitch_step_deg > 0:
            if self.pitch + self.params.pitch_step_deg > float(self.params.max_pitch) + 1e-6:
                mask[7] = 0
            if self.pitch - self.params.pitch_step_deg < float(self.params.min_pitch) - 1e-6:
                mask[8] = 0
        zs = self.params.z_values
        if zs.size <= 1:
            mask[9] = 0
            mask[10] = 0
        else:
            if self.z_idx + 1 >= len(zs):
                mask[9] = 0
            if self.z_idx - 1 < 0:
                mask[10] = 0
        return state, mask

    def step(self, deterministic: bool = False, store_buffer: bool = True) -> tuple[np.ndarray, int, tuple[int, int, int]]:
        state, mask = self.get_state_and_mask()
        action, logp = self.shared.act(state, mask, deterministic=deterministic)
        if store_buffer:
            self.temp_s, self.temp_m, self.temp_a, self.temp_lp = state, mask, action, logp
        step_cells = self._xy_step_cells()
        if action == 1:
            self.x += step_cells
        elif action == 2:
            self.x -= step_cells
        elif action == 3:
            self.y += step_cells
        elif action == 4:
            self.y -= step_cells
        elif action == 5:
            self.yaw += self.params.yaw_step_deg
        elif action == 6:
            self.yaw -= self.params.yaw_step_deg
        elif action == 7:
            self.pitch += self.params.pitch_step_deg
        elif action == 8:
            self.pitch -= self.params.pitch_step_deg
        elif action == 9:
            self.z_idx += 1
        elif action == 10:
            self.z_idx -= 1
        self._clamp_yaw_to_target()
        self._clamp_pitch_to_range()
        self._clamp_z_to_range()
        return state.detach().cpu().numpy(), action, (self.x, self.y, self.z_idx)

    def update(self, reward: float, global_state: torch.Tensor) -> None:
        if self.temp_s is None or self.temp_m is None or self.temp_a is None or self.temp_lp is None:
            return
        self.shared.store(self.temp_s, self.temp_m, self.temp_a, self.temp_lp, reward, global_state)

    def pose(self) -> list[float]:
        return [(self.x + 0.5) * self.geometry.grid_size, (self.y + 0.5) * self.geometry.grid_size, self.z, self.yaw, self.pitch]


def _allowed_yaws(wx: float, wy: float, geometry: MapGeometry, params: LegacyMAPPOParams) -> np.ndarray:
    return np.asarray(allowed_mount_yaws_deg(wx, wy, geometry, params.yaw_step_deg, params.yaw_axis_min_deg, params.yaw_axis_max_deg), dtype=np.float32)


def _optimize_yaws(env: LegacyMAPPOEnv, positions: np.ndarray) -> tuple[np.ndarray, float]:
    pos = _ensure_positions_array(positions)
    if pos.size == 0:
        return pos, 0.0
    best_cov, _ = env.evaluate_positions(pos)
    for idx in range(pos.shape[0]):
        wx, wy = pos[idx, 0], pos[idx, 1]
        yaws = _allowed_yaws(wx, wy, env.geometry, env.params)
        if yaws.size == 0:
            continue
        best_yaw = pos[idx, 3]
        best_cov_local = best_cov
        for yaw in yaws:
            if abs(float(yaw) - float(pos[idx, 3])) < 1e-6:
                continue
            candidate = pos.copy()
            candidate[idx, 3] = float(yaw)
            cov, _ = env.evaluate_positions(candidate)
            if cov > best_cov_local + 1e-6:
                best_cov_local = cov
                best_yaw = float(yaw)
        pos[idx, 3] = best_yaw
        best_cov = best_cov_local
    return pos, float(best_cov)


def _save_train_log(path: Path, logs: dict[str, list[float] | list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        episode=np.array(logs["episode"], dtype=np.int32),
        coverage=np.array(logs["coverage"], dtype=np.float32),
        coverage_pct=np.array(logs["coverage"], dtype=np.float32),
        score=np.array(logs["score"], dtype=np.float32),
        reward=np.array(logs["reward"], dtype=np.float32),
        overlap_bonus=np.array(logs["overlap_bonus"], dtype=np.float32),
        team_reward=np.array(logs["team_reward"], dtype=np.float32),
        delta_score_reward=np.array(logs["delta_score_reward"], dtype=np.float32),
        abs_cov_reward=np.array(logs["abs_cov_reward"], dtype=np.float32),
        overlap_penalty=np.array(logs["overlap_penalty"], dtype=np.float32),
        dist_bonus=np.array(logs["dist_bonus"], dtype=np.float32),
        dist_penalty=np.array(logs["dist_penalty"], dtype=np.float32),
        uncovered_bonus=np.array(logs["uncovered_bonus"], dtype=np.float32),
        outward_penalty=np.array(logs["outward_penalty"], dtype=np.float32),
        marginal_gain=np.array(logs["marginal_gain"], dtype=np.float32),
        indiv_reward_mean=np.array(logs["indiv_reward_mean"], dtype=np.float32),
        eval_episode=np.array(logs["eval_episode"], dtype=np.int32),
        eval_coverage=np.array(logs["eval_coverage"], dtype=np.float32),
        eval_score=np.array(logs["eval_score"], dtype=np.float32),
    )


def _train_one_row(row: MappoPlanRow, config: ExperimentConfig) -> dict[str, Any]:
    random.seed(int(row.seed))
    np.random.seed(int(row.seed))
    torch.manual_seed(int(row.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(row.seed))
    params = _legacy_params(row, config)
    scenario = get_scenario(row.scenario_id)
    env = LegacyMAPPOEnv(scenario, row.lidar_profile, config, params)
    valid_spawns = sum(env.quadrants, [])
    valid_spawns = _subsample_spawns_by_spacing(valid_spawns, env.geometry, params.xy_step_m)
    if not valid_spawns:
        raise RuntimeError(f"No installable spawn points for scenario {row.scenario_id}")
    warm_candidates = env.rank_spawns_by_visibility(valid_spawns, params.spawn_top_k) if params.warm_start_enabled else valid_spawns
    spawn_points = _select_spread_spawns(warm_candidates, params.num_agents)
    if len(spawn_points) < params.num_agents:
        base_spawns = warm_candidates if warm_candidates else valid_spawns
        spawn_points += random.choices(base_spawns, k=params.num_agents - len(spawn_points))
    state_dim = 5 + (params.local_map_size**2) + HINT_DIM
    global_dim = state_dim * params.num_agents
    shared = SharedMAPPO(state_dim, ACTION_DIM, global_dim, params.local_map_size, params, env.device)
    agents = [PPOAgent(env, [spawn_points[i]], shared) for i in range(params.num_agents)]
    base = config.results_dir / "mappo" / f"mappo_{row.lidar_profile.lower()}_{row.output_tag}_id{row.scenario_id}_k{row.budget_k}_t{row.trial}"
    result_npy_path = base.with_suffix(".npy")
    train_log_path = base.with_name(base.name + "_train_log.npz")
    global_best_cov = -1.0
    global_best_score = float("-inf")
    if result_npy_path.exists():
        try:
            previous = np.load(result_npy_path)
            global_best_cov, global_best_score = env.evaluate_positions(previous)
        except Exception:
            global_best_cov = -1.0
            global_best_score = float("-inf")
    best_cov = 0.0
    best_positions: np.ndarray | None = None
    best_saved_cov = float(global_best_cov)
    best_saved_score = float(global_best_score)
    best_eval_cov = -1.0
    logs: dict[str, list[float] | list[int]] = {
        "episode": [],
        "coverage": [],
        "score": [],
        "reward": [],
        "overlap_bonus": [],
        "team_reward": [],
        "delta_score_reward": [],
        "abs_cov_reward": [],
        "overlap_penalty": [],
        "dist_bonus": [],
        "dist_penalty": [],
        "uncovered_bonus": [],
        "outward_penalty": [],
        "marginal_gain": [],
        "indiv_reward_mean": [],
        "eval_episode": [],
        "eval_coverage": [],
        "eval_score": [],
    }
    plateau_count = 0
    plateau_active = False
    base_entropy_coef = float(params.ppo_entropy_coef)
    base_uncovered_move = float(params.w_uncovered_move)
    start_t = time.time()
    det: dict[str, float] = {"score": 0.0, "cov": 0.0}
    for ep in range(1, params.episodes + 1):
        g_states = []
        for ag in agents:
            state, _mask = ag.get_state_and_mask()
            g_states.append(state)
        g_vec = torch.cat(g_states)
        for ag in agents:
            ag.step()
        team_reward, indiv_rewards, det = env.calculate_reward(agents)
        for idx, ag in enumerate(agents):
            ag.update(team_reward + float(indiv_rewards[idx]), g_vec)
        logs["episode"].append(ep)
        logs["coverage"].append(det["cov"])
        logs["score"].append(det["score"])
        logs["reward"].append(team_reward + float(np.mean(indiv_rewards)) if len(indiv_rewards) else team_reward)
        logs["overlap_bonus"].append(det.get("overlap_q", 0.0))
        logs["team_reward"].append(det.get("team_reward", team_reward))
        logs["delta_score_reward"].append(det.get("delta_score_reward", 0.0))
        logs["abs_cov_reward"].append(det.get("abs_cov_reward", 0.0))
        logs["overlap_penalty"].append(det.get("overlap_penalty", 0.0))
        logs["dist_bonus"].append(det.get("dist_bonus", 0.0))
        logs["dist_penalty"].append(det.get("dist_penalty", 0.0))
        logs["uncovered_bonus"].append(det.get("uncovered_bonus", 0.0))
        logs["outward_penalty"].append(det.get("outward_penalty", 0.0))
        logs["marginal_gain"].append(det.get("marginal_gain", 0.0))
        logs["indiv_reward_mean"].append(float(np.mean(indiv_rewards)) if len(indiv_rewards) else 0.0)
        if det["cov"] > best_cov:
            best_cov = float(det["cov"])
            best_positions = np.asarray([ag.pose() for ag in agents], dtype=np.float32)
            if det["cov"] > global_best_cov + 1e-6:
                save_positions_npy(result_npy_path, best_positions)
                global_best_cov = float(det["cov"])
                global_best_score = float(det["score"])
                best_saved_cov = float(global_best_cov)
                best_saved_score = float(global_best_score)
                print(f"Saved best @ Ep {ep} | Cov: {det['cov']:.2f}% | Score: {det['score']:.1f} -> {result_npy_path}")
            plateau_count = 0
            if plateau_active:
                params.w_uncovered_move = base_uncovered_move
                shared.entropy_coef = base_entropy_coef
                plateau_active = False
        else:
            plateau_count += 1
            if (not plateau_active) and params.plateau_patience > 0 and plateau_count >= params.plateau_patience:
                params.w_uncovered_move = base_uncovered_move * params.plateau_uncovered_scale
                shared.entropy_coef = base_entropy_coef * params.plateau_entropy_scale
                plateau_active = True
        if params.eval_interval > 0 and ep % params.eval_interval == 0:
            saved_states = [(ag.x, ag.y, ag.z, ag.z_idx, ag.yaw, ag.pitch) for ag in agents]
            for _ in range(max(0, params.eval_steps)):
                for ag in agents:
                    ag.step(deterministic=True, store_buffer=False)
            eval_positions = np.asarray([ag.pose() for ag in agents], dtype=np.float32)
            eval_cov, eval_score = env.evaluate_positions(eval_positions)
            logs["eval_episode"].append(ep)
            logs["eval_coverage"].append(float(eval_cov))
            logs["eval_score"].append(float(eval_score))
            best_eval_cov = max(best_eval_cov, float(eval_cov))
            if eval_cov > global_best_cov + 1e-6:
                save_positions_npy(result_npy_path, eval_positions)
                global_best_cov = float(eval_cov)
                global_best_score = float(eval_score)
                best_saved_cov = float(global_best_cov)
                best_saved_score = float(global_best_score)
                print(f"Saved best (eval) @ Ep {ep} | Cov: {eval_cov:.2f}% | Score: {eval_score:.1f} -> {result_npy_path}")
            print(f"Eval Ep {ep} | Cov: {eval_cov:.2f}% | Score: {eval_score:.1f}")
            for ag, st in zip(agents, saved_states):
                ag.x, ag.y, ag.z, ag.z_idx, ag.yaw, ag.pitch = st
        if params.print_interval > 0 and ep % params.print_interval == 0:
            print(
                f"Ep {ep} | Cov: {det['cov']:.2f}% | Score: {det['score']:.1f} | "
                f"Team: {det.get('team_reward', team_reward):.2f} "
                f"(dScore*25={det.get('delta_score_reward', 0.0):.2f}, abs={det.get('abs_cov_reward', 0.0):.2f}, "
                f"overlap=-{det.get('overlap_penalty', 0.0):.2f}) | "
                f"Indiv: dist+={det.get('dist_bonus', 0.0):.2f}, dist-={det.get('dist_penalty', 0.0):.2f}, "
                f"uncov={det.get('uncovered_bonus', 0.0):.2f}, out={det.get('outward_penalty', 0.0):.2f}, "
                f"mg={det.get('marginal_gain', 0.0):.2f} | Time: {time.time() - start_t:.0f}s"
            )
            _save_train_log(train_log_path, logs)
            if ep % 500 == 0:
                gc.collect()
    shared.train_net()
    if best_positions is None:
        best_positions = np.asarray([ag.pose() for ag in agents], dtype=np.float32)
    tuned_positions, tuned_cov = _optimize_yaws(env, best_positions)
    if tuned_positions.size > 0:
        if tuned_cov > global_best_cov + 1e-6:
            tuned_score = env.evaluate_positions(tuned_positions)[1]
            save_positions_npy(result_npy_path, tuned_positions)
            global_best_cov = float(tuned_cov)
            global_best_score = float(tuned_score)
            best_saved_cov = float(global_best_cov)
            best_saved_score = float(global_best_score)
            print(f"Saved best (yaw-tuned) | Cov: {tuned_cov:.2f}% -> {result_npy_path}")
        best_cov = max(float(best_cov), float(tuned_cov))
    if not result_npy_path.exists():
        save_positions_npy(result_npy_path, tuned_positions if tuned_positions.size else best_positions)
    final_positions = np.asarray(np.load(result_npy_path), dtype=np.float32)
    _save_train_log(train_log_path, logs)
    evaluation = evaluate_coverage(final_positions, scenario, row.lidar_profile, config)
    profile = get_lidar_profile(row.lidar_profile, config)
    config_hash = compute_config_hash(config, scenario, profile)
    npy_path = save_positions_npy(result_npy_path, final_positions)
    record = build_result_record(
        method="MAPPO",
        scenario_id=row.scenario_id,
        topology=row.topology,
        lidar_profile=row.lidar_profile,
        lidar_model=row.lidar_profile,
        max_range_m=profile.max_range_m,
        budget_k=row.budget_k,
        budget_mode="at_most",
        candidate_mode=f"legacy_grid_xy{config.xy_step_m}_yaw{config.yaw_step_deg}_pitch{config.pitch_step_deg}",
        action_space="legacy_11_action_grid_move_yaw_pitch_z_mappo",
        num_selected=len(final_positions),
        positions_npy=str(npy_path),
        coverage_pct=evaluation.coverage_pct,
        coverage_ratio=evaluation.coverage_ratio,
        covered_count=evaluation.covered_count,
        num_targets=evaluation.num_targets,
        episodes=row.episodes,
        seed=row.seed,
        trial=row.trial,
        mappo_tag=row.output_tag,
        config_hash=config_hash,
        cache_key=evaluation.cache_key,
        train_log_npz=str(train_log_path),
        extra={
            "trainer": "legacy_mappo_native_port",
            "legacy_action_dim": ACTION_DIM,
            "best_train_coverage_pct": float(best_cov),
            "best_eval_coverage_pct": float(best_eval_cov) if best_eval_cov >= 0 else None,
            "best_internal_saved_coverage_pct": float(best_saved_cov),
            "best_internal_saved_score": float(best_saved_score) if np.isfinite(best_saved_score) else None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    save_result_json(base.with_suffix(".json"), record)
    return record


def run_mappo(rows: Sequence[MappoPlanRow], config: ExperimentConfig) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(_train_one_row(row, config))
    return {"records": records}

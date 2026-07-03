"""Single source of truth for experiment config, maps, LiDAR geometry, cache, and coverage."""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VISIBILITY_MODEL = "open3d_raycast_v1"


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def _simple_yaml_load(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)
    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return _simple_yaml_load(path)


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]
    source_path: str = ""
    lidar_profile: str = "VELODYNE64_R30"
    scenario_ids: list[int] = field(default_factory=list)
    k_values: list[int] = field(default_factory=list)
    experiment_tag: str = "default"

    @property
    def candidate(self) -> dict[str, Any]:
        return self.raw.setdefault("candidate", {})

    @property
    def target(self) -> dict[str, Any]:
        return self.raw.setdefault("target", {})

    @property
    def solver(self) -> dict[str, Any]:
        return self.raw.setdefault("solver", {})

    @property
    def mappo(self) -> dict[str, Any]:
        return self.raw.setdefault("mappo", {})

    @property
    def output(self) -> dict[str, Any]:
        return self.raw.setdefault("output", {})

    @property
    def results_dir(self) -> Path:
        return Path(str(self.output.get("results_dir", "results/active")))

    @property
    def cache_dir(self) -> Path:
        return Path(str(self.output.get("cache_dir", "data/cache")))

    @property
    def xy_step_m(self) -> float:
        return float(self.candidate.get("xy_step_m", 2.0))

    @property
    def z_min(self) -> float:
        return float(self.candidate.get("z_min", 3.0))

    @property
    def z_max(self) -> float:
        return float(self.candidate.get("z_max", 6.0))

    @property
    def z_step(self) -> float:
        return float(self.candidate.get("z_step", 1.0))

    @property
    def yaw_step_deg(self) -> float:
        return float(self.candidate.get("yaw_step_deg", 10.0))

    @property
    def yaw_axis_min_deg(self) -> float:
        return float(self.candidate.get("yaw_axis_min_deg", 35.0))

    @property
    def yaw_axis_max_deg(self) -> float:
        return float(self.candidate.get("yaw_axis_max_deg", 145.0))

    @property
    def pitch_min_deg(self) -> float:
        return float(self.candidate.get("pitch_min_deg", -25.0))

    @property
    def pitch_max_deg(self) -> float:
        return float(self.candidate.get("pitch_max_deg", -5.0))

    @property
    def pitch_step_deg(self) -> float:
        return float(self.candidate.get("pitch_step_deg", 2.5))

    @property
    def max_orientations_per_position(self) -> int:
        return int(self.candidate.get("max_orientations_per_position", 0))

    @property
    def solver_name(self) -> str:
        return str(self.solver.get("solver", "gurobi"))

    @property
    def threads(self) -> int:
        return int(self.solver.get("threads", 4))

    @property
    def time_limit_sec(self) -> float:
        return float(self.solver.get("time_limit_sec", 1800))

    @property
    def mip_gap(self) -> float:
        return float(self.solver.get("mip_gap", 0.05))

    @property
    def episodes(self) -> int:
        return int(self.mappo.get("episodes", 10000))

    @property
    def trials(self) -> int:
        return int(self.mappo.get("trials", 3))

    @property
    def seed_base(self) -> int:
        return int(self.mappo.get("seed_base", 0))


def load_config(path: str | Path = "configs/experiment.yaml") -> ExperimentConfig:
    path = Path(path)
    data = load_yaml(path)
    return ExperimentConfig(
        raw=data,
        source_path=str(path),
        lidar_profile=str(data.get("lidar_profile", "VELODYNE64_R30")),
        scenario_ids=[int(v) for v in data.get("scenario_ids", [])],
        k_values=[int(v) for v in data.get("k_values", [])],
        experiment_tag=str(data.get("experiment_tag", "default")),
    )


def config_hash_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "visibility_model": VISIBILITY_MODEL,
        "lidar_profile": config.lidar_profile,
        "candidate": config.candidate,
        "target": config.target,
        "mappo_visibility": {
            "score_base_voxel": float(config.mappo.get("score_base_voxel", 0.01)),
        },
    }


@dataclass(frozen=True)
class Scenario:
    id: int
    topology: str
    road_width: float
    num_lanes: int
    raw: dict[str, Any]

    @property
    def scenario_id(self) -> int:
        return self.id


def load_scenarios(path: str | Path = "data/scenarios/yeongjong_final.json") -> list[Scenario]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("scenarios", data.get("scenario_list", [])) if isinstance(data, dict) else data
    scenarios: list[Scenario] = []
    for idx, row in enumerate(rows or []):
        sid = int(row.get("id", row.get("scenario_id", idx)))
        scenarios.append(
            Scenario(
                id=sid,
                topology=str(row.get("topology", "STRAIGHT")),
                road_width=float(row.get("road_width", 25.0)),
                num_lanes=int(row.get("num_lanes", row.get("lanes", 4))),
                raw=dict(row),
            )
        )
    return scenarios


def get_scenario(scenario_id: int, path: str | Path = "data/scenarios/yeongjong_final.json") -> Scenario:
    for scenario in load_scenarios(path):
        if int(scenario.scenario_id) == int(scenario_id):
            return scenario
    raise KeyError(f"Scenario id not found: {scenario_id}")


def select_scenarios(scenarios: list[Scenario], scenario_ids: Iterable[int]) -> list[Scenario]:
    wanted = {int(v) for v in scenario_ids}
    return [scenario for scenario in scenarios if int(scenario.scenario_id) in wanted]


@dataclass(frozen=True)
class LidarProfile:
    name: str
    h_fov_deg: float
    v_fov_deg: float
    max_range_m: float


def get_lidar_profile(name: str, config: ExperimentConfig | None = None) -> LidarProfile:
    key = str(name).upper()
    profiles = (config.raw.get("lidar_profiles", {}) if config is not None else load_config().raw.get("lidar_profiles", {}))
    if key not in profiles:
        raise KeyError(f"Unknown LiDAR profile: {name}")
    row = profiles[key]
    return LidarProfile(key, float(row["h_fov_deg"]), float(row["v_fov_deg"]), float(row["max_range_m"]))


def is_velodyne_family(name: str) -> bool:
    return str(name).upper().startswith("VELODYNE64")


@dataclass
class MapGeometry:
    scenario: Scenario
    map_len: float = 100.0
    map_width: float = 100.0
    grid_size: float = 0.5
    sidewalk_width: float = 5.0
    building_height: float = 15.0
    median_width: float = 0.5
    road_centers_y: list[float] = field(default_factory=lambda: [50.0])
    road_centers_x: list[float] = field(default_factory=list)
    negative_zones: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scenario.topology in {"CROSS", "T_JUNCTION", "COMPLEX"}:
            self.road_centers_x = [50.0]
        self.half_road = self.scenario.road_width / 2.0
        self.num_lanes = self.scenario.num_lanes
        self.sidewalk_center_offset = self.half_road + self.sidewalk_width / 2.0
        self.road_x_centers = self.road_centers_x
        self.road_y_centers = self.road_centers_y
        self._set_building_zones()

    def _set_building_zones(self) -> None:
        topo = self.scenario.topology
        mid = 50.0
        offset = self.half_road + self.sidewalk_width
        if topo in {"STRAIGHT", "UNKNOWN"}:
            self.negative_zones = [[0, 0, 100, mid - offset], [0, mid + offset, 100, 100]]
        elif topo in {"CROSS", "COMPLEX"}:
            self.negative_zones = [
                [0, 0, mid - offset, mid - offset],
                [mid + offset, 0, 100, mid - offset],
                [0, mid + offset, mid - offset, 100],
                [mid + offset, mid + offset, 100, 100],
            ]
        elif topo == "T_JUNCTION":
            self.negative_zones = [
                [0, mid + offset, 100, 100],
                [0, 0, mid - offset, mid - offset],
                [mid + offset, 0, 100, mid - offset],
            ]

    def is_road(self, x: float, y: float) -> bool:
        topo = self.scenario.topology
        for rc in self.road_centers_x:
            if abs(x - rc) < self.half_road:
                if topo == "T_JUNCTION" and y > 50 + self.half_road:
                    continue
                if self.median_width / 2 < abs(x - rc) < self.half_road:
                    return True
        for rc in self.road_centers_y:
            if abs(y - rc) < self.half_road and self.median_width / 2 < abs(y - rc) < self.half_road:
                return True
        return topo in {"CROSS", "T_JUNCTION", "COMPLEX"} and abs(x - 50.0) < self.half_road and abs(y - 50.0) < self.half_road

    def _is_on_median_line(self, x: float, y: float) -> bool:
        median_half = self.median_width / 2.0 + 1e-6
        offset = min(max(self.half_road * 0.5, 1e-3), self.median_width / 2.0 + 1e-3)
        return any(abs(x - rc) <= median_half and self.is_road(rc + offset, y) for rc in self.road_centers_x) or any(
            abs(y - rc) <= median_half and self.is_road(x, rc + offset) for rc in self.road_centers_y
        )

    def is_installable(self, x: float, y: float, z: float | None = None, tolerance_m: float | None = None) -> bool:
        if any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in self.negative_zones):
            return False
        if self._is_on_median_line(x, y) or self.is_road(x, y):
            return False
        tolerance = self.grid_size / 2.0 if tolerance_m is None else max(float(tolerance_m), self.grid_size / 2.0)
        center_min = self.sidewalk_center_offset - tolerance
        center_max = self.sidewalk_center_offset + tolerance
        offset = min(max(self.half_road * 0.5, 1e-3), self.median_width / 2.0 + 1e-3)
        return any(self.is_road(rc + offset, y) and center_min <= abs(x - rc) <= center_max for rc in self.road_centers_x) or any(
            self.is_road(x, rc + offset) and center_min <= abs(y - rc) <= center_max for rc in self.road_centers_y
        )

    def road_normal_yaw_deg(self, wx: float, wy: float) -> float:
        offset = min(max(self.half_road * 0.5, 1e-3), self.median_width / 2.0 + 1e-3)
        best_h: tuple[float, float | None] = (float("inf"), None)
        for rc in self.road_centers_y:
            if self.is_road(wx, rc + offset):
                best_h = min(best_h, (abs(wy - rc), rc), key=lambda t: t[0])
        best_v: tuple[float, float | None] = (float("inf"), None)
        for rc in self.road_centers_x:
            if self.is_road(rc + offset, wy):
                best_v = min(best_v, (abs(wx - rc), rc), key=lambda t: t[0])
        if best_h[1] is not None and (best_v[1] is None or best_h[0] <= best_v[0]):
            return 90.0 if wy < float(best_h[1]) else 270.0
        if best_v[1] is not None:
            return 0.0 if wx < float(best_v[1]) else 180.0
        return 90.0 if wy < self.map_width * 0.5 else 270.0

    def target_points(self, coverage_z_slices: list[float]) -> np.ndarray:
        cols = int(self.map_len / self.grid_size)
        rows = int(self.map_width / self.grid_size)
        pts: list[list[float]] = []
        for gx in range(cols):
            wx = (gx + 0.5) * self.grid_size
            for gy in range(rows):
                wy = (gy + 0.5) * self.grid_size
                if self.is_road(wx, wy):
                    for wz in coverage_z_slices:
                        pts.append([wx, wy, float(wz)])
        return np.asarray(pts, dtype=np.float32)

    def create_environment_mesh(self):
        try:
            import open3d as o3d  # type: ignore
        except Exception as exc:
            raise RuntimeError("Open3D is required for raycast visibility baking. Install open3d or use an existing valid cache.") from exc

        geoms = []
        base = o3d.geometry.TriangleMesh.create_box(self.map_len, self.map_width, 0.05)
        base.paint_uniform_color([0.1, 0.1, 0.1])
        geoms.append(base)
        deep_yellow = [0.8, 0.5, 0.0]

        def draw_infrastructure(center: float, horizontal: bool = True) -> None:
            length = self.map_len if horizontal else self.map_width
            if (not horizontal) and self.scenario.topology == "T_JUNCTION":
                length = 50.0 + self.half_road

            road = o3d.geometry.TriangleMesh.create_box(length if horizontal else self.scenario.road_width, self.scenario.road_width if horizontal else length, 0.08)
            road.translate([0.0 if horizontal else center - self.half_road, center - self.half_road if horizontal else 0.0, 0.02])
            road.paint_uniform_color([0.15, 0.15, 0.15])
            geoms.append(road)

            median = o3d.geometry.TriangleMesh.create_box(length if horizontal else self.median_width, self.median_width if horizontal else length, 0.12)
            median.translate([0.0 if horizontal else center - self.median_width / 2.0, center - self.median_width / 2.0 if horizontal else 0.0, 0.05])
            median.paint_uniform_color(deep_yellow)
            geoms.append(median)

            lane_width = (self.scenario.road_width - self.median_width) / self.scenario.num_lanes if self.scenario.num_lanes > 0 else 3.5
            for i in range(1, self.scenario.num_lanes):
                dist = self.median_width / 2.0 + (i * lane_width)
                for side in [-1, 1]:
                    if dist >= self.half_road:
                        continue
                    lane_pos = center + side * dist
                    lane = o3d.geometry.TriangleMesh.create_box(length if horizontal else 0.15, 0.15 if horizontal else length, 0.02)
                    lane.translate([0.0 if horizontal else lane_pos, lane_pos if horizontal else 0.0, 0.07])
                    lane.paint_uniform_color([0.5, 0.5, 0.5])
                    geoms.append(lane)

            for side in [-1, 1]:
                sw_c = center + side * (self.half_road + self.sidewalk_width / 2.0)
                if horizontal:
                    if self.scenario.topology == "T_JUNCTION" and side == 1:
                        segments = [[0.0, self.map_len]]
                    else:
                        segments = [[0.0, 50.0 - self.half_road], [50.0 + self.half_road, self.map_len]] if self.road_centers_x else [[0.0, self.map_len]]
                    for start, end in segments:
                        if end - start <= 0:
                            continue
                        sidewalk = o3d.geometry.TriangleMesh.create_box(end - start, self.sidewalk_width, 0.15)
                        sidewalk.translate([start, sw_c - self.sidewalk_width / 2.0, 0.08])
                        sidewalk.paint_uniform_color([1.0, 0.9, 0.0])
                        geoms.append(sidewalk)
                else:
                    v_limit = 50.0 + self.half_road if self.scenario.topology == "T_JUNCTION" else self.map_width
                    segments = [[0.0, 50.0 - self.half_road], [50.0 + self.half_road, v_limit]]
                    for start, end in segments:
                        if end - start <= 0:
                            continue
                        sidewalk = o3d.geometry.TriangleMesh.create_box(self.sidewalk_width, end - start, 0.15)
                        sidewalk.translate([sw_c - self.sidewalk_width / 2.0, start, 0.08])
                        sidewalk.paint_uniform_color([1.0, 0.9, 0.0])
                        geoms.append(sidewalk)

        for road_y in self.road_centers_y:
            draw_infrastructure(float(road_y), horizontal=True)
        for road_x in self.road_centers_x:
            draw_infrastructure(float(road_x), horizontal=False)

        for x1, y1, x2, y2 in self.negative_zones:
            building = o3d.geometry.TriangleMesh.create_box(x2 - x1, y2 - y1, self.building_height)
            building.translate([x1, y1, 0.1])
            building.paint_uniform_color([0.2, 0.2, 0.2])
            geoms.append(building)

        combined = geoms[0]
        for geom in geoms[1:]:
            combined += geom
        return combined


def relative_yaw_range_deg(yaw_axis_min_deg: float, yaw_axis_max_deg: float) -> tuple[float, float]:
    lo, hi = sorted([float(yaw_axis_min_deg), float(yaw_axis_max_deg)])
    return lo - 90.0, hi - 90.0


def allowed_mount_yaws_deg(wx: float, wy: float, map_geom: MapGeometry, yaw_step_deg: float, yaw_axis_min_deg: float, yaw_axis_max_deg: float) -> np.ndarray:
    step = max(float(yaw_step_deg), 1e-6)
    base_yaw = map_geom.road_normal_yaw_deg(float(wx), float(wy))
    lo, hi = relative_yaw_range_deg(yaw_axis_min_deg, yaw_axis_max_deg)
    rels = np.arange(lo, hi + 1e-6, step, dtype=np.float32)
    return np.unique(np.round((base_yaw + (rels if rels.size else np.array([0.0], dtype=np.float32))) % 360.0, 6))


def compute_view_angles_numpy(dx, dy, dz, yaw_deg, pitch_deg):
    pitch_rad = np.deg2rad(float(pitch_deg))
    yaw_rad = np.deg2rad(float(yaw_deg))
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    dx_y = dx * cy + dy * sy
    dy_y = -dx * sy + dy * cy
    dx_f = dx_y * cp + dz * sp
    dz_f = -dx_y * sp + dz * cp
    return np.arctan2(dy_y, dx_f), np.arctan2(dz_f, np.sqrt(dx_f * dx_f + dy_y * dy_y)), dx_f, dist


def z_values_from_config(config: ExperimentConfig) -> np.ndarray:
    return np.arange(config.z_min, config.z_max + 0.1, config.z_step, dtype=np.float32)


def nearest_z_index(z: float, zs: np.ndarray) -> int | None:
    if zs.size == 0:
        return None
    return int(np.argmin(np.abs(zs - float(z))))


def unpack_visible_indices(visibility: dict[str, Any], key: tuple[int, int, int]) -> np.ndarray:
    total = int(len(visibility.get("target_points", [])))
    packed = visibility.get("table", {}).get(key)
    if packed is None or total == 0:
        return np.array([], dtype=np.int64)
    mask = np.unpackbits(packed, count=total, bitorder="big").astype(bool)
    if not np.any(mask):
        return np.array([], dtype=np.int64)
    return np.flatnonzero(mask).astype(np.int64)


def coverage_mask_for_pose(
    pose: Iterable[float],
    visibility: dict[str, Any],
    geometry: MapGeometry,
    profile: LidarProfile,
    config: ExperimentConfig,
) -> np.ndarray:
    targets = np.asarray(visibility["target_points"], dtype=np.float32)
    mask = np.zeros(len(targets), dtype=bool)
    if len(targets) == 0:
        return mask
    x, y, z, yaw, pitch = map(float, list(pose)[:5])
    gx = int(np.floor(x / geometry.grid_size))
    gy = int(np.floor(y / geometry.grid_size))
    z_idx = nearest_z_index(z, np.asarray(visibility.get("zs", z_values_from_config(config)), dtype=np.float32))
    if z_idx is None:
        return mask
    idx = unpack_visible_indices(visibility, (gx, gy, int(z_idx)))
    if idx.size == 0:
        return mask
    pts = targets[idx]
    dx = pts[:, 0] - x
    dy = pts[:, 1] - y
    dz = pts[:, 2] - z
    azimuth, elevation, forward, dist = compute_view_angles_numpy(dx, dy, dz, yaw, pitch)
    half_h = np.deg2rad(profile.h_fov_deg) / 2.0
    half_v = np.deg2rad(profile.v_fov_deg) / 2.0
    front = np.ones_like(dist, dtype=bool) if half_h >= np.pi - 1e-6 else forward > 0.5
    valid = (np.abs(azimuth) <= half_h) & (np.abs(elevation) <= half_v) & front & (dist <= profile.max_range_m)
    mask[idx[valid]] = True
    return mask


def compute_config_hash(config: ExperimentConfig, scenario: Scenario, profile: LidarProfile) -> str:
    payload = repr(
        {
            "scenario": {
                "id": scenario.id,
                "topology": scenario.topology,
                "road_width": scenario.road_width,
                "num_lanes": scenario.num_lanes,
            },
            "profile": profile.__dict__,
            "config": config_hash_payload(config),
        }
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def make_cache_key(scenario_or_id: Scenario | int, lidar_profile: str | LidarProfile, config_or_hash: ExperimentConfig | str) -> str:
    scenario_id = scenario_or_id.id if isinstance(scenario_or_id, Scenario) else int(scenario_or_id)
    profile_name = lidar_profile.name if isinstance(lidar_profile, LidarProfile) else str(lidar_profile)
    if isinstance(config_or_hash, ExperimentConfig):
        config_hash = hashlib.sha1(repr(config_hash_payload(config_or_hash)).encode("utf-8")).hexdigest()
    else:
        config_hash = str(config_or_hash)
    return f"visibility_s{scenario_id}_{profile_name.lower()}_{config_hash[:12]}"


def visibility_cache_path(config: ExperimentConfig, scenario: Scenario, profile: LidarProfile) -> tuple[str, Path]:
    h = compute_config_hash(config, scenario, profile)
    key = make_cache_key(scenario.id, profile.name, h)
    return key, config.cache_dir / f"{key}.pkl"


def bake_visibility_map(config: ExperimentConfig, scenario: Scenario, profile: LidarProfile, force: bool = False) -> dict[str, Any]:
    expected_hash = compute_config_hash(config, scenario, profile)
    cache_key, path = visibility_cache_path(config, scenario, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        with path.open("rb") as f:
            data = pickle.load(f)
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        if data.get("cache_key") == cache_key and meta.get("config_hash") == expected_hash and data.get("visibility_model") == VISIBILITY_MODEL and "table" in data:
            return data
        raise RuntimeError(f"Visibility cache metadata mismatch: {path}")
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        raise RuntimeError("Open3D is required to build raycast visibility cache.") from exc

    geometry = MapGeometry(scenario)
    target_points = geometry.target_points([float(v) for v in config.target.get("coverage_z_slices", [1.0, 1.5, 2.0])])
    target_weights = np.ones(len(target_points), dtype=np.float32) * float(config.mappo.get("score_base_voxel", 0.01))
    mesh = geometry.create_environment_mesh()
    device = o3d.core.Device("CUDA:0")
    try:
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh, vertex_dtype=o3d.core.Dtype.Float32, device=device)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
    except Exception:
        device = o3d.core.Device("CPU:0")
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh, vertex_dtype=o3d.core.Dtype.Float32, device=device)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)

    cols = int(geometry.map_len / geometry.grid_size)
    rows = int(geometry.map_width / geometry.grid_size)
    zs = z_values_from_config(config)
    table: dict[tuple[int, int, int], np.ndarray] = {}
    counts: dict[tuple[int, int, int], int] = {}
    if target_points.size > 0:
        for gx in range(cols):
            for gy in range(rows):
                wx = (gx + 0.5) * geometry.grid_size
                wy = (gy + 0.5) * geometry.grid_size
                if not geometry.is_installable(wx, wy):
                    continue
                for z_idx, wz in enumerate(zs):
                    if not geometry.is_installable(wx, wy, float(wz)):
                        continue
                    origin = np.array([wx, wy, float(wz)], dtype=np.float32)
                    vecs = target_points - origin
                    dist = np.linalg.norm(vecs, axis=1)
                    dist = np.maximum(dist, 1e-6)
                    dirs = vecs / dist[:, None]
                    origins = np.repeat(origin[None, :], target_points.shape[0], axis=0)
                    rays_np = np.concatenate([origins, dirs], axis=1)
                    rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32, device=device)
                    try:
                        result = scene.cast_rays(rays)
                    except RuntimeError:
                        result = scene.cast_rays(rays.to(o3d.core.Device("CPU:0")))
                    t_hit = result["t_hit"].numpy()
                    visible = ((~np.isfinite(t_hit)) | (t_hit >= dist - 1e-3)) & (dist <= profile.max_range_m)
                    if np.any(visible):
                        key = (gx, gy, int(z_idx))
                        table[key] = np.packbits(visible, bitorder="big")
                        counts[key] = int(np.count_nonzero(visible))
    data = {
        "cache_key": cache_key,
        "visibility_model": VISIBILITY_MODEL,
        "table": table,
        "counts": counts,
        "target_points": target_points,
        "target_weights": target_weights,
        "max_voxel_est": max(counts.values(), default=1),
        "total_road_voxels": int(len(target_points)),
        "zs": zs,
        "target_zs": np.asarray(config.target.get("coverage_z_slices", [1.0, 1.5, 2.0]), dtype=np.float32),
        "meta": {
            "scenario_id": int(scenario.id),
            "topology": scenario.topology,
            "road_width": float(scenario.road_width),
            "num_lanes": int(scenario.num_lanes),
            "lidar_profile": profile.name,
            "config_hash": expected_hash,
            "visibility_model": VISIBILITY_MODEL,
            "grid_size": float(geometry.grid_size),
            "map_len": float(geometry.map_len),
            "map_width": float(geometry.map_width),
        },
    }
    with path.open("wb") as f:
        pickle.dump(data, f)
    return data


@dataclass
class CoverageResult:
    covered_count: int
    num_targets: int
    coverage_ratio: float
    coverage_pct: float
    target_points: np.ndarray
    covered_mask: np.ndarray
    density_map: np.ndarray
    cache_key: str


def load_positions(path: str | Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 5:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 5 - arr.shape[1]), dtype=np.float32)], axis=1)
    return arr[:, :5]


def evaluate_coverage(positions: np.ndarray, scenario: Scenario | int, lidar_profile: str, config: ExperimentConfig) -> CoverageResult:
    scenario_obj = scenario if isinstance(scenario, Scenario) else get_scenario(int(scenario))
    profile = get_lidar_profile(lidar_profile, config)
    data = bake_visibility_map(config, scenario_obj, profile)
    targets = np.asarray(data["target_points"], dtype=np.float32)
    density = np.zeros(len(targets), dtype=np.float32)
    if len(targets) == 0 or np.asarray(positions).size == 0:
        return CoverageResult(0, len(targets), 0.0, 0.0, targets, density.astype(bool), density, data["cache_key"])
    geometry = MapGeometry(scenario_obj)
    for row in np.atleast_2d(positions):
        mask = coverage_mask_for_pose(row[:5], data, geometry, profile, config)
        density[mask] += 1.0
    road_detection = int(config.target.get("road_detection", 1))
    covered = density >= float(road_detection)
    count = int(np.count_nonzero(covered))
    total = int(len(targets))
    ratio = count / total if total else 0.0
    return CoverageResult(count, total, ratio, ratio * 100.0, targets, covered, density, data["cache_key"])

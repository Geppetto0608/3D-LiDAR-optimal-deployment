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
OPTICAL_MODEL = "angular_fov_range_v2"


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
    def xy_domain(self) -> str:
        value = self.candidate.get("xy_domain", "bin_representatives")
        if value not in ("bin_representatives", "installable_grid"):
            raise ValueError(f"Unknown candidate.xy_domain: {value!r}")
        return value

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


def optical_model_metadata(profile: LidarProfile) -> dict[str, str]:
    # Preserve existing 360-degree library/hash contracts; limited FOV changed.
    return {"optical_model": OPTICAL_MODEL} if profile.h_fov_deg < 360.0 else {}


def config_hash_payload(config: ExperimentConfig, profile: LidarProfile | None = None) -> dict[str, Any]:
    profile = profile or get_lidar_profile(config.lidar_profile, config)
    return {
        "visibility_model": VISIBILITY_MODEL,
        **optical_model_metadata(profile),
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
        settings = self.scenario.raw.get('geometry', {})
        if not isinstance(settings, dict) or set(settings) - {'junction_x_m', 'junction_y_m'}:
            raise ValueError('Unsupported scenario geometry settings')
        if settings and self.scenario.topology not in {'CROSS', 'T_JUNCTION'}:
            raise ValueError('Junction offsets are supported only for CROSS/T_JUNCTION')
        self.junction_x_m = float(settings.get('junction_x_m', 50.0))
        self.junction_y_m = float(settings.get('junction_y_m', self.road_centers_y[0]))
        if settings:
            margin = self.scenario.road_width / 2.0 + self.sidewalk_width + self.grid_size
            if not (np.isfinite([self.junction_x_m, self.junction_y_m]).all()
                    and margin < self.junction_x_m < self.map_len - margin
                    and margin < self.junction_y_m < self.map_width - margin):
                raise ValueError('Junction must leave nonempty road/sidewalk/building arms inside the map')
            self.road_centers_y = [self.junction_y_m]
        if self.scenario.topology in {"CROSS", "T_JUNCTION", "COMPLEX"}:
            self.road_centers_x = [self.junction_x_m]
        self.half_road = self.scenario.road_width / 2.0
        self.num_lanes = self.scenario.num_lanes
        self.sidewalk_center_offset = self.half_road + self.sidewalk_width / 2.0
        self.road_x_centers = self.road_centers_x
        self.road_y_centers = self.road_centers_y
        self._set_building_zones()

    def _set_building_zones(self) -> None:
        topo = self.scenario.topology
        cx, cy = self.junction_x_m, self.junction_y_m
        offset = self.half_road + self.sidewalk_width
        if topo in {"STRAIGHT", "UNKNOWN"}:
            self.negative_zones = [[0, 0, self.map_len, cy - offset], [0, cy + offset, self.map_len, self.map_width]]
        elif topo in {"CROSS", "COMPLEX"}:
            self.negative_zones = [
                [0, 0, cx - offset, cy - offset],
                [cx + offset, 0, self.map_len, cy - offset],
                [0, cy + offset, cx - offset, self.map_width],
                [cx + offset, cy + offset, self.map_len, self.map_width],
            ]
        elif topo == "T_JUNCTION":
            self.negative_zones = [
                [0, cy + offset, self.map_len, self.map_width],
                [0, 0, cx - offset, cy - offset],
                [cx + offset, 0, self.map_len, cy - offset],
            ]

    def is_road(self, x: float, y: float) -> bool:
        topo = self.scenario.topology
        for rc in self.road_centers_x:
            if abs(x - rc) < self.half_road:
                if topo == "T_JUNCTION" and y > self.junction_y_m + self.half_road:
                    continue
                if self.median_width / 2 < abs(x - rc) < self.half_road:
                    return True
        for rc in self.road_centers_y:
            if abs(y - rc) < self.half_road and self.median_width / 2 < abs(y - rc) < self.half_road:
                return True
        return topo in {"CROSS", "T_JUNCTION", "COMPLEX"} and abs(x - self.junction_x_m) < self.half_road and abs(y - self.junction_y_m) < self.half_road

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
                length = self.junction_y_m + self.half_road

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
                        segments = [[0.0, self.junction_x_m - self.half_road], [self.junction_x_m + self.half_road, self.map_len]] if self.road_centers_x else [[0.0, self.map_len]]
                    for start, end in segments:
                        if end - start <= 0:
                            continue
                        sidewalk = o3d.geometry.TriangleMesh.create_box(end - start, self.sidewalk_width, 0.15)
                        sidewalk.translate([start, sw_c - self.sidewalk_width / 2.0, 0.08])
                        sidewalk.paint_uniform_color([1.0, 0.9, 0.0])
                        geoms.append(sidewalk)
                else:
                    v_limit = self.junction_y_m + self.half_road if self.scenario.topology == "T_JUNCTION" else self.map_width
                    segments = [[0.0, self.junction_y_m - self.half_road], [self.junction_y_m + self.half_road, v_limit]]
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


@dataclass(frozen=True)
class ConnectedCorridorGeometry:
    """Opt-in straight/arc/straight corridor; unrelated to legacy scenario IDs.

    The two modules meet at length/2. A circular transition straddles this
    interface. Rays remain Euclidean; only the road and mounting lines bend.
    """
    length_m: float = 80.0
    road_width_m: float = 7.0
    turn_deg: float = 0.0
    bend_length_m: float = 20.0
    sidewalk_width_m: float = 5.0
    building_height_m: float = 15.0
    wall_thickness_m: float = 2.0
    mesh_step_m: float = 0.5
    median_width_m: float = 0.5

    def __post_init__(self):
        values = list(self.__dict__.values())
        if any(isinstance(value, bool) for value in values) or not np.isfinite(values).all():
            raise ValueError('Corridor parameters must be finite')
        for name in ('length_m', 'road_width_m', 'sidewalk_width_m', 'building_height_m',
                     'wall_thickness_m', 'mesh_step_m'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not (0 <= self.median_width_m < self.road_width_m):
            raise ValueError('Median must fit inside road')
        if not (0 <= self.bend_length_m <= self.length_m):
            raise ValueError('Transition must fit inside corridor')
        if abs(self.turn_deg) > 90 or (self.turn_deg and not self.bend_length_m):
            raise ValueError('Supported turn: [-90,90] degrees with a nonzero transition')
        outer = self.road_width_m / 2 + self.sidewalk_width_m + self.wall_thickness_m
        if abs(self.curvature) * outer >= 1:
            raise ValueError('Bend radius must exceed the outer building offset')

    @property
    def curvature(self):
        return np.deg2rad(self.turn_deg) / self.bend_length_m if self.bend_length_m else 0.

    @property
    def mounting_offset_m(self):
        return self.road_width_m / 2 + self.sidewalk_width_m / 2

    def centerline(self, stations):
        s = np.asarray(stations, dtype=float)
        if not np.isfinite(s).all() or np.any(s < 0) or np.any(s > self.length_m):
            raise ValueError('Station outside corridor')
        if abs(self.curvature) < 1e-12:
            return np.stack([s, np.zeros_like(s)], axis=-1), np.zeros_like(s)
        start = (self.length_m - self.bend_length_m) / 2
        u = np.clip(s - start, 0., self.bend_length_m)
        theta = self.curvature * u
        x = start + np.sin(theta) / self.curvature
        y = (1 - np.cos(theta)) / self.curvature
        tail = np.maximum(s - start - self.bend_length_m, 0.)
        x += tail * np.cos(theta)
        y += tail * np.sin(theta)
        x = np.where(s < start, s, x)
        y = np.where(s < start, 0., y)
        return np.stack([x, y], axis=-1), theta

    def points(self, station_lateral_height):
        coordinates = np.asarray(station_lateral_height, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
            raise ValueError('Expected finite [N,3] station/lateral/height')
        xy, heading = self.centerline(coordinates[:, 0])
        normal = np.column_stack([-np.sin(heading), np.cos(heading)])
        return np.column_stack([xy + coordinates[:, 1, None] * normal, coordinates[:, 2]])

    def poses(self, canonical):
        """[s, signed lateral offset, z, road-normal-relative yaw, pitch]."""
        canonical = np.asarray(canonical, dtype=float)
        self.validate_canonical(canonical)
        _, heading = self.centerline(canonical[:, 0])
        inward = np.where(canonical[:, 1] > 0, -90., 90.)
        yaw = (np.rad2deg(heading) + inward + canonical[:, 3]) % 360
        return np.column_stack([self.points(canonical[:, :3]), yaw, canonical[:, 4]])

    def validate_canonical(self, canonical):
        p = np.asarray(canonical, dtype=float)
        if p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all():
            raise ValueError('Expected finite [K,5] canonical poses')
        if (np.any(p[:, 0] < 0) or np.any(p[:, 0] > self.length_m)
                or not np.allclose(np.abs(p[:, 1]), self.mounting_offset_m, atol=1e-8, rtol=0)
                or np.any(p[:, 2] < 3) or np.any(p[:, 2] > 6)
                or np.any(np.abs(p[:, 3]) > 55)
                or np.any(p[:, 4] < -25) or np.any(p[:, 4] > -5)):
            raise ValueError('Canonical pose violates corridor mounting domain')

    def target_samples(self, step_m=1., heights=(1., 1.5, 2.)):
        if not np.isfinite(step_m) or step_m <= 0:
            raise ValueError('Positive finite sampling step required')
        counts = np.array([self.length_m, self.road_width_m]) / step_m
        if not np.allclose(counts, np.round(counts), atol=1e-8, rtol=0):
            raise ValueError('Target step must divide both length and width')
        heights = np.asarray(heights, dtype=float)
        if (heights.ndim != 1 or not len(heights) or not np.isfinite(heights).all()
                or np.any(heights <= .23) or len(np.unique(heights)) != len(heights)):
            raise ValueError('Targets must be above road infrastructure')
        stations = (np.arange(int(round(counts[0]))) + .5) * step_m
        lateral = (np.arange(int(round(counts[1]))) + .5) * step_m - self.road_width_m / 2
        lateral = lateral[np.abs(lateral) > self.median_width_m / 2]
        if not len(stations) or not len(lateral):
            raise ValueError('Sampling produced no road targets')
        canonical = np.array(np.meshgrid(stations, lateral, heights, indexing='ij')).reshape(3, -1).T
        return self.points(canonical).astype(np.float32), canonical

    def create_environment_mesh(self):
        """Closed swept slabs, open road ends, and opaque roadside wall strips."""
        import open3d as o3d
        breaks = [(self.length_m - self.bend_length_m) / 2,
                  (self.length_m + self.bend_length_m) / 2]
        stations = np.unique(np.r_[np.linspace(0, self.length_m,
            int(np.ceil(self.length_m / self.mesh_step_m)) + 1), breaks])
        combined = o3d.geometry.TriangleMesh()

        def slab(lo, hi, bottom, top, color):
            vertices = []
            for d, z in ((lo, bottom), (hi, bottom), (lo, top), (hi, top)):
                vertices.extend(self.points(np.column_stack([stations,
                    np.full(len(stations), d), np.full(len(stations), z)])))
            n = len(stations)
            faces = []

            def quad(a, b, c, d):
                faces.extend([[a, b, c], [a, c, d]])

            for i in range(n - 1):
                quad(i, i + 1, n + i + 1, n + i)
                quad(2*n + i, 3*n + i, 3*n + i + 1, 2*n + i + 1)
                quad(i, 2*n + i, 2*n + i + 1, i + 1)
                quad(n + i, n + i + 1, 3*n + i + 1, 3*n + i)
            quad(0, n, 3*n, 2*n)
            quad(n - 1, 3*n - 1, 4*n - 1, 2*n - 1)
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(vertices)),
                o3d.utility.Vector3iVector(np.asarray(faces)))
            mesh.paint_uniform_color(color)
            return mesh

        half = self.road_width_m / 2
        outer = half + self.sidewalk_width_m
        combined += slab(-half, half, 0., .1, [.2, .2, .2])
        if self.median_width_m:
            combined += slab(-self.median_width_m / 2, self.median_width_m / 2,
                             .1, .17, [.8, .6, .1])
        for side in (-1, 1):
            lo, hi = sorted([side * half, side * outer])
            combined += slab(lo, hi, .08, .23, [.7, .7, .7])
            lo, hi = sorted([side * outer, side * (outer + self.wall_thickness_m)])
            combined += slab(lo, hi, .1, .1 + self.building_height_m, [.4, .4, .4])
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


def installation_base_positions(geometry: MapGeometry, config: ExperimentConfig) -> list[tuple[float, float, float]]:
    """Geometry-only candidate domain; no coverage or deployment inputs."""
    mode = config.xy_domain
    if not np.isfinite([config.z_min, config.z_max, config.z_step, config.xy_step_m]).all():
        raise ValueError("Installation grid parameters must be finite")
    if config.z_step <= 0 or config.z_max < config.z_min or config.xy_step_m <= 0:
        raise ValueError("Invalid installation grid spacing or height range")
    zs = []
    z = config.z_min
    while z <= config.z_max + 1e-9:
        zs.append(round(z, 10))
        z += config.z_step
    positions = []
    seen_bins = set()
    step_m = max(config.xy_step_m, geometry.grid_size)
    for gx in range(int(geometry.map_len / geometry.grid_size)):
        for gy in range(int(geometry.map_width / geometry.grid_size)):
            x, y = (gx + .5) * geometry.grid_size, (gy + .5) * geometry.grid_size
            if not geometry.is_installable(float(x), float(y)):
                continue
            if mode == "bin_representatives":
                key = (int(np.floor(x / step_m)), int(np.floor(y / step_m)))
                if key in seen_bins:
                    continue
                seen_bins.add(key)
            positions.extend((float(x), float(y), float(z)) for z in zs)
    return positions


def validate_installation_pose(pose, geometry: MapGeometry, config: ExperimentConfig) -> None:
    """Validate the full discrete physical domain, before pruning/dedup/search limits."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (5,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite [x,y,z,yaw,pitch] pose")
    x, y, z, yaw, pitch = pose
    cell = pose[:2] / geometry.grid_size - .5
    if not np.allclose(cell, np.round(cell), atol=1e-5, rtol=0):
        raise ValueError("XY must be grid cell centers")
    if not (0 <= x < geometry.map_len and 0 <= y < geometry.map_width):
        raise ValueError("XY outside map bounds")
    if not geometry.is_installable(x, y, z):
        raise ValueError("Pose outside installation region")
    if config.z_step <= 0 or config.pitch_step_deg <= 0 or config.yaw_step_deg <= 0:
        raise ValueError("Discrete pose steps must be positive")
    zs = np.arange(config.z_min, config.z_max + 1e-6, config.z_step)
    pitches = np.arange(config.pitch_min_deg, config.pitch_max_deg + 1e-6, config.pitch_step_deg)
    if not np.any(np.isclose(zs, z, atol=1e-5, rtol=0)) or not np.any(np.isclose(pitches, pitch, atol=1e-5, rtol=0)):
        raise ValueError("Z/pitch outside allowed discrete grid")
    yaws = allowed_mount_yaws_deg(x, y, geometry, config.yaw_step_deg,
                                  config.yaw_axis_min_deg, config.yaw_axis_max_deg)
    if not np.any(np.abs((yaws - yaw + 180) % 360 - 180) < 1e-4):
        raise ValueError("Yaw outside allowed discrete grid")


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


def angular_visibility_mask(azimuth, elevation, distance, half_h, half_v, max_range):
    """Angular/range visibility for NumPy arrays or Torch tensors, without LOS.

    No forward slab or sensor minimum range is implied by horizontal FOV.
    Coincident targets are excluded because their viewing direction is undefined.
    """
    return ((abs(azimuth) <= half_h) & (abs(elevation) <= half_v)
            & (distance <= max_range) & (distance > 1e-6))


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
    azimuth, elevation, _, dist = compute_view_angles_numpy(dx, dy, dz, yaw, pitch)
    half_h = np.deg2rad(profile.h_fov_deg) / 2.0
    half_v = np.deg2rad(profile.v_fov_deg) / 2.0
    valid = angular_visibility_mask(azimuth, elevation, dist, half_h, half_v, profile.max_range_m)
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
                **({'geometry': {
                    'model': 'axis_aligned_junction_v1',
                    'junction_x_m': MapGeometry(scenario).junction_x_m,
                    'junction_y_m': MapGeometry(scenario).junction_y_m,
                }} if scenario.raw.get('geometry') else {}),
            },
            "profile": profile.__dict__,
            "config": config_hash_payload(config, profile),
        }
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def make_cache_key(scenario_or_id: Scenario | int, lidar_profile: str | LidarProfile, config_or_hash: ExperimentConfig | str) -> str:
    scenario_id = scenario_or_id.id if isinstance(scenario_or_id, Scenario) else int(scenario_or_id)
    profile_name = lidar_profile.name if isinstance(lidar_profile, LidarProfile) else str(lidar_profile)
    if isinstance(config_or_hash, ExperimentConfig):
        profile = (lidar_profile if isinstance(lidar_profile, LidarProfile)
                   else get_lidar_profile(profile_name, config_or_hash))
        config_hash = hashlib.sha1(repr(config_hash_payload(config_or_hash, profile)).encode("utf-8")).hexdigest()
    else:
        config_hash = str(config_or_hash)
    return f"visibility_s{scenario_id}_{profile_name.lower()}_{config_hash[:12]}"


def visibility_cache_path(config: ExperimentConfig, scenario: Scenario, profile: LidarProfile) -> tuple[str, Path]:
    h = compute_config_hash(config, scenario, profile)
    key = make_cache_key(scenario.id, profile.name, h)
    return key, config.cache_dir / f"{key}.pkl"


def bake_visibility_map(config: ExperimentConfig, scenario: Scenario, profile: LidarProfile, force: bool = False, *, work=None) -> dict[str, Any]:
    expected_hash = compute_config_hash(config, scenario, profile)
    cache_key, path = visibility_cache_path(config, scenario, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        with path.open("rb") as f:
            data = pickle.load(f)
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        if data.get("cache_key") == cache_key and meta.get("config_hash") == expected_hash and data.get("visibility_model") == VISIBILITY_MODEL and "table" in data:
            if work is not None:
                work.add('full_cache_loads')
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
                        if work is not None:
                            work.add('raycast_calls')
                            work.add('rays_submitted', len(rays_np))
                        result = scene.cast_rays(rays)
                    except RuntimeError:
                        if work is not None:
                            work.add('raycast_calls')
                            work.add('raycast_retries')
                            work.add('rays_submitted', len(rays_np))
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

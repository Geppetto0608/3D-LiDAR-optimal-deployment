"""Lightweight integrity checks for the research pipeline.

Run with:
    .venv/bin/python tests/test_pipeline.py

These tests intentionally avoid pytest so the workspace can verify itself with
only the project virtual environment.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import (
    VISIBILITY_MODEL,
    MapGeometry,
    bake_visibility_map,
    evaluate_coverage,
    get_lidar_profile,
    load_config,
    load_scenarios,
    make_cache_key,
    visibility_cache_path,
)
from src.methods.mappo import plan_mappo_runs
from src.outputs import build_result_record, validate_result_record


class PipelineIntegrityTest(unittest.TestCase):
    def test_cache_key_and_plan_schema(self) -> None:
        config = load_config("configs/experiment.yaml")
        scenarios = load_scenarios()
        self.assertNotEqual(
            make_cache_key(scenarios[0], "VELODYNE64_R30", config),
            make_cache_key(scenarios[1], "VELODYNE64_R30", config),
        )
        profile = get_lidar_profile("VELODYNE64_R30", config)
        self.assertEqual(profile.max_range_m, 30.0)

        rows = plan_mappo_runs([scenarios[0]], config.lidar_profile, [1], config, tag="dryrun", episodes=10, trials=1)
        self.assertEqual(rows[0].scenario_id, 0)
        self.assertEqual(rows[0].budget_k, 1)

    def test_result_schema(self) -> None:
        record = build_result_record(
            method="TEST",
            scenario_id=0,
            topology="STRAIGHT",
            lidar_profile="VELODYNE64_R30",
            lidar_model="VELODYNE64_R30",
            max_range_m=30.0,
            budget_k=1,
            budget_mode="at_most",
            candidate_mode="unit",
            action_space="[x,y,z,yaw,pitch]",
            num_selected=0,
            positions_npy="none.npy",
            coverage_pct=0.0,
            coverage_ratio=0.0,
            covered_count=0,
            num_targets=1,
            episodes=None,
            seed=None,
            trial=None,
            mappo_tag=None,
            config_hash="abc",
            cache_key="cache",
        )
        validate_result_record(record)
        self.assertIn("created_at", record)

    def test_raycast_cache_and_coverage_smoke(self) -> None:
        config = load_config("configs/experiment.yaml")
        config.candidate["z_min"] = 3.0
        config.candidate["z_max"] = 3.0
        with tempfile.TemporaryDirectory(prefix="hj_ws_cache_") as tmp:
            config.output["cache_dir"] = tmp
            scenario = load_scenarios()[0]
            profile = get_lidar_profile(config.lidar_profile, config)
            data = bake_visibility_map(config, scenario, profile, force=True)
            self.assertEqual(data["visibility_model"], VISIBILITY_MODEL)
            self.assertGreater(len(data["target_points"]), 0)
            self.assertGreater(len(data["table"]), 0)

            gx, gy, z_idx = next(iter(data["table"]))
            geometry = MapGeometry(scenario)
            pose = np.asarray(
                [[(gx + 0.5) * geometry.grid_size, (gy + 0.5) * geometry.grid_size, float(data["zs"][z_idx]), 270.0, -15.0]],
                dtype=np.float32,
            )
            evaluation = evaluate_coverage(pose, scenario, config.lidar_profile, config)
            self.assertGreaterEqual(evaluation.coverage_pct, 0.0)
            self.assertLessEqual(evaluation.coverage_pct, 100.0)

            _cache_key, cache_path = visibility_cache_path(config, scenario, profile)
            self.assertTrue(Path(cache_path).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

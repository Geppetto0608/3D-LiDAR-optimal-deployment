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
from types import SimpleNamespace

import numpy as np
import torch

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
from src.methods.mappo import (
    ACTION_DIM,
    HINT_DIM,
    LegacyMAPPOParams,
    SharedMAPPO,
    _PoseCandidate,
    _compute_time_agent_gae,
    _evaluate_deterministic_policy,
    _select_best_candidate,
    plan_mappo_runs,
)
from src.outputs import build_result_record, validate_result_record


class PipelineIntegrityTest(unittest.TestCase):
    @staticmethod
    def _mappo_test_params(num_agents: int) -> LegacyMAPPOParams:
        return LegacyMAPPOParams(
            num_agents=num_agents,
            episodes=2,
            xy_step_m=2.0,
            z_min=3.0,
            z_max=6.0,
            z_step=1.0,
            yaw_step_deg=5.0,
            yaw_axis_min_deg=35.0,
            yaw_axis_max_deg=145.0,
            pitch_step_deg=2.5,
            min_pitch=-25.0,
            max_pitch=-5.0,
            local_map_size=3,
            ppo_k_epochs=1,
            ppo_update_timestep=2,
            ppo_minibatch_size=4,
        )

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

    def test_mappo_gae_preserves_agent_axis(self) -> None:
        rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        dones = torch.tensor([0.0, 1.0])
        values = torch.zeros_like(rewards)
        advantages, returns = _compute_time_agent_gae(
            rewards,
            dones,
            values,
            bootstrap_values=torch.zeros(2),
            gamma=1.0,
            gae_lambda=1.0,
        )
        expected = torch.tensor([[3.0, 30.0], [2.0, 20.0]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(returns, expected)

        continuing_advantages, _ = _compute_time_agent_gae(
            rewards=torch.zeros(2, 2),
            dones=torch.zeros(2),
            values=torch.zeros(2, 2),
            bootstrap_values=torch.tensor([3.0, 30.0]),
            gamma=1.0,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(
            continuing_advantages,
            torch.tensor([[3.0, 30.0], [3.0, 30.0]]),
        )

    def test_mappo_joint_rollout_update_shapes(self) -> None:
        torch.manual_seed(0)
        for num_agents in (1, 3):
            with self.subTest(num_agents=num_agents):
                params = self._mappo_test_params(num_agents)
                state_dim = 5 + (params.local_map_size**2) + HINT_DIM
                shared = SharedMAPPO(
                    state_dim,
                    ACTION_DIM,
                    state_dim * num_agents,
                    params.local_map_size,
                    params,
                    torch.device("cpu"),
                )
                for _ in range(params.ppo_update_timestep):
                    states = torch.randn(num_agents, state_dim)
                    masks = torch.ones(num_agents, ACTION_DIM)
                    sampled = [shared.act(states[i], masks[i]) for i in range(num_agents)]
                    actions = torch.tensor([item[0] for item in sampled], dtype=torch.long)
                    log_probs = torch.stack([item[1] for item in sampled])
                    self.assertEqual(log_probs.shape, (num_agents,))
                    next_states = torch.randn(num_agents, state_dim)
                    shared.store_timestep(
                        states,
                        masks,
                        actions,
                        log_probs,
                        rewards=np.arange(1, num_agents + 1, dtype=np.float32),
                        global_state=states.flatten(),
                        next_global_state=next_states.flatten(),
                        done=False,
                    )
                self.assertEqual(len(shared.buffer["r"]), 0)
                self.assertEqual(len(shared.update_history), 1)
                self.assertEqual(shared.update_history[0]["rollout_steps"], params.ppo_update_timestep)
                for value in shared.update_history[0].values():
                    self.assertTrue(np.isfinite(value))
                for parameter in shared.policy.parameters():
                    self.assertTrue(torch.isfinite(parameter).all())

    def test_mappo_deterministic_evaluation_isolated_from_training_state(self) -> None:
        class FakeAgent:
            def __init__(self, x: int) -> None:
                self.x = x
                self.y = 0
                self.z = 3.0
                self.z_idx = 0
                self.yaw = 0.0
                self.pitch = -10.0

            def step(self, deterministic: bool = False, store_buffer: bool = True):
                self.assert_deterministic = deterministic and not store_buffer
                self.x += 1

            def pose(self) -> list[float]:
                return [float(self.x), float(self.y), self.z, self.yaw, self.pitch]

        class FakeEnv:
            def __init__(self) -> None:
                self.prev_score = 7.0
                self._prev_uncovered_dists = torch.tensor([9.0])
                self.params = SimpleNamespace(uncovered_centroid=np.array([8.0, 8.0], dtype=np.float32))
                self.target_points_np = np.array([[0.0, 0.0, 1.0], [4.0, 4.0, 1.0]], dtype=np.float32)
                self.total_voxels = 2

            def calculate_reward(self, agents):
                self.prev_score = float(sum(agent.x for agent in agents))
                self._prev_uncovered_dists = torch.tensor([self.prev_score])
                self.params.uncovered_centroid = np.array([1.0, 1.0], dtype=np.float32)
                return 0.0, np.zeros(len(agents), dtype=np.float32), {}

            @staticmethod
            def evaluate_positions(positions):
                score = float(np.asarray(positions)[:, 0].sum())
                return score, score

        env = FakeEnv()
        agents = [FakeAgent(9)]
        initial_states = [(1, 0, 3.0, 0, 0.0, -10.0)]
        positions, coverage, score = _evaluate_deterministic_policy(env, agents, initial_states, steps=2)

        self.assertEqual(float(positions[0, 0]), 3.0)
        self.assertEqual(coverage, 3.0)
        self.assertEqual(score, 3.0)
        self.assertEqual(agents[0].x, 9)
        self.assertEqual(env.prev_score, 7.0)
        torch.testing.assert_close(env._prev_uncovered_dists, torch.tensor([9.0]))
        np.testing.assert_array_equal(env.params.uncovered_centroid, np.array([8.0, 8.0], dtype=np.float32))

    def test_mappo_candidate_selection_keeps_result_provenance(self) -> None:
        pose = np.zeros((1, 5), dtype=np.float32)
        raw = _PoseCandidate("eval_raw", pose, coverage_pct=80.0, score=8.0)
        tied_postprocessed = _PoseCandidate(
            "eval_yaw_tuned", pose, coverage_pct=80.0, score=8.0, postprocessed=True
        )
        improved_postprocessed = _PoseCandidate(
            "train_yaw_tuned", pose, coverage_pct=81.0, score=8.1, postprocessed=True
        )

        self.assertEqual(_select_best_candidate([raw, tied_postprocessed]).source, "eval_raw")
        self.assertEqual(
            _select_best_candidate([raw, tied_postprocessed, improved_postprocessed]).source,
            "train_yaw_tuned",
        )

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

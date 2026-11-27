"""Deterministic best-improvement search with the MAPPO local action domain."""
import time

import numpy as np
import torch

from src.environment import MapGeometry, get_lidar_profile
from src.methods.local_mappo import (LocalPPOAgent, MeasuredMAPPOEnv, OnDemandVisibility,
                                     validate_poses)
from src.methods.mappo import (MappoPlanRow, _legacy_params, _snapshot_agent_states,
                               _restore_agent_states)


class _ActionDriver:
    def __init__(self, device):
        self.device = device
        self.action = 0

    def act(self, state, mask, deterministic=False):
        if not mask[self.action]:
            raise ValueError('Invalid local-search action')
        return self.action, torch.zeros((), device=self.device)


def refine_local_search(initial, scenario, config, *, radius_m=10., max_evaluations=2000,
                        time_limit_sec=None, work=None):
    """One-agent-at-a-time ascent; stop at a local optimum or soft limit.

    The initial deployment is retained. Includes the observation construction
    overhead of LocalPPOAgent to reuse exactly its masks, actions and clamping.
    """
    if max_evaluations < 1 or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('Positive evaluation limit and radius required')
    if time_limit_sec is not None and (not np.isfinite(time_limit_sec) or time_limit_sec <= 0):
        raise ValueError('Positive finite time limit required')
    if config.target.get('road_detection', 1) != 1:
        raise ValueError('Only union coverage supported')
    started = time.perf_counter()
    geometry = MapGeometry(scenario)
    initial = validate_poses(initial, geometry, config)
    row = MappoPlanRow(scenario.id, scenario.topology, config.lidar_profile, len(initial), 0, 0, 1, 'local_search')
    params = _legacy_params(row, config)
    visibility = OnDemandVisibility(geometry, config, get_lidar_profile(config.lidar_profile, config), work=work)
    env = MeasuredMAPPOEnv(scenario, config.lidar_profile, config, params, work=work,
        visibility_data=visibility.metadata(params.score_base_voxel), visibility_query=visibility.query)
    driver = _ActionDriver(env.device)
    agents = [LocalPPOAgent(env, driver, p, radius_m) for p in initial]
    best = initial.copy()
    best_cov, _ = env.evaluate_positions(best)
    initial_cov = best_cov
    evaluated, sweeps = 0, 0
    reason = 'local_optimum'
    if work is not None:
        work.observe(best_cov, 'local_search_initial')
    while evaluated < max_evaluations:
        states = _snapshot_agent_states(agents)
        next_states, next_cov, next_pose = states, best_cov, best
        stopped = False
        for i, agent in enumerate(agents):
            _, mask = agent.get_state_and_mask()
            for action in range(1, 11):
                if evaluated >= max_evaluations:
                    stopped, reason = True, 'evaluation_limit'
                    break
                if time_limit_sec is not None and time.perf_counter() - started >= time_limit_sec:
                    stopped, reason = True, 'time_limit'
                    break
                if not mask[action]:
                    continue
                _restore_agent_states(agents, states)
                driver.action = action
                agent.step(deterministic=True, store_buffer=False)
                poses = np.asarray([ag.pose() for ag in agents], dtype=np.float32)
                if len(np.unique(poses, axis=0)) != len(poses):
                    continue
                cov, _ = env.evaluate_positions(poses)
                evaluated += 1
                if cov > next_cov + 1e-9:
                    next_cov, next_pose, next_states = cov, poses.copy(), _snapshot_agent_states(agents)
            _restore_agent_states(agents, states)
            if stopped:
                break
        sweeps += 1
        if next_cov > best_cov + 1e-9:
            best, best_cov = next_pose, next_cov
            if work is not None:
                work.observe(best_cov, 'local_search_improvement')
        else:
            break
        _restore_agent_states(agents, next_states)
        if stopped:
            break
    if evaluated >= max_evaluations:
        reason = 'evaluation_limit'
    validate_poses(best, geometry, config)
    if np.any(np.linalg.norm(best[:, :2] - initial[:, :2], axis=1) > radius_m + 1e-5):
        raise RuntimeError('Local search escaped radius')
    return best, dict(method='coordinate_best_improvement_v1', coverage_pct=best_cov,
        initial_coverage_pct=initial_cov, neighbor_evaluations=evaluated, sweeps=sweeps,
        stop_reason=reason, time_limit_sec=time_limit_sec, max_evaluations=max_evaluations,
        local_radius_m=radius_m, elapsed_sec=time.perf_counter() - started,
        memoized_xyz_entries=len(env._visible_idx_cache), full_visibility_cache=False,
        semantics='best initial/neighbor deployment; one-agent moves, no learning or global optimality claim')

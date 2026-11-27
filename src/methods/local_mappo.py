"""Fixed-start, bounded MAPPO refinement using on-demand shared mesh visibility."""
from dataclasses import asdict
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, get_lidar_profile, validate_installation_pose
from src.methods.mappo import (
    ACTION_DIM, HINT_DIM, LegacyMAPPOEnv, MappoPlanRow, PPOAgent, SharedMAPPO,
    _evaluate_deterministic_policy, _legacy_params, _snapshot_agent_states,
)


def validate_poses(poses, geometry, config):
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 5 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Expected nonempty finite [K,5] poses')
    if len(np.unique(poses, axis=0)) != len(poses):
        raise ValueError('Duplicate complete poses')
    for pose in poses:
        validate_installation_pose(pose, geometry, config)
    return poses


class OnDemandVisibility:
    def __init__(self, geometry, config, profile, *, work=None):
        self.geometry = geometry
        self.targets = geometry.target_points(config.target['coverage_z_slices'])
        self.zs = np.arange(config.z_min, config.z_max + 1e-6, config.z_step, dtype=np.float32)
        self.provider = MeshVisibility(geometry.create_environment_mesh(), self.targets, profile, work=work)
        self.raycast_queries = 0
        self.raycast_sec = 0.

    def metadata(self, score_weight):
        return dict(target_points=self.targets, zs=self.zs,
                    target_weights=np.full(len(self.targets), score_weight, dtype=np.float32),
                    max_voxel_est=max(len(self.targets), 1))

    def query(self, gx, gy, zi):
        x, y = (gx+.5)*self.geometry.grid_size, (gy+.5)*self.geometry.grid_size
        if not 0 <= zi < len(self.zs) or not self.geometry.is_installable(x, y):
            return np.empty(0, dtype=np.int64)
        started = time.perf_counter()
        mask = self.provider.line_of_sight_mask([x, y, self.zs[zi]])
        self.raycast_queries += 1
        self.raycast_sec += time.perf_counter()-started
        return np.flatnonzero(mask)


class LocalPPOAgent(PPOAgent):
    def __init__(self, env, shared, pose, radius_m):
        self.anchor_xy = np.asarray(pose[:2], dtype=float)
        self.radius_m = radius_m
        gx, gy = np.floor(self.anchor_xy/env.geometry.grid_size).astype(int)
        super().__init__(env, [(int(gx), int(gy))], shared)
        self.x, self.y = int(gx), int(gy)
        self.z_idx = int(np.argmin(abs(env.zs-pose[2])))
        self.z, self.yaw, self.pitch = map(float, pose[2:])

    def get_state_and_mask(self):
        state, mask = super().get_state_and_mask()
        step = self._xy_step_cells()
        for i, (dx, dy) in enumerate(((step,0),(-step,0),(0,step),(0,-step))):
            next_xy = (np.array([self.x+dx, self.y+dy])+.5)*self.geometry.grid_size
            if np.linalg.norm(next_xy-self.anchor_xy) > self.radius_m+1e-6:
                mask[i+1] = 0
        return state, mask


class MeasuredMAPPOEnv(LegacyMAPPOEnv):
    """Instrumentation only; reward, optics and cache behavior remain inherited."""
    def __init__(self, *args, work=None, **kwargs):
        self.work = work
        super().__init__(*args, **kwargs)

    def evaluate_positions(self, positions):
        if self.work is not None:
            self.work.add('deployment_evaluations')
            self.work.add('cached_pose_evaluations', len(positions))
        return super().evaluate_positions(positions)

    def calculate_reward(self, agents):
        if self.work is not None:
            self.work.add('reward_evaluations')
            self.work.add('cached_pose_evaluations', len(agents))
        return super().calculate_reward(agents)


def refine_placement(initial, scenario, config, output, *, steps=2000, seed=0, radius_m=10.,
                     work=None, time_limit_sec=None):
    """Train a fresh policy; preserve initial best and separate final-policy output.

    No exhaustive cache, warm spawn search, yaw postprocessing or checkpoint loading.
    V3 reward and 11 actions are retained; XY movement is additionally radius limited.
    """
    if steps < 1 or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('Positive steps and local radius required')
    if time_limit_sec is not None and (not np.isfinite(time_limit_sec) or time_limit_sec <= 0):
        raise ValueError('Positive finite refinement time limit required')
    if config.target.get('road_detection', 1) != 1:
        raise ValueError('Only binary union coverage supported')
    geometry = MapGeometry(scenario)
    initial = validate_poses(initial, geometry, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    row = MappoPlanRow(scenario.id, scenario.topology, config.lidar_profile,
                      len(initial), 0, seed, steps, 'local_refinement')
    params = _legacy_params(row, config)
    visibility = OnDemandVisibility(geometry, config, get_lidar_profile(config.lidar_profile, config), work=work)
    env = MeasuredMAPPOEnv(scenario, config.lidar_profile, config, params, work=work,
                        visibility_data=visibility.metadata(params.score_base_voxel),
                        visibility_query=visibility.query)
    state_dim = 5+params.local_map_size**2+HINT_DIM
    shared = SharedMAPPO(state_dim, ACTION_DIM, state_dim*len(initial), params.local_map_size, params, env.device)
    agents = [LocalPPOAgent(env, shared, p, radius_m) for p in initial]
    initial_states = _snapshot_agent_states(agents)
    env.calculate_reward(agents)
    initial_cov, _ = env.evaluate_positions(initial)
    best, best_cov, best_source = initial.copy(), initial_cov, 'initial'
    history, evaluations = [], []
    best_eval, best_eval_cov = initial.copy(), -1.
    if work is not None:
        work.observe(initial_cov, 'refinement_initial')

    def consider(poses, cov, source):
        nonlocal best, best_cov, best_source
        # Invalid simultaneous duplicate proposals are not deployable best solutions.
        if len(np.unique(poses, axis=0)) == len(poses) and cov > best_cov+1e-9:
            best, best_cov, best_source = poses.copy(), float(cov), source
            if work is not None:
                work.observe(best_cov, source)

    def evaluate(step):
        nonlocal best_eval, best_eval_cov
        poses, cov, _ = _evaluate_deterministic_policy(env, agents, initial_states, params.eval_steps)
        evaluations.append((step, cov))
        if cov > best_eval_cov:
            best_eval, best_eval_cov = poses.copy(), cov
        consider(poses, cov, 'deterministic_evaluation')
        return poses, cov

    completed_steps = 0
    for step in range(1, steps+1):
        if time_limit_sec is not None and time.perf_counter()-started >= time_limit_sec:
            break
        global_state = torch.cat([ag.get_state_and_mask()[0] for ag in agents])
        for ag in agents:
            ag.step()
        team, indiv, details = env.calculate_reward(agents)
        transitions = [ag.consume_transition() for ag in agents]
        next_global = torch.cat([ag.get_state_and_mask()[0] for ag in agents])
        shared.store_timestep(
            states=torch.stack([t[0] for t in transitions]), masks=torch.stack([t[1] for t in transitions]),
            actions=torch.tensor([t[2] for t in transitions], device=env.device),
            log_probs=torch.stack([t[3] for t in transitions]), rewards=team+indiv,
            global_state=global_state, next_global_state=next_global, done=False)
        completed_steps = step
        if work is not None:
            work.add('joint_training_steps')
            work.add('agent_training_transitions', len(agents))
        poses = np.asarray([ag.pose() for ag in agents], dtype=np.float32)
        consider(poses, details['cov'], 'stochastic_training')
        history.append((step, details['cov'], best_cov, team+float(np.mean(indiv))))
        if params.eval_interval > 0 and step % params.eval_interval == 0 and step < steps:
            evaluate(step)
        if step % 100 == 0:
            print(f'{output.name} step={step}/{steps} coverage={details["cov"]:.3f} best={best_cov:.3f}', flush=True)
    shared.train_net()
    final_policy, final_cov = evaluate(completed_steps)
    validate_poses(best, geometry, config)
    if np.any(np.linalg.norm(best[:,:2]-initial[:,:2], axis=1) > radius_m+1e-5):
        raise RuntimeError('Local refinement escaped its XY radius')
    density = np.stack([visibility.provider.mask(p) for p in best]).sum(axis=0)
    direct_cov = 100*float((density>0).mean())
    if not np.isclose(direct_cov, best_cov, atol=1e-6, rtol=0):
        raise RuntimeError('MAPPO and shared direct raycasting disagree')
    for name, poses in [('initial',initial), ('deployment',best), ('final_policy',final_policy), ('best_eval',best_eval)]:
        np.save(output/f'{name}.npy', poses)
    h, e = np.asarray(history).reshape(-1, 4), np.asarray(evaluations).reshape(-1, 2)
    np.savez_compressed(output/'train_log.npz', step=h[:,0], coverage=h[:,1], best_coverage=h[:,2],
                        reward=h[:,3], eval_step=e[:,0], eval_coverage=e[:,1])
    torch.save(dict(policy=shared.policy_old.state_dict(), params={k:v for k,v in asdict(params).items()
                    if k != 'uncovered_centroid'}, state_dim=state_dim, action_dim=ACTION_DIM,
                    seed=seed, initial_pose=initial.tolist(), local_radius_m=radius_m), output/'policy.pt')
    return best, dict(trainer='mappo_v3_local_on_demand_v1', training_steps=completed_steps,
        requested_training_steps=steps, time_limit_sec=time_limit_sec,
        stopped_early=completed_steps < steps,
        timing_note='Soft deadline includes refinement initialization; final PPO flush/evaluation/save can overrun', seed=seed,
        local_radius_m=radius_m, xy_step_m=params.xy_step_m, ppo_updates=len(shared.update_history),
        ppo_diagnostics=shared.update_history, params={k:v for k,v in asdict(params).items() if k != 'uncovered_centroid'},
        initial_coverage_pct=initial_cov, coverage_pct=direct_cov, reported_solution_source=best_source,
        final_policy_coverage_pct=final_cov, best_eval_coverage_pct=best_eval_cov,
        result_semantics='best initial/stochastic/evaluated deployment, not final policy performance',
        full_visibility_cache=False, reused_checkpoint=False, yaw_postprocessing=False,
        observation_normalizer='number of target points', plateau_schedule=False,
        raycast_xyz_queries=visibility.raycast_queries, raycast_sec=visibility.raycast_sec,
        memoized_xyz_entries=len(env._visible_idx_cache), elapsed_sec=time.perf_counter()-started,
        device=str(env.device), train_log='train_log.npz', policy='policy.pt')

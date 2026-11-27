"""Fresh, fixed-K local refinement with a shared objective and hard constraints."""
from dataclasses import asdict
from functools import partial
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.core.coverage_requirements import CoverageRequirements
from src.environment import MapGeometry, get_lidar_profile, validate_installation_pose
from src.methods.local_mappo import LocalPPOAgent, OnDemandVisibility, validate_poses
from src.methods.local_search import _ActionDriver
from src.methods.mappo import (ACTION_DIM, HINT_DIM, LegacyMAPPOEnv, MAPPOActorCritic,
    MappoPlanRow, SharedMAPPO, _legacy_params, _restore_agent_states, _snapshot_agent_states)


class InitialInfeasible(ValueError):
    def __init__(self, assessment):
        self.assessment = assessment
        super().__init__('Initial deployment violates: ' + ', '.join(assessment.violations))


class RefinementSession:
    """Per-run visibility memo and objective; no cache shared between methods."""
    def __init__(self, initial, scenario, config, requirements, radius_m, work=None, seed=0, steps=1,
                 *, allow_infeasible=False, anchor_poses=None):
        if config.target.get('road_detection', 1) != 1:
            raise ValueError('Only union coverage is supported')
        self.geometry, self.config, self.work = MapGeometry(scenario), config, work
        self.initial = validate_poses(initial, self.geometry, config)
        self.anchors = validate_poses(anchor_poses if anchor_poses is not None else initial, self.geometry, config)
        if self.anchors.shape != self.initial.shape:
            raise ValueError('Anchors must match sensor count and ordering')
        self.radius_m = radius_m
        row = MappoPlanRow(scenario.id, scenario.topology, config.lidar_profile,
                          len(initial), 0, seed, steps, 'constrained_refinement')
        self.params = _legacy_params(row, config)
        self.visibility = OnDemandVisibility(self.geometry, config,
            get_lidar_profile(config.lidar_profile, config), work=work)
        self.evaluator = requirements.bind(self.visibility.targets)
        self.env = LegacyMAPPOEnv(scenario, config.lidar_profile, config, self.params,
            visibility_data=self.visibility.metadata(self.params.score_base_voxel),
            visibility_query=self.visibility.query)
        self.current, self.density = self.assess(self.initial)
        if not self.current.feasible and not allow_infeasible:
            raise InitialInfeasible(self.current)
        self.initial_assessment = self.current
        self.set_current(self.current, self.density)

    def assess(self, poses):
        for pose in poses:
            validate_installation_pose(pose, self.geometry, self.config)
        delta = poses[:, :2].astype(float) - self.anchors[:, :2].astype(float)
        if np.any(np.linalg.norm(delta, axis=1) > self.radius_m + 1e-5):
            raise RuntimeError('Refinement escaped its anchor radius')
        masks = None
        if self.evaluator.needs_sensor_masks:
            masks = np.stack([self.env.coverage_density(np.asarray([pose])) > 0 for pose in poses])
        density = masks.sum(axis=0) if masks is not None else self.env.coverage_density(poses)
        if self.work is not None:
            self.work.add('deployment_evaluations')
            self.work.add('cached_pose_evaluations', len(poses))
        return self.evaluator.evaluate(poses, density, sensor_masks=masks), density

    def set_current(self, assessment, density):
        self.current, self.density = assessment, density
        unseen = self.visibility.targets[density == 0, :2]
        self.params.uncovered_centroid = unseen.mean(axis=0) if len(unseen) else None

    def features(self, pose):
        a, spec = self.current, self.evaluator.spec
        cost = float(self.evaluator.installation_costs(np.asarray([pose]))[0])
        budget = spec.get('constraints', {}).get('max_installation_cost')
        slack = (budget - a.components['installation_cost']) / max(budget, 1.) if budget is not None else 0.
        features = [a.coverage_pct / 100., a.overlap_ratio,
                    a.objective_score / len(self.density), cost / (1. + cost), slack]
        if a.robustness:
            features.append(a.robustness['single_failure_coverage_pct'] / 100.)
        for region in a.regions.values():
            features.extend([region['coverage_pct'] / 100., region['overlap_ratio']])
            if 'single_failure_coverage_pct' in region:
                features.append(region['single_failure_coverage_pct'] / 100.)
        hubs = spec.get('constraints', {}).get('service_hubs', [])
        for hub in hubs:
            distance = float(np.linalg.norm(np.asarray(pose[:2]) - hub['xy']))
            features.append(distance / max(hub['max_horizontal_distance_m'], 1.))
        return features


class _ConstrainedAgent(LocalPPOAgent):
    def __init__(self, session, shared, pose, anchor_pose=None):
        self.session = session
        super().__init__(session.env, shared, pose, session.radius_m)
        if anchor_pose is not None:
            self.anchor_xy = np.asarray(anchor_pose[:2], dtype=float)

    def get_state_and_mask(self):
        state, mask = super().get_state_and_mask()
        extra = torch.tensor(self.session.features(self.pose()), dtype=state.dtype, device=self.device)
        return torch.cat((state, extra)), mask


def _poses(agents):
    return np.asarray([agent.pose() for agent in agents], dtype=np.float32)


def _joint_step(session, agents, *, deterministic=False, store_buffer=True):
    before = _snapshot_agent_states(agents)
    previous, previous_density = session.current, session.density
    for agent in agents:
        agent.step(deterministic=deterministic, store_buffer=store_buffer)
    proposed, density = session.assess(_poses(agents))
    rejected = not proposed.feasible
    if rejected:
        # The sampled action and its log probability stay in PPO; the environment
        # maps this infeasible joint proposal to a penalized self-loop.
        _restore_agent_states(agents, before)
        session.set_current(previous, previous_density)
        return -1., True, proposed
    session.set_current(proposed, density)
    return proposed.objective_score - previous.objective_score, False, proposed


def _evaluate_policy(session, agents, initial_states):
    states = _snapshot_agent_states(agents)
    current, density = session.current, session.density
    _restore_agent_states(agents, initial_states)
    initial, initial_density = session.assess(session.initial)
    session.set_current(initial, initial_density)
    rejected = 0
    try:
        for _ in range(session.params.eval_steps):
            _, invalid, _ = _joint_step(session, agents, deterministic=True, store_buffer=False)
            rejected += int(invalid)
        return _poses(agents), session.current, rejected
    finally:
        _restore_agent_states(agents, states)
        session.set_current(current, density)


def refine_constrained(initial, scenario, config, requirements, *, method='local_search',
                       steps=2000, max_evaluations=2000, radius_m=10., seed=0,
                       time_limit_sec=60., output=None, work=None, anchor_poses=None):
    """Select best feasible J, not best coverage. No infeasible-start repair.

    K is fixed during refinement. A sensor-count penalty therefore cannot change
    K here. Joint MAPPO constraints use rejection, not individual resampling.
    """
    if method not in ('local_search', 'mappo'):
        raise ValueError('Expected local_search or mappo')
    for value in (steps, max_evaluations):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Positive integer steps and evaluation limit required')
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('Positive finite radius required')
    if not np.isfinite(time_limit_sec) or time_limit_sec <= 0:
        raise ValueError('Positive finite soft deadline required')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('Seed must be an integer in [0, 2**32)')
    if not isinstance(requirements, CoverageRequirements):
        requirements = CoverageRequirements(requirements)
    root = Path(output) if output is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    session = RefinementSession(initial, scenario, config, requirements, radius_m,
                                work=work, seed=seed, steps=steps, anchor_poses=anchor_poses)
    env, params = session.env, session.params
    shared = _ActionDriver(env.device)
    if method == 'mappo':
        extra_dim = len(session.features(session.initial[0]))
        dim = 5 + params.local_map_size ** 2 + HINT_DIM + extra_dim
        factory = partial(MAPPOActorCritic, hint_dim=HINT_DIM + extra_dim)
        shared = SharedMAPPO(dim, ACTION_DIM, dim * len(initial), params.local_map_size,
                             params, env.device, policy_factory=factory)
    agents = [_ConstrainedAgent(session, shared, p, anchor) for p, anchor in zip(session.initial, session.anchors)]
    initial_states = _snapshot_agent_states(agents)
    best, best_assessment, source = session.initial.copy(), session.current, 'initial'
    trace, training, evaluations = [], [], []
    attempts = rejected = completed = eval_rejected = 0
    stop_reason = 'local_optimum'

    def consider(poses, assessment, origin):
        nonlocal best, best_assessment, source
        if assessment.feasible and assessment.objective_score > best_assessment.objective_score + 1e-9:
            best, best_assessment, source = poses.copy(), assessment, origin
            observe(origin)

    def observe(origin):
        trace.append(dict(elapsed_sec=time.perf_counter() - started, source=origin,
            objective_score=best_assessment.objective_score, coverage_pct=best_assessment.coverage_pct))

    def expired():
        return time.perf_counter() - started >= time_limit_sec

    observe('initial')
    final_policy = final_assessment = None
    if method == 'local_search':
        while attempts < max_evaluations and not expired():
            states = _snapshot_agent_states(agents)
            current, density = session.current, session.density
            next_states, next_assessment, next_density = states, current, density
            for agent in agents:
                _, mask = agent.get_state_and_mask()
                for action in range(1, ACTION_DIM):
                    if attempts >= max_evaluations or expired():
                        break
                    if not mask[action]:
                        continue
                    _restore_agent_states(agents, states)
                    shared.action = action
                    agent.step(deterministic=True, store_buffer=False)
                    assessment, candidate_density = session.assess(_poses(agents))
                    attempts += 1
                    rejected += int(not assessment.feasible)
                    if assessment.feasible and assessment.objective_score > next_assessment.objective_score + 1e-9:
                        next_states = _snapshot_agent_states(agents)
                        next_assessment, next_density = assessment, candidate_density
                _restore_agent_states(agents, states)
            completed += 1
            _restore_agent_states(agents, next_states)
            session.set_current(next_assessment, next_density)
            consider(_poses(agents), next_assessment, 'local_search')
            if next_assessment.objective_score <= current.objective_score + 1e-9:
                break
        if expired():
            stop_reason = 'time_limit'
        elif attempts >= max_evaluations:
            stop_reason = 'evaluation_limit'
    else:
        for step in range(1, steps + 1):
            if expired():
                break
            global_state = torch.cat([ag.get_state_and_mask()[0] for ag in agents])
            reward, invalid, _ = _joint_step(session, agents)
            attempts += 1
            rejected += int(invalid)
            transitions = [ag.consume_transition() for ag in agents]
            next_global = torch.cat([ag.get_state_and_mask()[0] for ag in agents])
            shared.store_timestep(states=torch.stack([t[0] for t in transitions]),
                masks=torch.stack([t[1] for t in transitions]),
                actions=torch.tensor([t[2] for t in transitions], device=env.device),
                log_probs=torch.stack([t[3] for t in transitions]),
                rewards=np.full(len(agents), reward), global_state=global_state,
                next_global_state=next_global, done=False)
            completed = step
            consider(_poses(agents), session.current, 'stochastic_training')
            training.append((step, session.current.objective_score, best_assessment.objective_score,
                             session.current.coverage_pct, reward, int(invalid)))
            if params.eval_interval > 0 and step % params.eval_interval == 0 and step < steps:
                pose, assessment, invalid_count = _evaluate_policy(session, agents, initial_states)
                eval_rejected += invalid_count
                evaluations.append((step, assessment.objective_score, assessment.coverage_pct))
                consider(pose, assessment, 'deterministic_evaluation')
        shared.train_net()
        final_policy, final_assessment, invalid_count = _evaluate_policy(session, agents, initial_states)
        eval_rejected += invalid_count
        evaluations.append((completed, final_assessment.objective_score, final_assessment.coverage_pct))
        consider(final_policy, final_assessment, 'final_policy')
        stop_reason = 'step_limit' if completed == steps else 'time_limit'
    optimization_sec = time.perf_counter() - started

    # Validate every reported deployment with independent direct pose raycasts.
    validation_started = time.perf_counter()
    validated = {}
    for name, pose, expected in [('initial', session.initial, session.initial_assessment),
                                 ('deployment', best, best_assessment),
                                 ('final_policy', final_policy, final_assessment)]:
        if pose is None:
            continue
        validate_poses(pose, session.geometry, config)
        _, cached = session.assess(pose)
        direct_masks = np.stack([session.visibility.provider.mask(p) for p in pose])
        direct = direct_masks.sum(axis=0)
        assessment = session.evaluator.evaluate(pose, direct, sensor_masks=direct_masks)
        if (not np.array_equal(cached, direct) or not assessment.feasible
                or assessment.components != expected.components
                or assessment.robustness != expected.robustness or assessment.regions != expected.regions
                or not np.isclose(assessment.objective_score, expected.objective_score, atol=1e-6, rtol=0)):
            raise RuntimeError(f'Independent validation failed: {name}')
        validated[name] = assessment.to_dict()
        if root is not None:
            np.save(root / f'{name}.npy', pose)
            extra = dict(sensor_masks=direct_masks) if session.evaluator.needs_sensor_masks else {}
            np.savez_compressed(root / f'{name}_coverage.npz', targets=session.visibility.targets, density=direct, **extra)
    if root is not None and method == 'mappo':
        h, e = np.asarray(training).reshape(-1, 6), np.asarray(evaluations).reshape(-1, 3)
        np.savez_compressed(root / 'train_log.npz', step=h[:, 0], objective=h[:, 1],
            best_objective=h[:, 2], coverage=h[:, 3], reward=h[:, 4], rejected=h[:, 5],
            eval_step=e[:, 0], eval_objective=e[:, 1], eval_coverage=e[:, 2])
        torch.save(dict(policy=shared.policy_old.state_dict(), state_dim=dim,
            action_dim=ACTION_DIM, hint_dim=HINT_DIM + extra_dim,
            local_map_size=params.local_map_size, num_agents=len(initial), seed=seed,
            requirements=requirements.specification, initial_pose=session.initial.tolist(),
            anchor_poses=session.anchors.tolist(), local_radius_m=radius_m), root / 'policy.pt')
    if root is not None:
        np.save(root / 'anchors.npy', session.anchors)
    return best, dict(method=method, status='complete', assessments=validated,
        reported_solution_source=source, objective_trace=trace,
        training_steps=completed if method == 'mappo' else 0,
        sweeps=completed if method == 'local_search' else 0,
        proposals=attempts, rejected_proposals=rejected, rejected_evaluation_proposals=eval_rejected,
        stop_reason=stop_reason, seed=seed, local_radius_m=radius_m,
        time_limit_sec=time_limit_sec, requested_steps=steps, max_evaluations=max_evaluations,
        optimization_sec=optimization_sec, validation_and_save_sec=time.perf_counter() - validation_started,
        elapsed_sec=time.perf_counter() - started, device=str(env.device),
        raycast_xyz_queries=session.visibility.raycast_queries,
        validated_direct_raycast=True,
        memoized_xyz_entries=len(env._visible_idx_cache), full_visibility_cache=False,
        reused_checkpoint=False, yaw_postprocessing=False, legacy_reward_shaping=False,
        warm_spawn_search=False, budget_k=requirements.specification['budget_k'], num_selected=len(initial),
        anchor_semantics='fixed sensor-wise transfer anchors; not recentered after repair',
        ppo_updates=len(shared.update_history) if method == 'mappo' else 0,
        ppo_diagnostics=shared.update_history if method == 'mappo' else [],
        params={k: v for k, v in asdict(params).items() if k != 'uncovered_centroid'},
        reward=('J(next)-J(current); infeasible joint proposal: self-loop with -1 reward'
                if method == 'mappo' else None),
        semantics='best feasible J deployment; final policy separate; fresh policy, fixed K',
        constraint_handling='feasible initial required; no repair or relaxation',
        timing_note='Soft limit includes initialization; PPO flush, final evaluation, validation and saving may overrun')

"""Scene-specific, one-step joint placement MAPPO for the workbench.

This is not the roadside 11-action policy or a pretrained transferable model.
Agents propose a candidate (or abstain); duplicate proposals install one sensor.
The shared team reward is exactly the common weighted placement objective.
"""
from dataclasses import asdict
import time
import numpy as np
import torch
from torch import nn
from src.core.evaluation import evaluate_selection
from src.methods.mappo import LegacyMAPPOParams, MAPPOActorCritic, SharedMAPPO


class CandidateActorCritic(MAPPOActorCritic):
    def __init__(self, state_dim, action_dim, global_dim, local_map_size, num_agents):
        nn.Module.__init__(self)
        self.cnn = nn.Identity()
        self.actor_body = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor_head = nn.Linear(64, action_dim)
        self.critic = nn.Sequential(nn.Linear(global_dim, 64), nn.Tanh(), nn.Linear(64, num_agents))

    def _encode_actor(self, state):
        return self.actor_body(state)


def solve_problem(problem, steps=1000, seed=0, progress=None):
    if not isinstance(steps, int) or not 32 <= steps <= 20000:
        raise ValueError('MAPPO steps must be an integer in [32, 20000]')
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an integer in [0, 2**32)')
    k, m = problem.budget_k, len(problem.candidates.candidates)
    if not 1 <= k <= 32 or not 1 <= m <= 5000:
        raise ValueError('UI MAPPO requires K=1..32 and 1..5000 candidates')
    started = time.monotonic()
    params = LegacyMAPPOParams(num_agents=k, episodes=steps, xy_step_m=1,
        z_min=0, z_max=0, z_step=1, yaw_step_deg=1, yaw_axis_min_deg=0,
        yaw_axis_max_deg=360, pitch_step_deg=1, min_pitch=-90, max_pitch=90,
        ppo_update_timestep=64, ppo_k_epochs=4, ppo_reward_scale=1,
        ppo_reward_clip=0, ppo_entropy_coef=0.01)
    best = evaluate_selection(problem, [])
    history = []
    # Fixed-scene contextual bandit: agent identity breaks symmetry. Each joint
    # proposal terminates, so GAE must never bootstrap into another proposal.
    states = torch.eye(k)
    global_state = states.flatten()
    masks = torch.ones((k, m + 1))
    scale = max(1., problem.objective.coverage_weight * len(problem.importance)
                + problem.objective.importance_weight * problem.importance.sum()
                + problem.objective.overlap_penalty * k * len(problem.importance)
                + problem.objective.sensor_count_penalty * k
                + problem.objective.installation_cost_weight * k * problem.costs.max())

    def evaluate(actions):
        return evaluate_selection(problem, sorted({a for a in actions if a < m}))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        learner = SharedMAPPO(k, m + 1, k*k, 0, params, torch.device('cpu'), CandidateActorCritic)
        initial = [p.detach().clone() for p in learner.policy.parameters()]
        for step in range(1, steps + 1):
            samples = [learner.act(states[i], masks[i]) for i in range(k)]
            actions = [a for a, _ in samples]
            result = evaluate(actions)
            if result.objective_score > best.objective_score:
                best = result
            learner.store_timestep(states, masks, torch.tensor(actions),
                torch.stack([lp for _, lp in samples]),
                np.full(k, result.objective_score / scale), global_state, global_state, done=True)
            if step == 1 or step % 64 == 0 or step == steps:
                history.append(dict(step=step, objective=result.objective_score,
                    best_objective=best.objective_score, best_coverage_pct=best.coverage_pct))
                if progress:
                    progress(step, steps)
        learner.train_net()
        policy_result = evaluate([learner.act(states[i], masks[i], deterministic=True)[0] for i in range(k)])
        changed = any(not torch.equal(a, b) for a, b in zip(initial, learner.policy.parameters()))
    best.metadata.update(trainer='candidate_mappo_bandit_v1', result_kind='best_sampled_placement',
        steps=steps, seed=seed, ppo_updates=len(learner.update_history), parameters_changed=changed,
        runtime_sec=time.monotonic()-started, reward_scale=float(scale),
        deterministic_policy=dict(selected_ids=policy_result.selected_ids,
            objective_score=policy_result.objective_score, coverage_pct=float(policy_result.coverage_pct)),
        history=history, update_history=learner.update_history,
        training_params=asdict(params), duplicate_proposals='install_once', pretrained=False)
    return best

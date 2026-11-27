"""Bounded feasibility-first local repair. Failure to find is not a proof."""
from pathlib import Path
import time

import numpy as np

from src.core.coverage_requirements import CoverageRequirements
from src.methods.constrained_refinement import RefinementSession, _ConstrainedAgent, _poses
from src.methods.local_mappo import validate_poses
from src.methods.local_search import _ActionDriver
from src.methods.mappo import ACTION_DIM, _restore_agent_states, _snapshot_agent_states


def _violation(assessment):
    return float(sum(assessment.constraint_residuals.values()))


def _better(candidate, incumbent):
    if candidate.feasible != incumbent.feasible:
        return candidate.feasible
    a, b = _violation(candidate), _violation(incumbent)
    return a < b - 1e-12 or (abs(a - b) <= 1e-12 and
                            candidate.objective_score > incumbent.objective_score + 1e-9)


def _necessary_failures(session):
    spec, poses = session.evaluator.spec, session.initial
    c, cost = spec.get('constraints', {}), spec.get('cost', {})
    n, failures = len(poses), []
    if n > spec['budget_k']:
        failures.append(dict(condition='fixed_sensor_count_exceeds_budget', count=n, budget_k=spec['budget_k']))
    # Include the common validator's tolerance and the original accepted poses,
    # not just the levels used by the action engine.
    pose_tolerance = 1e-5
    height_lower = max(0., min(float(session.config.z_min), float(session.env.zs.min()),
                               float(poses[:, 2].min())) - pose_tolerance)
    lower = n * (cost.get('base_per_sensor', 0.) + cost.get('per_height_m', 0.) * height_lower)
    if lower > c.get('max_installation_cost', float('inf')) + 1e-9:
        failures.append(dict(condition='installation_cost_lower_bound', lower_bound=lower,
                             budget=c['max_installation_cost'], height_lower_bound=height_lower,
                             note='Pose tolerance included; nonnegative location surcharges omitted'))
    if n < 2 and (c.get('min_overlap_ratio', 0.) > 1e-9 or
                 any(r.get('min_overlap_ratio', 0.) > 1e-9 for r in spec.get('regions', []))):
        failures.append(dict(condition='double_coverage_requires_two_sensors', count=n))
    minimum = c.get('min_xy_separation_m', 0.)
    for i in range(n):
        for j in range(i + 1, n):
            delta = session.anchors[i, :2].astype(float) - session.anchors[j, :2].astype(float)
            upper = float(np.linalg.norm(delta) + 2 * (session.radius_m + pose_tolerance))
            if minimum > upper + 1e-9:
                failures.append(dict(condition='pair_separation_upper_bound', pair=[i, j],
                                     upper_bound=upper, required=minimum))
    return failures


def repair_initial_placement(initial, scenario, config, requirements, *, radius_m=10.,
                             max_evaluations=1000, time_limit_sec=30., output=None, work=None):
    """Return a feasible pose array or None; never relax constraints or change K.

    One-agent moves use the existing 11-action domain. Rank by the sum of
    normalized hard-constraint residuals first, then J for ties. No policy is
    created. This local heuristic may miss a feasible solution.
    """
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or max_evaluations < 1:
        raise ValueError('Positive integer repair evaluation limit required')
    if not np.isfinite(radius_m) or radius_m <= 0 or not np.isfinite(time_limit_sec) or time_limit_sec <= 0:
        raise ValueError('Positive finite repair radius and time limit required')
    if not isinstance(requirements, CoverageRequirements):
        requirements = CoverageRequirements(requirements)
    root = Path(output) if output is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    session = RefinementSession(initial, scenario, config, requirements, radius_m,
                                work=work, allow_infeasible=True)
    original = session.initial_assessment
    driver = _ActionDriver(session.env.device)
    agents = [_ConstrainedAgent(session, driver, p, anchor) for p, anchor in zip(session.initial, session.anchors)]
    trace = []
    evaluated = sweeps = 0
    failures = [] if original.feasible else _necessary_failures(session)
    reason = 'already_feasible' if original.feasible else 'necessary_condition' if failures else 'local_stall'

    def expired():
        return time.perf_counter() - started >= time_limit_sec

    def observe():
        a = session.current
        trace.append(dict(elapsed_sec=time.perf_counter() - started, evaluations=evaluated,
            total_violation=_violation(a), objective_score=a.objective_score,
            coverage_pct=a.coverage_pct, violations=list(a.violations)))

    observe()
    while not session.current.feasible and not failures and evaluated < max_evaluations and not expired():
        states = _snapshot_agent_states(agents)
        previous = session.current
        next_states, next_assessment, next_density = states, previous, session.density
        for agent in agents:
            _, mask = agent.get_state_and_mask()
            for action in range(1, ACTION_DIM):
                if evaluated >= max_evaluations or expired():
                    break
                if not mask[action]:
                    continue
                _restore_agent_states(agents, states)
                driver.action = action
                agent.step(deterministic=True, store_buffer=False)
                candidate, density = session.assess(_poses(agents))
                evaluated += 1
                if _better(candidate, next_assessment):
                    next_states = _snapshot_agent_states(agents)
                    next_assessment, next_density = candidate, density
            _restore_agent_states(agents, states)
        sweeps += 1
        _restore_agent_states(agents, next_states)
        session.set_current(next_assessment, next_density)
        if not _better(next_assessment, previous):
            break
        observe()
    best, expected = _poses(agents), session.current
    if expected.feasible:
        status = 'already_feasible' if original.feasible else 'repaired'
        reason = status
    elif failures:
        status = 'infeasible_fixed_count'
    else:
        status = 'no_feasible_found'
        if expired():
            reason = 'time_limit'
        elif evaluated >= max_evaluations:
            reason = 'evaluation_limit'
    search_sec = time.perf_counter() - started
    validation_started = time.perf_counter()
    assessments = {}
    for name, poses, saved in [('initial', session.initial, original), ('best_candidate', best, expected)]:
        if saved.feasible:
            validate_poses(poses, session.geometry, config)
        _, cached = session.assess(poses)
        direct_masks = np.stack([session.visibility.provider.mask(p) for p in poses])
        direct = direct_masks.sum(axis=0)
        assessed = session.evaluator.evaluate(poses, direct, sensor_masks=direct_masks)
        if (not np.array_equal(cached, direct) or assessed.components != saved.components
                or assessed.violations != saved.violations
                or assessed.robustness != saved.robustness or assessed.regions != saved.regions
                or not np.isclose(assessed.objective_score, saved.objective_score, atol=1e-6, rtol=0)):
            raise RuntimeError(f'Independent repair validation failed: {name}')
        assessments[name] = assessed.to_dict()
        if root is not None:
            np.save(root / f'{name}.npy', poses)
            extra = dict(sensor_masks=direct_masks) if session.evaluator.needs_sensor_masks else {}
            np.savez_compressed(root / f'{name}_coverage.npz', targets=session.visibility.targets, density=direct, **extra)
    if root is not None:
        np.save(root / 'anchors.npy', session.anchors)
        if expected.feasible:
            np.save(root / 'feasible.npy', best)
    return (best if expected.feasible else None), dict(status=status, stop_reason=reason,
        assessments=assessments, trace=trace, neighbor_evaluations=evaluated, sweeps=sweeps,
        necessary_condition_failures=failures, certified_infeasible_fixed_count=bool(failures),
        certificate_scope='fixed sensor count, specified physical height grid and original XY anchor radius only',
        validated_direct_raycast=True, full_visibility_cache=False, learning=False,
        local_radius_m=radius_m, max_evaluations=max_evaluations, time_limit_sec=time_limit_sec,
        repair_search_sec=search_sec, validation_and_save_sec=time.perf_counter() - validation_started,
        elapsed_sec=time.perf_counter() - started,
        raycast_xyz_queries=session.visibility.raycast_queries,
        memoized_xyz_entries=len(session.env._visible_idx_cache),
        ranking='normalized hard-constraint residual sum first; J breaks ties',
        timing_note='Soft deadline includes initialization; direct validation and saving can overrun',
        semantics='common initial feasibility repair, not a globally optimal placement or infeasibility solver')

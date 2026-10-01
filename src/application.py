"""Application CLI adapters. Legacy CLI and result schema remain supported."""
from dataclasses import asdict
import json
from pathlib import Path
from src.environment import load_config, get_scenario


def run(args):
    config = load_config(args.config)
    if config.raw.get('application', args.command) != args.command:
        raise ValueError('Config application does not match CLI command')
    method = config.raw.get('optimizer', {}).get('method', 'greedy')
    budget = config.raw.get('optimizer', {}).get('budget_k', 1)
    if method not in ('greedy', 'milp'):
        raise NotImplementedError('Application-conditioned MAPPO requires a new trained policy; use legacy mappo for roadside v3')
    source = config.raw.get('roadside', {}).get('source', 'parametric')
    if args.command == 'roadside' and source != 'parametric':
        raise NotImplementedError('OSM/custom map adapter is not implemented yet')
    if args.dry_run:
        print(json.dumps(dict(application=args.command, method=method, budget_k=budget,
                              objective=config.raw.get('objective', {}), dry_run=True)))
        return 0
    if args.command == 'roadside':
        from src.roadside.problem_builder import build_problem
        sid = config.raw.get('roadside', {}).get('scenario_id', 0)
        problem = build_problem(get_scenario(sid), config.lidar_profile, config, budget)
    else:
        from src.robot.pipeline import build_problem
        problem = build_problem(config, budget)
    if method == 'greedy':
        from src.methods.greedy import solve_problem
        result = solve_problem(problem)
    else:
        from src.methods.milp import solve_problem
        result = solve_problem(problem, config)
    record = asdict(result)
    record['poses'] = result.poses.tolist()
    record.update(schema_version='placement_v1', application=args.command, method=method,
                  pose_columns=['x', 'y', 'z', 'yaw', 'pitch'], objective=asdict(problem.objective),
                  config=config.raw, num_candidates=len(problem.candidates.candidates),
                  num_targets=len(problem.candidates.target_points))
    text = json.dumps(record, indent=2)
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as f:
            f.write(text + '\n')
    print(text)
    return 0

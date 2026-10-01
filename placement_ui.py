"""Local interactive placement workbench. Run with the project virtualenv."""
from __future__ import annotations
import argparse
import json
import threading
from dataclasses import asdict
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, request, send_file, Response
from src.core.models import PlacementProblem
from src.core.objective import ObjectiveWeights
from src.core.mesh_problem import build_mesh_problem
from src.environment import load_config, MapGeometry, get_scenario

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
state = {}
lock = threading.Lock()
progress = {'step': 0, 'total': 0}


def bounded(payload, key, default, minimum, maximum, integer=False):
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'Invalid {key}')
    if not minimum <= value <= maximum or (integer and int(value) != value):
        raise ValueError(f'{key} must be in [{minimum}, {maximum}]')
    return int(value) if integer else float(value)


@app.get('/api/progress')
def training_progress():
    return jsonify(progress)


def scene(problem, mesh):
    return dict(vertices=np.asarray(mesh.vertices).tolist(), triangles=np.asarray(mesh.triangles).tolist(),
                targets=problem.candidates.target_points.tolist(),
                candidates=problem.candidates.positions_array(range(len(problem.candidates.candidates))).tolist(),
                importance=problem.importance.tolist(), candidate_count=len(problem.candidates.candidates))


@app.get('/')
def index():
    return send_file(ROOT / 'ui' / 'index.html')


@app.get('/plotly.js')
def plotly_js():
    from plotly.offline import get_plotlyjs
    return Response(get_plotlyjs(), mimetype='application/javascript')


@app.post('/api/load')
def load_scene():
    payload = request.get_json()
    with lock:
        mode = payload.get('mode', 'robot')
        if mode == 'custom':
            problem, mesh = build_mesh_problem(payload['data'])
            source = 'custom mesh: explicit approved mount candidates'
        elif mode == 'robot':
            import open3d as o3d
            config = load_config(ROOT / 'configs/robot.yaml')
            robot = config.raw['robot']
            mesh = o3d.geometry.TriangleMesh.create_box(*robot['box_dimensions_m'])
            problem, mesh = build_mesh_problem(dict(vertices=np.asarray(mesh.vertices).tolist(),
                triangles=np.asarray(mesh.triangles).tolist(), poses=robot['mount_poses'],
                targets=robot['target_points'], importance=robot['target_weights'],
                sensor=dict(name='DEMO_LIDAR', **config.raw['lidar_profiles']['DEMO_LIDAR'])))
            source = 'box smoke example, not a validated wheelchair model'
        elif mode == 'roadside':
            from src.roadside.problem_builder import build_problem
            sid = int(payload.get('scenario_id', 0))
            if sid not in [0, 1, 2, 3, 6, 9]:
                raise ValueError('Unsupported scenario')
            config = load_config(ROOT / 'configs/experiment.yaml')
            config.candidate.update(xy_step_m=10., z_min=3., z_max=3., yaw_step_deg=55., pitch_step_deg=20., max_orientations_per_position=0)
            # Include all demo candidates even when users later change weights.
            config.raw['objective'] = {'importance_weight': 1.0}
            problem = build_problem(get_scenario(sid), config.lidar_profile, config, 1)
            mesh = MapGeometry(get_scenario(sid)).create_environment_mesh()
            source = f'parametric s{sid}; interactive coarse grid: XY 10m, Z 3m, yaw 55deg, pitch 20deg; no top-M'
        else:
            raise ValueError('Unknown input mode')
        state.update(problem=problem, mesh=mesh, source=source)
        return jsonify({**scene(problem, mesh), 'source': source})


@app.post('/api/solve')
def solve():
    payload = request.get_json()
    with lock:
        if 'problem' not in state:
            raise ValueError('Load a scene first')
        base = state['problem']
        importance = base.importance.copy()
        region = payload.get('region')
        if region:
            bounds = np.asarray(region['bounds'], dtype=float)
            factor = float(region['factor'])
            if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] > bounds[2] or bounds[1] > bounds[3] or not np.isfinite(factor) or factor < 0:
                raise ValueError('Invalid importance region')
            xy = base.candidates.target_points[:, :2]
            mask = ((xy[:, 0] >= bounds[0]) & (xy[:, 0] <= bounds[2]) & (xy[:, 1] >= bounds[1]) & (xy[:, 1] <= bounds[3]))
            importance[mask] *= factor
        budget = bounded(payload, 'budget_k', 1, 0, 100, True)
        problem = PlacementProblem(base.candidates, budget, ObjectiveWeights(**payload['weights']), importance, base.costs)
        method = payload.get('method', 'greedy')
        if method == 'greedy':
            from src.methods.greedy import solve_problem
            limit = bounded(payload, 'greedy_steps', 100, 0, 100, True)
            limited = PlacementProblem(base.candidates, min(budget, limit), problem.objective, importance, base.costs)
            result = solve_problem(limited)
            result.metadata.update(max_selection_steps=limit, effective_budget=min(budget, limit))
        elif method == 'milp':
            from src.methods.milp import solve_problem
            config = load_config(ROOT / 'configs/experiment.yaml')
            config.solver.update(time_limit_sec=bounded(payload, 'milp_seconds', 30, 1, 3600),
                mip_gap=bounded(payload, 'milp_gap', 0.05, 0, 1), threads=1)
            result = solve_problem(problem, config)
            result.metadata.update(time_limit_sec=config.solver['time_limit_sec'], requested_mip_gap=config.solver['mip_gap'])
        elif method == 'mappo':
            import torch
            from src.methods.candidate_mappo import solve_problem
            steps = bounded(payload, 'mappo_steps', 1000, 32, 20000, True)
            seed = bounded(payload, 'seed', 0, 0, 2**32-1, True)
            progress.update(step=0, total=steps)
            old_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                result = solve_problem(problem, steps, seed,
                    lambda step, total: progress.update(step=step, total=total))
            finally:
                torch.set_num_threads(old_threads)
        else:
            raise ValueError('Supported methods: greedy, milp, mappo')
        density = np.zeros(len(importance), dtype=int)
        for i in result.selected_ids:
            density += problem.candidates.candidates[i].covered
        record = dict(schema_version='placement_ui_v1', source=state['source'], method=method,
                      budget_k=problem.budget_k, weights=asdict(problem.objective), region=region,
                      objective_score=result.objective_score, coverage_pct=float(result.coverage_pct),
                      components=result.components, metadata=result.metadata,
                      poses=result.poses.tolist(), pose_columns=['x','y','z','yaw','pitch'],
                      selected_ids=result.selected_ids, density=density.tolist(), importance=importance.tolist())
        return jsonify(record)


@app.errorhandler(ValueError)
@app.errorhandler(KeyError)
@app.errorhandler(TypeError)
@app.errorhandler(NotImplementedError)
def invalid(error):
    return jsonify(error=str(error)), 400


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    app.run(host='127.0.0.1', port=args.port, threaded=True, debug=False)

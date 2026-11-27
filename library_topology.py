"""Opt-in whole-topology source library; no target transfer or training.

The default CLI plans only. Execution uses a new directory and frozen source.
This frontier format is not a drop-in replacement for legacy kK directories.
"""
from argparse import ArgumentParser, SUPPRESS
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from math import comb
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import yaml

from prototype_offline import absolute, canonical, describe, digest, explicit_scenario, save
from src.core.deployment_library import build_frontier
from src.core.runtime import RuntimeMeter
from src.core.visibility import MeshVisibility
from src.environment import (MapGeometry, Scenario, allowed_mount_yaws_deg,
    get_lidar_profile, get_scenario, installation_base_positions, load_config,
    optical_model_metadata, validate_installation_pose)


BASE = Path(__file__).resolve().parent
SCHEMA = 'topology_source_library_run_v1'
ARTIFACTS = ('candidates.npz', 'mesh.npz', 'geometry.json', 'canonical_candidates.json',
             'frontier.json', 'validation.json')


def validate_scenario_fields(row):
    """Reject unsupported physical inputs in this opt-in, unconstrained adapter."""
    allowed = {'id', 'topology', 'road_width', 'num_lanes', 'geometry',
               'parent_scenario_id', 'provenance', 'name', 'changed_fields',
               'num_lanes_note', 'count_in_map', 'locations'}
    if not isinstance(row, dict):
        raise ValueError('Track A scenario must be an object')
    unknown = set(row) - allowed
    if unknown:
        raise ValueError(f'Unsupported Track A scenario fields: {sorted(unknown)}; '
                         'custom obstacles and operational constraints are not implemented')


def _domain(config, scenario):
    validate_scenario_fields(scenario.raw)
    if scenario.topology not in ('STRAIGHT', 'T_JUNCTION', 'CROSS'):
        raise ValueError('Only existing parametric straight/T/cross topologies are supported')
    if config.target.get('road_detection', 1) != 1:
        raise ValueError('Only single-observation union coverage is supported')
    if config.max_orientations_per_position != 0:
        raise ValueError('Set max_orientations_per_position=0 explicitly; no implicit pruning')
    axes = [(config.z_min, config.z_max, config.z_step, 3., 6., 16),
            (config.pitch_min_deg, config.pitch_max_deg, config.pitch_step_deg, -25., -5., 512),
            (config.yaw_axis_min_deg, config.yaw_axis_max_deg, config.yaw_step_deg, 35., 145., 512)]
    for low, high, step, allowed_low, allowed_high, max_levels in axes:
        if (not np.isfinite([low, high, step]).all() or step < 1e-6
                or not allowed_low <= low <= high <= allowed_high
                or (high - low + 1e-6) / step + 1 > max_levels):
            raise ValueError('Invalid or excessively dense installation axis')
    heights = np.asarray(config.target.get('coverage_z_slices', [1., 1.5, 2.]), dtype=float)
    if (heights.ndim != 1 or not 1 <= len(heights) <= 16 or not np.isfinite(heights).all()
            or (heights <= 0).any() or len(np.unique(heights)) != len(heights)):
        raise ValueError('Expected distinct positive finite target heights')
    geometry = MapGeometry(scenario)
    bases = installation_base_positions(geometry, config)
    pitches = np.arange(config.pitch_min_deg, config.pitch_max_deg + 1e-9, config.pitch_step_deg)
    poses = []
    for x, y, z in bases:
        yaws = allowed_mount_yaws_deg(x, y, geometry, config.yaw_step_deg,
                                     config.yaw_axis_min_deg, config.yaw_axis_max_deg)
        if len(poses) + len(yaws) * len(pitches) > 2000:
            raise ValueError('Source smoke guard: at most 2000 poses; explicitly coarsen config')
        poses.extend([x, y, z, float(yaw), float(pitch)] for yaw in yaws for pitch in pitches)
    poses = np.asarray(poses, dtype=float).reshape(-1, 5)
    targets = geometry.target_points(heights)
    if not len(poses) or not len(targets) or len(targets) > 50000 or len(poses) * len(targets) > 4000000:
        raise ValueError('Source smoke guard: nonempty domain, <=50000 targets and <=4000000 mask entries')
    for pose in poses:
        validate_installation_pose(pose, geometry, config)
    return geometry, poses, targets


def plan_library(config, scenario, k_values, *, alternatives_per_k=3, enumeration_limit=100000):
    ks = list(k_values)
    if (not ks or any(type(k) is not int or not 1 <= k <= 64 for k in ks)
            or ks != sorted(set(ks))):
        raise ValueError('Expected increasing unique integer K values in 1..64')
    if (type(alternatives_per_k) is not int or not 1 <= alternatives_per_k <= 100
            or type(enumeration_limit) is not int or not 0 <= enumeration_limit <= 1000000):
        raise ValueError('Invalid alternatives count or enumeration limit')
    geometry, poses, targets = _domain(config, scenario)
    profile = get_lidar_profile(config.lidar_profile, config)
    if (not np.isfinite([profile.h_fov_deg, profile.v_fov_deg, profile.max_range_m]).all()
            or not 0 < profile.h_fov_deg <= 360 or not 0 < profile.v_fov_deg <= 180
            or profile.max_range_m <= 0):
        raise ValueError('Invalid sensor profile')
    subsets = sum(comb(len(poses), r) for r in range(min(max(ks), len(poses)) + 1))
    return dict(schema='topology_source_library_plan_v1', track='A', execute=False,
        scenario=dict(id=scenario.id, topology=scenario.topology, road_width=scenario.road_width,
                      num_lanes=scenario.num_lanes, geometry=scenario.raw.get('geometry')),
        candidate_count=len(poses), target_count=len(targets),
        k_values=ks, alternatives_per_k=alternatives_per_k, enumeration_limit=enumeration_limit,
        subset_count=subsets, enumeration_will_complete=subsets <= enumeration_limit,
        sensor=asdict(profile), **optical_model_metadata(profile),
        candidate_spec=config.candidate, target_spec=config.target,
        objective='unweighted_union_count', budget_mode='at_most_k', operational_constraints=[],
        source_only=True, legacy_k_directory_compatible=False,
        scope='Explicit finite source grid only; no transfer, arbitrary angles, target solver or learning')


def build_library(config, scenario, k_values, *, alternatives_per_k=3, enumeration_limit=100000):
    started = time.perf_counter()
    plan = plan_library(config, scenario, k_values, alternatives_per_k=alternatives_per_k,
                        enumeration_limit=enumeration_limit)
    geometry, poses, targets = _domain(config, scenario)
    mesh = geometry.create_environment_mesh()
    profile = get_lidar_profile(config.lidar_profile, config)
    descriptor = describe(geometry, config)
    descriptor['discretization'] = f'Explicit {config.xy_domain} XYZ grid and all declared orientations; no mask dedup'
    descriptor['sensor'] = asdict(profile)
    relative = canonical(poses, geometry)
    rebuilt = absolute(relative, geometry)
    if (not np.allclose(rebuilt[:, :3], poses[:, :3], atol=1e-7, rtol=0)
            or not np.allclose((rebuilt[:, 3] - poses[:, 3] + 180) % 360 - 180, 0, atol=1e-7)
            or not np.allclose(rebuilt[:, 4], poses[:, 4], atol=1e-7, rtol=0)):
        raise RuntimeError('Source canonical/world roundtrip failed')
    phases = dict(geometry_domain_mesh=time.perf_counter() - started)
    work = RuntimeMeter()
    t0 = time.perf_counter()
    provider = MeshVisibility(mesh, targets, profile, work=work)
    masks = np.asarray([provider.mask(pose) for pose in poses], dtype=bool)
    phases['candidate_raycasts'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    report = build_frontier(poses, masks, targets, plan['k_values'],
        alternatives_per_k=alternatives_per_k, enumeration_limit=enumeration_limit,
        context=dict(track='A', geometry=descriptor, sensor=asdict(profile), **optical_model_metadata(profile),
                     candidate_spec=config.candidate, target_spec=config.target,
                     canonical_format='branch_frame_records_v1',
                     source_only=True, operational_constraints=[]))
    phases['frontier'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    solutions = [solution for row in report['frontier'] for solution in [row, *row['alternatives']]]
    ids = sorted({i for solution in solutions for i in solution['selected_ids']})
    validation_work = RuntimeMeter()
    independent = MeshVisibility(mesh, targets, profile, work=validation_work)
    checked = np.zeros_like(masks)
    for i in ids:
        checked[i] = independent.mask(poses[i])
        if not np.array_equal(checked[i], masks[i]):
            raise RuntimeError('Fresh source raycast differs from archived candidate mask')
    for solution in solutions:
        selected = solution['selected_ids']
        if int(np.any(checked[selected], axis=0).sum()) != solution['covered_count']:
            raise RuntimeError('Saved union differs from independent source raycast')
        solution['canonical'] = [relative[i] for i in selected]
        solution['validated_direct_raycast'] = True
    phases['independent_validation'] = time.perf_counter() - t0
    phases['build_total'] = time.perf_counter() - started
    validation = dict(passed=True, canonical_roundtrip=True, fresh_scene=True,
        checked_solution_records=len(solutions), unique_candidate_ids=ids,
        scope='All stored layouts, via fresh provider for their distinct candidate IDs; same physical implementation',
        build_work=work.report(), validation_work=validation_work.report(), phase_wall_sec=phases)
    report['physical_validation'] = validation
    return dict(frontier=report, validation=validation, geometry=descriptor, canonical=relative,
        archive=dict(poses=poses, targets=targets, masks=masks),
        mesh=dict(vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles)))


def _worker(root):
    root = Path(root).resolve()
    if BASE != root / 'source':
        raise ValueError('Worker must execute frozen source')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'prepared':
        raise ValueError('Worker requires a new prepared run')
    for folder, key in ((root / 'source', 'source_hashes'), (root, 'input_hashes')):
        for name, expected in manifest[key].items():
            if digest(folder / name) != expected:
                raise ValueError(f'Frozen input/source changed: {name}')
    if any((root / name).exists() for name in ARTIFACTS):
        raise FileExistsError('Refusing existing artifacts')
    started = time.perf_counter()
    try:
        manifest['status'] = 'running'
        save(root / 'manifest.json', manifest)
        config = load_config(root / 'inputs/config.yaml')
        scenario = Scenario(**json.loads((root / 'inputs/scenario.json').read_text()))
        result = build_library(config, scenario, manifest['plan']['k_values'],
            alternatives_per_k=manifest['plan']['alternatives_per_k'],
            enumeration_limit=manifest['plan']['enumeration_limit'])
        np.savez_compressed(root / 'candidates.npz', **result['archive'])
        np.savez_compressed(root / 'mesh.npz', **result['mesh'])
        for name, key in (('frontier.json', 'frontier'), ('geometry.json', 'geometry'),
                          ('canonical_candidates.json', 'canonical'), ('validation.json', 'validation')):
            save(root / name, result[key])
        manifest.update(status='complete', domain_sha256=result['frontier']['domain_sha256'],
                        packages={name: package_metadata.version(name) for name in ('numpy', 'open3d', 'PyYAML')},
                        artifact_hashes={name: digest(root / name) for name in ARTIFACTS})
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest.update(worker_work_sec=time.perf_counter() - started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        save(root / 'manifest.json', manifest)


def execute_library(config, scenario, output, k_values, *, alternatives_per_k=3,
                    enumeration_limit=100000, command=None):
    root = Path(output).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'Output must be new: {root}')
    plan = plan_library(config, scenario, k_values, alternatives_per_k=alternatives_per_k,
                        enumeration_limit=enumeration_limit)
    plan['execute'] = True
    sources = ['library_topology.py', 'prototype_offline.py',
               *sorted(str(p.relative_to(BASE)) for p in (BASE / 'src').rglob('*.py'))]
    contents = {name: (BASE / name).read_bytes() for name in sources}
    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve()
    (root / 'inputs').mkdir()
    (root / 'inputs/config.yaml').write_text(yaml.safe_dump(config.raw, sort_keys=False))
    save(root / 'inputs/scenario.json', asdict(scenario))
    for name, content in contents.items():
        path = root / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = dict(schema=SCHEMA, status='prepared', track='A', source_only=True,
        created_at=datetime.now(timezone.utc).isoformat(), command=command, plan=plan,
        python=sys.version, numpy_version=np.__version__,
        input_hashes={name: digest(root / name) for name in ('inputs/config.yaml', 'inputs/scenario.json')},
        source_hashes={name: digest(root / 'source' / name) for name in sources})
    save(root / 'manifest.json', manifest)
    started = time.perf_counter()
    with (root / 'worker.log').open('x') as log:
        completed = subprocess.run([sys.executable, '-B', str(root / 'source/library_topology.py'),
            '--worker-root', str(root), '--execute'], cwd=root / 'source', stdout=log, stderr=subprocess.STDOUT)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['worker_process_wall_sec'] = time.perf_counter() - started
    if completed.returncode:
        manifest.update(status='failed', worker_returncode=completed.returncode)
    save(root / 'manifest.json', manifest)
    if completed.returncode:
        raise RuntimeError(f'Frozen source worker failed: {root / "worker.log"}')
    return root


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=False, default=str(BASE / 'configs/library_topology_smoke.yaml'))
    parser.add_argument('--scenario-id', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--scenario-json')
    parser.add_argument('--output', default='results/library_topology_smoke')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--alternatives-per-k', type=int, default=3)
    parser.add_argument('--enumeration-limit', type=int, default=100000)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-root', help=SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return _worker(args.worker_root)
    config = load_config(args.config)
    scenario = (explicit_scenario(json.loads(Path(args.scenario_json).read_text())) if args.scenario_json
                else get_scenario(args.scenario_id, BASE / 'data/scenarios/yeongjong_final.json'))
    plan = plan_library(config, scenario, args.k_values, alternatives_per_k=args.alternatives_per_k,
                        enumeration_limit=args.enumeration_limit)
    plan['execute'] = args.execute
    print(json.dumps(plan, indent=2, allow_nan=False))
    if not args.execute:
        return plan
    result = execute_library(config, scenario, args.output, args.k_values,
        alternatives_per_k=args.alternatives_per_k, enumeration_limit=args.enumeration_limit,
        command=[sys.executable, str(BASE / 'library_topology.py'), *(sys.argv[1:] if argv is None else argv)])
    print(f'Completed: {result / "frontier.json"}')
    return result


if __name__ == '__main__':
    main()

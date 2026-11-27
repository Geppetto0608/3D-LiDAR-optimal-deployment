"""Bounded two-module deployment reuse pilot, with exact fixed-K=2 references.

Default is a no-write/no-raycast plan. Execution uses a frozen source subprocess.
There is no MAPPO, periodic boundary, learned response transfer or citywide claim.
"""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from src.core.coverage_requirements import CoverageRequirements
from src.environment import ConnectedCorridorGeometry, get_lidar_profile, load_config
from src.methods.connected_pairs import assess_ids, canonical_local_search, exact_pairs


BASE = Path(__file__).resolve().parent
SOURCE_FILES = [
    'connected_corridor_study.py', 'src/__init__.py', 'src/environment.py',
    'src/core/__init__.py', 'src/core/visibility.py', 'src/core/coverage_requirements.py',
    'src/core/evaluation.py', 'src/core/objective.py', 'src/core/models.py',
    'src/core/candidate.py', 'src/methods/__init__.py', 'src/methods/connected_pairs.py',
]
METHODS = ('independent_transfer', 'transfer_ls', 'library_joint', 'library_joint_ls',
           'reference_one_per_module', 'reference_global')


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def candidates(geometry, spec):
    step = spec['station_step_m']
    if isinstance(step, bool) or not np.isfinite(step) or step <= 0:
        raise ValueError('Positive finite candidate step required')
    n = geometry.length_m / step
    if not np.isclose(n, round(n), atol=1e-8, rtol=0):
        raise ValueError('Candidate step must divide each module length')
    count = int(round(n)) * 2 * len(spec['heights_m']) * len(spec['relative_yaws_deg']) * len(spec['pitches_deg'])
    if not 1 <= count <= 600:
        raise ValueError('Pilot candidate allocation limit: 600')
    stations = (np.arange(int(round(n))) + .5) * step
    result = np.array(list(product(stations, [-geometry.mounting_offset_m, geometry.mounting_offset_m],
        spec['heights_m'], spec['relative_yaws_deg'], spec['pitches_deg'])), dtype=float)
    geometry.validate_canonical(result)
    if len(result) != len(np.unique(result, axis=0)):
        raise ValueError('Duplicate candidate coordinates')
    return result


def validate_suite(data):
    required = {'schema', 'geometry', 'angles_deg', 'target_step_m', 'target_heights_m',
                'candidate', 'alternatives_per_site', 'interface_half_length_m',
                'local_station_radius_m', 'requirements'}
    if not isinstance(data, dict) or set(data) != required or data['schema'] != 'connected_corridor_pilot_v1':
        raise ValueError('Invalid connected corridor suite schema/keys')
    if 'turn_deg' in data['geometry']:
        raise ValueError('Specify angles separately')
    grid = data['candidate']
    if set(grid) != {'station_step_m', 'heights_m', 'relative_yaws_deg', 'pitches_deg'}:
        raise ValueError('Invalid candidate fields')
    for key in ('heights_m', 'relative_yaws_deg', 'pitches_deg'):
        if not isinstance(grid[key], list) or not grid[key] or not np.isfinite(grid[key]).all():
            raise ValueError(f'Invalid {key}')
    for value in (grid['station_step_m'], data['target_step_m'], data['interface_half_length_m']):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError('Sampling steps and interface length must be positive')
    if (not isinstance(data['alternatives_per_site'], int)
            or isinstance(data['alternatives_per_site'], bool) or data['alternatives_per_site'] < 1):
        raise ValueError('Positive integer alternatives_per_site required')
    radius = data['local_station_radius_m']
    if isinstance(radius, bool) or not np.isfinite(radius) or radius < 0:
        raise ValueError('Invalid station radius')
    angles = data['angles_deg']
    if (not isinstance(angles, list) or not angles or len(angles) > 21
            or not np.isfinite(angles).all() or len(set(angles)) != len(angles)):
        raise ValueError('Expected 1-21 unique finite angles')
    requirements = CoverageRequirements(data['requirements'])
    if requirements.specification['budget_k'] != 2:
        raise ValueError('This pilot supports fixed K=2 only')
    # Global constraints only: rotated interface ROIs require explicit road masks.
    if data['requirements'].get('regions') or data['requirements'].get('cost', {}).get('regions'):
        raise ValueError('Regional requirements/costs are not part of this pilot')
    source = ConnectedCorridorGeometry(**{**data['geometry'],
        'length_m': data['geometry']['length_m'] / 2, 'turn_deg': 0, 'bend_length_m': 0})
    canonical = candidates(source, grid)
    for angle in angles:
        geometry = ConnectedCorridorGeometry(**data['geometry'], turn_deg=angle)
        upper_targets = (geometry.length_m * geometry.road_width_m / data['target_step_m']**2
                         * len(data['target_heights_m']))
        if (len(canonical) * 2 > 600 or upper_targets > 15000
                or geometry.length_m / geometry.mesh_step_m > 5000):
            raise ValueError('Pilot allocation limit: grid/target/mesh size too large')
        target_points, target_canonical = geometry.target_samples(data['target_step_m'], data['target_heights_m'])
        if len(canonical) * 2 > 600 or len(target_points) > 10000:
            raise ValueError('Pilot limit: 600 candidates and 10000 targets')
        interface = np.abs(target_canonical[:, 0] - geometry.length_m / 2) <= data['interface_half_length_m']
        if not interface.any() or interface.all():
            raise ValueError('Interface must be a nonempty proper subset of targets')
        requirements.bind(target_points)
    return data


def load_suite(path):
    return validate_suite(json.loads(Path(path).read_text()))


def select_source_alternatives(canonical, scores, per_site):
    """Nested source-only pools, preserving the original site/score/ID order."""
    canonical, scores = np.asarray(canonical), np.asarray(scores)
    if (canonical.ndim != 2 or canonical.shape[1] != 5 or scores.shape != (len(canonical),)
            or not np.isfinite(canonical).all() or not np.isfinite(scores).all()
            or isinstance(per_site, bool) or not isinstance(per_site, int) or per_site < 1):
        raise ValueError('Expected canonical poses, finite scores and positive integer pool size')
    selected = []
    for site in np.unique(canonical[:, :2], axis=0):
        ids = np.flatnonzero(np.all(canonical[:, :2] == site, axis=1))
        selected.extend(ids[np.argsort(-scores[ids], kind='stable')[:per_site]].tolist())
    return selected


class MaskCache:
    def __init__(self, mesh, targets, profile, poses):
        from src.core.visibility import MeshVisibility
        self.counts = Counter()
        self.poses = poses
        self.cache = {}
        self.provider = MeshVisibility(mesh, targets, profile, work=self)

    def add(self, name, amount=1):
        self.counts[name] += int(amount)

    def __call__(self, candidate_id):
        cid = int(candidate_id)
        if cid not in self.cache:
            self.cache[cid] = self.provider.mask(self.poses[cid])
        return self.cache[cid]

    def report(self):
        return dict(unique_candidate_queries=len(self.cache), **dict(self.counts))


def source_library(suite, profile, directory):
    directory.mkdir()
    geometry = ConnectedCorridorGeometry(**{**suite['geometry'],
        'length_m': suite['geometry']['length_m'] / 2, 'turn_deg': 0, 'bend_length_m': 0})
    canonical = candidates(geometry, suite['candidate'])
    poses = geometry.poses(canonical)
    targets, target_coordinates = geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
    spec = deepcopy(suite['requirements'])
    spec['budget_k'] = 1
    # An isolated one-sensor source is selected by J, not the two-sensor target thresholds.
    spec['constraints'] = {}
    req = CoverageRequirements(spec).bind(targets)
    started = time.perf_counter()
    cache = MaskCache(geometry.create_environment_mesh(), targets, profile, poses)
    scores = np.array([assess_ids([i], poses, cache, req).objective_score for i in range(len(poses))])
    selected = select_source_alternatives(canonical, scores, suite['alternatives_per_site'])
    best = int(np.argmax(scores))
    if best not in selected:
        raise RuntimeError('Library must include the independent source winner')
    masks = np.asarray([cache(i) for i in range(len(poses))])
    np.savez_compressed(directory / 'library.npz', canonical=canonical, poses=poses,
        masks=masks, targets=targets, target_coordinates=target_coordinates, scores=scores,
        alternative_ids=selected)
    metadata = dict(geometry=asdict(geometry), requirements=spec, best_source_id=best,
        source_candidate_count=len(poses), alternative_count=len(selected),
        selection='Top J per station and installation side, source data only; stable candidate-ID ties',
        runtime_sec=time.perf_counter() - started, work=cache.report(),
        source_pose_mask_hash=digest(directory / 'library.npz'),
        optimality='Best single pose in source grid; alternatives are not target optima')
    save(directory / 'metadata.json', metadata)
    return canonical, selected, best


def metrics(density, coordinates, half_length, interface_half_length, step):
    interface = np.abs(coordinates[:, 0] - half_length) <= interface_half_length
    stations = np.unique(coordinates[:, 0])
    blind_runs = []
    # One sampled longitudinal track per lateral offset/height, no periodic wrap.
    for d, z in np.unique(coordinates[:, 1:], axis=0):
        track = density[(coordinates[:, 1] == d) & (coordinates[:, 2] == z)]
        if len(track) != len(stations):
            raise ValueError('Inconsistent longitudinal sample tracks')
        padded = np.r_[False, track == 0, False].astype(int)
        starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
        blind_runs.extend((ends - starts).tolist())
    return dict(interface_coverage_pct=100. * float(np.mean(density[interface] > 0)),
        interface_overlap_ratio=float(np.mean(density[interface] >= 2)),
        module_coverage_pct=[100. * float(np.mean(density[part] > 0))
            for part in (coordinates[:, 0] < half_length, coordinates[:, 0] >= half_length)],
        max_sampled_blind_run_m=max(blind_runs, default=0) * step,
        blind_run_semantics='Longest consecutive uncovered station-bin run on any fixed d/z track; '
            'includes finite outer ends, sample-based arc length, not continuous visibility/detection')


def persist_method(directory, result, canonical, poses, targets, coordinates, cache, mesh, profile, req, suite):
    from src.core.visibility import MeshVisibility
    directory.mkdir()
    ids = result['selected_ids']
    result['work'] = cache.report()
    if ids is None:
        result['validated_direct_raycast'] = None
        save(directory / 'result.json', result)
        return result
    ids = list(ids)
    if len(ids) != 2 or len(set(ids)) != 2:
        raise RuntimeError('All evaluated deployments must have exactly two distinct candidates')
    masks = np.asarray([cache(i) for i in ids])
    check = MeshVisibility(mesh, targets, profile)
    started = time.perf_counter()
    direct = np.asarray([check.mask(poses[i]) for i in ids])
    if not np.array_equal(masks, direct):
        raise RuntimeError('Saved masks disagree with fresh direct raycasts')
    density = direct.sum(axis=0)
    assessment = req.evaluate(poses[ids], density, sensor_masks=direct)
    if assessment.to_dict() != result['assessment']:
        raise RuntimeError('Saved assessment disagrees with shared evaluator')
    result.update(validated_direct_raycast=True, validation_sec=time.perf_counter() - started,
        **metrics(density, coordinates, suite['geometry']['length_m'] / 2,
                  suite['interface_half_length_m'], suite['target_step_m']))
    np.savez_compressed(directory / 'deployment.npz', candidate_ids=ids, poses=poses[ids],
        canonical=canonical[ids], targets=targets, target_coordinates=coordinates,
        sensor_masks=direct, density=density)
    result['deployment_sha256'] = digest(directory / 'deployment.npz')
    save(directory / 'result.json', result)
    return result


def run_angle(suite, profile, source, alternatives, source_best, angle, directory):
    directory.mkdir()
    started = time.perf_counter()
    geometry = ConnectedCorridorGeometry(**suite['geometry'], turn_deg=angle)
    mesh = geometry.create_environment_mesh()
    targets, coordinates = geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])
    shifted = source.copy()
    shifted[:, 0] += geometry.length_m / 2
    canonical = np.vstack([source, shifted])
    poses = geometry.poses(canonical)
    modules = np.repeat([0, 1], len(source))
    pool = alternatives + [i + len(source) for i in alternatives]
    initial = [source_best, source_best + len(source)]
    req = CoverageRequirements(suite['requirements']).bind(targets)
    np.savez_compressed(directory / 'geometry.npz', vertices=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.triangles), targets=targets, target_coordinates=coordinates,
        canonical_candidates=canonical, poses=poses, module_ids=modules, library_pool=pool)
    save(directory / 'geometry.json', asdict(geometry))
    save(directory / 'requirements.json', suite['requirements'])
    outputs = {}
    for name in METHODS[:4]:
        start = time.perf_counter()
        cache = MaskCache(mesh, targets, profile, poses)
        if name.startswith('library'):
            result = exact_pairs(pool, poses, cache, req, module_ids=modules)
        else:
            assessment = assess_ids(initial, poses, cache, req)
            result = dict(status='complete' if assessment.feasible else 'infeasible_transfer',
                          selected_ids=initial.copy(), assessment=assessment.to_dict())
        initialization = deepcopy(result)
        if name.endswith('_ls'):
            if result['selected_ids'] is None:
                result = dict(status='not_run_no_feasible_library_pair', selected_ids=None, assessment=None)
            else:
                result = canonical_local_search(result['selected_ids'], canonical, poses, cache, req,
                    station_radius_m=suite['local_station_radius_m'])
            result['initialization'] = initialization
        result['online_sec'] = time.perf_counter() - start
        result['timing_scope'] = 'Fresh per-method scene, target mask queries and selection/refinement; '
        result['timing_scope'] += 'shared geometry/source transfer and independent validation/export excluded'
        outputs[name] = persist_method(directory / name, result, canonical, poses, targets,
            coordinates, cache, mesh, profile, req, suite)

    # Only after online results are committed: independent full-domain reference cache.
    cache_start = time.perf_counter()
    reference = MaskCache(mesh, targets, profile, poses)
    masks = np.asarray([reference(i) for i in range(len(poses))])
    np.savez_compressed(directory / 'reference_masks.npz', masks=masks)
    build_sec = time.perf_counter() - cache_start
    for name, quota in [('reference_one_per_module', modules), ('reference_global', None)]:
        result = exact_pairs(range(len(poses)), poses, reference, req, module_ids=quota)
        result['reference_build_sec'] = build_sec
        result['reference_masks_sha256'] = digest(directory / 'reference_masks.npz')
        outputs[name] = persist_method(directory / name, result, canonical, poses, targets,
            coordinates, reference, mesh, profile, req, suite)
    save(directory / 'manifest.json', dict(status='complete', angle_deg=angle,
        completed_at=now(), wall_sec=time.perf_counter() - started, outputs=list(outputs),
        geometry_sha256=digest(directory / 'geometry.npz')))
    plot_case(directory, mesh, targets, outputs, poses, angle)
    return outputs


def plot_case(directory, mesh, targets, outputs, poses, angle, *, methods=METHODS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = int(np.ceil(len(methods)/3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4*rows), constrained_layout=True, squeeze=False)
    vertices = np.asarray(mesh.vertices)
    for ax, name in zip(axes.flat, methods):
        ax.scatter(vertices[:, 0], vertices[:, 1], s=1, c='#aaaaaa', rasterized=True)
        value = outputs[name]
        if value['selected_ids'] is not None:
            with np.load(directory / name / 'deployment.npz', allow_pickle=False) as data:
                density = data['density']
            # Show the middle height slice, while annotations use all targets.
            visible = np.isclose(targets[:, 2], 1.5)
            ax.scatter(targets[visible, 0], targets[visible, 1], s=3,
                c=np.where(density[visible] > 0, '#178a80', '#ce4051'))
            selected = poses[value['selected_ids']]
            ax.scatter(selected[:, 0], selected[:, 1], marker='^', s=65, c='#252525')
            a = value['assessment']
            label = f"{a['coverage_pct']:.2f}%  J={a['objective_score']:.2f}"
            ax.set_title(f"{name}\n{label}; feasible={a['feasible']}", fontsize=10)
        else:
            ax.set_title(f"{name}\n{value['status']}", fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    fig.suptitle(f'Connected corridor: {angle:g} deg, fixed K=2, coarse-grid development pilot')
    fig.savefig(directory / 'comparison.png', dpi=160)
    plt.close(fig)


def summarize(root, suite, jobs):
    rows = []
    for job in jobs:
        if job['status'] != 'complete':
            continue
        directory = root / job['directory']
        values = {name: json.loads((directory / name / 'result.json').read_text()) for name in METHODS}
        optimum = values['reference_global']['assessment']
        for name, result in values.items():
            a = result['assessment']
            if result['selected_ids'] is not None:
                if digest(directory / name / 'deployment.npz') != result['deployment_sha256']:
                    raise ValueError('Deployment hash mismatch')
                with np.load(directory / name / 'deployment.npz', allow_pickle=False) as data:
                    if not np.array_equal(data['sensor_masks'].sum(axis=0), data['density']):
                        raise ValueError('Invalid saved sensor masks/density')
                    repeated = CoverageRequirements(suite['requirements']).bind(data['targets']).evaluate(
                        data['poses'], data['density'], sensor_masks=data['sensor_masks'])
                if json.loads(json.dumps(repeated.to_dict())) != a:
                    raise ValueError('Saved assessment mismatch')
            feasible = a is not None and a['feasible']
            gap = optimum['objective_score'] - a['objective_score'] if feasible and optimum else None
            if gap is not None and gap < -1e-7:
                raise RuntimeError('Feasible online objective exceeds exhaustive reference')
            rows.append(dict(angle_deg=job['angle_deg'], method=name, status=result['status'],
                feasible=feasible, coverage_pct=a['coverage_pct'] if a else None,
                objective_score=a['objective_score'] if a else None,
                objective_gap_to_grid_optimum=gap,
                interface_coverage_pct=result.get('interface_coverage_pct'),
                interface_overlap_ratio=result.get('interface_overlap_ratio'),
                max_sampled_blind_run_m=result.get('max_sampled_blind_run_m'),
                installation_cost=a['components']['installation_cost'] if a else None,
                single_failure_coverage_pct=a['robustness'].get('single_failure_coverage_pct') if a else None,
                online_sec=result.get('online_sec'), reference_solve_sec=result.get('elapsed_sec')
                    if name.startswith('reference') else None,
                candidate_queries=result['work']['unique_candidate_queries'],
                rays_submitted=result['work'].get('rays_submitted', 0),
                validated_direct_raycast=result['validated_direct_raycast'],
                result_json=str((directory / name / 'result.json').relative_to(root))))
    if not rows:
        return
    with (root / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(root / 'summary.json', dict(completed_angles=sum(j['status'] == 'complete' for j in jobs),
        expected_angles=len(jobs), method_rows=len(rows), results=rows,
        note='Development sample points only. Exact means fixed K=2 in this coarse finite candidate '
             'grid and static mesh. No interval/continuous optimum, policy, or speedup guarantee.'))
    lines = ['# Connected Corridor Pilot', '',
        'Fixed two real sensors on a finite road. No periodic copies, detection or MAPPO.',
        'The source is a newly computed straight 40m module at default settings, not the old 410xxx library.',
        'J and all hard constraints are shared. Candidate grids/actions differ from legacy experiments.',
        'All thresholds are illustrative. Exact references enumerate this coarse grid only.',
        'Gap is an objective difference, not a coverage percentage-point difference.', '',
        '| Angle | Method | Feasible | Coverage % | Interface % | J gap | Queries |',
        '|---:|---|---|---:|---:|---:|---:|']
    for row in rows:
        def fmt(key):
            return '-' if row[key] is None else f'{row[key]:.3f}'
        lines.append(f"| {row['angle_deg']:g} | {row['method']} | {row['feasible']} | "
            f"{fmt('coverage_pct')} | {fmt('interface_coverage_pct')} | "
            f"{fmt('objective_gap_to_grid_optimum')} | {row['candidate_queries']} |")
    lines.extend(['', 'Each angle directory contains comparison.png, geometry and per-method deployments.',
        'Timing is not a repeated cold-start benchmark. Reference methods share one exhaustive mask build.',
        'Online methods have independent memoization and cannot access that later-built reference cache.',
        'Library joint selection retains one sensor per module; the global reference has no module quota.',
        'Local search can cross the module interface within its original station radius, but cannot switch sides.',
        'Infeasible/missing deployments are not reported as zero-coverage successes.', ''])
    (root / 'REPORT.md').write_text('\n'.join(lines))


def worker(root):
    suite = load_suite(root / 'inputs/suite.json')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        if digest(BASE / name) != expected:
            raise RuntimeError(f'Frozen source mismatch: {name}')
    for name, expected in manifest['input_hashes'].items():
        if digest(root / name) != expected:
            raise RuntimeError(f'Frozen input mismatch: {name}')
    cfg = load_config(root / 'inputs/experiment.yaml')
    profile = get_lidar_profile(cfg.lidar_profile, cfg)
    started = time.perf_counter()
    try:
        manifest.update(status='running', stage='source_library', worker_started_at=now())
        save(root / 'manifest.json', manifest)
        source, alternatives, best = source_library(suite, profile, root / 'library')
        for job in manifest['jobs']:
            manifest.update(stage='connected_target', current_angle=job['angle_deg'])
            job['status'] = 'running'
            save(root / 'manifest.json', manifest)
            print(f"Angle {job['angle_deg']:g}: start", flush=True)
            results = run_angle(suite, profile, source, alternatives, best, job['angle_deg'], root / job['directory'])
            job['status'] = 'complete'
            save(root / 'manifest.json', manifest)
            summarize(root, suite, manifest['jobs'])
            print(f"Angle {job['angle_deg']:g}: complete, grid optimum "
                  f"{results['reference_global']['assessment']}", flush=True)
        manifest.update(status='complete', stage='complete', finished_at=now())
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['worker_wall_sec'] = time.perf_counter() - started
        save(root / 'manifest.json', manifest)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default=str(BASE / 'configs/connected_corridor_pilot.json'))
    parser.add_argument('--config', default=str(BASE / 'configs/experiment.yaml'))
    parser.add_argument('--output', default='results/connected_corridor_pilot')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-root', help=__import__('argparse').SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return worker(Path(args.worker_root).resolve())
    suite = load_suite(args.suite)
    geometry = ConnectedCorridorGeometry(**suite['geometry'])
    source_geometry = ConnectedCorridorGeometry(**{**suite['geometry'], 'length_m': geometry.length_m / 2,
                                                    'bend_length_m': 0})
    n = len(candidates(source_geometry, suite['candidate']))
    target_count = len(geometry.target_samples(suite['target_step_m'], suite['target_heights_m'])[0])
    config = load_config(args.config)
    profile = get_lidar_profile(config.lidar_profile, config)
    plan = dict(angles_deg=suite['angles_deg'], source_candidates=n, target_candidates=2*n,
        target_count=target_count, pairs_per_global_reference=n*(2*n - 1), fixed_sensor_count=2,
        methods=METHODS, sensor=asdict(profile), execute=args.execute,
        limitation='New coarse-grid corridor geometry; not periodic, not old source library, not MAPPO')
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'inputs').mkdir()
    shutil.copy2(args.suite, root / 'inputs/suite.json')
    shutil.copy2(args.config, root / 'inputs/experiment.yaml')
    for name in SOURCE_FILES:
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, destination)
    packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
                              capture_output=True, text=True).stdout
    (root / 'packages.txt').write_text(packages)
    manifest = dict(status='prepared', created_at=now(), command=sys.argv if argv is None else argv,
        plan=plan, source_hashes={name: digest(root / 'source' / name) for name in SOURCE_FILES},
        input_hashes={name: digest(root / name) for name in ('inputs/suite.json', 'inputs/experiment.yaml')},
        jobs=[dict(angle_deg=angle, directory=f'angle_{i:02d}', status='pending')
              for i, angle in enumerate(suite['angles_deg'])])
    save(root / 'manifest.json', manifest)
    command = [sys.executable, '-u', str(root / 'source/connected_corridor_study.py'),
               '--worker-root', str(root), '--execute']
    with (root / 'worker.log').open('w') as log:
        completed = subprocess.run(command, cwd=root / 'source', stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest.update(status='failed', worker_returncode=completed.returncode)
        save(root / 'manifest.json', manifest)
        raise RuntimeError(f'Worker failed; see {root / "worker.log"}')
    print(f'Completed: {root / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()

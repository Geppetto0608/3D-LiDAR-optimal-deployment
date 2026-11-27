"""Opt-in Track B library for ONE finite straight or bent source component.

Default CLI behavior is a no-write/no-mesh/no-raycast/no-solve plan. --execute
requires a NEW output directory and runs a frozen source snapshot. This is a
bounded smoke workflow, not network decomposition or a transfer benchmark.
The frontier/NPZ format is not a drop-in replacement for legacy kK libraries.
"""
from argparse import ArgumentParser, SUPPRESS
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
from itertools import product
import json
import math
from numbers import Real
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.environment import (ConnectedCorridorGeometry, LidarProfile, get_lidar_profile,
                             load_config, optical_model_metadata)


BASE = Path(__file__).resolve().parent
SPEC_SCHEMA = 'component_source_library_config_v1'
MANIFEST_SCHEMA = 'component_source_library_run_v1'
SCOPE = 'single_finite_component_smoke_only'
SOURCE_FILES = (
    'library_components.py', 'src/__init__.py', 'src/environment.py',
    'src/core/__init__.py', 'src/core/visibility.py', 'src/core/deployment_library.py',
)
ARTIFACTS = ('library.npz', 'geometry.npz', 'metadata.json', 'frontier.json')
LIMITS = dict(max_candidates=600, max_targets=10000, max_mask_entries=2000000,
              max_mesh_sections=5000)
CANONICAL_FIELDS = ['station_m', 'signed_offset_m', 'z_m', 'relative_yaw_deg', 'pitch_deg']
BOUNDARY_CONTEXT = dict(
    source_occluders='Local finite corridor slabs and roadside walls only; no neighboring geometry.',
    neighbor_coverage_must_be_reevaluated=True,
    reuse_rule='Reevaluate all transferred poses against actual neighbor geometry and targets; '
              'source masks and source optimality certificates do not certify composed deployments.',
    road_ends='Open, finite entry and exit; not stitched to another component.',
    ray_physics='Euclidean Open3D MeshVisibility rays; no station-distance or wrapped rays.',
    periodic=False, wrap=False,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def default_spec():
    """New opt-in defaults, independent of legacy source winners and experiments."""
    geometry = ConnectedCorridorGeometry(length_m=40, road_width_m=6, mesh_step_m=1)
    return dict(schema=SPEC_SCHEMA, geometry=asdict(geometry),
        candidate=dict(station_step_m=10, heights_m=[3], relative_yaws_deg=[0], pitches_deg=[-10]),
        target=dict(step_m=1, heights_m=[1.5]), max_k=3,
        alternatives_per_k=3, enumeration_limit=100000, limits=dict(LIMITS))


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum or "unbounded"}]')
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def _axis(values, name, low, high):
    if not isinstance(values, list) or not values or len(values) > LIMITS['max_candidates']:
        raise ValueError(f'{name} must be a nonempty bounded list')
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           or not low <= v <= high for v in values):
        raise ValueError(f'{name} must contain finite values in [{low}, {high}]')
    if len(set(values)) != len(values):
        raise ValueError(f'{name} contains duplicate coordinates')


def validate_spec(spec):
    """Reject unsupported topology/constraints, invalid sampling and oversized work."""
    if not isinstance(spec, dict) or set(spec) != set(default_spec()) or spec['schema'] != SPEC_SCHEMA:
        raise ValueError('Expected single-component source-library config keys/schema')
    spec = json.loads(json.dumps(spec, allow_nan=False))
    try:
        geometry = ConnectedCorridorGeometry(**spec['geometry'])
    except (TypeError, OverflowError) as exc:
        raise ValueError('Invalid single corridor geometry') from exc
    spec['geometry'] = asdict(geometry)
    candidate, target = spec['candidate'], spec['target']
    if (not isinstance(candidate, dict)
            or set(candidate) != {'station_step_m', 'heights_m', 'relative_yaws_deg', 'pitches_deg'}
            or not isinstance(target, dict) or set(target) != {'step_m', 'heights_m'}):
        raise ValueError('Explicit candidate and target sampling fields required')
    _positive(candidate['station_step_m'], 'candidate.station_step_m')
    _positive(target['step_m'], 'target.step_m')
    _axis(candidate['heights_m'], 'candidate.heights_m', 3, 6)
    _axis(candidate['relative_yaws_deg'], 'candidate.relative_yaws_deg', -55, 55)
    _axis(candidate['pitches_deg'], 'candidate.pitches_deg', -25, -5)
    _axis(target['heights_m'], 'target.heights_m', np.nextafter(.23, 1), sys.float_info.max)
    _integer(spec['max_k'], 'max_k', maximum=64)
    _integer(spec['alternatives_per_k'], 'alternatives_per_k', maximum=100)
    _integer(spec['enumeration_limit'], 'enumeration_limit', minimum=0, maximum=1000000)
    if not isinstance(spec['limits'], dict) or set(spec['limits']) != set(LIMITS):
        raise ValueError('All explicit size guard fields are required')
    for key, ceiling in LIMITS.items():
        _integer(spec['limits'][key], key, maximum=ceiling)
    _counts(spec, geometry)
    return spec


def _bins(length, step, label):
    ratio = length / step
    if not math.isfinite(ratio) or not math.isclose(ratio, round(ratio), abs_tol=1e-8, rel_tol=0):
        raise ValueError(f'{label} step must divide length/width exactly')
    if round(ratio) < 1:
        raise ValueError(f'{label} sampling has no bins')
    return int(round(ratio))


def _counts(spec, geometry):
    """Check counts before allocating a Cartesian domain, mesh, or mask table."""
    candidate, target, limits = spec['candidate'], spec['target'], spec['limits']
    stations = _bins(geometry.length_m, candidate['station_step_m'], 'Candidate station')
    m = stations * 2 * math.prod(len(candidate[key])
        for key in ('heights_m', 'relative_yaws_deg', 'pitches_deg'))
    if m > limits['max_candidates']:
        raise ValueError(f'max_candidates exceeded: {m}; explicitly coarsen sampling, no pruning is applied')
    ns = _bins(geometry.length_m, target['step_m'], 'Target station')
    nd = _bins(geometry.road_width_m, target['step_m'], 'Target lateral')
    if max(ns, nd) > limits['max_targets']:
        raise ValueError('max_targets exceeded by a target sampling axis')
    lateral = (np.arange(nd) + .5) * target['step_m'] - geometry.road_width_m / 2
    n = ns * int(np.count_nonzero(np.abs(lateral) > geometry.median_width_m / 2)) * len(target['heights_m'])
    if n < 1:
        raise ValueError('Sampling produced no road targets outside the median')
    if n > limits['max_targets']:
        raise ValueError(f'max_targets exceeded: {n}')
    if m * n > limits['max_mask_entries']:
        raise ValueError(f'max_mask_entries exceeded: {m * n}')
    sections = geometry.length_m / geometry.mesh_step_m
    if not math.isfinite(sections) or math.ceil(sections) + 2 > limits['max_mesh_sections']:
        raise ValueError('max_mesh_sections exceeded (including up to two bend breakpoints)')
    subset_count = sum(math.comb(m, size) for size in range(min(spec['max_k'], m) + 1))
    return dict(candidate_count=m, target_count=n, mask_entries=m*n,
                mesh_sections_upper_bound=math.ceil(sections) + 2, subset_count=subset_count)


def _validate_profile(profile):
    if not isinstance(profile, LidarProfile) or not isinstance(profile.name, str) or not profile.name:
        raise ValueError('A named LidarProfile is required')
    for name in ('h_fov_deg', 'v_fov_deg', 'max_range_m'):
        _positive(getattr(profile, name), f'sensor.{name}')
    if profile.h_fov_deg > 360 or profile.v_fov_deg > 180:
        raise ValueError('Sensor field of view exceeds 360 horizontal / 180 vertical degrees')


def component_ports(geometry):
    """Centerline ports at z=0; heading returned by centerline() is in radians."""
    stations = np.array([0., geometry.length_m])
    xy, headings = geometry.centerline(stations)
    ports = []
    for i, name in enumerate(('entry', 'exit')):
        tangent = np.array([np.cos(headings[i]), np.sin(headings[i]), 0.])
        ports.append(dict(name=name, station_m=float(stations[i]), position_m=[*xy[i].tolist(), 0.],
            heading_deg=float(np.rad2deg(headings[i])), tangent=tangent.tolist(),
            outward_tangent=((-1 if i == 0 else 1) * tangent).tolist(),
            road_width_m=geometry.road_width_m))
    return ports


def plan_component(spec, profile):
    spec = validate_spec(spec)
    _validate_profile(profile)
    geometry = ConnectedCorridorGeometry(**spec['geometry'])
    counts = _counts(spec, geometry)
    return dict(schema='component_source_library_plan_v1', scope=SCOPE, execute=False,
        **counts, k_values=list(range(1, spec['max_k'] + 1)),
        enumeration_will_complete=counts['subset_count'] <= spec['enumeration_limit'],
        config=spec, sensor=asdict(profile), **optical_model_metadata(profile), ports=component_ports(geometry),
        canonical_fields=CANONICAL_FIELDS,
        candidate_domain='Full Cartesian product of station-bin centers, both mounting offsets, '
                         'heights, inward-road-normal-relative yaws and pitches; no score pruning or mask dedup.',
        target_domain='Station/lateral bin centers at explicit heights, excluding the geometry median.',
        budget_mode='at_most_k', objective='unweighted_union_count',
        operational_constraints=[], boundary_context=dict(BOUNDARY_CONTEXT),
        legacy_k_directory_compatible=False,
        optimality_scope='Only this explicit discrete candidate and target domain; '
                         'not continuous placement or whole-road-network optimality.')


def _domain(spec):
    geometry = ConnectedCorridorGeometry(**spec['geometry'])
    counts = _counts(spec, geometry)
    grid = spec['candidate']
    stations = (np.arange(_bins(geometry.length_m, grid['station_step_m'], 'Candidate')) + .5) * grid['station_step_m']
    canonical = np.array(list(product(stations, [-geometry.mounting_offset_m, geometry.mounting_offset_m],
        grid['heights_m'], grid['relative_yaws_deg'], grid['pitches_deg'])), dtype=float)
    poses = geometry.poses(canonical)
    targets, coordinates = geometry.target_samples(spec['target']['step_m'], spec['target']['heights_m'])
    if len(poses) != counts['candidate_count'] or len(targets) != counts['target_count']:
        raise ValueError('Geometry sampling disagrees with preflight counts')
    if not np.isfinite(poses).all() or not np.isfinite(targets).all():
        raise ValueError('World coordinates must be finite at raycaster precision')
    return geometry, dict(canonical=canonical, poses=poses, targets=targets, target_coordinates=coordinates)


class _Work:
    def __init__(self):
        self.counts = Counter()

    def add(self, name, amount=1):
        self.counts[name] += int(amount)


def _solutions(report):
    for row in report['frontier']:
        yield row
        yield from row['alternatives']


def _check_solutions(report, archive, masks):
    geometry = ConnectedCorridorGeometry(**report['context']['geometry'])
    for row in report['frontier']:
        for solution in [row, *row['alternatives']]:
            ids = solution['selected_ids']
            if (len(ids) > row['budget_k'] or len(set(ids)) != len(ids)
                    or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(masks) for i in ids)):
                raise ValueError('Invalid saved solution candidate IDs')
            covered = int(np.any(masks[ids], axis=0).sum())
            # JSON encodes an empty [0,5] layout as []; restore its second axis.
            canonical = np.asarray(solution['canonical'], dtype=float).reshape(-1, 5)
            world = np.asarray(solution['poses'], dtype=float).reshape(-1, 5)
            if (solution['covered_count'] != covered or solution['num_selected'] != len(ids)
                    or not np.isclose(solution['coverage_pct'], 100. * covered / len(archive['targets']))
                    or solution['poses'] != archive['poses'][ids].tolist()
                    or solution['canonical'] != archive['canonical'][ids].tolist()
                    or not np.array_equal(geometry.poses(canonical), world)):
                raise ValueError('Saved solution disagrees with the pose/mask domain')


def build_component_library(spec, profile, *, proposals=()):
    """Explicit in-memory execution: actual rays, shared frontier, fresh validation.

    No files are written here. The CLI uses execute_library() for frozen-source
    execution and persistence. All candidate rows, including equal masks, remain.
    """
    from src.core.deployment_library import build_frontier
    from src.core.visibility import MeshVisibility

    started = time.perf_counter()
    plan = plan_component(spec, profile)
    spec = plan['config']
    geometry, archive = _domain(spec)
    mesh = geometry.create_environment_mesh()
    geometry_archive = dict(vertices=np.asarray(mesh.vertices).copy(), triangles=np.asarray(mesh.triangles).copy())
    phase_wall_sec = dict(geometry_domain_mesh=time.perf_counter() - started)

    started = time.perf_counter()
    work = _Work()
    provider = MeshVisibility(mesh, archive['targets'], profile, work=work)
    archive['masks'] = np.asarray([provider.mask(pose) for pose in archive['poses']], dtype=bool)
    phase_wall_sec['candidate_raycasts'] = time.perf_counter() - started

    started = time.perf_counter()
    context = dict(track='B', scope=SCOPE, geometry=asdict(geometry), sensor=asdict(profile),
        **optical_model_metadata(profile),
        candidate_sampling=spec['candidate'], target_sampling=spec['target'],
        canonical_fields=CANONICAL_FIELDS, ports=plan['ports'], boundary_context=dict(BOUNDARY_CONTEXT),
        operational_constraints=[], optimality_scope=plan['optimality_scope'])
    report = build_frontier(archive['poses'], archive['masks'], archive['targets'], plan['k_values'],
        alternatives_per_k=spec['alternatives_per_k'], enumeration_limit=spec['enumeration_limit'],
        proposals=proposals, context=context)
    phase_wall_sec['frontier'] = time.perf_counter() - started

    started = time.perf_counter()
    ids = sorted({cid for solution in _solutions(report) for cid in solution['selected_ids']})
    validation_work = _Work()
    independent = MeshVisibility(mesh, archive['targets'], profile, work=validation_work)
    checked_masks = np.zeros_like(archive['masks'])
    for cid in ids:
        checked_masks[cid] = independent.mask(archive['poses'][cid])
        if not np.array_equal(checked_masks[cid], archive['masks'][cid]):
            raise ValueError(f'Independent raycast mismatch for candidate {cid}')
    for solution in _solutions(report):
        solution['canonical'] = archive['canonical'][solution['selected_ids']].tolist()
    _check_solutions(report, archive, checked_masks)
    for solution in _solutions(report):
        solution['validated_direct_raycast'] = True
    validation = dict(fresh_scene=True, unique_candidate_ids=ids,
        checked_solution_records=sum(1 for _ in _solutions(report)), work=dict(validation_work.counts),
        scope='Every saved row and alternative union, including empty layouts; '
              'distinct selected candidate IDs independently raycast once against local geometry.')
    report['physical_validation'] = validation
    phase_wall_sec['independent_validation'] = time.perf_counter() - started
    metadata = dict(schema='component_source_library_metadata_v1', scope=SCOPE,
        legacy_k_directory_compatible=False,
        config=spec, geometry=asdict(geometry), sensor=asdict(profile), **optical_model_metadata(profile),
        ports=plan['ports'],
        canonical_fields=CANONICAL_FIELDS, target_coordinate_fields=CANONICAL_FIELDS[:3],
        relative_yaw_reference='Inward road normal: heading + 90 deg on negative offset, '
                               'heading - 90 deg on positive offset; then add relative yaw.',
        port_tangent_reference='tangent points toward increasing station; outward_tangent '
                               'points outside the component at each port.',
        boundary_context=dict(BOUNDARY_CONTEXT), candidate_domain=plan['candidate_domain'],
        target_domain=plan['target_domain'], optimality_scope=plan['optimality_scope'],
        alternatives_semantics='Incumbent first, followed by other feasible source layouts; '
                               'not all claimed near-optimal and not target-scene optima.',
        candidate_count=len(archive['poses']), target_count=len(archive['targets']),
        build_work=dict(work.counts), physical_validation=validation,
        phase_wall_sec=phase_wall_sec,
        timing_scope='Nonoverlapping construction phases measured with time.perf_counter; '
                     'raycast phases include scene setup. Frozen execution adds worker_total, '
                     'measured by the parent around the subprocess, including startup and artifact writes '
                     'but excluding parent source freezing and final timing/hash annotation. '
                     'In-memory builds have no worker_total. Single-run diagnostics, not a benchmark.',
        source_hashes={name: digest(BASE / name) for name in SOURCE_FILES},
        domain_sha256=report['domain_sha256'])
    return dict(frontier=report, metadata=metadata, archive=archive, geometry_archive=geometry_archive)


def _verify_hashes(root, hashes):
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid provenance path')
        if not (root / relative).is_file() or digest(root / relative) != expected:
            raise ValueError(f'File hash mismatch: {name}')


def _worker(root):
    if BASE != (root / 'source').resolve():
        raise ValueError('Worker must run from its frozen source directory')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('status') != 'prepared':
        raise ValueError('Worker requires a newly prepared run')
    if any((root / name).exists() for name in ARTIFACTS):
        raise FileExistsError('Worker refuses existing library artifacts')
    _verify_hashes(BASE, manifest['source_hashes'])
    _verify_hashes(root, manifest['input_hashes'])
    manifest['status'] = 'running'
    _save(root / 'manifest.json', manifest)
    try:
        spec = json.loads((root / 'inputs/config.json').read_text())
        profile = LidarProfile(**json.loads((root / 'inputs/sensor.json').read_text()))
        result = build_component_library(spec, profile)
        np.savez_compressed(root / 'library.npz', **result['archive'])
        np.savez_compressed(root / 'geometry.npz', **result['geometry_archive'])
        _save(root / 'frontier.json', result['frontier'])
        _save(root / 'metadata.json', result['metadata'])
        manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
            domain_sha256=result['frontier']['domain_sha256'],
            artifact_hashes={name: digest(root / name) for name in ARTIFACTS},
            runtime=dict(python=sys.version, executable=sys.executable,
                packages={name: package_metadata.version(name) for name in ('numpy', 'open3d')}))
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        _save(root / 'manifest.json', manifest)


def execute_library(spec, profile, output, *, config_path=None, command=None):
    """Freeze inputs/source, then execute in a NEW directory; never reuse a run."""
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output must be NEW: {output}')
    plan = plan_component(spec, profile)
    # Read every dependency before creating output; no partial run if core is absent.
    sources = {name: (BASE / name).read_bytes() for name in SOURCE_FILES}
    config_bytes = Path(config_path).read_bytes() if config_path is not None else None
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    (output / 'inputs').mkdir()
    _save(output / 'inputs/config.json', plan['config'])
    _save(output / 'inputs/sensor.json', asdict(profile))
    input_names = ['inputs/config.json', 'inputs/sensor.json']
    if config_bytes is not None:
        (output / 'inputs/experiment.yaml').write_bytes(config_bytes)
        input_names.append('inputs/experiment.yaml')
    for name, content in sources.items():
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    manifest = dict(schema=MANIFEST_SCHEMA, scope=SCOPE, status='prepared',
        created_at=datetime.now(timezone.utc).isoformat(),
        command=list(command) if command is not None else None, source_root=str(BASE),
        config_path=str(Path(config_path).resolve()) if config_path is not None else None,
        source_hashes={name: digest(output / 'source' / name) for name in SOURCE_FILES},
        input_hashes={name: digest(output / name) for name in input_names}, plan={**plan, 'execute': True})
    _save(output / 'manifest.json', manifest)
    worker_command = [sys.executable, '-B', str(output / 'source/library_components.py'),
                      '--worker-root', str(output), '--execute']
    with (output / 'worker.log').open('x') as log:
        started = time.perf_counter()
        completed = subprocess.run(worker_command, cwd=output / 'source', stdout=log, stderr=subprocess.STDOUT)
        worker_total = time.perf_counter() - started
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['worker_wall_sec'] = worker_total
    if completed.returncode:
        manifest.update(status='failed', worker_returncode=completed.returncode)
        _save(output / 'manifest.json', manifest)
        raise RuntimeError(f'Component worker failed; see {output / "worker.log"}')
    metadata = json.loads((output / 'metadata.json').read_text())
    metadata['phase_wall_sec']['worker_total'] = worker_total
    _save(output / 'metadata.json', metadata)
    manifest['artifact_hashes']['metadata.json'] = digest(output / 'metadata.json')
    _save(output / 'manifest.json', manifest)
    return output


def load_component_library(directory):
    """Hash-checked, pickle-free roundtrip; does not raycast or solve on load."""
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('Expected a complete component source library')
    if set(manifest.get('artifact_hashes', {})) != set(ARTIFACTS):
        raise ValueError('Incomplete artifact provenance')
    if set(manifest.get('source_hashes', {})) != set(SOURCE_FILES):
        raise ValueError('Incomplete source provenance')
    _verify_hashes(root, manifest['artifact_hashes'])
    _verify_hashes(root / 'source', manifest['source_hashes'])
    _verify_hashes(root, manifest['input_hashes'])
    with np.load(root / 'library.npz', allow_pickle=False) as stored:
        archive = {name: stored[name] for name in stored.files}
    with np.load(root / 'geometry.npz', allow_pickle=False) as stored:
        geometry_archive = {name: stored[name] for name in stored.files}
    report = json.loads((root / 'frontier.json').read_text())
    metadata = json.loads((root / 'metadata.json').read_text())
    expected_optics = optical_model_metadata(LidarProfile(**metadata['sensor']))
    for key, value in expected_optics.items():
        if metadata.get(key) != value or report['context'].get(key) != value:
            raise ValueError('Source optical model mismatch; rebuild limited-FOV library')
    if report['domain_sha256'] != manifest['domain_sha256'] or metadata['domain_sha256'] != report['domain_sha256']:
        raise ValueError('Inconsistent domain provenance')
    if archive['masks'].dtype != bool or archive['masks'].shape != (len(archive['poses']), len(archive['targets'])):
        raise ValueError('Invalid archived mask table')
    geometry = ConnectedCorridorGeometry(**metadata['geometry'])
    if not np.array_equal(geometry.poses(archive['canonical']), archive['poses']):
        raise ValueError('Canonical/world pose roundtrip mismatch')
    if not np.array_equal(geometry.points(archive['target_coordinates']).astype(np.float32), archive['targets']):
        raise ValueError('Canonical/world target roundtrip mismatch')
    _check_solutions(report, archive, archive['masks'])
    return dict(frontier=report, metadata=metadata, archive=archive, geometry_archive=geometry_archive)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--suite', help='Single-component JSON config; CLI sampling overrides are explicit')
    parser.add_argument('--config', default=str(BASE / 'configs/experiment.yaml'), help='Sensor profile config only')
    parser.add_argument('--output', default='results/library_components_smoke')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker-root', help=SUPPRESS)
    for field in ('length_m', 'road_width_m', 'turn_deg', 'bend_length_m', 'mesh_step_m'):
        parser.add_argument('--' + field.replace('_', '-'), type=float)
    parser.add_argument('--station-step-m', type=float)
    parser.add_argument('--candidate-heights-m', type=float, nargs='+')
    parser.add_argument('--relative-yaws-deg', type=float, nargs='+')
    parser.add_argument('--pitches-deg', type=float, nargs='+')
    parser.add_argument('--target-step-m', type=float)
    parser.add_argument('--target-heights-m', type=float, nargs='+')
    parser.add_argument('--max-k', type=int)
    parser.add_argument('--alternatives-per-k', type=int)
    parser.add_argument('--enumeration-limit', type=int)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        raise ValueError('--execute and --dry-run are mutually exclusive')
    if args.worker_root:
        if not args.execute:
            raise ValueError('Worker requires --execute')
        return _worker(Path(args.worker_root).resolve())
    spec = json.loads(Path(args.suite).read_text()) if args.suite else default_spec()
    for field in ('length_m', 'road_width_m', 'turn_deg', 'bend_length_m', 'mesh_step_m'):
        if getattr(args, field) is not None:
            spec['geometry'][field] = getattr(args, field)
    overrides = [('candidate', 'station_step_m', args.station_step_m),
        ('candidate', 'heights_m', args.candidate_heights_m),
        ('candidate', 'relative_yaws_deg', args.relative_yaws_deg),
        ('candidate', 'pitches_deg', args.pitches_deg),
        ('target', 'step_m', args.target_step_m), ('target', 'heights_m', args.target_heights_m)]
    for section, field, value in overrides:
        if value is not None:
            spec[section][field] = value
    for field in ('max_k', 'alternatives_per_k', 'enumeration_limit'):
        if getattr(args, field) is not None:
            spec[field] = getattr(args, field)
    config = load_config(args.config)
    profile = get_lidar_profile(config.lidar_profile, config)
    plan = plan_component(spec, profile)
    plan['execute'] = args.execute
    print(json.dumps(plan, indent=2, allow_nan=False), flush=True)
    if not args.execute:
        return plan
    command = [sys.executable, str(BASE / 'library_components.py'), *(sys.argv[1:] if argv is None else argv)]
    output = execute_library(spec, profile, args.output, config_path=args.config, command=command)
    print(f'Completed: {output / "frontier.json"}', flush=True)
    return output


if __name__ == '__main__':
    main()

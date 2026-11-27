"""Geometry-only Track A target preparation for the shared target evaluator.

Source masks are read only to check archived layouts, never to choose target
poses. No meshes, visibility providers, solvers, files or workers are created.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from library_topology import ARTIFACTS, SCHEMA, _domain, plan_library, validate_scenario_fields
from prototype_offline import absolute, canonical, describe, digest, explicit_scenario
from prototype_segments import (installation_segments, segment_coordinates,
                                segment_key, segment_to_target)
from prototype_transfer import angle_delta, project_poses
from src.environment import Scenario, get_lidar_profile, load_config, optical_model_metadata


def _json(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    json.dumps(value, allow_nan=False)
    return value


def _verify_hashes(root, hashes, required):
    if not isinstance(hashes, dict) or set(hashes) != set(required):
        raise ValueError('Incomplete provenance hashes')
    for name, expected in hashes.items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or '..' in relative.parts
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError('Invalid provenance path')
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'File hash mismatch: {name}')


def _ks(values):
    values = list(values)
    if (not values or any(type(k) is not int or not 1 <= k <= 64 for k in values)
            or values != sorted(set(values))):
        raise ValueError('Expected increasing unique integer K values in 1..64')
    return values


def _poses_equal(actual, expected):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.size == 0:
        actual = actual.reshape(0, 5)
    return (actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual[:, :3], expected[:, :3], atol=1e-7, rtol=0)
            and np.allclose(angle_delta(actual[:, 3], expected[:, 3]), 0, atol=1e-7, rtol=0)
            and np.allclose(actual[:, 4], expected[:, 4], atol=1e-7, rtol=0))


def _check_canonical(records, poses, geometry):
    expected = canonical(poses, geometry)
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError('Invalid canonical candidate count')
    for actual, wanted in zip(records, expected):
        if not isinstance(actual, dict) or set(actual) != set(wanted):
            raise ValueError('Invalid canonical record fields')
        for key, value in wanted.items():
            if isinstance(value, str):
                matches = actual[key] == value
            else:
                matches = (type(actual[key]) in (int, float)
                           and np.isclose(actual[key], value, atol=1e-7, rtol=0))
            if not matches:
                raise ValueError('Canonical/world branch or coordinate mismatch')
    if not _poses_equal(absolute(records, geometry), poses):
        raise ValueError('Canonical/world roundtrip mismatch')


def _domain_digest(poses, targets, masks, context):
    # Same finite-domain fingerprint as build_frontier, without running it.
    result = sha256()
    for array in (np.asarray(poses, dtype=np.float64),
                  np.asarray(targets, dtype=np.float64), np.packbits(masks, axis=1)):
        result.update(str(array.shape).encode('ascii'))
        result.update(np.ascontiguousarray(array).tobytes())
    result.update(json.dumps(context, sort_keys=True).encode('utf-8'))
    return result.hexdigest()


def load_topology_library(source_directory):
    """Validate the new complete source format without executing frozen code."""
    root = Path(source_directory).expanduser().resolve()
    manifest = _json(root / 'manifest.json')
    if (manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete'
            or manifest.get('track') != 'A' or manifest.get('source_only') is not True):
        raise ValueError('Expected a complete Track A source library')
    _verify_hashes(root, manifest.get('artifact_hashes'), ARTIFACTS)
    _verify_hashes(root, manifest.get('input_hashes'),
                   ('inputs/config.yaml', 'inputs/scenario.json'))
    source_files = {str(p.relative_to(root / 'source'))
                    for p in (root / 'source').rglob('*.py')}
    required = {'library_topology.py', 'prototype_offline.py', 'src/environment.py',
                'src/core/deployment_library.py', 'src/core/visibility.py'}
    if not required <= source_files:
        raise ValueError('Incomplete frozen source')
    _verify_hashes(root / 'source', manifest.get('source_hashes'), source_files)
    config = load_config(root / 'inputs/config.yaml')
    scenario = Scenario(**_json(root / 'inputs/scenario.json'))
    plan = manifest['plan']
    profile = get_lidar_profile(config.lidar_profile, config)
    if any(plan.get(key) != value for key, value in optical_model_metadata(profile).items()):
        raise ValueError('Source optical model mismatch; rebuild limited-FOV library')
    expected_plan = plan_library(config, scenario, _ks(plan['k_values']),
        alternatives_per_k=plan['alternatives_per_k'], enumeration_limit=plan['enumeration_limit'])
    expected_plan['execute'] = True
    if plan != expected_plan:
        raise ValueError('Frozen inputs disagree with source plan')
    geometry, expected_poses, expected_targets = _domain(config, scenario)
    with np.load(root / 'candidates.npz', allow_pickle=False) as stored:
        if set(stored.files) != {'poses', 'targets', 'masks'}:
            raise ValueError('Invalid source candidate archive fields')
        poses, targets, masks = (stored[key] for key in ('poses', 'targets', 'masks'))
    if (not np.array_equal(poses, expected_poses)
            or not np.array_equal(targets, expected_targets)
            or masks.dtype != bool or masks.shape != (len(poses), len(targets))):
        raise ValueError('Source domain differs from frozen geometry/configuration')
    with np.load(root / 'mesh.npz', allow_pickle=False) as stored:
        if set(stored.files) != {'vertices', 'triangles'}:
            raise ValueError('Invalid source mesh fields')
        vertices, triangles = stored['vertices'], stored['triangles']
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(vertices).all() or triangles.ndim != 2
            or triangles.shape[1:] != (3,) or not len(triangles)
            or triangles.dtype.kind not in 'iu' or (triangles < 0).any()
            or (triangles >= len(vertices)).any()):
        raise ValueError('Invalid archived mesh')
    descriptor = _json(root / 'geometry.json')
    expected_descriptor = describe(geometry, config)
    expected_descriptor['discretization'] = (
        f'Explicit {config.xy_domain} XYZ grid and all declared orientations; no mask dedup')
    expected_descriptor['sensor'] = asdict(profile)
    if descriptor != expected_descriptor:
        raise ValueError('Source geometry descriptor disagrees with frozen inputs')
    _check_canonical(_json(root / 'canonical_candidates.json'), poses, geometry)
    report = _json(root / 'frontier.json')
    expected_context = dict(track='A', geometry=descriptor, sensor=asdict(profile), **optical_model_metadata(profile),
        candidate_spec=config.candidate, target_spec=config.target,
        canonical_format='branch_frame_records_v1', source_only=True, operational_constraints=[])
    if (report.get('schema') != 'deployment_library_frontier_v1'
            or report.get('context') != expected_context
            or report.get('k_values') != plan['k_values']
            or report.get('candidate_count') != len(poses)
            or report.get('target_count') != len(targets)
            or report.get('budget_mode') != 'at_most_k'
            or report.get('objective') != 'unweighted_union_count'):
        raise ValueError('Invalid source frontier schema, domain counts or context')
    fingerprint = _domain_digest(poses, targets, masks, report['context'])
    if fingerprint != report.get('domain_sha256') or fingerprint != manifest.get('domain_sha256'):
        raise ValueError('Source domain hash mismatch')
    rows = report['frontier']
    if [row['budget_k'] for row in rows] != plan['k_values']:
        raise ValueError('Source frontier budgets mismatch')
    checked, used = 0, set()
    for row in rows:
        if not isinstance(row['alternatives'], list):
            raise ValueError('Invalid source alternatives')
        for solution in [row, *row['alternatives']]:
            ids = solution['selected_ids']
            if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(poses) for i in ids)
                    or ids != sorted(set(ids)) or type(solution['num_selected']) is not int
                    or solution['num_selected'] != len(ids) or len(ids) > row['budget_k']):
                raise ValueError('Invalid source layout IDs or sensor count')
            if not _poses_equal(solution['poses'], poses[ids]):
                raise ValueError('Source layout poses disagree with selected IDs')
            _check_canonical(solution['canonical'], poses[ids], geometry)
            count = int(np.any(masks[ids], axis=0).sum())
            if (type(solution['covered_count']) is not int or solution['covered_count'] != count
                    or not np.isclose(solution['coverage_pct'], 100. * count / len(targets))
                    or solution.get('validated_direct_raycast') is not True):
                raise ValueError('Source layout coverage or validation mismatch')
            checked += 1
            used.update(ids)
    validation = _json(root / 'validation.json')
    if (validation != report.get('physical_validation')
            or any(validation.get(key) is not True for key in ('passed', 'canonical_roundtrip', 'fresh_scene'))
            or validation.get('checked_solution_records') != checked
            or validation.get('unique_candidate_ids') != sorted(used)):
        raise ValueError('Source validation records mismatch')
    return dict(manifest=manifest, manifest_sha256=digest(root / 'manifest.json'),
        directory=str(root), geometry=geometry, config=config, profile=profile,
        poses=poses, targets=targets, frontier=report)


def _physical_geometry(geometry):
    result = asdict(geometry)
    result['scenario'] = dict(topology=geometry.scenario.topology,
                             road_width=geometry.scenario.road_width,
                             num_lanes=geometry.scenario.num_lanes)
    return result


def prepare_problem(source_directory, target_spec, k_values) -> dict:
    """Return the full target domain and transferred FULL-domain candidate IDs.

    target_spec is an explicit_scenario JSON object. k_values must be increasing,
    unique source budgets. Each incumbent/alternative is transferred separately;
    collisions are unavailable, not repaired using other candidates or coverage.
    proposals use build_frontier's format, but contain no target coverage claims.
    The caller must raycast/recombine the pool before evaluating a full reference.
    """
    ks = _ks(k_values)
    validate_scenario_fields(target_spec)
    source = load_topology_library(source_directory)
    if not set(ks) <= set(source['frontier']['k_values']):
        raise ValueError('Requested K missing from source library')
    target_spec = json.loads(json.dumps(target_spec, allow_nan=False))
    scenario = explicit_scenario(target_spec)
    if scenario.topology != source['geometry'].scenario.topology:
        raise ValueError('Transfer requires the same topology')
    geometry, poses, targets = _domain(source['config'], scenario)
    if _physical_geometry(geometry) == _physical_geometry(source['geometry']):
        raise ValueError('Source and target must not have identical geometry, even with different IDs')
    source_segments = installation_segments(source['geometry'])
    target_segments = installation_segments(geometry)
    if source_segments.keys() != target_segments.keys():
        raise ValueError('Installation branch/side structure changed')
    transfers, proposals, pool = [], [], set()
    for row in source['frontier']['frontier']:
        if row['budget_k'] not in ks:
            continue
        for index, solution in enumerate([row, *row['alternatives']]):
            record = dict(budget_k=row['budget_k'], source_budget_k=row['budget_k'],
                source_alternative=None if index == 0 else index - 1,
                source_layout='incumbent' if index == 0 else 'alternative',
                source_selected_ids=list(solution['selected_ids']),
                matching_mode='segment_fraction_v1')
            try:
                coords = segment_coordinates(source['poses'][solution['selected_ids']],
                                             source['geometry'], source_segments)
                raw = segment_to_target(coords, geometry, target_segments)
                record.update(source_coordinates=coords, unprojected_poses=raw.tolist())
                projected = project_poses(raw, geometry, source['config'], coords) if len(raw) else raw
                ids = []
                for pose in projected:
                    matches = np.flatnonzero(np.all(np.isclose(poses, pose, atol=1e-7, rtol=0), axis=1))
                    if len(matches) != 1:
                        raise ValueError('Projected pose has no unique full target-domain ID')
                    ids.append(int(matches[0]))
                if len(set(ids)) != len(ids):
                    raise ValueError('Projection collapsed distinct sensors to identical target IDs')
                target_canonical = canonical(poses[ids], geometry)
                if [segment_key(p) for p in coords] != [segment_key(p) for p in target_canonical]:
                    raise ValueError('Projection changed installation branch/side')
                record.update(status='complete', selected_ids=ids, num_selected=len(ids),
                    target_canonical=target_canonical,
                    projection_displacement_m=np.linalg.norm(poses[ids, :3] - raw[:, :3], axis=1).tolist(),
                    projection_yaw_change_deg=angle_delta(raw[:, 3], poses[ids, 3]).tolist(),
                    projection_pitch_change_deg=(poses[ids, 4] - raw[:, 4]).tolist())
                pool.update(ids)
                proposals.append(dict(selected_ids=ids, budget_k=row['budget_k'],
                    origin=f'track_A_source_k{row["budget_k"]}_layout{index}'))
            except ValueError as exc:
                record.update(status='unavailable', reason=str(exc))
            transfers.append(record)
    context = dict(track='A', mapping='same_topology_segment_fraction_v1',
        source=dict(directory=source['directory'], manifest=source['manifest'],
                    manifest_sha256=source['manifest_sha256'],
                    domain_sha256=source['frontier']['domain_sha256']),
        target_geometry=describe(geometry, source['config']),
        sensor=asdict(source['profile']), **optical_model_metadata(source['profile']),
        candidate_count=len(poses), target_count=len(targets),
        requested_k_values=ks, operational_constraints=[],
        limitations=[
            'Same-topology finite parametric installation branch/side mapping only.',
            'Full target domain generated independently from geometry and frozen source configuration.',
            'Pool contains only successfully transferred stored layouts; no oracle additions or collision repair.',
            'No target visibility, coverage, optimization or source optimality certificate is transferred.',
            'Parent evaluator must evaluate and freeze pool frontier before independent full-domain reference.'])
    json.dumps(dict(context=context, transfers=transfers, proposals=proposals), allow_nan=False)
    return dict(geometry=geometry, poses=poses, targets=targets, profile=source['profile'],
                k_values=ks, pool_ids=sorted(pool), transfers=transfers,
                context=context, proposals=proposals)

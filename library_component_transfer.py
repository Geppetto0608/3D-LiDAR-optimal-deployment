"""Geometry-only Track B composition adapter for a finite connected corridor.

Public API::

    prepare_problem(source_directory, target_spec, k_values) -> dict
    target_spec = {
        'geometry': {'length_m': 160, 'road_width_m': 10, 'turn_deg': 45,
                     'bend_length_m': 32, 'mesh_step_m': 2},
        'modules': 2,
    }

``geometry`` overrides the saved source geometry; {} explicitly retains it.
``modules`` is an integer in [2, 8]. Modules are equal contiguous station
intervals on ONE corridor, not general T/C graph decomposition. Internal
boundaries belong to the following interval. Each interval must contain target
candidates and samples. ``k_values`` is a nonempty increasing sequence of unique
positive integers, interpreted globally as at-most K, without module quotas.

The full target Cartesian lattice is generated BEFORE examining layouts, using
the source's physical station spacing, height/yaw/pitch axes and target sampling.
Source spacing is NOT scaled per module and transferred poses never seed it.
Sampling steps must divide the full target length/width, as in the source API.
The source builder's private validate/count/domain helpers are deliberately
reused so its guards and sampling rules remain identical (M <= 600, M*N <= 2e6).

Each source frontier row and alternative with budget <= max(k_values) is mapped
to every interval by numeric canonical station fraction, mounting-side offset,
height, inward-normal-relative yaw and pitch. Projection minimizes Euclidean
position displacement on the independently generated candidates, constrained to
the same module, side, height, relative yaw and pitch; ties use the first full
domain ID. Exact aligned grids therefore map without displacement. A collision
makes the entire layout unavailable, not a zero-coverage result. Empty layouts
remain valid. Only complete layouts contribute to the sorted unique pool.

Returns geometry (ConnectedCorridorGeometry), poses[M,5], targets[N,3], profile
(unchanged LidarProfile), k_values, pool_ids, transfers, context and proposals.
Every ID is in the FULL target domain, including proposal IDs; a pool evaluator
must remap proposals when slicing poses. Proposals use only the core fields
selected_ids/budget_k/origin. Transfer alternative_index is None for the frontier
row, then zero-based for its saved alternatives (including a repeated incumbent).
Complete transfers have status/selected_ids/budget_k/num_selected/module_index;
unavailable transfers have a reason and null selected_ids/num_selected.
Per-pose projection records contain numeric mapped/projected canonical poses,
target IDs, and position/station/yaw displacement. Context and transfers are JSON
serializable and describe ports, boundaries, domains and verified provenance.

Preparation loads with load_component_library and verifies source hashes before
returning. It creates no mesh, raycaster, coverage matrix, solver or files. The
loader validates its saved masks internally; NO source masks, coverage counts,
bounds or certificates are returned or used to rank target candidates. The parent
must independently raycast target geometry for both pool and full reference.
"""
from dataclasses import asdict
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from library_components import (
    _domain, _validate_profile, _verify_hashes, digest, load_component_library,
    validate_spec,
)
from src.environment import ConnectedCorridorGeometry, LidarProfile, optical_model_metadata


SCOPE = 'equal_station_interval_component_composition_only'


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} must be an integer >= {minimum}'
                         + (f' and <= {maximum}' if maximum is not None else ''))
    return int(value)


def _target_config(source_spec, target_spec):
    if not isinstance(target_spec, dict) or set(target_spec) != {'geometry', 'modules'}:
        raise ValueError('target_spec requires exactly geometry and modules')
    modules = _integer(target_spec['modules'], 'modules', 2, 8)
    overrides = target_spec['geometry']
    if not isinstance(overrides, dict) or set(overrides) - set(source_spec['geometry']):
        raise ValueError('geometry must contain only ConnectedCorridorGeometry fields')
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           or not np.isfinite(value) for value in overrides.values()):
        raise ValueError('geometry values must be finite numbers')
    spec = json.loads(json.dumps(source_spec, allow_nan=False))
    spec['geometry'].update(overrides)
    # Float32 is the eventual raycaster precision, even though mapping uses float64.
    if any(abs(value) > np.finfo(np.float32).max / 4 for value in spec['geometry'].values()):
        raise ValueError('geometry exceeds finite raycaster precision')
    try:
        spec = validate_spec(spec)
    except (TypeError, OverflowError) as exc:
        raise ValueError('Invalid target geometry or sampling') from exc
    return spec, modules


def _module_metadata(geometry, count, canonical, coordinates):
    boundaries = np.linspace(0., geometry.length_m, count + 1)
    ownership = np.searchsorted(boundaries[1:-1], canonical[:, 0], side='right')
    target_ownership = np.searchsorted(boundaries[1:-1], coordinates[:, 0], side='right')
    xy, headings = geometry.centerline(boundaries)
    modules = []
    for index in range(count):
        candidates = np.flatnonzero(ownership == index).tolist()
        targets = np.flatnonzero(target_ownership == index).tolist()
        if boundaries[index + 1] <= boundaries[index] or not candidates or not targets:
            raise ValueError('Each module interval must contain candidates and targets')
        ports = []
        for offset, name in enumerate(('entry', 'exit')):
            port_index = index + offset
            heading = headings[port_index]
            tangent = np.array([np.cos(heading), np.sin(heading), 0.])
            ports.append(dict(name=name, station_m=float(boundaries[port_index]),
                position_m=[*xy[port_index].tolist(), 0.], heading_deg=float(np.rad2deg(heading)),
                tangent=tangent.tolist(), outward_tangent=((2 * offset - 1) * tangent).tolist(),
                road_width_m=geometry.road_width_m, internal=0 < port_index < count))
        modules.append(dict(module_index=index, station_start_m=float(boundaries[index]),
            station_end_m=float(boundaries[index + 1]), end_inclusive=index == count - 1,
            candidate_ids=candidates, target_ids=targets, ports=ports))
    return modules


def _transfer(solution, budget, alternative, module, source_geometry, geometry, archive):
    source_ids = solution['selected_ids']
    canonical = np.asarray(solution['canonical'], dtype=float).reshape(-1, 5)
    source_geometry.validate_canonical(canonical)
    if len(canonical) != len(source_ids) or len(source_ids) > budget:
        raise ValueError('Source layout dimensions or budget mismatch')
    mapped = canonical.copy()
    start, end = module['station_start_m'], module['station_end_m']
    mapped[:, 0] = start + canonical[:, 0] / source_geometry.length_m * (end - start)
    mapped[:, 1] = np.sign(canonical[:, 1]) * geometry.mounting_offset_m
    mapped_poses = geometry.poses(mapped)
    record = dict(module_index=module['module_index'], budget_k=budget,
                  alternative_index=alternative, source_selected_ids=list(source_ids), projection=[])
    target_ids = []
    module_ids = np.asarray(module['candidate_ids'], dtype=int)
    for source_id, desired, pose in zip(source_ids, mapped, mapped_poses):
        # Preserve all nonstation axes exactly; only station position is projected.
        eligible = module_ids[np.all(archive['canonical'][module_ids, 1:] == desired[1:], axis=1)]
        if not len(eligible):
            return dict(record, status='unavailable', reason='no_matching_module_side_orientation',
                        selected_ids=None, num_selected=None)
        distances = np.linalg.norm(archive['poses'][eligible, :3] - pose[:3], axis=1)
        cid = int(eligible[int(np.argmin(distances))])
        projected = archive['canonical'][cid]
        target_ids.append(cid)
        record['projection'].append(dict(source_id=int(source_id), target_id=cid,
            mapped_canonical=desired.tolist(), projected_canonical=projected.tolist(),
            station_displacement_m=float(projected[0] - desired[0]),
            position_displacement_m=float(np.linalg.norm(archive['poses'][cid, :3] - pose[:3])),
            yaw_displacement_deg=float((archive['poses'][cid, 3] - pose[3] + 180) % 360 - 180)))
    if len(set(target_ids)) != len(target_ids):
        return dict(record, status='unavailable', reason='projection_collision',
                    selected_ids=None, num_selected=None)
    return dict(record, status='complete', selected_ids=sorted(target_ids), num_selected=len(target_ids))


def prepare_problem(source_directory, target_spec, k_values):
    """Prepare an immutable-source, geometry-only target problem; see module API."""
    if not isinstance(k_values, (list, tuple, np.ndarray)):
        raise ValueError('k_values must be a nonempty increasing integer sequence')
    try:
        ks = [_integer(k, 'K') for k in k_values]
    except TypeError as exc:
        raise ValueError('k_values must be a nonempty increasing integer sequence') from exc
    if not ks or ks != sorted(set(ks)):
        raise ValueError('k_values must be nonempty, increasing and unique')
    root = Path(source_directory).expanduser().resolve()
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    loaded = load_component_library(root)
    metadata = loaded['metadata']
    source_spec = validate_spec(metadata['config'])
    if (metadata['source_hashes'] != manifest['source_hashes']
            or metadata['geometry'] != source_spec['geometry']
            or json.loads((root / 'inputs/config.json').read_text()) != source_spec
            or json.loads((root / 'inputs/sensor.json').read_text()) != metadata['sensor']
            or loaded['frontier']['context']['sensor'] != metadata['sensor']):
        raise ValueError('Source configuration, profile or provenance mismatch')
    profile = LidarProfile(**metadata['sensor'])
    _validate_profile(profile)
    spec, count = _target_config(source_spec, target_spec)
    # Complete baseline domain is fixed before reading any frontier layout.
    try:
        with np.errstate(over='raise', invalid='raise'):
            geometry, archive = _domain(spec)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError('Target domain exceeds finite raycaster precision') from exc
    if np.any(np.abs(archive['poses'][:, :3]) > np.finfo(np.float32).max):
        raise ValueError('Target poses exceed finite raycaster precision')
    modules = _module_metadata(geometry, count, archive['canonical'], archive['target_coordinates'])
    source_geometry = ConnectedCorridorGeometry(**source_spec['geometry'])
    transfers = []
    for module in modules:
        for row in loaded['frontier']['frontier']:
            budget = _integer(row['budget_k'], 'source budget')
            if budget > max(ks):
                continue
            for alternative, solution in [(None, row), *enumerate(row['alternatives'])]:
                transfers.append(_transfer(solution, budget, alternative, module,
                                           source_geometry, geometry, archive))
    complete = [record for record in transfers if record['status'] == 'complete']
    pool_ids = sorted({cid for record in complete for cid in record['selected_ids']})
    proposals = [dict(selected_ids=list(record['selected_ids']), budget_k=record['budget_k'],
        origin=f"component_module_{record['module_index']}_k{record['budget_k']}_"
               f"alternative_{record['alternative_index']}") for record in complete]
    context = dict(track='B', scope=SCOPE, schema='component_composition_transfer_v1',
        source_directory=str(root), source_hashes=dict(manifest['source_hashes']),
        source_artifact_hashes=dict(manifest['artifact_hashes']),
        source_input_hashes=dict(manifest['input_hashes']),
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        source_domain_sha256=metadata['domain_sha256'], source_geometry=asdict(source_geometry),
        sensor=asdict(profile), **optical_model_metadata(profile),
        target=dict(geometry=asdict(geometry), modules=count,
            candidate=spec['candidate'], target=spec['target']), modules=modules,
        geometry_changed=geometry != source_geometry,
        length_extended=geometry.length_m > source_geometry.length_m,
        candidate_count=len(archive['poses']), target_count=len(archive['targets']),
        budget_mode='at_most_k', budget_scope='global', operational_constraints=[],
        module_quotas=None, candidate_domain='Independent full source-spacing Cartesian station lattice',
        projection_rule='Nearest Euclidean position within module, side, height, relative yaw and pitch; '
                        'first full-domain ID breaks ties',
        pool_rule='Sorted union of complete mapped source rows and alternatives at all available budgets '
                  '<= max requested K; no module quotas or score pruning',
        proposal_id_space='full_target_domain',
        boundary_context=dict(module_model='Equal contiguous station intervals, not T/C graph decomposition',
            ownership='[start,end), except the last interval includes the corridor exit',
            internal_ports='Shared target centerline positions with opposite outward tangents; '
                           'no artificial internal walls or endcaps',
            road_ends='Open finite target corridor entry and exit', periodic=False, wrap=False,
            neighbor_coverage_must_be_reevaluated=True,
            source_masks_transferred=False, source_bounds_transferred=False,
            reuse_rule='Parent must raycast actual full target geometry and targets independently; '
                       'source coverage and optimality do not certify target deployments'))
    # Recheck every frozen input/artifact/source, including changes during preparation.
    if digest(root / 'manifest.json') != context['source_manifest_sha256']:
        raise ValueError('Source manifest changed during preparation')
    _verify_hashes(root, manifest['artifact_hashes'])
    _verify_hashes(root / 'source', manifest['source_hashes'])
    _verify_hashes(root, manifest['input_hashes'])
    context = json.loads(json.dumps(context, allow_nan=False))
    transfers = json.loads(json.dumps(transfers, allow_nan=False))
    return dict(geometry=geometry, poses=archive['poses'], targets=archive['targets'],
                profile=profile, k_values=ks, pool_ids=pool_ids, transfers=transfers,
                context=context, proposals=proposals)

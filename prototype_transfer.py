"""Held-out road width transfer; no online full cache or training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from prototype_offline import describe, digest, save, width_scenario, canonical, absolute
from src.environment import (MapGeometry, Scenario, allowed_mount_yaws_deg,
    evaluate_coverage, get_lidar_profile, load_config)
from src.roadside.candidate_generator import _base_positions, _frange
from src.core.visibility import MeshVisibility


def angle_delta(a, b):
    return (np.asarray(b) - np.asarray(a) + 180) % 360 - 180


class IncompatiblePrototypes(ValueError):
    """No correspondence under the pilot's fixed installation-line rule."""


def align_poses(low, high, geometry, low_keys=None, matching='strict'):
    """Match sensors using template reflections, never held-out coverage."""
    if matching not in ('strict', 'geometry'):
        raise ValueError('Unknown matching mode')
    if low.shape != high.shape or len(low) == 0:
        raise IncompatiblePrototypes('Interpolation requires equal nonzero actual sensor counts')
    cy = geometry.road_centers_y[0]
    best = None
    for reflect_x in (False, True):
        for reflect_y in (False, True):
            if geometry.scenario.topology != 'STRAIGHT':
                if reflect_x and not np.isclose(geometry.junction_x_m, geometry.map_len / 2):
                    continue
                if reflect_y and not np.isclose(geometry.junction_y_m, geometry.map_width / 2):
                    continue
            if geometry.scenario.topology == 'T_JUNCTION' and reflect_y:
                continue
            p = high.copy()
            if reflect_x:
                p[:, 0] = geometry.map_len - p[:, 0]
                p[:, 3] = (180 - p[:, 3]) % 360
            if reflect_y:
                p[:, 1] = 2 * cy - p[:, 1]
                p[:, 3] = -p[:, 3] % 360
            side_match = (low[:, 1, None] > cy) == (p[None, :, 1] > cy)
            if geometry.scenario.topology != 'STRAIGHT':
                # Width-normalized prototypes supplied by interpolate share this geometry.
                a, b = low_keys if low_keys is not None else canonical(low, geometry), canonical(p, geometry)
                side_match = np.array([[ (u['branch_id'],u['installation_line_id']) ==
                    (v['branch_id'],v['installation_line_id']) for v in b] for u in a])
            cost = ((low[:, 0, None] - p[None, :, 0]) / geometry.map_len) ** 2
            cost += ((low[:, 1, None] - p[None, :, 1]) / geometry.map_width) ** 2
            cost += ((low[:, 2, None] - p[None, :, 2]) / 6) ** 2
            cost += (angle_delta(low[:, 3, None], p[None, :, 3]) / 180) ** 2
            cost += ((low[:, 4, None] - p[None, :, 4]) / 90) ** 2
            if matching == 'strict':
                cost[~side_match] = 1e6
            rows, cols = linear_sum_assignment(cost)
            if matching == 'strict' and not side_match[rows, cols].all():
                continue
            score = float(cost[rows, cols].sum())
            if best is None or score < best[0]:
                best = (score, p[cols], dict(reflect_x=reflect_x, reflect_y=reflect_y,
                    high_indices=cols.tolist(), matching_cost=score))
    if best is None:
        raise IncompatiblePrototypes('No same-installation-side correspondence; add compatible prototypes')
    return best[1], best[2]


def project_to_target(poses: np.ndarray, geometry: MapGeometry, config) -> np.ndarray:
    """Minimum total squared XYZ displacement onto valid, distinct target poses.

    Each XYZ uses that sensor's nearest allowed yaw/pitch. Uniqueness applies to
    the complete pose, not the pole. No visibility scores enter this assignment.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 5 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Expected finite nonempty [K,5] poses')
    base = np.asarray(_base_positions(geometry, config), dtype=float).reshape(-1, 3)
    if not len(base):
        raise ValueError('Target map has no valid installation candidates')
    pitches = np.asarray(list(_frange(config.pitch_min_deg, config.pitch_max_deg, config.pitch_step_deg)))
    yaws = [np.asarray(allowed_mount_yaws_deg(x, y, geometry, config.yaw_step_deg,
        config.yaw_axis_min_deg, config.yaw_axis_max_deg)) for x, y, z in base]
    alternatives = []
    for pose in poses:
        alternatives.append(np.column_stack([base,
            [v[np.argmin(np.abs(angle_delta(pose[3], v)))] for v in yaws],
            np.full(len(base), pitches[np.argmin(np.abs(pitches-pose[4]))])]))
    pool, inverse = np.unique(np.concatenate(alternatives), axis=0, return_inverse=True)
    cost = np.full((len(poses), len(pool)), np.inf)
    for i, ids in enumerate(inverse.reshape(len(poses), len(base))):
        cost[i, ids] = np.sum((base-poses[i, :3])**2, axis=1)
    try:
        rows, cols = linear_sum_assignment(cost)
    except ValueError as exc:
        raise ValueError('No distinct feasible projection for all sensors') from exc
    if len(rows) != len(poses):
        raise ValueError('Too few distinct target poses')
    result = pool[cols]
    if not all(geometry.is_installable(*p[:3]) for p in result):
        raise RuntimeError('Target projection violated installation constraints')
    return result


def transfer_raw(poses, source, target):
    if source.scenario.topology != target.scenario.topology:
        raise ValueError('Cannot transfer between different topologies')
    if source.scenario.topology != 'STRAIGHT':
        mapped = canonical(poses, source)
        for p in mapped:
            p['d_m'] += np.sign(p['d_m']) * (target.sidewalk_center_offset-source.sidewalk_center_offset)
        return absolute(mapped, target)
    result = poses.copy()
    side = np.where(poses[:, 1] > source.road_centers_y[0], 1., -1.)
    offset = np.abs(poses[:, 1] - source.road_centers_y[0]) - source.sidewalk_center_offset
    result[:, 0] *= target.map_len / source.map_len
    result[:, 1] = target.road_centers_y[0] + side * (target.sidewalk_center_offset + offset)
    return result


def project_poses(raw, geometry, config, branch_keys=None):
    """Project to the existing XYZ/yaw/pitch grid without visibility scoring."""
    base = np.asarray(_base_positions(geometry, config))
    pitches = np.asarray(list(_frange(config.pitch_min_deg, config.pitch_max_deg, config.pitch_step_deg)))
    result = raw.copy()
    cy = geometry.road_centers_y[0]
    for i, pose in enumerate(raw):
        same_side = base[(base[:, 1] > cy) == (pose[1] > cy)]
        if geometry.scenario.topology != 'STRAIGHT':
            key = branch_keys[i] if branch_keys is not None else canonical([pose], geometry)[0]
            base_keys = canonical(np.column_stack([base, np.zeros((len(base),2))]), geometry)
            mask = [(p['branch_id'],p['installation_line_id']) ==
                (key['branch_id'],key['installation_line_id']) for p in base_keys]
            same_side = base[mask]
        if not len(same_side):
            raise ValueError('No valid installation positions on required side')
        result[i, :3] = same_side[np.argmin(np.sum((same_side - pose[:3]) ** 2, axis=1))]
        yaws = allowed_mount_yaws_deg(*result[i, :2], geometry, config.yaw_step_deg,
            config.yaw_axis_min_deg, config.yaw_axis_max_deg)
        result[i, 3] = yaws[np.argmin(np.abs(angle_delta(pose[3], yaws)))]
        result[i, 4] = pitches[np.argmin(np.abs(pitches - pose[4]))]
        if not geometry.is_installable(*result[i, :3]):
            raise ValueError('Projection produced an invalid installation')
    if len(np.unique(result, axis=0)) != len(result):
        raise ValueError('Projection collapsed distinct sensors to identical poses')
    return result


def interpolate(low, high, low_geometry, high_geometry, target, config, matching='strict'):
    if len({g.scenario.topology for g in (low_geometry, high_geometry, target)}) != 1:
        raise ValueError('Interpolation requires matching topologies')
    w0, w1, w = [g.scenario.road_width for g in (low_geometry, high_geometry, target)]
    if not w0 < w < w1:
        raise ValueError('Test width must be strictly between distinct library widths')
    if low_geometry.scenario.topology == 'STRAIGHT':
        aligned, mapping = align_poses(low, high, high_geometry, matching=matching)
    else:
        aligned, mapping = align_poses(transfer_raw(low,low_geometry,high_geometry), high, high_geometry,
            low_keys=canonical(low,low_geometry), matching=matching)
    a = (w - w0) / (w1 - w0)
    p0 = transfer_raw(low, low_geometry, target)
    p1 = transfer_raw(aligned, high_geometry, target)
    raw = (1-a)*p0 + a*p1
    raw[:, 3] = (p0[:, 3] + a*angle_delta(p0[:, 3], p1[:, 3])) % 360
    projected = (project_to_target(raw, target, config) if matching == 'geometry' else
        project_poses(raw, target, config, canonical(low,low_geometry)))
    return projected, dict(alpha=a, correspondence=mapping, matching_mode=matching,
        projection_rule='global target grid' if matching == 'geometry' else 'same installation line',
        projection_displacement_m=np.linalg.norm(projected[:, :3]-raw[:, :3], axis=1).tolist(),
        projection_yaw_change_deg=angle_delta(raw[:, 3], projected[:, 3]).tolist(),
        low_canonical=canonical(low,low_geometry), aligned_high_canonical=canonical(aligned,high_geometry),
        target_canonical=canonical(projected,target),
        unprojected_poses=raw.tolist(), aligned_high_poses=aligned.tolist())


def load_prototype(root, k):
    root = Path(root)
    result = json.loads((root/f'k{k}/result.json').read_text())
    desc = json.loads((root/'geometry.json').read_text())
    scenario = Scenario(**desc['geometry']['scenario'])
    if scenario.topology not in ('STRAIGHT','T_JUNCTION','CROSS') or result['budget_k'] != k:
        raise ValueError('Unsupported prototype topology or budget')
    poses = np.load(root/result['positions_npy'], allow_pickle=False)
    if poses.shape != (result['num_selected'], 5) or not np.isfinite(poses).all():
        raise ValueError('Invalid stored prototype poses')
    config = load_config(root/'config.yaml')
    if config.target.get('road_detection', 1) != 1:
        raise ValueError('Only binary union coverage supported')
    return root, result, desc, MapGeometry(scenario), poses, config


def prepare_placements(low, high, low_geometry, high_geometry, target, config, matching='strict'):
    """Retain single-source controls when strict interpolation is unavailable."""
    if matching == 'road_relative':
        return prepare_relative_placements(low, high, low_geometry, high_geometry, target, config)
    placements = {}
    try:
        proposed, metadata = interpolate(low, high, low_geometry, high_geometry, target, config, matching)
        placements['interpolation'] = proposed
        metadata['status'] = 'available'
    except IncompatiblePrototypes as exc:
        metadata = dict(status='unavailable', reason=str(exc),
            low_canonical=canonical(low, low_geometry), high_canonical=canonical(high, high_geometry),
            matching_mode=matching, matching_rule='template reflections and specified matching mode')
    for label, poses, geometry in [('low', low, low_geometry), ('high', high, high_geometry)]:
        placements[f'transfer_{label}'] = project_poses(
            transfer_raw(poses, geometry, target), target, config, canonical(poses, geometry))
    return placements, metadata


def relative_coordinates(poses, geometry):
    """Coordinates along each template installation line, anchored at junction edge.

    Straight roads keep signed distance from the map center. Junction arms use
    outward distance from the road intersection boundary, not a global XY mean.
    """
    coordinates = canonical(poses, geometry)
    for p in coordinates:
        p['s_anchor_m'] = 0. if geometry.scenario.topology == 'STRAIGHT' else geometry.scenario.road_width/2
        p['s_from_anchor_m'] = p['s_m']-p['s_anchor_m']
        p['line_offset_m'] = abs(p['d_m'])-geometry.sidewalk_center_offset
    return coordinates


def relative_to_target(coordinates, target):
    mapped = []
    for p in coordinates:
        q = dict(p)
        anchor = 0. if target.scenario.topology == 'STRAIGHT' else target.scenario.road_width/2
        q['s_m'] = p['s_from_anchor_m']+anchor
        q['d_m'] = np.sign(p['d_m'])*(target.sidewalk_center_offset+p['line_offset_m'])
        mapped.append(q)
    return absolute(mapped, target)


def prepare_relative_placements(low, high, low_geometry, high_geometry, target, config):
    """Use the same road-relative transfer for both controls and interpolation.

    Incompatible branch/side allocations are explicitly unavailable, never
    replaced with a best single transfer labelled as interpolation.
    """
    if len({g.scenario.topology for g in (low_geometry, high_geometry, target)}) != 1:
        raise ValueError('Road-relative transfer requires matching topologies')
    w0, w1, w = [g.scenario.road_width for g in (low_geometry,high_geometry,target)]
    if not w0 < w < w1:
        raise ValueError('Held-out width must lie strictly between prototype widths')
    c0, c1 = relative_coordinates(low,low_geometry), relative_coordinates(high,high_geometry)
    placements = {
        'transfer_low': project_poses(relative_to_target(c0,target),target,config,c0),
        'transfer_high': project_poses(relative_to_target(c1,target),target,config,c1),
    }
    meta = dict(matching_mode='road_relative', alpha=(w-w0)/(w1-w0),
                coordinate_rule='same branch/side; s from junction edge (straight: map center)',
                low_coordinates=c0, high_coordinates=c1)
    try:
        normalized_low = relative_to_target(c0,high_geometry)
        aligned, correspondence = align_poses(normalized_low,high,high_geometry,low_keys=c0)
        aligned_c1 = relative_coordinates(aligned,high_geometry)
        alpha = meta['alpha']
        interpolated = []
        for a, b in zip(c0,aligned_c1):
            if (a['branch_id'],a['installation_line_id']) != (b['branch_id'],b['installation_line_id']):
                raise IncompatiblePrototypes('No same branch/installation-line correspondence')
            p = dict(a)
            for field in ('s_from_anchor_m','line_offset_m','z_m','pitch_deg'):
                p[field] = (1-alpha)*a[field]+alpha*b[field]
            p['relative_yaw_deg'] = float(a['relative_yaw_deg']+alpha*angle_delta(a['relative_yaw_deg'],b['relative_yaw_deg']))
            interpolated.append(p)
        raw = relative_to_target(interpolated,target)
        poses = project_poses(raw,target,config,c0)
        placements['interpolation'] = poses
        meta.update(status='available', correspondence=correspondence,
                    aligned_high_coordinates=aligned_c1, interpolated_coordinates=interpolated,
                    unprojected_poses=raw.tolist(),
                    projection_displacement_m=np.linalg.norm(poses[:,:3]-raw[:,:3],axis=1).tolist())
    except IncompatiblePrototypes as exc:
        meta.update(status='unavailable', reason=str(exc))
    return placements, meta


def run(args):
    input_started = time.perf_counter()
    p0, p1 = load_prototype(args.low, args.k), load_prototype(args.high, args.k)
    config = p0[5]
    if p0[3].scenario.topology != p1[3].scenario.topology:
        raise ValueError('Library topologies must match')
    for field in ('candidate', 'target'):
        if getattr(config, field) != getattr(p1[5], field):
            raise ValueError(f'Prototype {field} settings differ')
    profile = get_lidar_profile(config.lidar_profile, config)
    if asdict(profile) != asdict(get_lidar_profile(p1[5].lidar_profile, p1[5])):
        raise ValueError('Prototype sensor models differ')
    fields = ('map_len', 'map_width', 'grid_size', 'sidewalk_width', 'building_height', 'median_width', 'road_centers_y')
    if any(getattr(p0[3], f) != getattr(p1[3], f) for f in fields) or p0[3].num_lanes != p1[3].num_lanes:
        raise ValueError('Pilot requires width-only variants')
    parent = {'STRAIGHT':0,'T_JUNCTION':1,'CROSS':2}[p0[3].scenario.topology]
    target = MapGeometry(width_scenario(args.width, parent))
    if any(getattr(target, f) != getattr(p0[3], f) for f in fields):
        raise ValueError('Held-out map must share the fixed geometry parameters')
    input_sec = time.perf_counter() - input_started
    started = time.perf_counter()
    placements, metadata = prepare_placements(p0[4], p1[4], p0[3], p1[3], target, config, args.matching)
    placement_sec = time.perf_counter() - started
    print(json.dumps(dict(width=args.width, k=args.k, interpolation=metadata,
        methods=list(placements), online_full_visibility_cache=False)), flush=True)
    if not args.execute:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version='road_width_transfer_v3', status='running', topology=target.scenario.topology,
        matching_mode=args.matching, local_refinement='not_run',
        command=[sys.executable, *sys.argv], held_out_width_m=args.width, budget_k=args.k,
        online_full_visibility_cache=False, training=False, placement_generation_sec=placement_sec,
        online_input_load_and_geometry_sec=input_sec,
        timing_excludes=['Python/import startup', 'offline source parity checks', 'artifact/HTML export'],
        interpolation=metadata, sources=[], results={},
        limitations=['Two-prototype single-map pilot, not generalization evidence',
            'No held-out MILP reference or MAPPO refinement in this stage',
            'Reflections assume template geometry; arbitrary roads unsupported'])
    save(output/'report.json', report)
    try:
        # Keep a small self-contained prototype library, not the entire offline cache.
        for label, p in [('low', p0), ('high', p1)]:
            root, result, desc, geometry, poses, cfg = p
            dest = output/'library'/label
            dest.mkdir(parents=True)
            for name in ('geometry.json', 'config.yaml'):
                shutil.copy2(root/name, dest/name)
            np.save(dest/'poses.npy', poses)
            save(dest/'result.json', dict(result, positions_npy='poses.npy'))
            # Check direct online raycasting against the shared cached evaluator on source maps.
            pts = geometry.target_points(cfg.target['coverage_z_slices'])
            provider = MeshVisibility(geometry.create_environment_mesh(), pts, profile)
            source_density = np.stack([provider.mask(pose) for pose in poses]).sum(axis=0)
            direct = source_density > 0
            cached = evaluate_coverage(poses, geometry.scenario, cfg.lidar_profile, cfg)
            if not np.array_equal(direct, cached.covered_mask) or int(direct.sum()) != result['covered_count']:
                raise RuntimeError('Direct raycast/source cached evaluator parity failed')
            export_view(dest/'view.html', geometry.create_environment_mesh(), pts, source_density,
                poses, f'Library {geometry.scenario.road_width:g}m', result['coverage_pct'])
            report['sources'].append(dict(label=label, width_m=geometry.scenario.road_width,
                source_root=str(root.resolve()), result_sha256=digest(root/f'k{args.k}/result.json'),
                poses_sha256=digest(root/result['positions_npy']), direct_cached_parity=True,
                certification=result['certification'], coverage_pct=result['coverage_pct']))
        save(output/'library/index.json', dict(schema_version='road_width_library_v2', topology=target.scenario.topology,
            budget_k=args.k, prototypes=report['sources'], test_width_excluded=args.width))
        setup = time.perf_counter()
        mesh = target.create_environment_mesh()
        targets = target.target_points(config.target['coverage_z_slices'])
        provider = MeshVisibility(mesh, targets, profile)
        report['online_scene_setup_sec'] = time.perf_counter() - setup
        save(output/'test_geometry.json', describe(target, config))
        np.savez_compressed(output/'test_mesh.npz', vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles))
        for name, poses in placements.items():
            begin = time.perf_counter()
            density = np.stack([provider.mask(p) for p in poses]).sum(axis=0)
            elapsed = time.perf_counter() - begin
            rec = dict(coverage_pct=100*float((density>0).mean()),
                covered_count=int((density>0).sum()), target_count=len(targets),
                uncovered_count=int((density==0).sum()), overlap_target_count=int((density>1).sum()),
                num_selected=len(poses), raycast_pose_queries=len(poses), evaluation_sec=elapsed,
                poses_npy=f'{name}.npy', validation='installability/grid + direct mesh raycast')
            np.save(output/f'{name}.npy', poses)
            np.savez_compressed(output/f'{name}_coverage.npz', target_points=targets, density=density)
            report['results'][name] = rec
            save(output/'report.json', report)
            export_view(output/f'{name}.html', mesh, targets, density, poses, name, rec['coverage_pct'])
            print(name, rec, flush=True)
        report['online_comparison_total_sec'] = input_sec + placement_sec + report['online_scene_setup_sec'] + sum(
            r['evaluation_sec'] for r in report['results'].values())
        report['source_hashes'] = {str(p):digest(p) for p in [Path(__file__), Path('prototype_offline.py'), *Path('src').rglob('*.py')]}
        snapshot = output/'source'; snapshot.mkdir()
        for name in report['source_hashes']:
            path = Path(name)
            dest = snapshot/(path.name if path.is_absolute() else path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        report['status'] = 'complete' if metadata['status'] == 'available' else 'partial'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        save(output/'report.json', report)


def export_view(path, mesh, targets, density, poses, name, coverage):
    import plotly.graph_objects as go
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    fig = go.Figure(go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
        opacity=0.22, color='gray', name='Road environment'))
    for label, mask, color in [('Covered', density>0, '#16856a'), ('Uncovered', density==0, '#d34a4a')]:
        indices = np.flatnonzero(mask)
        indices = indices[::max(1, int(np.ceil(len(indices)/6000)))]
        points = targets[indices]
        fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers',
            marker=dict(size=2, color=color), name=label))
    fig.add_trace(go.Scatter3d(x=poses[:,0], y=poses[:,1], z=poses[:,2], mode='markers+text',
        text=[f'L{i+1}' for i in range(len(poses))], marker=dict(size=7, color='#212121'), name='LiDAR'))
    for i, pose in enumerate(poses):
        yaw, pitch = np.deg2rad(pose[3:])
        end = pose[:3] + 6*np.array([np.cos(yaw)*np.cos(pitch), np.sin(yaw)*np.cos(pitch), np.sin(pitch)])
        fig.add_trace(go.Scatter3d(x=[pose[0], end[0]], y=[pose[1], end[1]], z=[pose[2], end[2]],
            mode='lines', line=dict(width=5, color='#212121'), showlegend=False,
            name=f'L{i+1} yaw={pose[3]:g}, pitch={pose[4]:g}'))
    fig.update_layout(title=dict(text=f'{name}: {coverage:.3f}%', font=dict(size=14)),
        margin=dict(l=10, r=10, t=80, b=10), legend=dict(orientation='h', x=0, y=1.04),
        scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
            xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='z (m)'))
    fig.write_html(path, include_plotlyjs=True, default_height='min(85vh, 95vw)', config=dict(responsive=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low', required=True)
    parser.add_argument('--high', required=True)
    parser.add_argument('--width', type=float, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--matching', choices=['strict', 'geometry', 'road_relative'], default='strict',
        help='Geometry mode permits cross-line correspondence; target installation constraints remain mandatory')
    run(parser.parse_args())

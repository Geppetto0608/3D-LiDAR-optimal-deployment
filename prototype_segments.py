"""Width transfer in fractional installation-segment coordinates.

Segments are extracted from the shared feasible grid, not from visibility.
Only one connected longitudinal interval per template branch/side is supported.
"""
import numpy as np

from prototype_offline import absolute, canonical
from prototype_transfer import (IncompatiblePrototypes, align_poses, angle_delta,
                                project_poses, relative_coordinates)


def segment_key(point):
    return point['branch_id'], point['installation_line_id']


def installation_segments(geometry):
    cells = []
    for gx in range(int(geometry.map_len / geometry.grid_size)):
        for gy in range(int(geometry.map_width / geometry.grid_size)):
            x, y = (gx + .5) * geometry.grid_size, (gy + .5) * geometry.grid_size
            if geometry.is_installable(x, y):
                cells.append([x, y, 3., 0., 0.])
    groups = {}
    for p in canonical(cells, geometry):
        groups.setdefault(segment_key(p), []).append(round(p['s_m'], 8))
    segments = {}
    for key, values in groups.items():
        values = np.unique(values)
        if len(values) < 2 or np.any(np.diff(values) > geometry.grid_size + 1e-6):
            raise IncompatiblePrototypes(f'Disconnected or degenerate installation segment: {key}')
        segments[key] = (float(values[0]), float(values[-1]))
    if not segments:
        raise IncompatiblePrototypes('No installation segments')
    return segments


def segment_coordinates(poses, geometry, segments):
    coordinates = relative_coordinates(poses, geometry)
    for p in coordinates:
        lo, hi = segments[segment_key(p)]
        u = (p['s_m'] - lo) / (hi - lo)
        if not -1e-6 <= u <= 1 + 1e-6:
            raise IncompatiblePrototypes('Source pose outside its installation segment')
        p.update(segment_start_m=lo, segment_end_m=hi, u=float(np.clip(u, 0, 1)))
    return coordinates


def segment_to_target(coordinates, target, segments):
    mapped = []
    for p in coordinates:
        if segment_key(p) not in segments:
            raise IncompatiblePrototypes(f'Missing corresponding segment: {segment_key(p)}')
        lo, hi = segments[segment_key(p)]
        if not 0 <= p['u'] <= 1:
            raise ValueError('Fractional coordinate must lie in [0,1]')
        mapped.append(dict(p, s_m=lo + p['u'] * (hi - lo),
                           d_m=np.sign(p['d_m']) * (target.sidewalk_center_offset + p['line_offset_m'])))
    return absolute(mapped, target)


def prepare_segment_placements(low, high, low_geometry, high_geometry, target, config):
    if len({g.scenario.topology for g in (low_geometry, high_geometry, target)}) != 1:
        raise ValueError('Matching topology required')
    w0, w1, w = [g.scenario.road_width for g in (low_geometry, high_geometry, target)]
    if not w0 < w < w1:
        raise ValueError('Width must lie strictly between prototypes')
    s0, s1, st = [installation_segments(g) for g in (low_geometry, high_geometry, target)]
    if not s0.keys() == s1.keys() == st.keys():
        raise IncompatiblePrototypes('Installation branch/side structure changed')
    c0, c1 = segment_coordinates(low, low_geometry, s0), segment_coordinates(high, high_geometry, s1)
    placements = {}
    projections = {}
    for name, coords in [('transfer_low', c0), ('transfer_high', c1)]:
        raw = segment_to_target(coords, target, st)
        placements[name] = project_poses(raw, target, config, coords)
        projections[name] = np.linalg.norm(placements[name][:, :3] - raw[:, :3], axis=1).tolist()
    alpha = (w - w0) / (w1 - w0)
    meta = dict(matching_mode='segment_fraction_v1', alpha=alpha, low_coordinates=c0,
                high_coordinates=c1, target_segments=[dict(branch_id=k[0], installation_line_id=k[1],
                s_start_m=v[0], s_end_m=v[1]) for k, v in st.items()], projection_displacement_m=projections)
    try:
        aligned, mapping = align_poses(segment_to_target(c0, high_geometry, s1), high, high_geometry, low_keys=c0)
        aligned_c1 = segment_coordinates(aligned, high_geometry, s1)
        interpolated = []
        for a, b in zip(c0, aligned_c1):
            if segment_key(a) != segment_key(b):
                raise IncompatiblePrototypes('Sensor branch/side correspondence failed')
            p = dict(a)
            for field in ('u', 'line_offset_m', 'z_m', 'pitch_deg'):
                p[field] = (1 - alpha) * a[field] + alpha * b[field]
            p['relative_yaw_deg'] = float(a['relative_yaw_deg'] + alpha * angle_delta(a['relative_yaw_deg'], b['relative_yaw_deg']))
            interpolated.append(p)
        raw = segment_to_target(interpolated, target, st)
        placements['interpolation'] = project_poses(raw, target, config, interpolated)
        projections['interpolation'] = np.linalg.norm(placements['interpolation'][:, :3] - raw[:, :3], axis=1).tolist()
        meta.update(status='available', correspondence=mapping, aligned_high_coordinates=aligned_c1,
                    interpolated_coordinates=interpolated, unprojected_poses=raw.tolist())
    except IncompatiblePrototypes as exc:
        meta.update(status='unavailable', reason=str(exc))
    return placements, meta

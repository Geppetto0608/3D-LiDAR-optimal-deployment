"""Larger synthetic RSU stress domains, using existing geometry and optics.

Object footprints are removed from the target support in EVERY scene level.
This keeps clear/mixed/dense comparisons on identical targets; objects are
opaque static surrogates, not a vehicle simulation or detection model.
"""
from itertools import product

import numpy as np

from prototype_segments import installation_segments, segment_coordinates, segment_to_target
from src.environment import (ConnectedCorridorGeometry, ExperimentConfig, LidarProfile,
    MapGeometry, Scenario, allowed_mount_yaws_deg, installation_base_positions,
    validate_installation_pose)
from src.core.scene_occlusion import with_occlusion, _rotation


def config(sampling):
    y, z, p = (sampling[key] for key in ('relative_yaws_deg', 'heights_m', 'pitches_deg'))
    return ExperimentConfig(raw=dict(candidate=dict(xy_domain='bin_representatives',
        xy_step_m=sampling['installation_step_m'], z_min=min(z), z_max=max(z),
        z_step=z[1]-z[0] if len(z)>1 else 1., yaw_axis_min_deg=90+min(y),
        yaw_axis_max_deg=90+max(y), yaw_step_deg=y[1]-y[0] if len(y)>1 else 1.,
        pitch_min_deg=min(p), pitch_max_deg=max(p), pitch_step_deg=p[1]-p[0] if len(p)>1 else 1.,
        max_orientations_per_position=0), target=dict(coverage_z_slices=sampling['target_heights_m'])))


def object_boxes(geometry, descriptor):
    rng = np.random.default_rng(descriptor['seed'])
    length = descriptor['size_m'] if descriptor['track']=='A' else descriptor['length_m']
    count = max(4, int(length // 20))
    boxes = []
    for i in range(count):
        side = -1 if i % 2 else 1
        width = descriptor['width_m']
        if descriptor['track']=='A':
            branches = [('east', 0.), ('west', np.pi), ('south', 1.5*np.pi)]
            if descriptor['topology']=='CROSS':
                branches.append(('north', .5*np.pi))
            name, angle = branches[i % len(branches)]
            lengths = dict(east=geometry.map_len-geometry.junction_x_m, west=geometry.junction_x_m,
                           south=geometry.junction_y_m, north=geometry.map_width-geometry.junction_y_m)
            station = rng.uniform(width/2+8, lengths[name]-8)
            center = np.array([geometry.junction_x_m, geometry.junction_y_m]) + station*np.array([np.cos(angle), np.sin(angle)])
        else:
            station = (i+.5)*geometry.length_m/count
            xy, angles = geometry.centerline(np.array([station]))
            center, angle = xy[0], angles[0]
        kind = i % 3
        size = ([4.5, 1.8, 1.8], [10., 2.5, 3.5], [6., .25, 2.5])[kind]
        offset = side * (width/2-1.4 if kind != 2 else width/2+1.)
        center = center + offset*np.array([-np.sin(angle), np.cos(angle)])
        boxes.append(dict(center_m=[float(center[0]), float(center[1]), size[2]/2+.1],
                          size_m=size, yaw_deg=float(np.rad2deg(angle))))
    return boxes


def outside_footprints(points, boxes, margin=.05):
    keep = np.ones(len(points), dtype=bool)
    for box in boxes:
        local = (points[:, :3]-box['center_m']) @ _rotation(box['yaw_deg'])
        keep &= ~np.all(np.abs(local[:, :2]) <= np.array(box['size_m'][:2])/2+margin, axis=1)
    return keep


def domain(descriptor, sampling):
    cfg = config(sampling)
    if descriptor['track']=='A':
        size = descriptor['size_m']
        cx, cy = np.array(descriptor['center_fraction']) * size
        raw = dict(geometry=dict(junction_x_m=float(cx), junction_y_m=float(cy)))
        scenario = Scenario(460001, descriptor['topology'], descriptor['width_m'], 2, raw)
        geometry = MapGeometry(scenario, map_len=size, map_width=size, grid_size=.5)
        bases = installation_base_positions(geometry, cfg)
        poses = np.array([[x,y,z,float(yaw),pitch] for x,y,z in bases
            for yaw in allowed_mount_yaws_deg(x,y,geometry,cfg.yaw_step_deg,cfg.yaw_axis_min_deg,cfg.yaw_axis_max_deg)
            for pitch in sampling['pitches_deg']], dtype=float)
        for pose in poses:
            validate_installation_pose(pose, geometry, cfg)
        targets = geometry.target_points(sampling['target_heights_m'])
        segments = installation_segments(geometry)
        coords = segment_coordinates(poses, geometry, segments)
        critical = np.linalg.norm(targets[:,:2]-[cx,cy], axis=1) <= 20.
        allowed = np.array([not (p['branch_id']=='east' and .32 <= p['u'] <= .43)
                            and not (p['branch_id']=='west' and p['installation_line_id']=='left'
                                     and .70 <= p['u'] <= .80) for p in coords])
    else:
        length = descriptor['length_m']
        geometry = ConnectedCorridorGeometry(length_m=length, road_width_m=descriptor['width_m'],
            turn_deg=descriptor['turn_deg'], bend_length_m=descriptor.get('bend_length_m', 48.), mesh_step_m=1.)
        step = sampling['installation_step_m']
        stations = np.arange(step/2, length, step)
        coords = np.array(list(product(stations, [-geometry.mounting_offset_m, geometry.mounting_offset_m],
            sampling['heights_m'], sampling['relative_yaws_deg'], sampling['pitches_deg'])))
        poses = geometry.poses(coords)
        targets, target_coords = geometry.target_samples(sampling['target_b_step_m'], sampling['target_heights_m'])
        interfaces = np.linspace(0, length, descriptor.get('modules', 2)+1)[1:-1]
        critical = np.any(np.abs(target_coords[:,0,None]-interfaces) <= 12., axis=1)
        allowed = ~(((coords[:,0]/length >= .18) & (coords[:,0]/length <= .23)) |
                    ((coords[:,0]/length >= .66) & (coords[:,0]/length <= .70) & (coords[:,1]>0)))
        segments = None
    boxes = object_boxes(geometry, descriptor)
    target_keep = outside_footprints(targets, boxes)
    candidate_keep = outside_footprints(poses, boxes, margin=.25) & allowed
    poses = poses[candidate_keep]
    coords = ([c for c, use in zip(coords, candidate_keep) if use] if descriptor['track']=='A' else coords[candidate_keep])
    targets, critical = targets[target_keep], critical[target_keep]
    if (not len(poses) or not len(targets) or not critical.any() or len(poses)>12000
            or len(targets)>200000 or len(poses)*len(targets)>1_500_000_000):
        raise ValueError('Explicit stress-domain size/critical-region guard failed')
    return dict(descriptor=descriptor, geometry=geometry, poses=poses, targets=targets,
        critical=critical, canonical=coords, segments=segments, boxes=boxes,
        profile=LidarProfile('VELODYNE64_R30',360.,26.9,30.), context={},
        removed_targets=int((~target_keep).sum()), removed_candidates=int((~candidate_keep).sum()))


def scene(problem, level):
    if level not in ('clear', 'mixed', 'dense'):
        raise ValueError('Unknown obstacle level')
    boxes = problem['boxes'] if level=='dense' else problem['boxes'][:len(problem['boxes'])//2] if level=='mixed' else []
    return with_occlusion(problem, dict(schema='library_scene_occlusion_v1', boxes=boxes))['geometry'].create_environment_mesh()


def retrieve(sources, target, count=3):
    eligible = [s for s in sources if s['track']==target['track'] and
                (s['track']=='B' or s['topology']==target['topology'])]
    def distance(s):
        value = abs(s['width_m']-target['width_m'])/4
        if s['track']=='A':
            value += abs(s['size_m']-target['size_m'])/100
        else:
            value += abs(s['length_m']-target['length_m']/target['modules'])/80
            value += abs(s['turn_deg']-target['turn_deg'])/60
        return value, s['name']
    if len(eligible)<count:
        raise ValueError('Insufficient topology-compatible sources')
    return sorted(eligible, key=distance)[:count]


def transfer_ids(source, solutions, target):
    """Batch projection with cached target coordinates; no physical coverage input."""
    used = sorted({cid for ids in solutions for cid in ids})
    if not used:
        return [], []
    mapping, records = {}, []
    if target['descriptor']['track']=='A':
        coords = [source['canonical'][i] for i in used]
        raw = segment_to_target(coords, target['geometry'], target['segments'])
        groups = {}
        for i, c in enumerate(target['canonical']):
            groups.setdefault((c['branch_id'],c['installation_line_id'],c['z_m']), []).append(i)
        for sid, c, p in zip(used, coords, raw):
            candidates = np.array(groups[(c['branch_id'],c['installation_line_id'],c['z_m'])])
            squared = ((target['poses'][candidates,:2]-p[:2])**2).sum(axis=1)
            nearest = candidates[squared <= squared.min()+1e-8]
            yaw = (target['poses'][nearest,3]-p[3]+180)%360-180
            orientation = yaw**2+(target['poses'][nearest,4]-p[4])**2
            mapping[sid] = [int(nearest[np.argmin(orientation)])]
    else:
        length = target['geometry'].length_m
        modules = target['descriptor']['modules']
        source_length = source['descriptor']['length_m']
        canonical = np.asarray(source['canonical'])
        tc = target['canonical']
        for sid in used:
            c = canonical[sid]
            mapping[sid] = []
            for module in range(modules):
                lo, hi = module*length/modules, (module+1)*length/modules
                valid = (tc[:,0]>=lo)&(tc[:,0]<hi)&(np.sign(tc[:,1])==np.sign(c[1]))
                valid &= np.isclose(tc[:,2],c[2]) & np.isclose(tc[:,3],c[3]) & np.isclose(tc[:,4],c[4])
                candidates = np.flatnonzero(valid)
                if not len(candidates):
                    continue
                station = lo+c[0]/source_length*(hi-lo)
                mapping[sid].append(int(candidates[np.argmin(np.abs(tc[candidates,0]-station))]))
    order, seen = [], set()
    for ids in solutions:
        mapped = [cid for sid in ids for cid in mapping.get(sid, [])]
        records.append(dict(source_ids=ids, target_ids=sorted(set(mapped)),
            projected_collisions=len(mapped)-len(set(mapped))))
        for cid in mapped:
            if cid not in seen:
                order.append(cid)
                seen.add(cid)
    return order, records

"""Geometry-aligned transfer of source occlusion residuals, not learned detection.

Only source-optically-valid blocked rays are transferable labels. Unknown source
samples fall back to target optics, never to an observed invisible label.
"""
import numpy as np


def nearest_source(sources, target):
    if not sources:
        raise ValueError('A nonempty source family is required')
    def distance(item):
        return sum(((target[k]-item[k])/scale)**2 for k, scale in
            [('road_width_m', 7.), ('turn_deg', 90.), ('bend_length_m', 20.)])
    winner = min(sources, key=lambda s: (distance(s), s['name']))
    return winner['name'], float(distance(winner))


def correspondence(source_canonical, source_targets, target_canonical, target_targets, source_width, target_width):
    arrays = [np.asarray(v, dtype=float) for v in
        (source_canonical, source_targets, target_canonical, target_targets)]
    sc, st, tc, tt = arrays
    if (sc.ndim != 2 or sc.shape[1] != 5 or tc.ndim != 2 or tc.shape[1] != 5
            or st.ndim != 2 or st.shape[1] != 3 or tt.ndim != 2 or tt.shape[1] != 3
            or any(not len(v) or not np.isfinite(v).all() for v in arrays)
            or not np.isfinite([source_width, target_width]).all() or min(source_width, target_width) <= 0
            or np.any(sc[:, 1] == 0) or np.any(tc[:, 1] == 0)):
        raise ValueError('Finite nonempty road-relative arrays and positive widths required')
    keys = [(p[0], np.sign(p[1]), *p[2:]) for p in sc]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate source pose key')
    lookup = {k: i for i, k in enumerate(keys)}
    try:
        candidate_ids = np.array([lookup[(p[0], np.sign(p[1]), *p[2:])] for p in tc], int)
    except KeyError as exc:
        raise ValueError('Source pose grid does not support target pose grid') from exc
    target_ids = np.full(len(tt), -1, int)
    for s, z in np.unique(tt[:, [0, 2]], axis=0):
        source = np.flatnonzero((st[:, 0] == s) & (st[:, 2] == z))
        target = np.flatnonzero((tt[:, 0] == s) & (tt[:, 2] == z))
        if not len(source):
            continue
        lateral = st[source, 1]/source_width
        unique = np.unique(lateral)
        tolerance = np.min(np.diff(unique))/2+1e-9 if len(unique) > 1 else 0.
        distances = np.abs(tt[target, 1, None]/target_width-lateral)
        nearest = distances.argmin(axis=1)
        supported = ((distances[np.arange(len(target)), nearest] <= tolerance)
            & (np.sign(st[source[nearest], 1]) == np.sign(tt[target, 1])))
        target_ids[target[supported]] = source[nearest[supported]]
    return candidate_ids, target_ids


def transfer_residual(optical, source_optical, source_masks, candidate_ids, target_ids):
    optical, source_optical, source_masks = map(np.asarray, (optical, source_optical, source_masks))
    candidate_ids, target_ids = np.asarray(candidate_ids), np.asarray(target_ids)
    if (optical.ndim != 2 or source_optical.ndim != 2 or source_masks.shape != source_optical.shape
            or any(a.dtype != bool for a in (optical, source_optical, source_masks))
            or candidate_ids.shape != (len(optical),) or target_ids.shape != (optical.shape[1],)
            or candidate_ids.dtype.kind not in 'iu' or target_ids.dtype.kind not in 'iu'
            or np.any(candidate_ids < 0) or np.any(candidate_ids >= len(source_masks))
            or np.any(target_ids < -1) or np.any(target_ids >= source_masks.shape[1])):
        raise ValueError('Invalid residual-transfer masks or correspondence')
    predicted = optical.copy()
    supported = target_ids >= 0
    blocked = source_optical & ~source_masks
    predicted[:, supported] &= ~blocked[np.ix_(candidate_ids, target_ids[supported])]
    return predicted

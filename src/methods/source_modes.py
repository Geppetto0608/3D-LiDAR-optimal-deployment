"""Source-only geometric-mode and marginal-coverage-response compression."""
import numpy as np


def diverse_source_alternatives(canonical, scores, per_site):
    canonical, scores = np.asarray(canonical, dtype=float), np.asarray(scores, dtype=float)
    if (canonical.ndim != 2 or canonical.shape[1] != 5 or scores.shape != (len(canonical),)
            or not np.isfinite(canonical).all() or not np.isfinite(scores).all()
            or isinstance(per_site, bool) or not isinstance(per_site, int) or per_site < 1):
        raise ValueError('Expected canonical poses, finite scores and positive mode count')
    selected = []
    for site in np.unique(canonical[:, :2], axis=0):
        ids = np.flatnonzero(np.all(canonical[:, :2] == site, axis=1))
        modes = []
        for mode in np.unique(canonical[ids][:, [2, 3]], axis=0):
            group = ids[np.all(canonical[ids][:, [2, 3]] == mode, axis=1)]
            modes.append(int(group[np.argmax(scores[group])]))
        if per_site > len(modes):
            raise ValueError('Requested diversity budget exceeds distinct source height/yaw modes')
        modes = np.asarray(sorted(modes))
        features = canonical[modes][:, [2, 3]]
        span = np.ptp(features, axis=0)
        features = (features - features.min(axis=0)) / np.where(span > 0, span, 1.)
        chosen = [int(np.argmax(scores[modes]))]
        while len(chosen) < per_site:
            distance = ((features[:, None] - features[chosen][None])**2).sum(axis=2).min(axis=1)
            remaining = [i for i in range(len(modes)) if i not in chosen]
            winner = max(remaining, key=lambda i: (float(distance[i]), float(scores[modes[i]]), -int(modes[i])))
            chosen.append(winner)
        selected.extend(modes[chosen].tolist())
    return selected


def response_source_alternatives(canonical, masks, scores, per_site, *, intersection_weight):
    """Greedy worst-context regret compression, using source data only.

    Contexts are empty or one source sensor at a different station/side site.
    For union coverage minus excess overlap and additive costs, marginal J is
    single-sensor J minus (coverage weight + overlap penalty) times intersection.
    Hard constraints are NOT certified by this source response proxy.
    """
    canonical = np.asarray(canonical, dtype=float)
    masks, scores = np.asarray(masks), np.asarray(scores, dtype=float)
    if (canonical.ndim != 2 or canonical.shape[1] != 5 or len(canonical) == 0
            or scores.shape != (len(canonical),) or not np.isfinite(canonical).all()
            or not np.isfinite(scores).all() or masks.dtype != bool or masks.ndim != 2
            or masks.shape[0] != len(canonical) or masks.shape[1] == 0
            or isinstance(per_site, bool) or not isinstance(per_site, int) or per_site < 1
            or isinstance(intersection_weight, bool) or not np.isscalar(intersection_weight)
            or not np.isfinite(intersection_weight) or intersection_weight < 0):
        raise ValueError('Expected finite source poses/scores, bool masks and valid response budget/weight')
    # Integer multiplication counts intersections; bool matrix multiplication does not.
    integer_masks = masks.astype(np.int64)
    intersections = integer_masks @ integer_masks.T
    selected, traces = [], []
    for site in np.unique(canonical[:, :2], axis=0):
        same_site = np.all(canonical[:, :2] == site, axis=1)
        ids, contexts = np.flatnonzero(same_site), np.flatnonzero(~same_site)
        if per_site > len(ids):
            raise ValueError('Response budget exceeds source candidates at a site')
        response = scores[ids, None] - intersection_weight * np.column_stack([
            np.zeros(len(ids), dtype=np.int64), intersections[np.ix_(ids, contexts)]])
        best = response.max(axis=0)
        chosen = [int(np.argmax(scores[ids]))]
        history = []
        while True:
            envelope = response[chosen].max(axis=0)
            regret = np.maximum(best-envelope, 0.)
            history.append(dict(selected_id=int(ids[chosen[-1]]),
                worst_regret=float(regret.max()), mean_regret=float(regret.mean())))
            if len(chosen) == per_site:
                break
            remaining = [i for i in range(len(ids)) if i not in chosen]
            def rank(i):
                residual = np.maximum(best-np.maximum(envelope, response[i]), 0.)
                return (float(residual.max()), float(residual.mean()), -float(scores[ids[i]]), int(ids[i]))
            chosen.append(min(remaining, key=rank))
        selected.extend(ids[chosen].tolist())
        traces.append(dict(site=site.tolist(), context_count=len(contexts)+1,
                           context_ids=[None, *contexts.tolist()], steps=history))
    return selected, traces

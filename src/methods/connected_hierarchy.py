"""Opt-in information/budget contracts and two-level static placement baselines.

The optical surrogate ignores occlusion. Only measured masks certify an upper
selection or a final deployment. No learned policy or transfer guarantee.
"""
from collections import Counter
from dataclasses import dataclass
import time

import numpy as np

from src.environment import angular_visibility_mask, compute_view_angles_numpy
from src.methods.connected_pairs import assess_ids, exact_pairs


@dataclass
class SelectionContext:
    canonical: np.ndarray
    poses: np.ndarray
    targets: np.ndarray
    source_prior: np.ndarray
    requirements: object
    sensor: object

    def __post_init__(self):
        self.canonical = np.asarray(self.canonical, dtype=float)
        self.poses = np.asarray(self.poses, dtype=float)
        self.targets = np.asarray(self.targets, dtype=float)
        self.source_prior = np.asarray(self.source_prior, dtype=float)
        n = len(self.canonical)
        if (not n or self.canonical.shape != (n, 5) or self.poses.shape != (n, 5)
                or self.targets.ndim != 2 or self.targets.shape[1] != 3 or not len(self.targets)
                or self.source_prior.shape != (n,) or not all(np.isfinite(v).all() for v in
                    (self.canonical, self.poses, self.targets, self.source_prior))):
            raise ValueError('Invalid geometry/source-prior selection context')
        if self.requirements.spec['budget_k'] != 2:
            raise ValueError('This hierarchy pilot is fixed K=2')
        self.sites, self.site_ids = np.unique(self.canonical[:, :2], axis=0, return_inverse=True)
        self.allowed = np.asarray(self.requirements.allowed, dtype=bool).copy()
        if 'mounting_z_range_m' in self.requirements.spec.get('constraints', {}):
            lo, hi = self.requirements.spec['constraints']['mounting_z_range_m']
            self.allowed &= (self.poses[:, 2]+1e-9 >= lo) & (self.poses[:, 2] <= hi+1e-9)
        self.groups = [np.flatnonzero(self.allowed & (self.site_ids == i)).tolist() for i in range(len(self.sites))]


class BudgetedObservations:
    """Known masks plus a metered callback, with no full target-mask argument."""
    def __init__(self, context, query, *, per_site=None, max_extra=None, initial=None, initial_work=None):
        for value in (per_site, max_extra):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError('Query budgets must be nonnegative integers')
        self.context, self._query = context, query
        self.per_site, self.max_extra = per_site, max_extra
        self.masks = {}
        for cid, value in (initial or {}).items():
            if not self.can_install(cid):
                raise ValueError('Ineligible initial observation')
            self.masks[int(cid)] = self._mask(value)
        self.initial_count = len(self.masks)
        self.initial_work = dict(initial_work or {})
        self.events = []
        self.frozen = False

    def can_install(self, cid):
        return (isinstance(cid, (int, np.integer)) and not isinstance(cid, (bool, np.bool_))
                and 0 <= cid < len(self.context.poses) and bool(self.context.allowed[cid]))

    def can_query(self, cid):
        if not self.can_install(cid):
            return False
        if cid in self.masks:
            return True
        if self.frozen or (self.max_extra is not None and len(self.events) >= self.max_extra):
            return False
        if self.per_site is not None:
            count = sum(self.context.site_ids[i] == self.context.site_ids[cid] for i in self.masks)
            if count >= self.per_site:
                return False
        return True

    def measured(self, cid):
        if not self.can_install(cid) or cid not in self.masks:
            raise KeyError('Unmeasured target visibility is not available')
        return self.masks[cid]

    def get(self, cid, *, phase='refinement'):
        if not self.can_query(cid):
            raise ValueError('Candidate ineligible or target-query budget exhausted')
        cid = int(cid)
        if cid not in self.masks:
            mask = self._mask(self._query(cid))
            self.events.append(dict(candidate_id=cid, phase=phase, known_before=len(self.masks)))
            self.masks[cid] = mask
        return self.masks[cid]

    __call__ = get

    def _mask(self, value):
        mask = np.asarray(value)
        if mask.dtype != bool or mask.shape != (len(self.context.targets),):
            raise ValueError('Physical callback must return bool [target_count]')
        mask = mask.copy()
        mask.setflags(write=False)
        return mask

    def report(self, query_work=None):
        if query_work is None:
            query_work = self._query.report() if hasattr(self._query, 'report') else {}
        accounting = {'unique_candidate_queries', 'base_candidate_queries', 'extra_candidate_queries'}
        total = Counter({k: v for k, v in self.initial_work.items() if k not in accounting})
        total.update({k: v for k, v in query_work.items() if k not in accounting})
        return dict(unique_candidate_queries=len(self.masks), base_candidate_queries=self.initial_count,
                    extra_candidate_queries=len(self.events), **dict(total))


def optical_predictions(context):
    masks = np.zeros((len(context.poses), len(context.targets)), bool)
    sensor = context.sensor
    half_h, half_v = np.deg2rad([sensor.h_fov_deg, sensor.v_fov_deg])/2
    for cid in np.flatnonzero(context.allowed):
        pose = context.poses[cid]
        vectors = context.targets-pose[:3]
        az, el, _, distance = compute_view_angles_numpy(*vectors.T, pose[3], pose[4])
        masks[cid] = angular_visibility_mask(az, el, distance, half_h, half_v, sensor.max_range_m)
    return masks


def predicted_partner(cid, partners, predicted, observations, context, *, fast_screen=False):
    """Rank an unmeasured pose against measured partners; predictions never certify."""
    partners = [i for i in partners if context.site_ids[i] != context.site_ids[cid]]
    if not partners:
        return None, None
    req, pose = context.requirements, context.poses
    others = np.asarray([observations.measured(i) for i in partners])
    w = req.requirements.weights
    rewards = w.coverage_weight+w.importance_weight*req.importance+w.overlap_penalty
    costs = req.installation_costs(pose[[cid, *partners]])
    penalties = w.sensor_count_penalty+w.installation_cost_weight*costs
    penalties += w.overlap_penalty*np.r_[predicted[cid].sum(), others.sum(axis=1)]
    values = (others | predicted[cid]) @ rewards-penalties[0]-penalties[1:]
    eligible = np.ones(len(partners), dtype=bool)
    if fast_screen:
        # Necessary checks only. Surviving predictions still use the full evaluator.
        c, n = req.spec.get('constraints', {}), len(context.targets)
        first_count, other_count = predicted[cid].sum(), others.sum(axis=1)
        double = np.count_nonzero(others & predicted[cid], axis=1)
        covered = first_count+other_count-double
        eligible &= req.filter_visibility_pairs(predicted[cid], others)
        eligible &= costs[0]+costs[1:] <= c.get('max_installation_cost', float('inf'))+1e-9
        eligible &= np.linalg.norm(pose[partners, :2]-pose[cid, :2], axis=1)+1e-9 >= c.get('min_xy_separation_m', 0)
        eligible &= 100*covered/n+1e-9 >= c.get('min_coverage_pct', 0)
        eligible &= double/n+1e-9 >= c.get('min_overlap_ratio', 0)
        eligible &= double/n <= c.get('max_overlap_ratio', 1)+1e-9
        eligible &= 100*np.minimum(first_count, other_count)/n+1e-9 >= c.get('min_single_failure_coverage_pct', 0)
    order = sorted(np.flatnonzero(eligible).tolist(), key=lambda k: (-round(float(values[k]), 8), partners[k]))
    for k in order:
        pair_masks = np.asarray([predicted[cid], others[k]])
        assessment = req.evaluate(pose[[cid, partners[k]]], pair_masks.sum(axis=0), sensor_masks=pair_masks)
        if assessment.feasible:
            return float(assessment.objective_score), int(partners[k])
    return None, None


def batch_predicted_partners(candidates, partners, predicted, observations, context, *, fast_screen=False):
    """Screen a frozen round in packed blocks; preserve scalar objective ordering.

    No surrogate arithmetic replaces the objective dot product. With multiple
    survivors, its full partner vector and rounding remain identical to scalar.
    """
    from src.roadside.corridor_requirements import CorridorRequirements

    req, pose = context.requirements, context.poses
    if not fast_screen or type(req) is not CorridorRequirements:
        return {cid: predicted_partner(cid, partners, predicted, observations, context,
            fast_screen=fast_screen) for cid in candidates}
    if not candidates or not partners:
        return {cid: (None, None) for cid in candidates}
    ids, known = np.asarray(candidates, int), np.asarray(partners, int)
    # Observations are copied once, before any proposal in this round is queried.
    others = np.asarray([observations.measured(i) for i in known])
    first = predicted[ids]
    first_count, other_count = first.sum(axis=1), others.sum(axis=1)
    popcount = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)

    def intersections(a, b):
        packed_a, packed_b = np.packbits(a, axis=1), np.packbits(b, axis=1)
        counts = np.empty((len(a), len(b)), dtype=np.int64)
        # Bound temporary storage rather than materializing candidate x partner x target.
        for start in range(0, len(a), 16):
            bits = packed_a[start:start+16, None, :] & packed_b[None, :, :]
            counts[start:start+16] = popcount[bits].sum(axis=2)
        return counts

    double = intersections(first, others)
    covered = first_count[:, None]+other_count[None, :]-double
    c, n = req.spec.get('constraints', {}), len(context.targets)
    costs = req.installation_costs(pose)
    eligible = context.site_ids[ids, None] != context.site_ids[known][None, :]
    eligible &= costs[ids, None]+costs[known][None, :] <= c.get('max_installation_cost', float('inf'))+1e-9
    eligible &= np.linalg.norm(pose[ids, None, :2]-pose[known, :2][None, :, :], axis=2)+1e-9 >= c.get('min_xy_separation_m', 0)
    eligible &= 100*covered/n+1e-9 >= c.get('min_coverage_pct', 0)
    eligible &= double/n+1e-9 >= c.get('min_overlap_ratio', 0)
    eligible &= double/n <= c.get('max_overlap_ratio', 1)+1e-9
    eligible &= 100*np.minimum(first_count[:, None], other_count[None, :])/n+1e-9 >= c.get('min_single_failure_coverage_pct', 0)
    band = req.interface_mask
    a, b = first[:, band], others[:, band]
    band_double = intersections(a, b)
    band_covered = a.sum(axis=1)[:, None]+b.sum(axis=1)[None, :]-band_double
    thresholds = req.profile['interface']
    eligible &= 100*(band_covered/np.count_nonzero(band))+1e-9 >= thresholds.get('min_coverage_pct', 0)
    eligible &= band_double/np.count_nonzero(band)+1e-9 >= thresholds.get('min_overlap_ratio', 0)
    eligible &= band_double/np.count_nonzero(band) <= thresholds.get('max_overlap_ratio', 1)+1e-9
    w = req.requirements.weights
    rewards = w.coverage_weight+w.importance_weight*req.importance+w.overlap_penalty
    result = {}
    for row, cid in enumerate(candidates):
        surviving = np.flatnonzero(eligible[row])
        if len(surviving) > 1:
            different = np.flatnonzero(context.site_ids[known] != context.site_ids[cid])
            partner_ids = known[different]
            partner_masks = others[different]
            penalties = w.sensor_count_penalty+w.installation_cost_weight*costs[np.r_[cid, partner_ids]]
            penalties += w.overlap_penalty*np.r_[first_count[row], other_count[different]]
            values = (partner_masks | predicted[cid]) @ rewards-penalties[0]-penalties[1:]
            ranks = {int(k): j for j, k in enumerate(different)}
            surviving = sorted(surviving.tolist(), key=lambda k: (-round(float(values[ranks[k]]), 8), int(known[k])))
        result[cid] = None, None
        for k in surviving:
            pair_masks = np.asarray([predicted[cid], others[k]])
            assessment = req.evaluate(pose[[cid, known[k]]], pair_masks.sum(axis=0), sensor_masks=pair_masks)
            if assessment.feasible:
                result[cid] = float(assessment.objective_score), int(known[k])
                break
    return result


def probe_prior_values(mode, context, predicted, *, random_seed=None):
    """Only the source mode reads source scores; all modes rank seeds/ties/fallbacks."""
    if mode not in ('source', 'geometry', 'random'):
        raise ValueError('Unknown probe prior')
    if mode == 'random':
        if (isinstance(random_seed, bool) or not isinstance(random_seed, int)
                or not 0 <= random_seed < 2**32):
            raise ValueError('Random prior requires an integer seed in [0, 2**32)')
        return np.random.default_rng(random_seed).permutation(len(context.poses)).astype(float)
    if random_seed is not None:
        raise ValueError('Only random prior accepts a seed')
    if mode == 'source':
        values = np.asarray(context.source_prior, dtype=float).copy()
    else:
        req, w = context.requirements, context.requirements.requirements.weights
        reward = w.coverage_weight+w.importance_weight*req.importance
        values = predicted @ reward-w.sensor_count_penalty-w.installation_cost_weight*req.installation_costs(context.poses)
    if values.shape != (len(context.poses),) or not np.isfinite(values).all():
        raise ValueError('Invalid prior score vector')
    return values


def conditional_schedule(context, observations, per_site, predicted, prior, *,
                         fast_screen=False, partner_engine='scalar'):
    """Shared schedule; the observation provider determines the feedback source."""
    groups = [g for g in context.groups if g]
    trace, snapshots = [], []
    for group in groups:
        cid = min(group, key=lambda i: (-prior[i], i))
        observations.get(cid, phase='seed')
    for round_id in range(1, per_site):
        # Freeze the context for the entire round; no within-round order advantage.
        known = sorted(observations.masks)
        snapshots.append(dict(round=round_id, measured_context_ids=known))
        batched = None
        if partner_engine == 'batch':
            pending = [i for group in groups for i in group if i not in observations.masks]
            batched = batch_predicted_partners(pending, known, predicted, observations, context,
                fast_screen=fast_screen)
        proposals = []
        for group in groups:
            options = []
            for cid in group:
                if cid in observations.masks:
                    continue
                score, partner = (batched[cid] if batched is not None else predicted_partner(
                    cid, known, predicted, observations, context, fast_screen=fast_screen))
                options.append((score is None, -round(score, 8) if score is not None else 0.,
                                -float(prior[cid]), cid, score, partner))
            chosen = min(options)
            proposals.append(chosen[3])
            trace.append(dict(round=round_id, candidate_id=chosen[3], predicted_j=chosen[4],
                              measured_partner_id=chosen[5], source_fallback=chosen[0]))
        for cid in proposals:
            observations.get(cid, phase=f'round_{round_id}')
    return trace, snapshots


def select_pool(kind, context, observations, per_site, *, static_ids=None,
                probe_prior=None, random_seed=None, fast_screen=False, prediction_transform=None,
                partner_engine='scalar'):
    if kind not in ('source_response', 'geometry_single', 'probe_context', 'random_pool'):
        raise ValueError('Unknown hierarchy selector')
    if not isinstance(fast_screen, bool):
        raise ValueError('fast_screen must be boolean')
    if partner_engine not in ('scalar', 'batch') or (kind != 'probe_context' and partner_engine != 'scalar'):
        raise ValueError('Partner engine must be scalar, or batch for conditional probes')
    if kind != 'probe_context' and (probe_prior is not None or fast_screen):
        raise ValueError('Prior/screen options apply only to conditional probes')
    if kind == 'random_pool':
        if type(random_seed) is not int or not 0 <= random_seed < 2**32:
            raise ValueError('Random pool requires an integer seed in [0, 2**32)')
    elif kind != 'probe_context' and random_seed is not None:
        raise ValueError('Seed applies only to random pool or conditional random prior')
    if prediction_transform is not None and (kind != 'probe_context' or not callable(prediction_transform)):
        raise ValueError('Prediction transform requires conditional probes and a callable')
    if isinstance(per_site, bool) or not isinstance(per_site, int) or per_site < 1:
        raise ValueError('Positive per-site budget required')
    if observations.masks or observations.per_site != per_site or observations.frozen:
        raise ValueError('Selection requires an empty, matched per-site observation budget')
    groups = [g for g in context.groups if g]
    if any(len(g) < per_site for g in groups):
        raise ValueError('A site has fewer allowed poses than the matched budget')
    started = time.perf_counter()
    trace, snapshots = [], []
    predicted = None if kind in ('source_response', 'random_pool') else optical_predictions(context)
    prior = (probe_prior_values('source' if probe_prior is None else probe_prior, context, predicted,
        random_seed=random_seed) if kind == 'probe_context' else None)
    rejected = None
    if prediction_transform is not None:
        transformed = np.asarray(prediction_transform(predicted.copy()))
        if transformed.dtype != bool or transformed.shape != predicted.shape or (transformed & ~predicted).any():
            raise ValueError('Occlusion transform must return a boolean subset of optical predictions')
        rejected = int(np.count_nonzero(predicted & ~transformed))
        predicted = transformed.copy()
    if kind == 'source_response':
        if static_ids is None:
            raise ValueError('Source-only selector needs frozen source IDs')
        ids = [int(i) for i in static_ids if context.allowed[i]]
        if len(ids) != per_site*len(groups) or len(set(ids)) != len(ids):
            raise ValueError('Source pool does not satisfy matched site budget')
        for cid in ids:
            observations.get(cid, phase='source_pool')
    elif kind == 'random_pool':
        # Permute each whole group before slicing: budget changes preserve prefixes.
        rng = np.random.default_rng(random_seed)
        orders = [rng.permutation(group).tolist() for group in groups]
        for order in orders:
            for cid in order[:per_site]:
                observations.get(cid, phase='random_pool')
    elif kind == 'geometry_single':
        req, w = context.requirements, context.requirements.requirements.weights
        reward = w.coverage_weight+w.importance_weight*req.importance
        values = predicted @ reward-w.sensor_count_penalty-w.installation_cost_weight*req.installation_costs(context.poses)
        for group in groups:
            for cid in sorted(group, key=lambda i: (-round(float(values[i]), 8), i))[:per_site]:
                observations.get(cid, phase='geometry_pool')
    else:
        trace, snapshots = conditional_schedule(context, observations, per_site, predicted, prior,
            fast_screen=fast_screen, partner_engine=partner_engine)
    observations.frozen = True
    ids = sorted(observations.masks)
    for group in groups:
        if sum(i in observations.masks for i in group) != per_site:
            raise RuntimeError('Selector failed matched-budget contract')
    result = dict(candidate_ids=ids, events=observations.events, decisions=trace, contexts=snapshots,
        kind=kind, selection_sec=time.perf_counter()-started,
        optical_candidate_evaluations=0 if predicted is None else int(context.allowed.sum()),
        optical_point_tests=0 if predicted is None else int(context.allowed.sum()*len(context.targets)),
        optical_semantics='Unoccluded range/FOV prediction, not measured visibility or feasibility certificate')
    if probe_prior is not None:
        result.update(probe_prior=probe_prior, random_seed=random_seed, fast_screen=fast_screen,
            prior_values=prior.tolist())
        for decision in trace:
            decision['prior_fallback'] = decision.pop('source_fallback')
    if rejected is not None:
        result.update(predicted_occlusion_rejections=rejected,
            prediction_semantics='Transferred source LOS residual; geometry prior unchanged. Not measured target visibility.')
    if partner_engine != 'scalar':
        result['partner_engine'] = partner_engine
    if kind == 'random_pool':
        result.update(random_seed=random_seed,
            sampling='Uniform without replacement within each eligible site; fixed seed, nested budget prefixes')
    return result


def upper_select(pool, context, observations, module_ids, allocation):
    if allocation not in ('one_per_module', 'free'):
        raise ValueError('Unknown upper allocation mode')
    if len(pool) < 2:
        return dict(selected_ids=None, assessment=None, status='infeasible_in_queried_pool', quota=allocation)
    result = exact_pairs(pool, context.poses, observations.measured, context.requirements,
        module_ids=module_ids if allocation == 'one_per_module' else None,
        pair_mask_filter=context.requirements.filter_visibility_pairs)
    result['allocation_mode'] = allocation
    if result['selected_ids'] is not None:
        signature = Counter((int(module_ids[i]), 'left' if context.canonical[i, 1] > 0 else 'right')
                            for i in result['selected_ids'])
        result['installation_allocation'] = [dict(module=m, side=s, count=n) for (m, s), n in sorted(signature.items())]
    return result


def refine_poses(initial, context, observations, *, station_radius_m, max_sweeps=100):
    """Budget-aware coordinate descent, with fixed sites when radius is zero."""
    if initial['selected_ids'] is None:
        return dict(selected_ids=None, assessment=None, status='not_run_no_feasible_upper')
    if not np.isfinite(station_radius_m) or station_radius_m < 0 or max_sweeps < 1:
        raise ValueError('Invalid lower search domain')
    selected = list(initial['selected_ids'])
    current = assess_ids(selected, context.poses, observations, context.requirements)
    if not current.feasible:
        raise ValueError('Lower stage requires an actually feasible upper selection')
    canonical = context.canonical
    anchors = canonical[selected, :2].copy()
    lookup = {tuple(row): i for i, row in enumerate(canonical)}
    levels = {a: np.unique(canonical[:, a]) for a in (0, 2, 3, 4)}
    evaluations, blocked = 0, set()
    trace = [dict(selected_ids=selected.copy(), objective_score=current.objective_score)]
    for sweep in range(max_sweeps):
        best_ids, best = selected, current
        skipped = False
        for agent, cid in enumerate(selected):
            for axis, values in levels.items():
                index = int(np.searchsorted(values, canonical[cid, axis]))
                for offset in (-1, 1):
                    if not 0 <= index+offset < len(values):
                        continue
                    proposal = canonical[cid].copy()
                    proposal[axis] = values[index+offset]
                    replacement = lookup.get(tuple(proposal))
                    if (replacement is None or replacement in selected or not context.allowed[replacement]
                            or abs(proposal[0]-anchors[agent, 0]) > station_radius_m+1e-9):
                        continue
                    if not observations.can_query(replacement):
                        skipped = True
                        blocked.add(replacement)
                        continue
                    trial = selected.copy()
                    trial[agent] = replacement
                    assessment = assess_ids(trial, context.poses, observations, context.requirements)
                    evaluations += 1
                    if assessment.feasible and assessment.objective_score > best.objective_score+1e-8:
                        best_ids, best = trial, assessment
        if best_ids == selected:
            reason = 'query_budget_limited' if skipped else 'no_improving_one_coordinate_move'
            break
        selected, current = best_ids, best
        trace.append(dict(selected_ids=selected.copy(), objective_score=current.objective_score))
    else:
        reason = 'max_sweeps'
    return dict(selected_ids=selected, assessment=current.to_dict(), status='complete',
        station_radius_m=station_radius_m, original_anchors=anchors.tolist(),
        neighbor_evaluations=evaluations, sweeps=sweep+1, stopping_reason=reason,
        budget_blocked_candidate_ids=sorted(blocked), trace=trace)

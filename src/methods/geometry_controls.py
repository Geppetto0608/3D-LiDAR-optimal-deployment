"""Source-free, preplanned controls for the fixed-K=2 geometry experiment.

Optical pair feasibility is a ranking heuristic, never a physical certificate.
Neither planner receives measured masks or the physical query callback.
"""
import time

import numpy as np

from src.methods.connected_hierarchy import (BudgetedObservations, optical_predictions,
    probe_prior_values, predicted_partner, batch_predicted_partners, conditional_schedule)

CONTROL_METHODS = ('static_pair', 'no_feedback')
CONTROL_TIMINGS = ('selection_sec', 'planning_sec', 'physical_query_sec')


def plan_geometry_control(kind, context, per_site, *, partner_engine='batch'):
    """Return the complete query plan using geometry/constraints only."""
    if kind not in CONTROL_METHODS or partner_engine not in ('batch', 'scalar'):
        raise ValueError('Unknown geometry control or partner engine')
    if type(per_site) is not int or per_site < 1:
        raise ValueError('Positive per-site budget required')
    groups = [g for g in context.groups if g]
    if any(len(g) < per_site for g in groups):
        raise ValueError('A site has fewer allowed poses than the matched budget')
    predicted = optical_predictions(context)
    prior = probe_prior_values('geometry', context, predicted)
    virtual = BudgetedObservations(context, lambda cid: predicted[cid], per_site=per_site)
    eligible = np.flatnonzero(context.allowed).tolist()
    rankings = []
    if kind == 'no_feedback':
        trace, snapshots = conditional_schedule(context, virtual, per_site, predicted, prior,
            fast_screen=True, partner_engine=partner_engine)
        # The shared legacy names refer to the provider, not necessarily real measurements.
        for item in trace:
            item['predicted_partner_id'] = item.pop('measured_partner_id')
            item['prior_fallback'] = item.pop('source_fallback')
        for item in snapshots:
            item['predicted_context_ids'] = item.pop('measured_context_ids')
        plan = [dict(candidate_id=e['candidate_id'], phase=e['phase']) for e in virtual.events]
        pair_tests = sum(sum(context.site_ids[cid] != context.site_ids[p]
            for p in item['predicted_context_ids']) for item in snapshots
            for cid in eligible if cid not in item['predicted_context_ids'])
        virtual_count = len(virtual.masks)
    else:
        # All possible partner masks are optical, even if never scheduled for raycasting.
        predicted_observations = BudgetedObservations(context, lambda cid: predicted[cid],
            initial={cid: predicted[cid] for cid in eligible})
        if partner_engine == 'batch':
            scores = batch_predicted_partners(eligible, eligible, predicted,
                predicted_observations, context, fast_screen=True)
        else:
            scores = {cid: predicted_partner(cid, eligible, predicted, predicted_observations,
                context, fast_screen=True) for cid in eligible}
        def rank(cid):
            score, _ = scores[cid]
            return (score is None, -round(score, 8) if score is not None else 0., -float(prior[cid]), cid)
        plan = []
        for group in groups:
            order = sorted(group, key=rank)
            rankings.append(dict(site_id=int(context.site_ids[group[0]]), candidate_ids=order,
                predicted_j=[scores[cid][0] for cid in order],
                predicted_partner_ids=[scores[cid][1] for cid in order]))
            plan.extend(dict(candidate_id=cid, phase='static_pair_pool') for cid in order[:per_site])
        trace, snapshots = [], []
        pair_tests = sum(len(group)*(len(eligible)-len(group)) for group in groups)
        virtual_count = len(eligible)
    return dict(kind=kind, query_plan=plan, decisions=trace, contexts=snapshots, rankings=rankings,
        prior_values=prior.tolist(), partner_engine=partner_engine, planned_before_physical_queries=True,
        feedback='unoccluded_predictions_only', virtual_candidate_masks=virtual_count,
        directed_candidate_partner_tests=int(pair_tests), optical_candidate_evaluations=len(eligible),
        optical_point_tests=len(eligible)*len(context.targets),
        optical_semantics='Unoccluded range/FOV prediction; no physical visibility or feasibility certificate',
        virtual_work_semantics='Predicted masks supplied to planning and directed pairs considered before screening; '
            'neither count is a physical query, a ray, or an actual full constraint evaluation')


def select_geometry_control(kind, context, observations, per_site, *, partner_engine='batch'):
    if (observations.context is not context or observations.masks or observations.frozen
            or observations.per_site != per_site):
        raise ValueError('Control requires an empty, matched per-site observation budget')
    started = time.perf_counter()
    result = plan_geometry_control(kind, context, per_site, partner_engine=partner_engine)
    planned = time.perf_counter()
    # Only after every decision is frozen is the physical provider touched.
    for item in result['query_plan']:
        observations.get(item['candidate_id'], phase=item['phase'])
    queried = time.perf_counter()
    observations.frozen = True
    if any(g and len(set(g) & set(observations.masks)) != per_site for g in context.groups):
        raise RuntimeError('Control failed matched-budget contract')
    result.update(candidate_ids=sorted(observations.masks), events=observations.events,
        planning_sec=planned-started, physical_query_sec=queried-planned,
        selection_sec=time.perf_counter()-started)
    return result

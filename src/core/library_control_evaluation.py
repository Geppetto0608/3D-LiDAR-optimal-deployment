"""Equal-candidate-query controls for the bounded A/B development targets.

All query plans finish before any physical evaluation. All arm decisions finish
before the full reference. Source-free planners receive arrays/profile and a
matched scalar budget, never library poses, measured masks or reference labels.
"""
import copy
import json
import time

import numpy as np

from src.core.deployment_library import build_frontier
from src.core.library_transfer_evaluation import Work, _annotate_ids, _ids, plan_problem
from src.core.visibility import MeshVisibility
from src.methods.library_query_controls import plan_control


GEOMETRY_METHODS = ('geometry_single', 'geometry_union', 'geometry_portfolio')


def plan_matched_controls(problem, seeds, enumeration_limit=100000):
    plan = plan_problem(problem, enumeration_limit)
    seeds = list(seeds)
    if (not 2 <= len(seeds) <= 100
            or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seeds)
            or seeds != sorted(set(seeds))):
        raise ValueError('Expected 2..100 increasing distinct nonnegative integer random seeds')
    if not all(plan['enumeration_will_complete'].values()):
        raise ValueError('Matched-quality study requires exact subset enumeration for every domain')
    plan.update(control_seeds=seeds, expected_arm_count=4 + len(seeds),
        query_budget=plan['pool_candidate_count'], study='matched_candidate_query_controls_v1',
        worker_timeout_sec=600,
        geometry_methods=list(GEOMETRY_METHODS), full_reference_after_all_decisions=True,
        matching='Same unique candidate pose queries, not equal rays, FLOPs or end-to-end time',
        population='Previously inspected synthetic development targets; random seeds are not independent maps')
    return plan


def _query_plans(problem, seeds, enumeration_limit):
    poses, targets = np.asarray(problem['poses']), np.asarray(problem['targets'])
    q = len(problem['pool_ids'])
    plans = [dict(arm_id='library', method='library', seed=None,
                  selected_ids=list(problem['pool_ids']), query_order=list(problem['pool_ids']),
                  planning_sec=None, optical_candidate_evaluations=0, optical_point_tests=0,
                  semantics='Transferred pool; load/mapping time is recorded separately by the runner')]
    for method, seed in [(name, None) for name in GEOMETRY_METHODS] + [('random', seed) for seed in seeds]:
        proposal = plan_control(poses.copy(), targets.copy(), problem['profile'], q,
            list(problem['k_values']), method, seed=seed, enumeration_limit=enumeration_limit)
        ids = _ids(proposal['selected_ids'], len(poses))
        if len(ids) != q or ids != sorted(ids):
            raise ValueError('Source-free planner violated its matched query budget')
        order = _ids(proposal['query_order'], len(poses))
        if sorted(order) != ids:
            raise ValueError('Query order and selected candidate IDs disagree')
        if proposal['method'] != method or proposal['seed'] != seed:
            raise ValueError('Planner returned wrong method or seed')
        plans.append(dict(proposal, arm_id=method if seed is None else f'random_{seed}'))
    json.dumps(plans, allow_nan=False)
    return plans


def _query_pool(provider, poses, ids, order, target_count):
    masks = {}
    for cid in order:
        mask = np.asarray(provider.mask(poses[cid]))
        if mask.dtype != bool or mask.shape != (target_count,):
            raise ValueError('Physical provider must return boolean masks with target shape')
        masks[cid] = mask.copy()
    return np.asarray([masks[cid] for cid in ids], dtype=bool).reshape(len(ids), target_count)


def evaluate_matched_controls(problem, seeds, enumeration_limit=100000):
    plan = plan_matched_controls(problem, seeds, enumeration_limit)
    started = time.perf_counter()
    plans = _query_plans(problem, plan['control_seeds'], enumeration_limit)
    phases = dict(all_query_planning=time.perf_counter() - started)
    frozen_plans = copy.deepcopy(plans)
    poses, targets = np.array(problem['poses'], copy=True), np.array(problem['targets'], copy=True)
    poses.setflags(write=False)
    targets.setflags(write=False)
    started = time.perf_counter()
    mesh = problem['geometry'].create_environment_mesh()
    phases['shared_mesh'] = time.perf_counter() - started
    arms, queried_masks = [], []
    for query_plan in plans:
        ids = query_plan['selected_ids']
        work = Work()
        started = time.perf_counter()
        provider = MeshVisibility(mesh, targets, problem['profile'], work=work)
        masks = _query_pool(provider, poses, ids, query_plan['query_order'], len(targets))
        query_sec = time.perf_counter() - started
        if work.counts.get('direct_pose_evaluations', 0) != plan['query_budget']:
            raise RuntimeError('Physical pose evaluation count differs from matched budget')
        started = time.perf_counter()
        proposals = []
        if query_plan['method'] == 'library':
            local = {cid: index for index, cid in enumerate(ids)}
            proposals = [dict(selected_ids=[local[i] for i in row['selected_ids']],
                              budget_k=row['budget_k'], origin=f'transferred_layout_{number}')
                         for number, row in enumerate(problem['transfers']) if row['status'] == 'complete']
        frontier = build_frontier(poses[ids], masks, targets, problem['k_values'],
            alternatives_per_k=1, enumeration_limit=enumeration_limit, proposals=proposals,
            context=dict(track=plan['track'], arm_id=query_plan['arm_id'],
                         full_candidate_ids=ids, evaluation_domain='matched_queried_pool',
                         source_certificates_reused=False))
        if (not frontier['enumeration']['completed']
                or any(row['bound_source'] != 'exhaustive_subsets' for row in frontier['frontier'])):
            raise RuntimeError('Arm did not complete exact subset enumeration')
        _annotate_ids(frontier, ids)
        arms.append(dict(arm_id=query_plan['arm_id'], method=query_plan['method'], seed=query_plan['seed'],
                         query_plan=query_plan, frontier=frontier, work=dict(work.counts),
                         physical_query_ids=list(query_plan['query_order']),
                         phase_wall_sec=dict(visibility=query_sec, selection=time.perf_counter() - started)))
        queried_masks.append(masks)
    frozen_decisions = copy.deepcopy(arms)

    started = time.perf_counter()
    full_work = Work()
    reference = MeshVisibility(mesh, targets, problem['profile'], work=full_work)
    full_masks = _query_pool(reference, poses, list(range(len(poses))), list(range(len(poses))), len(targets))
    phases['full_visibility'] = time.perf_counter() - started
    for arm, masks in zip(arms, queried_masks):
        if not np.array_equal(full_masks[arm['query_plan']['selected_ids']], masks):
            raise RuntimeError('Arm and full reference disagree on physical masks')
    started = time.perf_counter()
    full = build_frontier(poses, full_masks, targets, problem['k_values'],
        alternatives_per_k=1, enumeration_limit=enumeration_limit,
        context=dict(track=plan['track'], evaluation_domain='geometry_generated_full_domain',
                     source_layout_warm_start=False, source_certificates_reused=False))
    if (not full['enumeration']['completed']
            or any(row['bound_source'] != 'exhaustive_subsets' for row in full['frontier'])):
        raise RuntimeError('Full reference did not complete exact enumeration')
    _annotate_ids(full, list(range(len(poses))))
    phases['full_selection'] = time.perf_counter() - started
    if plans != frozen_plans or arms != frozen_decisions:
        raise RuntimeError('Query plan or arm decision changed during reference evaluation')

    started = time.perf_counter()
    solutions = [solution for report in [full, *(arm['frontier'] for arm in arms)]
                 for row in report['frontier'] for solution in [row, *row['alternatives']]]
    validation_work = Work()
    independent = MeshVisibility(mesh, targets, problem['profile'], work=validation_work)
    selected = sorted({cid for solution in solutions for cid in solution['full_candidate_ids']})
    check_masks = _query_pool(independent, poses, selected, selected, len(targets))
    checked = dict(zip(selected, check_masks))
    for cid in selected:
        if not np.array_equal(checked[cid], full_masks[cid]):
            raise RuntimeError('Fresh target raycast disagrees with physical mask')
    for solution in solutions:
        union = np.zeros(len(targets), bool)
        for cid in solution['full_candidate_ids']:
            union |= checked[cid]
        if int(union.sum()) != solution['covered_count']:
            raise RuntimeError('Saved solution union disagrees with fresh target raycast')
        solution['validated_direct_raycast'] = True
    phases['independent_validation'] = time.perf_counter() - started
    comparison = []
    for arm in arms:
        for row, reference_row in zip(arm['frontier']['frontier'], full['frontier']):
            difference = reference_row['covered_count'] - row['covered_count']
            if difference < 0:
                raise RuntimeError('Arm exceeds exact full-domain optimum')
            comparison.append(dict(arm_id=arm['arm_id'], method=arm['method'], seed=arm['seed'],
                budget_k=row['budget_k'], query_count=plan['query_budget'],
                covered_count=row['covered_count'], coverage_pct=row['coverage_pct'],
                reference_covered_count=reference_row['covered_count'],
                reference_coverage_pct=reference_row['coverage_pct'], full_shortfall_count=difference))
    report = dict(schema='library_matched_query_evaluation_v1', plan=plan, arms=arms,
        full_frontier=full, comparison=comparison, phase_wall_sec=phases,
        validation=dict(passed=True, all_count_matched=True, all_optimal=True,
            plans_before_physical_queries=True, full_reference_after_all_decisions=True,
            shared_target_order=True, unique_candidates_recast=selected,
            saved_solution_records=len(solutions),
            scope='Fresh provider/scene, same physical implementation, with union verification'),
        work=dict(full_reference=dict(full_work.counts), independent_validation=dict(validation_work.counts)),
        timing_scope='One process and shared mesh; separate provider per arm; source loading/mapping '
            'reported separately once; geometry planning includes all unoccluded point tests. '
            'Equal physical pose queries do not equalize rays, CPU or total time; no speedup inference.',
        statistical_scope='Random seeds vary candidate selection on the same development maps. '
            'Descriptive conditional distribution only; no independent-map or superiority significance claim.',
        certificate_scope='Exact only in the declared finite grids; source certificates are not transferred.')
    return dict(report=report, archive=dict(poses=poses, targets=targets, masks=full_masks,
                                          pool_ids=np.asarray(problem['pool_ids'], dtype=np.int64)),
                mesh=dict(vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles)))

"""Bounded target-scene checks for new-format A/B source-library reuse.

Restricted-pool selection finishes before full-reference visibility is queried.
Timings are sequential diagnostics, not isolated cold-start benchmarks.
"""
from collections import Counter
import copy
import time

import numpy as np

from src.core.deployment_library import build_frontier
from src.core.visibility import MeshVisibility


class Work:
    def __init__(self):
        self.counts = Counter()

    def add(self, name, amount=1):
        self.counts[name] += int(amount)


def _ids(values, size):
    values = list(values)
    if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            or not 0 <= i < size for i in values) or len(set(values)) != len(values)):
        raise ValueError('Expected unique in-domain integer target candidate IDs')
    return [int(i) for i in values]


def plan_problem(problem, enumeration_limit=100000):
    from math import comb

    poses, targets = np.asarray(problem['poses']), np.asarray(problem['targets'])
    if (poses.ndim != 2 or poses.shape[1:] != (5,) or not len(poses)
            or targets.ndim != 2 or targets.shape[1:] != (3,) or not len(targets)
            or not np.isfinite(poses).all() or not np.isfinite(targets).all()
            or len(np.unique(poses, axis=0)) != len(poses)
            or len(np.unique(targets, axis=0)) != len(targets)):
        raise ValueError('Expected distinct finite nonempty poses[M,5] and targets[N,3]')
    if len(poses) > 2000 or len(targets) > 50000 or len(poses) * len(targets) > 4000000:
        raise ValueError('Target smoke domain exceeds its size guard')
    ks = list(problem['k_values'])
    if (not ks or any(type(k) is not int or not 1 <= k <= 64 for k in ks)
            or ks != sorted(set(ks))):
        raise ValueError('Expected increasing unique integer K values in 1..64')
    if type(enumeration_limit) is not int or not 0 <= enumeration_limit <= 1000000:
        raise ValueError('Invalid enumeration limit')
    pool = _ids(problem['pool_ids'], len(poses))
    if pool != sorted(pool):
        raise ValueError('Pool IDs must be sorted for deterministic tie resolution')
    for row in problem['transfers']:
        if row['status'] != 'complete':
            continue
        ids = _ids(row['selected_ids'], len(poses))
        if (not set(ids).issubset(pool) or type(row['budget_k']) is not int
                or row['budget_k'] < len(ids) or row['num_selected'] != len(ids)):
            raise ValueError('Transferred layout disagrees with pool or budget')
    counts = {name: sum(comb(m, r) for r in range(min(max(ks), m) + 1))
              for name, m in [('library_pool', len(pool)), ('full_reference', len(poses))]}
    return dict(track=problem['context']['track'], execute=False,
        candidate_count=len(poses), pool_candidate_count=len(pool), target_count=len(targets),
        k_values=ks, enumeration_limit=enumeration_limit, subset_counts=counts,
        enumeration_will_complete={key: count <= enumeration_limit for key, count in counts.items()},
        transfer_status_counts=dict(Counter(row['status'] for row in problem['transfers'])),
        context=problem['context'], budget_mode='at_most_k', objective='unweighted_union_count',
        scope='Development target smoke; no training, operational joint constraints or runtime superiority test')


def _annotate_ids(report, full_ids):
    for row in report['frontier']:
        for layout in [row, *row['alternatives']]:
            layout['full_candidate_ids'] = [full_ids[i] for i in layout['selected_ids']]


def evaluate_problem(problem, enumeration_limit=100000):
    """Return reports and arrays; no source writes, target cache reuse or export."""
    plan = plan_problem(problem, enumeration_limit)
    poses, targets = np.asarray(problem['poses']), np.asarray(problem['targets'])
    pool_ids, ks = list(problem['pool_ids']), list(problem['k_values'])
    started = time.perf_counter()
    mesh = problem['geometry'].create_environment_mesh()
    phases = dict(shared_target_mesh=time.perf_counter() - started)
    started = time.perf_counter()
    online_work = Work()
    online = MeshVisibility(mesh, targets, problem['profile'], work=online_work)
    pool_masks = np.asarray([online.mask(poses[i]) for i in pool_ids], dtype=bool).reshape(len(pool_ids), len(targets))
    phases['pool_visibility'] = time.perf_counter() - started
    started = time.perf_counter()
    index = {full: local for local, full in enumerate(pool_ids)}
    proposals = [dict(selected_ids=[index[i] for i in row['selected_ids']],
                      budget_k=row['budget_k'], origin=f'transferred_layout_{number}')
                 for number, row in enumerate(problem['transfers']) if row['status'] == 'complete']
    pool_report = build_frontier(poses[pool_ids], pool_masks, targets, ks,
        enumeration_limit=enumeration_limit, proposals=proposals,
        context=dict(**problem['context'], evaluation_domain='transferred_pool',
                     full_candidate_ids=pool_ids, source_certificates_reused=False))
    _annotate_ids(pool_report, pool_ids)
    transfers = copy.deepcopy(problem['transfers'])
    for row in transfers:
        if row['status'] == 'complete':
            ids = [index[i] for i in row['selected_ids']]
            count = int(np.any(pool_masks[ids], axis=0).sum())
            row.update(target_covered_count=count, target_coverage_pct=100. * count / len(targets))
    phases['pool_selection'] = time.perf_counter() - started

    # No full-scene candidate masks exist until the online decisions above finish.
    frozen_online = copy.deepcopy(pool_report)
    started = time.perf_counter()
    reference_work = Work()
    reference = MeshVisibility(mesh, targets, problem['profile'], work=reference_work)
    masks = np.asarray([reference.mask(pose) for pose in poses], dtype=bool)
    if not np.array_equal(masks[pool_ids], pool_masks):
        raise RuntimeError('Restricted and reference providers disagree on target masks')
    phases['reference_visibility'] = time.perf_counter() - started
    started = time.perf_counter()
    context = dict(**problem['context'], evaluation_domain='geometry_generated_full_domain',
                   source_certificates_reused=False, source_layout_warm_start=False)
    full_report = build_frontier(poses, masks, targets, ks,
        enumeration_limit=enumeration_limit, context=context)
    greedy_report = build_frontier(poses, masks, targets, ks,
        enumeration_limit=0, alternatives_per_k=1, context=dict(context, selection='greedy_only'))
    _annotate_ids(full_report, list(range(len(poses))))
    _annotate_ids(greedy_report, list(range(len(poses))))
    phases['reference_selection'] = time.perf_counter() - started
    if pool_report != frozen_online:
        raise RuntimeError('Online result changed during reference evaluation')

    started = time.perf_counter()
    validation_work = Work()
    independent = MeshVisibility(mesh, targets, problem['profile'], work=validation_work)
    reports = [pool_report, full_report, greedy_report]
    layouts = [layout for report in reports for row in report['frontier']
               for layout in [row, *row['alternatives']]]
    selected = sorted({i for row in layouts for i in row['full_candidate_ids']}
                      | {i for row in transfers if row['status'] == 'complete' for i in row['selected_ids']})
    checked = {i: independent.mask(poses[i]) for i in selected}
    for i in selected:
        if not np.array_equal(checked[i], masks[i]):
            raise RuntimeError('Fresh target raycast mask mismatch')
    for row in layouts:
        union = np.zeros(len(targets), dtype=bool)
        for i in row['full_candidate_ids']:
            union |= checked[i]
        if int(union.sum()) != row['covered_count']:
            raise RuntimeError('Saved layout union disagrees with fresh target raycast')
        row['validated_direct_raycast'] = True
    for row in transfers:
        if row['status'] != 'complete':
            continue
        union = np.zeros(len(targets), dtype=bool)
        for i in row['selected_ids']:
            union |= checked[i]
        if int(union.sum()) != row['target_covered_count']:
            raise RuntimeError('Transferred layout union disagrees with fresh target raycast')
        row['validated_direct_raycast'] = True
    phases['independent_validation'] = time.perf_counter() - started
    comparison = []
    for online_row, full_row, greedy_row in zip(pool_report['frontier'], full_report['frontier'], greedy_report['frontier']):
        if online_row['covered_count'] > full_row['upper_bound_count']:
            raise RuntimeError('Restricted feasible solution exceeds full-domain upper bound')
        comparison.append(dict(budget_k=online_row['budget_k'],
            library_pool_coverage_pct=online_row['coverage_pct'],
            library_pool_num_selected=online_row['num_selected'],
            library_pool_certification=online_row['certification'],
            full_reference_coverage_pct=full_row['coverage_pct'],
            full_reference_certification=full_row['certification'],
            full_reference_upper_bound_count=full_row['upper_bound_count'],
            greedy_coverage_pct=greedy_row['coverage_pct'],
            pool_minus_full_incumbent_pp=online_row['coverage_pct'] - full_row['coverage_pct'],
            pool_certified_shortfall_upper_count=full_row['upper_bound_count'] - online_row['covered_count']))
    report = dict(schema='library_target_evaluation_v1', plan=plan,
        pool_frontier=pool_report, full_frontier=full_report, greedy_frontier=greedy_report,
        transfers=transfers, comparison=comparison,
        validation=dict(passed=True, pool_is_subset=True, shared_target_order=True,
            full_reference_not_used_by_online=True, unique_candidates_recast=selected,
            saved_solution_records=len(layouts),
            transferred_layout_records=sum(row['status'] == 'complete' for row in transfers),
            scope='Fresh scene/provider with same physical implementation; no independent physics implementation'),
        work=dict(library_pool=dict(online_work.counts), full_reference=dict(reference_work.counts),
                  independent_validation=dict(validation_work.counts)), phase_wall_sec=phases,
        timing_scope='Same process, shared geometry/mesh and imports; online selection before reference; '
                     'source loading and geometric transfer excluded here, runner records preparation separately; '
                     'not a controlled cold-start speedup benchmark',
        certificate_scope='Pool certificates cover only the transferred pool. Full certificates cover only '
                          'the declared finite target grid. No source certificate is transferred.')
    return dict(report=report, archive=dict(poses=poses, targets=targets, masks=masks,
                                          pool_ids=np.asarray(pool_ids, dtype=np.int64)),
                mesh=dict(vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles)))

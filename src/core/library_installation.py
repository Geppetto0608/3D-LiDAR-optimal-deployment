"""Unary installation exclusions for frozen library transfer experiments.

Closed XY rectangles remove poses, not targets or mesh geometry. Candidate IDs
in the returned problem index the filtered domain; source records remain nested
and explicitly refer to the original domain.
"""
from copy import deepcopy

import numpy as np

from src.core.coverage_requirements import _bounds, _inside
from src.core.library_transfer_evaluation import _ids, plan_problem


SCHEMA = 'library_installation_exclusion_v1'


def validate_spec(spec):
    if (not isinstance(spec, dict) or set(spec) != {'schema', 'forbidden_regions'}
            or spec['schema'] != SCHEMA or not isinstance(spec['forbidden_regions'], list)
            or len(spec['forbidden_regions']) > 100):
        raise ValueError('Expected versioned installation exclusions with at most 100 rectangles')
    for bounds in spec['forbidden_regions']:
        _bounds(bounds)
    return deepcopy(spec)


def allowed_candidates(poses, spec):
    spec = validate_spec(spec)
    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1:] != (5,) or not np.isfinite(poses).all():
        raise ValueError('Expected finite poses[M,5]')
    allowed = np.ones(len(poses), dtype=bool)
    for bounds in spec['forbidden_regions']:
        allowed &= ~_inside(poses, bounds)
    return allowed


def with_installation(problem, spec):
    if 'installation' in problem['context']:
        raise ValueError('Installation exclusions already attached')
    plan_problem(problem)
    spec = validate_spec(spec)
    poses = np.asarray(problem['poses'])
    allowed = allowed_candidates(poses, spec)
    original_ids = np.flatnonzero(allowed).tolist()
    if not original_ids:
        raise ValueError('Empty full candidate domain is unsupported by this study runner')
    remap = {old: new for new, old in enumerate(original_ids)}
    pool = [remap[i] for i in problem['pool_ids'] if i in remap]
    transfers = []
    for original in problem['transfers']:
        row = dict(original_transfer=deepcopy(original),
                   original_id_space='unrestricted_target_domain',
                   budget_k=original['budget_k'], status=original['status'])
        if original['status'] == 'complete':
            excluded = [i for i in original['selected_ids'] if i not in remap]
            if excluded:
                row.update(status='unavailable', reason='installation_exclusion',
                           excluded_original_ids=excluded)
            else:
                ids = [remap[i] for i in original['selected_ids']]
                row.update(selected_ids=ids, num_selected=len(ids))
        transfers.append(row)
    proposals = []
    for row in problem.get('proposals', []):
        ids = _ids(row['selected_ids'], len(poses))
        if all(i in remap for i in ids):
            proposals.append(dict(selected_ids=[remap[i] for i in ids],
                                  budget_k=row['budget_k'], origin=row.get('origin', 'transfer')))
    context = dict(track=problem['context']['track'], source_context=deepcopy(problem['context']),
        installation=dict(spec=spec, original_candidate_count=len(poses),
            eligible_to_original_ids=original_ids, excluded_original_ids=np.flatnonzero(~allowed).tolist(),
            original_pool_ids=list(problem['pool_ids']), original_pool_count=len(problem['pool_ids']),
            eligible_pool_count=len(pool), empty_library_pool=not pool,
            semantics='Current IDs index eligible poses. Nested source_context/original_transfer IDs '
                      'index the unrestricted domain. Surviving pool candidates are retained even '
                      'when their original complete layout is unavailable; no repair or padding.'))
    result = dict(problem, poses=poses[original_ids].copy(), pool_ids=pool,
                  transfers=transfers, proposals=proposals, context=context)
    plan_problem(result)
    return result


def validate_result(result, spec):
    """Attach explicit ID mapping after verifying the entire evaluated domain."""
    archive, report = result['archive'], result['report']
    if not allowed_candidates(archive['poses'], spec).all():
        raise ValueError('Evaluated a forbidden installation candidate')
    meta = report['plan']['context']['installation']
    if meta['spec'] != validate_spec(spec):
        raise ValueError('Installation input changed during evaluation')
    ids = _ids(meta['eligible_to_original_ids'], meta['original_candidate_count'])
    if len(ids) != len(archive['poses']) or ids != sorted(ids):
        raise ValueError('Invalid eligible-to-original candidate mapping')
    archive['original_candidate_ids'] = np.asarray(ids, dtype=np.int64)
    report['validation']['installation_passed'] = True

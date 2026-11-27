"""Matched geometry controls: separate pair scoring from measured feedback.

Reads completed, audited artifacts only. Timing repeats are not new maps.
"""
import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

import geometry_placement as runner
from geometry_budget_report import average, compare, TOL

METHODS = ('adaptive_batch', 'static', 'static_pair', 'no_feedback')
CONTRASTS = (('adaptive_batch', 'no_feedback'), ('static_pair', 'static'),
             ('adaptive_batch', 'static_pair'))


def aggregate(rows, cases, budgets):
    if not cases or len(set(cases)) != len(cases) or not budgets or budgets != sorted(set(budgets)):
        raise ValueError('Unique cases and sorted nonempty budgets required')
    expected = {(c, m, b) for c in cases for m in METHODS for b in budgets}
    expected.update((c, 'full', None) for c in cases)
    indexed = {}
    for row in rows:
        key = row['case'], row['method'], row['query_budget']
        value = row['objective_score']
        if (key in indexed or row.get('random_seed') is not None
                or row['feasible'] != (value is not None) or (value is not None and not np.isfinite(value))):
            raise ValueError('Duplicate/invalid method or feasible objective')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('Incomplete or unexpected mechanism grid')
    paired, groups = [], {}
    for budget in budgets:
        by_method = {m: [] for m in METHODS}
        contrasts = {f'{a}_vs_{b}': Counter() for a, b in CONTRASTS}
        for case in cases:
            full = indexed[case, 'full', None]
            actual = {m: indexed[case, m, budget] for m in METHODS}
            if len({r['candidate_queries'] for r in actual.values()}) != 1:
                raise ValueError('Unmatched physical pose-query budget')
            for m, row in actual.items():
                value, ref = row['objective_score'], full['objective_score']
                if value is not None and (ref is None or value > ref+TOL):
                    raise ValueError('Subset exceeds full reference')
                by_method[m].append(dict(row, gap=ref-value if ref is not None and value is not None else None))
            for a, b in CONTRASTS:
                left, right = actual[a], actual[b]
                aj, bj = left['objective_score'], right['objective_score']
                outcome = compare(aj, bj).replace('adaptive_only', 'left_only').replace('other_only', 'right_only')
                contrasts[f'{a}_vs_{b}'][outcome] += 1
                paired.append(dict(case=case, query_budget=budget, left_method=a, right_method=b,
                    left_j=aj, right_j=bj, left_minus_right_j=aj-bj if aj is not None and bj is not None else None,
                    outcome=outcome, reference_j=full['objective_score'], candidate_queries=left['candidate_queries'],
                    left_rays=left['rays_submitted'], right_rays=right['rays_submitted'],
                    left_cold_sec=left['cold_ready_median_sec'], right_cold_sec=right['cold_ready_median_sec'],
                    left_worker_wall_sec=left['worker_wall_median_sec'], right_worker_wall_sec=right['worker_wall_median_sec']))
        methods = {}
        for m, values in by_method.items():
            gaps = [v['gap'] for v in values if v['gap'] is not None]
            methods[m] = dict(cases=len(values), feasible_cases=sum(v['feasible'] for v in values),
                comparable_reference_cases=len(gaps), matches_reference=sum(abs(g)<=TOL for g in gaps),
                mean_feasible_j_gap=average(gaps), mean_pose_queries=average([v['candidate_queries'] for v in values]),
                mean_rays=average([v['rays_submitted'] for v in values]),
                mean_cold_ready_sec=average([v['cold_ready_median_sec'] for v in values]),
                mean_worker_wall_sec=average([v['worker_wall_median_sec'] for v in values]))
        groups[str(budget)] = dict(methods=methods, contrasts={k:dict(v) for k,v in contrasts.items()})
    full = [indexed[c, 'full', None] for c in cases]
    return paired, dict(conditions=len(cases), budgets=budgets, groups=groups,
        full=dict(feasible_cases=sum(r['feasible'] for r in full),
            mean_pose_queries=average([r['candidate_queries'] for r in full]),
            mean_cold_ready_sec=average([r['cold_ready_median_sec'] for r in full]),
            mean_worker_wall_sec=average([r['worker_wall_median_sec'] for r in full])),
        semantics='Equal-weight known development cases, not independent map families. Timing repetitions are '
            'not additional quality samples. Null J means no feasible solution in the queried pool, not zero or '
            'global infeasibility. J differences are weighted objective units, not coverage percentage points. '
            'Matched pose queries do not imply matched rays, FLOPs or time. Cold-ready excludes final validation/export; '
            'worker wall includes them. No global bound or unseen-generalization claim.')


def report(root):
    root = Path(root).resolve()
    manifest = runner.read(root/'manifest.json')
    validation = runner.read(root/'independent_validation.json')
    if (manifest['status'] != 'complete' or validation['status'] != 'passed'
            or validation['workers'] != len(manifest['jobs'])):
        raise ValueError('Completed independent audit required')
    for prefix, hashes in [('source', manifest['source_hashes']), ('', manifest['input_hashes'])]:
        for name, expected in hashes.items():
            if runner.corridor.digest(root/prefix/name) != expected:
                raise ValueError(f'Frozen hash mismatch: {name}')
    pools, qualities, selection_work = {}, {}, []
    for job in manifest['jobs']:
        directory = root/'runs'/job['id']
        for name, expected in job['output_hashes'].items():
            if runner.corridor.digest(directory/name) != expected:
                raise ValueError(f'Worker hash mismatch: {name}')
        arm = job['arm']
        if arm['method'] == 'full':
            continue
        selection = runner.read(directory/'selection.json')
        key = job['case'], arm['method'], job['repeat']
        pools.setdefault(key, {})[arm['query_budget']] = set(selection['candidate_ids'])
        assessment = runner.read(directory/'solution/result.json')['assessment']
        qualities.setdefault(key, {})[arm['query_budget']] = assessment['objective_score'] if assessment else None
        selection_work.append(dict(case=job['case'], method=arm['method'], query_budget=arm['query_budget'],
            repeat=job['repeat'], selection_sec=selection['selection_sec'],
            planning_sec=selection.get('planning_sec'), physical_query_sec=selection.get('physical_query_sec'),
            virtual_candidate_masks=selection.get('virtual_candidate_masks'),
            directed_candidate_partner_tests=selection.get('directed_candidate_partner_tests')))
    _, rows, _ = runner.tables(root, manifest)
    budgets = sorted({a['query_budget'] for a in manifest['plan']['arms'] if a['query_budget'] is not None})
    paired, summary = aggregate(rows, [c['directory'] for c in manifest['cases']], budgets)
    checked, seed_checks = 0, 0
    for key, by_budget in pools.items():
        for small, large in zip(budgets, budgets[1:]):
            before, after = qualities[key][small], qualities[key][large]
            if not by_budget[small] <= by_budget[large] or (before is not None and (after is None or before > after+TOL)):
                raise ValueError('Nested pool/quality contract violated')
            checked += 1
        if key[1] == 'adaptive_batch' and 1 in budgets:
            partner = key[0], 'no_feedback', key[2]
            if by_budget[1] != pools[partner][1] or qualities[key][1] != qualities[partner][1]:
                raise ValueError('No-feedback seed control differs before feedback')
            seed_checks += 1
    summary.update(timing_repeats=manifest['plan']['repeats'], nested_pool_checks=checked, seed_parity_checks=seed_checks)
    output = root/'mechanism_report'
    output.mkdir(exist_ok=True)
    for filename, values in [('paired.csv', paired), ('selection_work.csv', selection_work)]:
        with (output/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    runner.corridor.save(output/'summary.json', summary)
    runner.corridor.save(output/'provenance.json', dict(manifest_sha256=runner.corridor.digest(root/'manifest.json'),
        validation_sha256=runner.corridor.digest(root/'independent_validation.json'),
        report_sha256=runner.corridor.digest(Path(__file__)), output_sha256={
            n:runner.corridor.digest(output/n) for n in ('paired.csv', 'selection_work.csv', 'summary.json')}))
    return summary


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(report(parser.parse_args().root), allow_nan=False))

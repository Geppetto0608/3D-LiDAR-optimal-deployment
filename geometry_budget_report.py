"""Paired equal-pose-query comparison; random seeds are not additional maps.

Run after geometry_placement.py and its frozen independent audit finish. This
report reads saved results only and never supplies reference masks to a selector.
"""
import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

import geometry_placement as runner

TOL = 1e-7


def average(values):
    return float(np.mean(values)) if values else None


def compare(a, b):
    if a is None or b is None:
        return 'both_infeasible' if a is b else 'adaptive_only_feasible' if b is None else 'other_only_feasible'
    return 'win' if a > b+TOL else 'loss' if a < b-TOL else 'tie'


def aggregate(rows, cases, budgets, seeds, adaptive='adaptive_batch'):
    """Every case/arm is required; infeasibility is separate from numeric quality."""
    if (not cases or len(set(cases)) != len(cases) or not budgets or budgets != sorted(set(budgets))
            or not seeds or seeds != sorted(set(seeds))):
        raise ValueError('Unique cases and sorted nonempty budgets/seeds required')
    expected = {(c, m, b, s) for c in cases for b in budgets
        for m, s in [(adaptive, None), ('static', None)]+[('random', s) for s in seeds]}
    expected.update((c, 'full', None, None) for c in cases)
    indexed = {}
    for row in rows:
        key = row['case'], row['method'], row['query_budget'], row.get('random_seed')
        if key in indexed:
            raise ValueError('Duplicate case/method/budget/seed')
        value = row['objective_score']
        if row['feasible'] != (value is not None) or (value is not None and not np.isfinite(value)):
            raise ValueError('Invalid feasible objective (missing is not zero)')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('Incomplete or unexpected paired arm grid')
    paired, groups = [], {}
    for budget in budgets:
        stats = {m: [] for m in (adaptive, 'static', 'random')}
        static_outcomes, random_outcomes, random_mean_outcomes = Counter(), Counter(), Counter()
        for case in cases:
            reference = indexed[case, 'full', None, None]['objective_score']
            a, b = [indexed[case, m, budget, None] for m in (adaptive, 'static')]
            random_rows = [indexed[case, 'random', budget, s] for s in seeds]
            all_rows = [a, b]+random_rows
            if len({r['candidate_queries'] for r in all_rows}) != 1:
                raise ValueError('Unmatched actual pose-query count')
            for row in all_rows:
                value = row['objective_score']
                if value is not None and (reference is None or value > reference+TOL):
                    raise ValueError('Subset exceeds full reference')
            random_values = [r['objective_score'] for r in random_rows if r['feasible']]
            aj, bj = a['objective_score'], b['objective_score']
            random_mean = average(random_values)
            all_random_feasible = len(random_values) == len(seeds)
            static_outcomes[compare(aj, bj)] += 1
            for row in random_rows:
                random_outcomes[compare(aj, row['objective_score'])] += 1
            if all_random_feasible and aj is not None:
                random_mean_outcomes[compare(aj, random_mean)] += 1
            paired.append(dict(case=case, query_budget=budget, candidate_queries=a['candidate_queries'],
                reference_j=reference, adaptive_j=aj, static_j=bj,
                adaptive_minus_static_j=aj-bj if aj is not None and bj is not None else None,
                adaptive_vs_static=compare(aj, bj), random_seeds=len(seeds),
                random_feasible_seeds=len(random_values), random_feasible_mean_j=random_mean,
                random_feasible_sd_j=float(np.std(random_values, ddof=1)) if len(random_values)>1 else None,
                adaptive_minus_random_mean_j=aj-random_mean if aj is not None and all_random_feasible else None,
                random_reference_matches=sum(reference is not None and abs(v-reference)<=TOL for v in random_values),
                adaptive_rays=a['rays_submitted'], static_rays=b['rays_submitted'],
                random_mean_rays=average([r['rays_submitted'] for r in random_rows])))
            for method, values in [(adaptive, [a]), ('static', [b]), ('random', random_rows)]:
                feasible = [r for r in values if r['feasible']]
                gaps = [reference-r['objective_score'] for r in feasible] if reference is not None else []
                stats[method].append(dict(feasible_fraction=len(feasible)/len(values),
                    reference_fraction=sum(abs(g)<=TOL for g in gaps)/len(values),
                    feasible_gap=average(gaps), all_feasible=len(feasible)==len(values),
                    all_reference=len(gaps)==len(values) and all(abs(g)<=TOL for g in gaps),
                    cold=average([r['cold_ready_median_sec'] for r in values]),
                    queries=values[0]['candidate_queries'], rays=average([r['rays_submitted'] for r in values])))
        methods = {}
        for method, values in stats.items():
            gaps = [v['feasible_gap'] for v in values if v['feasible_gap'] is not None]
            methods[method] = dict(cases=len(values), all_feasible_cases=sum(v['all_feasible'] for v in values),
                all_reference_match_cases=sum(v['all_reference'] for v in values),
                mean_case_feasible_fraction=average([v['feasible_fraction'] for v in values]),
                mean_case_reference_match_fraction=average([v['reference_fraction'] for v in values]),
                feasible_gap_cases=len(gaps), mean_case_feasible_j_gap=average(gaps),
                mean_pose_queries=average([v['queries'] for v in values]),
                mean_rays=average([v['rays'] for v in values]), mean_cold_ready_sec=average([v['cold'] for v in values]))
        groups[str(budget)] = dict(methods=methods, adaptive_vs_static=dict(static_outcomes),
            adaptive_vs_random_seed=dict(random_outcomes), adaptive_vs_complete_random_mean=dict(random_mean_outcomes),
            complete_random_mean_cases=sum(random_mean_outcomes.values()))
    return paired, dict(conditions=len(cases), budgets=budgets, random_seeds=seeds, adaptive_method=adaptive,
        groups=groups, semantics='Each case has equal weight; random seeds averaged within case, never oracle-selected. '
            'Infeasible outcomes have null J, not zero. Feasible-only gaps are conditional and include denominators. '
            'Case/seed outcome counts are descriptive, not independent-map sample sizes. Budget unit is unique '
            'pose visibility queries per eligible site, not rays, wall time or FLOPs. Full is an unrestricted reference.')


def report(root):
    root = Path(root).resolve()
    manifest = runner.read(root/'manifest.json')
    validation = runner.read(root/'independent_validation.json')
    if (manifest['status'] != 'complete' or validation['status'] != 'passed'
            or validation['workers'] != len(manifest['jobs'])):
        raise ValueError('Completed independent audit required before reporting')
    for prefix, hashes in [('source', manifest['source_hashes']), ('', manifest['input_hashes'])]:
        for name, expected in hashes.items():
            if runner.corridor.digest(root/prefix/name) != expected:
                raise ValueError(f'Frozen hash mismatch: {name}')
    for job in manifest['jobs']:
        for name, expected in job['output_hashes'].items():
            if runner.corridor.digest(root/'runs'/job['id']/name) != expected:
                raise ValueError(f'Worker hash mismatch: {job["id"]}/{name}')
    _, rows, _ = runner.tables(root, manifest)
    arms = manifest['plan']['arms']
    adaptive = [m for m in ('adaptive', 'adaptive_batch') if any(a['method']==m for a in arms)]
    if len(adaptive) != 1:
        raise ValueError('Exactly one adaptive implementation required in paired report')
    budgets = sorted({a['query_budget'] for a in arms if a['query_budget'] is not None})
    seeds = sorted({a['random_seed'] for a in arms if a['method']=='random'})
    paired, summary = aggregate(rows, [c['directory'] for c in manifest['cases']], budgets, seeds, adaptive[0])
    pools, qualities = {}, {}
    for job in manifest['jobs']:
        arm = job['arm']
        if arm['method']=='full':
            continue
        directory = root/'runs'/job['id']
        key = job['case'], arm['method'], arm.get('random_seed'), job['repeat']
        pools.setdefault(key, {})[arm['query_budget']] = set(runner.read(directory/'selection.json')['candidate_ids'])
        assessment = runner.read(directory/'solution/result.json')['assessment']
        qualities.setdefault(key, {})[arm['query_budget']] = assessment['objective_score'] if assessment else None
    checked = 0
    for key, by_budget in pools.items():
        for small, large in zip(budgets, budgets[1:]):
            if not by_budget[small] <= by_budget[large]:
                raise ValueError('Budget increase did not preserve queried pool')
            before, after = qualities[key][small], qualities[key][large]
            if before is not None and (after is None or before > after+TOL):
                raise ValueError('Nested exact pool optimum decreased')
            checked += 1
    summary['nested_pool_checks'] = checked
    summary['timing_repeats'] = manifest['plan']['repeats']
    output = root/'budget_report'
    output.mkdir(exist_ok=True)
    with (output/'paired.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    runner.corridor.save(output/'summary.json', summary)
    runner.corridor.save(output/'provenance.json', dict(
        manifest_sha256=runner.corridor.digest(root/'manifest.json'),
        validation_sha256=runner.corridor.digest(root/'independent_validation.json'),
        report_sha256=runner.corridor.digest(Path(__file__)), output_sha256={
            name:runner.corridor.digest(output/name) for name in ('paired.csv', 'summary.json')}))
    return summary


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(report(parser.parse_args().root), allow_nan=False))

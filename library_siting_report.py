"""Analyze completed siting experiments without raycasting, training or solving.

Reuses the existing matched-query saved-array validator. This report deliberately
requires the frozen K=1,2,3 / 30-seed / positive-query design, not arbitrary runs.
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

import numpy as np

import library_query_report as common


METHODS = (*common.DETERMINISTIC, 'random_mean')
COLUMNS = {name: f'{name}_coverage_pct' for name in common.DETERMINISTIC}
COLUMNS['random_mean'] = 'random_mean_pct'


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')


def summarize(rows):
    """Equal weight per map/condition/K, after averaging random seeds per cell."""
    result = dict(conditions=len(rows), reference_mean_pct=float(np.mean([
        row['reference_coverage_pct'] for row in rows])), methods={})
    for method, column in COLUMNS.items():
        values = np.array([row[column] for row in rows])
        gaps = np.array([row['reference_coverage_pct'] for row in rows]) - values
        data = dict(mean_coverage_pct=float(values.mean()), mean_shortfall_pp=float(gaps.mean()),
                    p95_shortfall_pp=float(np.quantile(gaps, .95)), max_shortfall_pp=float(gaps.max()))
        if method != 'random_mean':
            data['exact_match_count'] = sum(row[f'{method}_covered_count'] == row['reference_covered_count']
                                            for row in rows)
        else:
            data['exact_match_seed_rows'] = sum(row['random_full_reference_match_count'] for row in rows)
            data['seed_rows'] = sum(row['random_seed_count'] for row in rows)
        if method != 'library':
            data['library_vs_method'] = common._wins([row['library_coverage_pct'] - row[column] for row in rows])
        result['methods'][method] = data
    return result


def grouped_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        track, level, occlusion, k = (row[key] for key in ('track', 'installation', 'occlusion', 'budget_k'))
        for group in ('all', f'track/{track}', f'K/{k}', f'track_K/{track}/{k}',
                      f'factor/{track}/{level}/{occlusion}', f'factor_K/{track}/{level}/{occlusion}/{k}'):
            groups[group].append(row)
    return {key: summarize(value) for key, value in groups.items()}


def paired_effects(rows):
    indexed = {(r['base_map'], r['installation'], r['occlusion'], r['budget_k']): r for r in rows}
    common._require(len(indexed) == len(rows), 'Duplicate factor/K cell')
    effects = []
    for base_map, k in sorted({(r['base_map'], r['budget_k']) for r in rows}):
        for factor, fixed_levels in [('installation', ('clear', 'occluded')), ('occlusion', ('open', 'restricted'))]:
            for fixed in fixed_levels:
                before_key = (base_map, 'open', fixed, k) if factor == 'installation' else (base_map, fixed, 'clear', k)
                after_key = (base_map, 'restricted', fixed, k) if factor == 'installation' else (base_map, fixed, 'occluded', k)
                common._require(before_key in indexed and after_key in indexed, 'Missing paired factor cell')
                before, after = indexed[before_key], indexed[after_key]
                item = dict(base_map=base_map, track=before['track'], budget_k=k, factor=factor, fixed=fixed,
                    query_before=before['query_count'], query_after=after['query_count'],
                    reference_delta_pp=after['reference_coverage_pct'] - before['reference_coverage_pct'])
                for method, column in COLUMNS.items():
                    item[f'{method}_delta_pp'] = after[column] - before[column]
                    item[f'{method}_shortfall_delta_pp'] = item['reference_delta_pp'] - item[f'{method}_delta_pp']
                effects.append(item)
    return effects


def analyze(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Report output must be new')
    if (root / 'runs') in output.resolve().parents or root == output.resolve():
        raise ValueError('Report output cannot replace raw results')
    manifest = common._json_bytes((root / 'manifest.json').read_bytes())
    common._require(manifest.get('status') == 'complete'
                    and manifest.get('schema') == 'library_occlusion_siting_sweep_v1', 'Incomplete/wrong batch')
    for field, prefix in [('input_hashes', ''), ('source_hashes', 'source')]:
        for name, expected in manifest[field].items():
            common._require(common._digest(common._artifact(root, str(Path(prefix) / name))) == expected,
                            f'Batch {field} changed: {name}')
    validation = {}
    for name, key in [('pair_validation.json', 'validation_sha256'),
                      ('installation_validation.json', 'installation_validation_sha256')]:
        common._require(common._digest(root / name) == manifest[key], f'{name} changed')
        validation[name] = common._json_bytes((root / name).read_bytes())
        common._require(validation[name]['passed'] is True, f'{name} failed')
    spec = common._json_bytes((root / 'inputs/spec.json').read_bytes())
    cases = common._json_bytes((root / 'inputs/cases.json').read_bytes())
    names = [c['name'] for c in cases]
    common._require(len(names) == len(set(names)) == manifest['expected_case_count']
                    and set(names) == set(manifest['completed'])
                    and len(manifest['completed']) == len(names), 'Case completion mismatch')
    common._require(spec['k_values'] == [1, 2, 3] and spec['control_seeds'] == list(range(30)),
                    'This report requires the frozen K/seed design')
    conditions, raw, availability, provenance = [], [], [], []
    for c in cases:
        case = common._validate_case(common._load(root / 'runs' / c['name']))
        plan = case['report']['plan']
        common._require(case['seeds'] == spec['control_seeds'] and plan['track'] == c['track'],
                        'Case track/seeds changed')
        for key, value in c['plan'].items():
            if key not in ('context', 'execute'):
                common._require(plan[key] == value, f'Case plan changed: {key}')
        common._require(plan['context']['installation'] == c['plan']['context']['installation']
                        and plan['context']['scene_occlusion'] == c['occlusion'], 'Case factor input changed')
        meta = dict(base_map=c['base_map'], installation=c['installation_level'], occlusion=c['condition'])
        conditions.extend(dict(row, **meta) for row in common._condition_rows(case))
        raw.extend(dict(case_id=c['name'], track=c['track'], **meta, **row) for row in case['raw'])
        inst = plan['context']['installation']
        availability.append(dict(case_id=c['name'], track=c['track'], **meta,
            full_candidates=plan['candidate_count'], original_candidates=inst['original_candidate_count'],
            query_count=case['q'], original_query_count=inst['original_pool_count'],
            query_fraction=case['q'] / plan['candidate_count'],
            excluded_candidates=len(inst['excluded_original_ids']), empty_pool=inst['empty_library_pool'],
            unavailable_layout_records=plan['transfer_status_counts'].get('unavailable', 0)))
        provenance.append(case['provenance'])
    common._require(len(raw) == manifest['expected_comparison_rows'] and len(conditions) == len(cases) * 3,
                    'Raw/condition count mismatch')
    effects = paired_effects(conditions)
    groups = grouped_summary(conditions)
    effect_groups = defaultdict(list)
    for row in effects:
        effect_groups[f'{row["track"]}/{row["factor"]}/{row["fixed"]}'].append(row)
    effect_summary = {key: {field: float(np.mean([r[field] for r in rows]))
                           for field in rows[0] if field.endswith('_pp') or field.startswith('query_')}
                      for key, rows in effect_groups.items()}
    # Mechanism checks use candidate-set inclusion and exact-pool results, not new solves.
    all_availability = {}
    for track in ('A', 'B'):
        for level in ('open', 'restricted'):
            rows = [r for r in availability if r['track'] == track and r['installation'] == level
                    and r['occlusion'] == 'clear']
            all_availability[f'{track}/{level}'] = dict(maps=len(rows),
                mean_full_candidates=float(np.mean([r['full_candidates'] for r in rows])),
                mean_query_count=float(np.mean([r['query_count'] for r in rows])),
                mean_query_fraction=float(np.mean([r['query_fraction'] for r in rows])),
                empty_pool_maps=sum(r['empty_pool'] for r in rows))
    worst = sorted(conditions, key=lambda r: r['reference_coverage_pct'] - r['library_coverage_pct'], reverse=True)[:10]
    caveats = [
        'Equal weight per map/installation/occlusion/K cell; A has 24 maps and B 32.',
        'Random results are first averaged across 30 seeds within each cell; seeds are not independent maps.',
        'Installation changes both the eligible domain and the common query budget; not a fixed-q causal estimate.',
        'Full optima are exact only on coarse finite grids (22-72 poses here), not continuous or city-scale optima.',
        'This analysis verifies hashes, stored mask unions, saved certificates and CSV/JSON agreement; no fresh raycast or exhaustive solve.',
        'Coverage is geometric visibility, not detection, tracking or operational RSU readiness.',
        'No significance test or confidence interval; observed wins and mean gaps are descriptive.',
        'No runtime speedup claim; geometry planners perform additional optical work and preparation boundaries differ.',
        'All predeclared cases retained; these are developed synthetic families, not independent unseen real roads.',
        'Source library learning, RL refinement, joint costs and spacing constraints were not tested.',
    ]
    summary = dict(schema='library_siting_analysis_v1', created_at=datetime.now(timezone.utc).isoformat(),
        cases=len(cases), conditions=len(conditions), raw_rows=len(raw), groups=groups,
        paired_effects=effect_summary, availability=all_availability, worst_library_cases=worst,
        occlusion_effective_pairs=sum(p['effective_occlusion'] for p in validation['pair_validation.json']['pairs']),
        validation=dict(passed=True, hashes_verified=True, saved_array_unions_verified=True,
                        csv_json_agreement=True, fresh_physics=False, fresh_optimization=False),
        caveats=caveats, source_sha256=common._digest(Path(__file__)),
        validator_sha256=common._digest(Path(common.__file__)), batch_manifest_sha256=common._digest(root / 'manifest.json'),
        provenance=provenance,
        interpretation_checks=dict(
            simpson='Track, K and factor strata reported; pooled means are not a universal trend.',
            ecological='Geometric target-point coverage is not detection or population performance.',
            selection='Synthetic development geometries; no random real-road sampling claim.',
            collider='q depends on surviving library pool; comparisons condition on this budget.',
            base_rate='Not applicable: no classification sensitivity/specificity inference.',
            regression_to_mean='No cases selected for rerun by extreme scores; worst cases are descriptive only.',
            survivorship='All declared completed cases included, none removed by coverage outcome.',
            look_elsewhere='All declared methods/factors reported; no significance tests or cherry-picked winner claim.',
            forking_paths='Frozen design, descriptive post-run summaries; not claimed as preregistered inference.',
            causation='Installation contrasts change eligible domain and q jointly; not isolated fixed-q effects.',
            reverse_causality='Factors resolved before evaluation; no inferred real-world causal relationship.'),
        command=[sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--output', str(output)])
    lines = ['# Installation / Occlusion Study Analysis', '',
        f'{len(cases)} cases; {len(conditions)} condition/K cells; {len(raw)} method/K/seed rows.',
        'Each cell is weighted equally. Random means collapse seeds before averaging cells.', '',
        '| Method | Mean coverage % | Mean shortfall pp | Exact-reference matches |',
        '|---|---:|---:|---:|']
    for method, row in groups['all']['methods'].items():
        match = f"{row['exact_match_count']}/{len(conditions)}" if method != 'random_mean' else f"{row['exact_match_seed_rows']}/{row['seed_rows']} seed rows"
        lines.append(f"| {method} | {row['mean_coverage_pct']:.3f} | {row['mean_shortfall_pp']:.3f} | {match} |")
    lines += ['', '## Track and Factor Breakdown', '',
              '| Track / installation / occlusion | Reference % | Library % | Union % | Portfolio % | Random mean % |',
              '|---|---:|---:|---:|---:|---:|']
    for key, row in groups.items():
        if key.startswith('factor/'):
            vals = [row['reference_mean_pct'], *[row['methods'][m]['mean_coverage_pct']
                    for m in ('library', 'geometry_union', 'geometry_portfolio', 'random_mean')]]
            lines.append('| ' + key.removeprefix('factor/') + ' | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    lines += ['', '## Interpretation', '',
        'Compare the library against both geometry-union and geometry-portfolio, not only single-score or random controls.',
        'Each queried pool is solved exactly in this experiment. Any positive full-reference shortfall therefore means',
        'the retained pool cannot attain the full-domain optimum; it is not an optimization failure inside that pool.',
        'A refinement restricted to those same IDs cannot close that gap. Additional eligible poses or different',
        'candidate selection would be needed; the benefit of such an extension remains untested.', '',
        '## Scope', '', *[f'- {c}' for c in caveats], '',
        'Verification status: ANALYZED. Saved-array and provenance checks passed; no new simulation.', '',
        'Interpretation checks: 11/11 covered in summary.json (including inapplicable diagnostic metrics).', '',
        '## Reproduce', '', '```bash', shlex.join(summary['command']), '```', '',
        'Use a new output directory. Original experiment artifacts are never edited.', '']
    serialized = json.dumps(summary, indent=2, allow_nan=False, default=json_scalar) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('condition_summary.csv', conditions), ('raw_comparison.csv', raw),
                       ('paired_effects.csv', effects), ('availability.csv', availability)]:
        common._write_csv(output / name, rows)
    (output / 'summary.json').write_text(serialized)
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return output


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(analyze(args.root, args.output))


if __name__ == '__main__':
    main()

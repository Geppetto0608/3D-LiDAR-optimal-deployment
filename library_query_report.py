"""Read-only aggregation of completed matched-query development experiments.

API: aggregate_reports(roots, output, previous_root=None) -> pathlib.Path.
Only a new output directory is written. No simulator, planner, or solver imports.
"""
from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from math import comb, isfinite
from pathlib import Path
import shlex
import sys

import numpy as np


SCHEMA = 'library_matched_query_evaluation_v1'
DETERMINISTIC = ('library', 'geometry_single', 'geometry_union', 'geometry_portfolio')
KS = [1, 2, 3]
DEVELOPMENT_CASES = {'a_w9', 'a_w12', 'b_turn45', 'b_turn60'}
DEVELOPMENT_SEEDS = list(range(30))
RAW_FIELDS = ('arm_id', 'method', 'seed', 'budget_k', 'query_count', 'covered_count',
              'coverage_pct', 'reference_covered_count', 'reference_coverage_pct',
              'full_shortfall_count')
CAVEATS = [
    'The four known development targets are not a generalization evaluation.',
    'Equal pose queries do not imply equal rays, CPU work, or runtime.',
    'Random seeds describe conditional variation within a map, not independent maps.',
    'Many arms and comparisons are descriptive, not inferential; no p-values.',
    'Saved-array checks are not fresh physics or an independent physics implementation.',
    'Enumeration completion and discrete optimality are checked as recorded certificates; '
    'the aggregator does not rerun exhaustive optimization.',
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f'{label}: expected integer >= {minimum}')
    return value


def _ids(value, size, label):
    _require(isinstance(value, list), f'{label}: expected ID list')
    for item in value:
        _integer(item, label)
    _require(len(value) == len(set(value)) and all(i < size for i in value),
             f'{label}: duplicate or out-of-domain IDs')
    return value


def _same_number(value, expected, label):
    _require(type(value) in (int, float) and isfinite(value)
             and abs(value - expected) <= 1e-8, f'{label}: differs from saved arrays')


def _json_bytes(content):
    def reject(value):
        raise ValueError(f'Non-finite JSON number: {value}')
    return json.loads(content, parse_constant=reject)


def _digest(path):
    with path.open('rb') as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _artifact(root, name):
    _require(isinstance(name, str), 'Artifact name must be a string')
    relative = Path(name)
    _require(not relative.is_absolute() and '..' not in relative.parts and relative.parts,
             f'Unsafe artifact path: {name}')
    path = root
    for part in relative.parts:
        path = path / part
        _require(not path.is_symlink(), f'Symlink artifact: {name}')
    _require(path.is_file(), f'Missing artifact: {name}')
    return path


def _load(root):
    root = Path(root).expanduser().resolve(strict=True)
    manifest_bytes = _artifact(root, 'manifest.json').read_bytes()
    manifest = _json_bytes(manifest_bytes)
    _require(manifest.get('schema') == 'library_target_run_v1',
             f'{root}: unexpected manifest schema')
    _require(manifest.get('status') == 'complete', f'{root}: run is not complete')
    frozen_hashes = {}
    for field, prefix in [('input_hashes', ''), ('source_hashes', 'source'),
                          ('library_hashes', 'inputs/library')]:
        declared = manifest.get(field)
        _require(isinstance(declared, dict) and declared, f'{root}: missing/empty {field}')
        for name, expected in declared.items():
            _require(isinstance(name, str), f'{field}: invalid snapshot path')
            path = _artifact(root, str(Path(prefix) / name))
            _require(_digest(path) == expected, f'{root}: frozen {field} hash mismatch: {name}')
        frozen_hashes[field] = declared
    hashes = manifest.get('artifact_hashes')
    _require(isinstance(hashes, dict) and
             {'report.json', 'comparison.csv', 'target.npz'} <= hashes.keys(),
             f'{root}: incomplete artifact hashes')
    payloads = {}
    for name, expected in hashes.items():
        path = _artifact(root, name)
        if name in ('report.json', 'comparison.csv', 'target.npz'):
            content = path.read_bytes()
            actual = sha256(content).hexdigest()
            payloads[name] = content
        else:
            actual = _digest(path)
        _require(actual == expected, f'{root}: artifact hash mismatch: {name}')
    with np.load(io.BytesIO(payloads['target.npz']), allow_pickle=False) as archive:
        _require({'poses', 'targets', 'masks', 'pool_ids'} <= set(archive.files),
                 f'{root}: missing target arrays')
        arrays = {name: archive[name] for name in ('poses', 'targets', 'masks', 'pool_ids')}
    poses, targets, masks = (arrays[name] for name in ('poses', 'targets', 'masks'))
    for name, array, width in [('poses', poses, 5), ('targets', targets, 3)]:
        _require(array.dtype.kind in 'fiu' and array.ndim == 2 and array.shape[1] == width
                 and len(array) > 0 and np.isfinite(array).all(), f'{root}: invalid {name}')
        _require(len(np.unique(array, axis=0)) == len(array), f'{root}: duplicate {name}')
    _require(masks.dtype == np.bool_ and masks.shape == (len(poses), len(targets)),
             f'{root}: expected boolean masks[M,N]')
    pool = arrays['pool_ids']
    _require(pool.ndim == 1 and pool.dtype.kind in 'iu', f'{root}: invalid pool_ids array')
    _ids(pool.tolist(), len(poses), 'pool_ids')
    return dict(root=root, manifest=manifest, report=_json_bytes(payloads['report.json']),
                arrays=arrays, csv=payloads['comparison.csv'].decode('utf-8'),
                provenance=dict(root=str(root), manifest_sha256=sha256(manifest_bytes).hexdigest(),
                                artifact_hashes=hashes, frozen_hashes_verified=True, **frozen_hashes))


def _frontier(frontier, domain, arrays, label):
    poses, masks, targets = (arrays[name] for name in ('poses', 'masks', 'targets'))
    n = len(targets)
    _require(frontier.get('schema') == 'deployment_library_frontier_v1', f'{label}: schema')
    _require(frontier.get('k_values') == KS and all(type(k) is int for k in frontier['k_values']),
             f'{label}: expected K=1,2,3')
    _require(frontier.get('budget_mode') == 'at_most_k'
             and frontier.get('objective') == 'unweighted_union_count', f'{label}: objective')
    for field, expected in [('candidate_count', len(domain)), ('target_count', n)]:
        _require(_integer(frontier.get(field), field) == expected, f'{label}: {field}')
    enumeration = frontier.get('enumeration', {})
    _require(enumeration.get('completed') is True, f'{label}: enumeration not complete')
    subset_count = sum(comb(len(domain), k) for k in range(min(3, len(domain)) + 1))
    _require(_integer(enumeration.get('subset_count'), 'subset_count') == subset_count
             and _integer(enumeration.get('limit'), 'enumeration limit') >= subset_count,
             f'{label}: inconsistent enumeration certificate')
    ceiling = int(np.any(masks[domain], axis=0).sum())
    _require(_integer(frontier.get('coverage_ceiling_count'), 'ceiling') == ceiling,
             f'{label}: ceiling differs from masks')
    _same_number(frontier.get('coverage_ceiling_pct'), 100. * ceiling / n, f'{label}: ceiling')
    rows = frontier.get('frontier')
    _require(isinstance(rows, list) and len(rows) == 3, f'{label}: missing/duplicate K rows')
    by_k = {}
    previous = 0
    for row in rows:
        k = _integer(row.get('budget_k'), 'budget_k', 1)
        _require(k in KS and k not in by_k, f'{label}: missing/duplicate K rows')
        alternatives = row.get('alternatives')
        _require(isinstance(alternatives, list) and alternatives, f'{label}: missing alternatives')
        counts = []
        for layout in [row, *alternatives]:
            ids = _ids(layout.get('full_candidate_ids'), len(poses), f'{label}: full_candidate_ids')
            local = _ids(layout.get('selected_ids'), len(domain), f'{label}: selected_ids')
            _require(ids == [domain[i] for i in local], f'{label}: IDs disagree with query plan')
            _require(len(ids) <= k and _integer(layout.get('num_selected'), 'num_selected') == len(ids),
                     f'{label}: NumSelected/budget mismatch')
            if 'NumSelected' in layout:
                _require(_integer(layout['NumSelected'], 'NumSelected') == len(ids),
                         f'{label}: NumSelected mismatch')
            expected_poses = poses[ids].tolist()
            _require(layout.get('poses') == expected_poses, f'{label}: poses disagree with IDs')
            count = int(np.any(masks[ids], axis=0).sum())
            _require(_integer(layout.get('covered_count'), 'covered_count') == count,
                     f'{label}: coverage differs from masks')
            _same_number(layout.get('coverage_pct'), 100. * count / n, f'{label}: coverage_pct')
            counts.append(count)
        lower = counts[0]
        _require(all(count <= lower for count in counts), f'{label}: better saved alternative')
        _require(alternatives[0]['full_candidate_ids'] == row['full_candidate_ids'],
                 f'{label}: first alternative is not incumbent')
        _require(row.get('certification') == 'discrete_optimal'
                 and row.get('bound_source') == 'exhaustive_subsets'
                 and _integer(row.get('upper_bound_count'), 'upper_bound_count') == lower,
                 f'{label}: expected discrete optimal upper=L')
        _same_number(row.get('certified_gap'), 0, f'{label}: certified gap')
        _require(lower <= ceiling, f'{label}: lower exceeds ceiling')
        for alt, count in zip(alternatives, counts[1:]):
            _require(_integer(alt.get('incumbent_loss_count'), 'incumbent loss') == lower - count,
                     f'{label}: alternative loss mismatch')
        by_k[k] = row
    for k in KS:
        _require(by_k[k]['covered_count'] >= previous, f'{label}: decreasing frontier')
        previous = by_k[k]['covered_count']
    return by_k


def _validate_csv(content, expected):
    reader = csv.DictReader(io.StringIO(content))
    _require(reader.fieldnames is not None and set(RAW_FIELDS) <= set(reader.fieldnames)
             and len(reader.fieldnames) == len(set(reader.fieldnames)), 'comparison.csv: missing/duplicate columns')
    seen = set()
    for raw in reader:
        _require(None not in raw, 'comparison.csv: extra cells')
        try:
            key = (raw['arm_id'], int(raw['budget_k']))
        except (TypeError, ValueError) as exc:
            raise ValueError('comparison.csv: invalid row key') from exc
        _require(key in expected and key not in seen, 'comparison.csv: unexpected/duplicate row')
        seen.add(key)
        for field in RAW_FIELDS:
            value, wanted = raw[field], expected[key][field]
            if field in ('arm_id', 'method'):
                _require(value == wanted, f'comparison.csv: inconsistent {field}')
            elif field == 'seed' and wanted is None:
                _require(value in ('', 'None', 'null'), 'comparison.csv: inconsistent seed')
            elif field.endswith('_pct'):
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'comparison.csv: invalid {field}') from exc
                _same_number(value, wanted, f'comparison.csv: {field}')
            else:
                try:
                    value = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'comparison.csv: invalid {field}') from exc
                _require(value == wanted, f'comparison.csv: inconsistent {field}')
    _require(seen == set(expected), 'comparison.csv: missing arm/K rows')


def _validate_report_comparison(rows, expected):
    _require(isinstance(rows, list), 'report.comparison: expected raw rows')
    seen = set()
    for row in rows:
        _require(isinstance(row, dict) and set(RAW_FIELDS) <= row.keys(),
                 'report.comparison: missing fields')
        _require(isinstance(row['arm_id'], str), 'report.comparison: invalid arm ID')
        key = (row['arm_id'], _integer(row['budget_k'], 'report.comparison budget_k', 1))
        _require(key in expected and key not in seen, 'report.comparison: unexpected/duplicate row')
        seen.add(key)
        for field in RAW_FIELDS:
            value, wanted = row[field], expected[key][field]
            if field.endswith('_pct'):
                _same_number(value, wanted, f'report.comparison: {field}')
            else:
                _require(type(value) is type(wanted) and value == wanted,
                         f'report.comparison: inconsistent {field}')
    _require(seen == set(expected), 'report.comparison: missing arm/K rows')


def _validate_case(case):
    report, arrays = case['report'], case['arrays']
    _require(report.get('schema') == SCHEMA, 'Expected matched-query evaluation schema')
    _require(report.get('validation', {}).get('passed') is True, 'Validation did not pass')
    plan = report['plan']
    _require(isinstance(plan.get('context'), dict) and isinstance(plan.get('track'), str), 'Invalid plan context')
    _require(plan.get('k_values') == KS and all(type(k) is int for k in plan['k_values']),
             'Plan must have K=1,2,3')
    m, n, pool = len(arrays['poses']), len(arrays['targets']), arrays['pool_ids'].tolist()
    for field, count in [('candidate_count', m), ('target_count', n), ('pool_candidate_count', len(pool))]:
        _require(_integer(plan.get(field), field) == count, f'Plan {field} disagrees with arrays')
    seeds = plan.get('control_seeds')
    _require(isinstance(seeds, list) and seeds, 'Missing control seeds')
    for seed in seeds:
        _integer(seed, 'control seed')
    _require(len(seeds) == len(set(seeds)), 'Duplicate control seeds')
    expected = {name: None for name in DETERMINISTIC}
    expected.update({f'random_{seed}': seed for seed in seeds})
    _require(_integer(plan.get('expected_arm_count'), 'expected_arm_count') == len(expected),
             'Incorrect expected arm count')
    q = _integer(plan.get('query_budget'), 'query_budget', 1)
    _require(q <= m, 'Query budget exceeds candidate count')
    full = _frontier(report['full_frontier'], list(range(m)), arrays, 'full reference')
    arms = report.get('arms')
    _require(isinstance(arms, list) and len(arms) == len(expected), 'Missing/duplicate arms or seeds')
    seen, raw, by_arm = set(), [], {}
    for arm in arms:
        name = arm.get('arm_id')
        _require(name in expected and name not in seen, 'Unexpected/duplicate arm')
        seen.add(name)
        seed = expected[name]
        _require(arm.get('seed') == seed and (seed is None or type(arm.get('seed')) is int),
                 f'{name}: wrong seed')
        method = 'random' if seed is not None else name
        _require(arm.get('method') == method, f'{name}: wrong method')
        ids = _ids(arm['query_plan'].get('selected_ids'), m, f'{name}: query plan')
        _require(len(ids) == q and _integer(arm.get('work', {}).get('direct_pose_evaluations'),
                                           'direct_pose_evaluations') == q,
                 f'{name}: unequal query count')
        if 'query_order' in arm['query_plan'] or 'physical_query_ids' in arm:
            order = _ids(arm['query_plan'].get('query_order'), m, f'{name}: query order')
            physical = _ids(arm.get('physical_query_ids'), m, f'{name}: physical query IDs')
            _require(set(order) == set(ids) and physical == order,
                     f'{name}: physical query order disagrees with query plan')
        if name == 'library':
            _require(set(ids) == set(pool), 'Library query IDs differ from stored pool')
        rows = _frontier(arm['frontier'], ids, arrays, name)
        by_arm[name] = rows
        for k in KS:
            count, reference = rows[k]['covered_count'], full[k]['covered_count']
            _require(count <= reference, f'{name}: exceeds full reference upper bound')
            raw.append(dict(arm_id=name, method=method, seed=seed, budget_k=k, query_count=q,
                            covered_count=count, coverage_pct=100. * count / n,
                            reference_covered_count=reference, reference_coverage_pct=100. * reference / n,
                            full_shortfall_count=reference - count))
    _require(seen == set(expected), 'Missing expected arms/seeds')
    if 'saved_solution_records' in report['validation']:
        saved = sum(1 + len(row['alternatives'])
                    for frontier in [report['full_frontier'], *(arm['frontier'] for arm in arms)]
                    for row in frontier['frontier'])
        _require(_integer(report['validation']['saved_solution_records'], 'saved_solution_records') == saved,
                 'Validation saved solution record count mismatch')
    expected_rows = {(row['arm_id'], row['budget_k']): row for row in raw}
    _validate_csv(case['csv'], expected_rows)
    _validate_report_comparison(report.get('comparison'), expected_rows)
    case.update(raw=raw, by_arm=by_arm, full=full, seeds=seeds, q=q)
    return case


def _wins(differences):
    return dict(wins=sum(value > 1e-10 for value in differences),
                ties=sum(abs(value) <= 1e-10 for value in differences),
                losses=sum(value < -1e-10 for value in differences))


def _condition_rows(case):
    rows = []
    n = len(case['arrays']['targets'])
    for k in KS:
        arms, reference = case['by_arm'], case['full'][k]['covered_count']
        library = arms['library'][k]['covered_count']
        row = dict(case_id=case['root'].name, track=case['report']['plan']['track'], budget_k=k,
                   query_count=case['q'], target_count=n, reference_covered_count=reference,
                   reference_coverage_pct=100. * reference / n, reference_missing_count=n - reference,
                   library_missing_count=n - library, library_full_shortfall_count=reference - library)
        for name in DETERMINISTIC:
            count = arms[name][k]['covered_count']
            row[f'{name}_covered_count'] = count
            row[f'{name}_coverage_pct'] = 100. * count / n
            if name != 'library':
                row[f'library_minus_{name}_count'] = library - count
                row[f'library_minus_{name}_pp'] = 100. * (library - count) / n
        random = np.asarray([arms[f'random_{seed}'][k]['covered_count'] for seed in case['seeds']], dtype=float)
        row['random_seed_count'] = len(random)
        row['random_full_reference_match_count'] = int(np.count_nonzero(random == reference))
        for suffix, values in [('count', random), ('pct', 100. * random / n)]:
            row[f'random_mean_{suffix}'] = float(np.mean(values))
            row[f'random_sample_sd_{suffix}'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            for name, value in zip(('p10', 'median', 'p90'), np.quantile(values, [.1, .5, .9], method='linear')):
                row[f'random_{name}_{suffix}'] = float(value)
        row['library_minus_random_mean_pp'] = row['library_coverage_pct'] - row['random_mean_pct']
        row.update({f'library_vs_random_{key}': value for key, value in _wins(library - random).items()})
        rows.append(row)
    return rows


def _reproduction(cases, previous_root):
    if previous_root is None:
        return dict(status='not_provided', scope='No previous-run comparison requested')
    previous_root = Path(previous_root).expanduser().resolve(strict=True)
    checks = []
    for case in cases:
        old = _load(previous_root / case['root'].name)
        _require(old['report'].get('schema') == 'library_target_evaluation_v1'
                 and old['report'].get('validation', {}).get('passed') is True,
                 'Previous report is not a validated old-format evaluation')
        for name in ('poses', 'targets', 'masks'):
            _require(np.array_equal(old['arrays'][name], case['arrays'][name]),
                     f'Reproduction: {case["root"].name} {name} mismatch')
        old_pool = _frontier(old['report']['pool_frontier'], old['arrays']['pool_ids'].tolist(),
                             old['arrays'], 'previous library')
        old_full = _frontier(old['report']['full_frontier'], list(range(len(old['arrays']['poses']))),
                             old['arrays'], 'previous full')
        for label, before, after in [('library', old_pool, case['by_arm']['library']),
                                     ('full', old_full, case['full'])]:
            for k in KS:
                for field in ('full_candidate_ids', 'covered_count', 'num_selected'):
                    _require(before[k][field] == after[k][field],
                             f'Reproduction: {label} K={k} {field} mismatch')
        checks.append(dict(case_id=case['root'].name, status='passed', arrays=['poses', 'targets', 'masks'],
                           incumbent_fields=['full_candidate_ids', 'covered_count', 'num_selected'],
                           provenance=old['provenance']))
    return dict(status='passed', scope='Exact saved arrays and library/full incumbents after decisions; '
                'not a fresh physics rerun', cases=checks)


def _write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _full_reference_matches(raw):
    def counts(rows):
        return dict(match_count=sum(row['covered_count'] == row['reference_covered_count'] for row in rows),
                    row_count=len(rows))

    result = {}
    for method in (*DETERMINISTIC, 'random'):
        rows = [row for row in raw if row['method'] == method]
        result[method] = dict(**counts(rows), unit='case/K/seed' if method == 'random' else 'case/K',
                             by_k={str(k): counts([row for row in rows if row['budget_k'] == k]) for k in KS})
        if method == 'random':
            result[method]['by_seed'] = {
                str(seed): counts([row for row in rows if row['seed'] == seed])
                for seed in sorted({row['seed'] for row in rows})}
    return result


def aggregate_reports(roots, output, previous_root=None):
    """Validate all inputs, then emit a new report folder; return its absolute Path.

    Invalid/incomplete data or reproduction mismatches raise ValueError. Existing
    output raises FileExistsError. One random seed is allowed (sample SD is null).
    Cases must declare identical seed sets. Previous cases are matched by
    directory basename; seeds are never map units. Frozen hash maps are required.
    """
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output must be new: {output}')
    roots = [Path(root).expanduser().resolve(strict=True) for root in roots]
    _require(roots and len(set(roots)) == len(roots), 'Expected distinct case roots')
    _require(len({root.name for root in roots}) == len(roots), 'Duplicate case names')
    protected = roots + ([Path(previous_root).expanduser().resolve(strict=True)] if previous_root is not None else [])
    _require(all(root != output.resolve() and root not in output.resolve().parents for root in protected),
             'Output cannot be inside a raw input root')
    cases = [_validate_case(_load(root)) for root in sorted(roots)]
    seeds = sorted(cases[0]['seeds'])
    _require(all(sorted(case['seeds']) == seeds for case in cases), 'Cases must use the same control seeds')
    conditions = [row for case in cases for row in _condition_rows(case)]
    raw = [dict(case_id=case['root'].name, track=case['report']['plan']['track'], **row)
           for case in cases for row in sorted(case['raw'], key=lambda row: (row['arm_id'], row['budget_k']))]
    by_k = {}
    mean_fields = [f'{name}_coverage_pct' for name in DETERMINISTIC] + ['random_mean_pct', 'reference_coverage_pct']
    difference_fields = [f'library_minus_{name}_pp' for name in DETERMINISTIC[1:]] + ['library_minus_random_mean_pp']
    for k in KS:
        selected = [row for row in conditions if row['budget_k'] == k]
        by_k[str(k)] = dict(case_count=len(selected), weighting='equal weight per case/map',
                           **{field: float(np.mean([row[field] for row in selected]))
                              for field in mean_fields + difference_fields})
    utility = {name: _wins([row[f'library_minus_{name}_pp'] for row in conditions])
               for name in DETERMINISTIC[1:]}
    utility['random_mean'] = _wins([row['library_minus_random_mean_pp'] for row in conditions])
    reference_matches = _full_reference_matches(raw)
    provenance = [case['provenance'] for case in cases]
    source_hash = _digest(Path(__file__))
    provenance_hash = sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    command = [sys.executable, str(Path(__file__).resolve()), '--roots', *map(str, roots), '--output', str(output)]
    if previous_root is not None:
        command += ['--previous-root', str(Path(previous_root).expanduser().resolve())]
    # Prior results are loaded only after all new-result checks and aggregates.
    reproduction = _reproduction(cases, previous_root)
    summary = dict(schema='library_matched_query_report_v1', case_count=len(cases),
                   condition_count=len(conditions), expected_development_case_count=4,
                   expected_development_condition_count=12,
                   complete_four_case_design=({root.name for root in roots} == DEVELOPMENT_CASES
                                              and seeds == DEVELOPMENT_SEEDS),
                   k_values=KS, control_seeds=seeds,
                   by_k=by_k, library_utility=utility, full_reference_matches=reference_matches,
                   reproduction=reproduction,
                   validation=dict(passed=True, saved_array_checks='passed', fresh_physics_performed=False,
                                   parent_validation_record_passed=True, frozen_hash_checks='passed'),
                   material_passport=dict(OriginSkill='academic-research-suite', Agent='experiment-agent',
                                          OriginMode='run', VerificationStatus='ANALYZED: saved-array checks passed; '
                                          'parent physics record only, no fresh physics'),
                   random_summary=dict(unit='seed conditional on case', sample_sd_ddof=1,
                                       quantile_method='numpy linear', single_seed_sd=None),
                   source_sha256=source_hash, provenance_sha256=provenance_hash,
                   sources=provenance, scope_caveats=CAVEATS, command=command,
                   created_at=datetime.now(timezone.utc).isoformat())
    rendered = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    lines = ['## Material Passport', '', '- OriginSkill: academic-research-suite',
             '- Agent: experiment-agent; OriginMode: run',
             '- VerificationStatus: ANALYZED. Stored-array checks passed. Parent validation records passed;',
             '  no fresh raycast/physics performed by this aggregator.', '', '## Descriptive Results', '',
             f'{len(cases)} cases; {len(conditions)} case/K conditions (four-case design: 12).',
             'Library wins / ties / losses across conditions:']
    for name, result in utility.items():
        lines.append(f'- {name}: {result["wins"]} / {result["ties"]} / {result["losses"]}')
    lines += ['', 'Full-reference match counts (random rows are conditional case/K/seed observations):']
    for name, result in reference_matches.items():
        lines.append(f'- {name}: {result["match_count"]} / {result["row_count"]}')
    lines += ['', 'Per-K aggregates weight maps equally. Random sample SD uses ddof=1 (undefined for one seed);',
              'p10/median/p90 use NumPy linear quantiles. CSV differences are library minus comparator;',
              'missing counts are target count minus coverage; full shortfall is reference minus library.',
              '', '## Scope', '', *[f'- {caveat}' for caveat in CAVEATS],
              f'- Previous-run reproduction: {reproduction["status"]}; compared only after new decisions.',
              '', '## Reproduce', '', '```sh', shlex.join(command), '```', '',
              'Use a new --output path for every invocation. Raw artifacts are never modified.',
              'Outputs: condition_summary.csv, raw_comparison.csv, summary.json, REPORT.md.', '']
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / 'condition_summary.csv', conditions)
    _write_csv(output / 'raw_comparison.csv', raw)
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        stream.write(rendered)
    with (output / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    return output.resolve()


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True, help='Completed per-case run directories')
    parser.add_argument('--output', required=True, help='New report directory')
    parser.add_argument('--previous-root', help='Optional old run collection, e.g. results/library_ab_transfer_20261118')
    args = parser.parse_args(argv)
    output = aggregate_reports(args.roots, args.output, args.previous_root)
    print(output)
    return output


if __name__ == '__main__':
    main()

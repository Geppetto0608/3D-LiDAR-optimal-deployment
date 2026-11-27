"""Development-only, one-factor coverage-constraint sensitivity study."""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import mean, stdev
import subprocess
import sys
import time

import numpy as np

from prototype_constrained import freeze_inputs, make_plan as constrained_plan, parser as constrained_parser
from prototype_offline import digest, save
from prototype_refinement_study import pose_hash
from src.core.coverage_requirements import CoverageRequirements


METHODS = ('local_search', 'mappo')
SUCCESS = ('already_feasible', 'repaired')


def now():
    return datetime.now(timezone.utc).isoformat()


def materialize_cases(config, base):
    if (not isinstance(config, dict)
            or set(config) != {'schema', 'split', 'base_requirements', 'target_id', 'initialization', 'cases'}
            or config['schema'] != 'constrained_sensitivity_v1' or config['split'] != 'development'):
        raise ValueError('Expected constrained_sensitivity_v1 development specification')
    CoverageRequirements(base)
    if not isinstance(config['cases'], list) or not config['cases']:
        raise ValueError('Nonempty cases required')
    names, cases = set(), []
    for case in config['cases']:
        if not isinstance(case, dict) or set(case) - {'id', 'axis', 'value', 'region', 'center'}:
            raise ValueError('Invalid case keys')
        name, axis, value = case.get('id'), case.get('axis'), case.get('value')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name) or name in names:
            raise ValueError('Case IDs must be unique safe identifiers')
        names.add(name)
        spec = deepcopy(base)
        if axis == 'baseline':
            if value is not None or set(case) != {'id', 'axis', 'value'}:
                raise ValueError('Baseline must have null value and no extra fields')
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Case value must be finite and nonnegative')
            if axis in ('cost_budget', 'min_separation') and set(case) == {'id', 'axis', 'value'}:
                key = 'max_installation_cost' if axis == 'cost_budget' else 'min_xy_separation_m'
                spec.setdefault('constraints', {})[key] = value
            elif axis == 'region_overlap' and set(case) == {'id', 'axis', 'value', 'region'}:
                regions = [r for r in spec.get('regions', []) if r['name'] == case['region']]
                if len(regions) != 1:
                    raise ValueError('Named region absent or ambiguous')
                regions[0]['min_overlap_ratio'] = value
            elif axis == 'forbidden_half_extent' and set(case) == {'id', 'axis', 'value', 'center'}:
                center = case['center']
                if (value <= 0 or not isinstance(center, list) or len(center) != 2
                        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in center)):
                    raise ValueError('Positive half extent and finite XY center required')
                x, y = center
                spec.setdefault('constraints', {}).setdefault('forbidden_regions', []).append(
                    [x - value, y - value, x + value, y + value])
            else:
                raise ValueError('Unsupported axis or extra case fields')
        CoverageRequirements(spec)
        cases.append(dict(case, requirements=spec))
    if sum(c['axis'] == 'baseline' for c in cases) != 1:
        raise ValueError('Exactly one baseline is required')
    return cases


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--study', default='configs/constrained_sensitivity.json')
    p.add_argument('--suite', default='configs/library_expansion.json')
    p.add_argument('--library', default='results/prototype_library_shapes_20261008')
    p.add_argument('--cases', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--evaluations', type=int, default=128)
    p.add_argument('--seconds', type=float, default=6.)
    p.add_argument('--repair-evaluations', type=int, default=128)
    p.add_argument('--repair-seconds', type=float, default=6.)
    p.add_argument('--radius-m', type=float, default=4.)
    p.add_argument('--quality-times', nargs='+', type=float, default=[2., 4., 6.])
    p.add_argument('--output', default='results/constrained_sensitivity_pilot_20261020')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--report-only', action='store_true')
    return p


def make_plan(args):
    config = json.loads(Path(args.study).read_text())
    if not isinstance(config, dict) or not isinstance(config.get('base_requirements'), str):
        raise ValueError('Study must reference a base requirements JSON file')
    base_path = (Path(args.study).resolve().parent / config['base_requirements']).resolve()
    cases = materialize_cases(config, json.loads(base_path.read_text()))
    if args.cases is not None:
        if len(set(args.cases)) != len(args.cases) or not set(args.cases).issubset(c['id'] for c in cases):
            raise ValueError('Unknown or duplicate cases')
        cases = [c for c in cases if c['id'] in args.cases]
    if not args.seeds or args.seeds != sorted(set(args.seeds)) or any(s < 0 or s >= 2 ** 32 for s in args.seeds):
        raise ValueError('Use ascending unique seeds in [0, 2**32)')
    if (args.quality_times != sorted(set(args.quality_times))
            or any(not math.isfinite(t) or t <= 0 for t in args.quality_times)):
        raise ValueError('Use ascending positive finite quality times')
    child = constrained_parser().parse_args(['--library', args.library, '--suite', args.suite,
        '--target-id', str(config['target_id']), '--initialization', config['initialization'],
        '--requirements', str(base_path), '--steps', str(args.steps), '--evaluations', str(args.evaluations),
        '--seconds', str(args.seconds), '--radius-m', str(args.radius_m), '--seed', str(args.seeds[0]),
        '--repair-initial', '--repair-evaluations', str(args.repair_evaluations),
        '--repair-seconds', str(args.repair_seconds)])
    baseline = constrained_plan(child)
    jobs = [dict(id=f'{c["id"]}_seed{s}', case_id=c['id'], seed=s,
                 directory=f'runs/{c["id"]}_seed{s}', status='pending') for c in cases for s in args.seeds]
    if len(jobs) > 100:
        raise ValueError('This pilot runner supports at most 100 paired repeats')
    return dict(schema='constrained_sensitivity_run_v1', split='development', status='planned',
        cases=cases, seeds=args.seeds, quality_times=args.quality_times,
        target_id=config['target_id'], initialization=config['initialization'], jobs=jobs,
        limits=baseline['limits'], initial_repair=baseline['initial_repair'], baseline_plan=baseline,
        suite_path=str(Path(args.suite).resolve()),
        interpretation='One development geometry; repeated timings and MAPPO seeds, not independent maps; no detection')


def trace_at(trace, deadline):
    previous_time, previous_score, result = -1., -math.inf, None
    for row in trace:
        elapsed, score, coverage = (float(row[k]) for k in ('elapsed_sec', 'objective_score', 'coverage_pct'))
        if (not all(math.isfinite(x) for x in (elapsed, score, coverage)) or elapsed < 0 or elapsed < previous_time
                or score < previous_score - 1e-9 or not 0 <= coverage <= 100):
            raise ValueError('Invalid monotone best-objective trace')
        previous_time, previous_score = elapsed, score
        if elapsed <= deadline:
            result = row
    return result


def _statistics(values):
    values = [float(x) for x in values if x is not None]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite summary input')
    return (mean(values) if values else None, stdev(values) if len(values) > 1 else None)


def summarize(repairs, rows, case_ids, seeds):
    repair_summary, method_summary, paired = [], [], []
    if len({(r['case_id'], r['seed']) for r in repairs}) != len(repairs):
        raise ValueError('Duplicate repair repeat')
    identities = {(r['case_id'], r['seed'], r['method']): r for r in rows}
    if len(identities) != len(rows):
        raise ValueError('Duplicate method repeat')
    for case in case_ids:
        attempts = [r for r in repairs if r['case_id'] == case]
        success = [r for r in attempts if r['repair_status'] in SUCCESS]
        counts = Counter(r['repair_status'] for r in attempts)
        repair_summary.append(dict(case_id=case, requested_repeats=len(seeds), processed_repeats=len(attempts),
            repair_successes=len(success), repair_success_rate=len(success) / len(attempts) if attempts else None,
            already_feasible=counts['already_feasible'], repaired=counts['repaired'],
            no_feasible_found=counts['no_feasible_found'], infeasible_fixed_count=counts['infeasible_fixed_count'],
            distinct_repair_starts=len({r['feasible_pose_sha256'] for r in success}),
            repair_worker_wall_mean=_statistics([r['worker_wall_sec'] for r in attempts])[0]))
        for method in METHODS:
            requested = [r for r in rows if r['case_id'] == case and r['method'] == method]
            complete = [r for r in requested if r['status'] == 'complete']
            entry = dict(case_id=case, method=method, requested_repeats=len(seeds),
                completed=len(complete), not_run=sum(r['status'] == 'not_run' for r in requested))
            metrics = ['objective_score', 'coverage_pct', 'overlap_ratio', 'installation_cost',
                       'optimization_sec', 'worker_wall_sec']
            metrics += sorted({k for r in rows for k in r if k.startswith('region_')})
            for metric in metrics:
                entry[f'{metric}_mean'], entry[f'{metric}_sd'] = _statistics([r.get(metric) for r in complete])
            method_summary.append(entry)
        for seed in seeds:
            a, b = [identities.get((case, seed, method)) for method in METHODS]
            if a is None or b is None or a['status'] != 'complete' or b['status'] != 'complete':
                continue
            if a['initial_pose_sha256'] != b['initial_pose_sha256']:
                raise ValueError('Paired initial deployment mismatch')
            paired.append(dict(case_id=case, seed=seed, initial_pose_sha256=a['initial_pose_sha256'],
                mappo_minus_ls_objective=b['objective_score'] - a['objective_score'],
                mappo_minus_ls_coverage_pp=b['coverage_pct'] - a['coverage_pct'],
                mappo_minus_ls_optimization_sec=b['optimization_sec'] - a['optimization_sec']))
    return dict(repair_summary=repair_summary, method_summary=method_summary, paired=paired)


def summarize_quality(rows, case_ids, seeds, deadlines):
    keys = [(r['case_id'], r['seed'], r['method'], r['refinement_deadline_sec']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate quality-time observation')
    for row in rows:
        if (row['case_id'] not in case_ids or row['seed'] not in seeds or row['method'] not in METHODS
                or row['refinement_deadline_sec'] not in deadlines
                or (row['objective_score'] is None) != (row['coverage_pct'] is None)):
            raise ValueError('Invalid quality-time identity or missing value')
    result = []
    for case in case_ids:
        for method in METHODS:
            for deadline in deadlines:
                complete = [r for r in rows if r['case_id'] == case and r['method'] == method
                            and r['refinement_deadline_sec'] == deadline]
                observed = [r for r in complete if r['objective_score'] is not None]
                entry = dict(case_id=case, method=method, refinement_deadline_sec=deadline,
                    requested_repeats=len(seeds), completed_repeats=len(complete), observed_by_deadline=len(observed))
                for metric in ('objective_score', 'coverage_pct'):
                    entry[f'{metric}_mean'], entry[f'{metric}_sd'] = _statistics([r[metric] for r in observed])
                result.append(entry)
    return result


def quality_pairs(rows):
    index = {(r['case_id'], r['seed'], r['method'], r['refinement_deadline_sec']): r for r in rows}
    result = []
    for row in rows:
        if row['method'] != 'local_search' or row['objective_score'] is None:
            continue
        other = index.get((row['case_id'], row['seed'], 'mappo', row['refinement_deadline_sec']))
        if other is not None and other['objective_score'] is not None:
            result.append(dict(case_id=row['case_id'], seed=row['seed'],
                refinement_deadline_sec=row['refinement_deadline_sec'],
                mappo_minus_ls_objective=other['objective_score'] - row['objective_score'],
                mappo_minus_ls_coverage_pp=other['coverage_pct'] - row['coverage_pct']))
    return result


def _path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes run directory')
    return path


def collect(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['frozen_hashes'].items():
        if digest(_path(root, name)) != expected:
            raise ValueError(f'Frozen study input changed: {name}')
    frozen = json.loads((root / 'frozen/study_plan.json').read_text())
    for key in ('cases', 'seeds', 'limits', 'initial_repair', 'quality_times', 'target_id', 'initialization'):
        if manifest[key] != frozen[key]:
            raise ValueError(f'Study plan changed: {key}')
    expected_jobs = [(c['id'], s) for c in frozen['cases'] for s in frozen['seeds']]
    if [(j['case_id'], j['seed']) for j in manifest['jobs']] != expected_jobs:
        raise ValueError('Job completeness/order mismatch')
    hashes, repairs, rows, quality, source_hashes, targets_hashes, package_hashes = {}, [], [], [], set(), set(), set()
    specs = {c['id']: c['requirements'] for c in frozen['cases']}

    def read(path):
        hashes[str(path.relative_to(root))] = digest(path)
        return json.loads(path.read_text())

    def poses(path):
        hashes[str(path.relative_to(root))] = digest(path)
        return np.load(path, allow_pickle=False)

    def check_assessment(directory, phase, expected, evaluator, anchors, feasible=None):
        selected = poses(directory / f'{phase}.npy')
        path = directory / f'{phase}_coverage.npz'
        hashes[str(path.relative_to(root))] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            targets, density = data['targets'], data['density']
        targets_hashes.add(hashlib.sha256(np.asarray(targets, dtype='<f8').tobytes()).hexdigest())
        if not np.array_equal(targets, evaluator.targets):
            raise ValueError('Target ordering mismatch')
        actual = evaluator.evaluate(selected, density)
        if (actual.components != expected['components'] or actual.feasible != expected['feasible']
                or list(actual.violations) != expected['violations']
                or actual.regions != expected['regions']
                or any(not math.isclose(getattr(actual, key), expected[key], abs_tol=1e-6, rel_tol=0)
                       for key in ('objective_score', 'coverage_pct', 'overlap_ratio'))):
            raise ValueError('Saved density/assessment mismatch')
        if feasible is not None and actual.feasible != feasible:
            raise ValueError('Unexpected feasibility status')
        if selected.shape != anchors.shape or np.any(np.linalg.norm(selected[:, :2].astype(float) - anchors[:, :2], axis=1)
                                                     > frozen['limits']['radius_m'] + 1e-5):
            raise ValueError('Saved deployment changed K or escaped original anchors')
        return selected

    for job in manifest['jobs']:
        if job['status'] not in ('complete', 'no_feasible_initial'):
            continue
        directory = _path(root, job['directory'])
        child = read(directory / 'manifest.json')
        packages = directory / 'packages.txt'
        hashes[str(packages.relative_to(root))] = digest(packages)
        package_hashes.add(digest(packages))
        if child['status'] != job['status']:
            raise ValueError('Child status mismatch')
        for name, expected in child['frozen_hashes'].items():
            if digest(_path(directory, name)) != expected:
                raise ValueError(f'Child frozen input changed: {name}')
        spec = read(directory / 'requirements.json')
        if spec != specs[job['case_id']]:
            raise ValueError('Requirements differ from the one-factor case')
        if child['job']['scenario'] != frozen['baseline_plan']['job']['scenario']:
            raise ValueError('Child target geometry changed')
        settings = read(directory / 'settings.json')
        if (settings['limits'] != dict(frozen['limits'], seed=job['seed'])
                or settings['initial_repair'] != frozen['initial_repair'] or tuple(settings['methods']) != METHODS):
            raise ValueError('Child runtime settings changed')
        original = read(root / 'frozen/base_plan.json')['source_prototypes']
        if [s['hashes'] for s in child['source_prototypes']] != [s['hashes'] for s in original]:
            raise ValueError('Source prototype changed between repeats')
        repair_dir = directory / 'repair'
        repair = read(repair_dir / 'result.json')
        if (not repair['validated_direct_raycast'] or repair['full_visibility_cache']
                or repair['requirements_sha256'] != digest(directory / 'requirements.json')):
            raise ValueError('Missing direct validation or exhaustive target cache used')
        anchors = poses(repair_dir / 'initial.npy')
        if pose_hash(anchors) != repair['transferred_pose_sha256']:
            raise ValueError('Transferred pose hash mismatch')
        source_hashes.add(pose_hash(anchors))
        with np.load(repair_dir / 'initial_coverage.npz', allow_pickle=False) as data:
            evaluator = CoverageRequirements(spec).bind(data['targets'])
        check_assessment(repair_dir, 'initial', repair['assessments']['initial'], evaluator, anchors)
        best = check_assessment(repair_dir, 'best_candidate', repair['assessments']['best_candidate'], evaluator, anchors)
        success = repair['status'] in SUCCESS
        if success != (child['status'] == 'complete') or success != repair['assessments']['best_candidate']['feasible']:
            raise ValueError('Repair/child success mismatch')
        if success:
            np.testing.assert_array_equal(poses(repair_dir / 'feasible.npy'), best)
            if pose_hash(best) != repair['feasible_pose_sha256']:
                raise ValueError('Shared feasible pose hash mismatch')
        elif (repair_dir / 'feasible.npy').exists() or repair['status'] not in ('no_feasible_found', 'infeasible_fixed_count'):
            raise ValueError('Failed repair published a feasible artifact or unknown status')
        prefix = child['initial_repair']['worker_wall_sec']
        repairs.append(dict(case_id=job['case_id'], seed=job['seed'], repair_status=repair['status'],
            worker_wall_sec=prefix, feasible_pose_sha256=repair['feasible_pose_sha256'],
            neighbor_evaluations=repair['neighbor_evaluations'],
            repair_search_sec=repair['repair_search_sec'],
            rays_submitted=repair['runtime']['counts'].get('rays_submitted', 0),
            initial_coverage_pct=repair['assessments']['initial']['coverage_pct'],
            repaired_coverage_pct=repair['assessments']['best_candidate']['coverage_pct'] if success else None))
        for method in METHODS:
            entry = dict(case_id=job['case_id'], seed=job['seed'], method=method,
                status='complete' if success else 'not_run', repair_status=repair['status'],
                initial_pose_sha256=repair['feasible_pose_sha256'], repair_worker_wall_sec=prefix,
                objective_score=None, coverage_pct=None, overlap_ratio=None, installation_cost=None,
                optimization_sec=None, worker_wall_sec=None, soft_limit_overrun_sec=None,
                training_steps=None, ppo_updates=None, rays_submitted=None)
            for region in spec.get('regions', []):
                for metric in ('coverage_pct', 'overlap_ratio'):
                    entry[f'region_{region["name"]}_{metric}'] = None
            if not success:
                if child['jobs'][method]['status'] != 'not_run' or (directory / method).exists():
                    raise ValueError('Optimizer ran after unsuccessful common repair')
                rows.append(entry)
                continue
            out = directory / method
            result = read(out / 'result.json')
            if (result['status'] != 'complete' or not result['shared_initial_repair'] or result['full_visibility_cache']
                    or result['initial_pose_sha256'] != repair['feasible_pose_sha256']
                    or result['method'] != method or result['seed'] != job['seed']
                    or child['jobs'][method]['status'] != 'complete'
                    or result['requirements_sha256'] != digest(directory / 'requirements.json')):
                raise ValueError('Unpaired or unvalidated optimizer result')
            np.testing.assert_array_equal(poses(out / 'anchors.npy'), anchors)
            for phase, assessment in result['assessments'].items():
                selected = check_assessment(out, phase, assessment, evaluator, anchors, feasible=True)
                if phase == 'initial':
                    np.testing.assert_array_equal(selected, best)
            assessment = result['assessments']['deployment']
            trace = result['objective_trace']
            if (not trace or not math.isclose(trace[0]['objective_score'],
                    result['assessments']['initial']['objective_score'], abs_tol=1e-6, rel_tol=0)
                    or not math.isclose(trace[-1]['objective_score'], assessment['objective_score'], abs_tol=1e-6, rel_tol=0)
                    or trace[-1]['elapsed_sec'] > result['optimization_sec'] + 1e-6):
                raise ValueError('Quality trace does not match the reported deployment or clock')
            entry.update(objective_score=assessment['objective_score'], coverage_pct=assessment['coverage_pct'],
                overlap_ratio=assessment['overlap_ratio'], installation_cost=assessment['components']['installation_cost'],
                optimization_sec=result['optimization_sec'], worker_wall_sec=child['jobs'][method]['worker_wall_sec'],
                soft_limit_overrun_sec=max(result['optimization_sec'] - frozen['limits']['time_limit_sec'], 0.),
                training_steps=result['training_steps'], ppo_updates=result['ppo_updates'],
                rays_submitted=result['runtime']['counts'].get('rays_submitted', 0))
            for name, values in assessment['regions'].items():
                for metric in ('coverage_pct', 'overlap_ratio'):
                    entry[f'region_{name}_{metric}'] = values[metric]
            rows.append(entry)
            for deadline in frozen['quality_times']:
                point = trace_at(trace, deadline)
                quality.append(dict(case_id=job['case_id'], seed=job['seed'], method=method,
                    refinement_deadline_sec=deadline, objective_score=point['objective_score'] if point else None,
                    coverage_pct=point['coverage_pct'] if point else None,
                    last_observation_sec=point['elapsed_sec'] if point else None))
    if len(source_hashes) > 1 or len(targets_hashes) > 1 or len(package_hashes) > 1:
        raise ValueError('Repeats used different transferred poses, targets or package versions')
    return manifest, repairs, rows, quality, hashes


def _csv(path, rows, fallback):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fallback)
        writer.writeheader()
        writer.writerows(rows)


def write_report(root, *, plot=True):
    manifest, repairs, rows, quality, hashes = collect(root)
    ids = [c['id'] for c in manifest['cases']]
    result = summarize(repairs, rows, ids, manifest['seeds'])
    result['quality_summary'] = summarize_quality(quality, ids, manifest['seeds'], manifest['quality_times'])
    result['quality_pairs'] = quality_pairs(quality)
    for filename, values, fallback in [('raw_methods.csv', rows, ['case_id', 'seed', 'method', 'status']),
        ('raw_repairs.csv', repairs, ['case_id', 'seed', 'repair_status']),
        ('quality_time.csv', quality, ['case_id', 'seed', 'method', 'refinement_deadline_sec', 'objective_score']),
        ('repair_summary.csv', result['repair_summary'], ['case_id']),
        ('summary.csv', result['method_summary'], ['case_id', 'method']),
        ('quality_summary.csv', result['quality_summary'], ['case_id', 'method', 'refinement_deadline_sec']),
        ('paired_quality_time.csv', result['quality_pairs'], ['case_id', 'seed', 'refinement_deadline_sec']),
        ('paired.csv', result['paired'], ['case_id', 'seed', 'mappo_minus_ls_objective'])]:
        _csv(root / filename, values, fallback)
    report_source = root / 'report_source'
    report_source.mkdir(exist_ok=True)
    if Path(__file__).resolve() != (report_source / 'constrained_sensitivity.py').resolve():
        shutil.copy2(Path(__file__), report_source / 'constrained_sensitivity.py')
    save(root / 'validation_summary.json', dict(status=manifest['status'], processed_repeats=len(repairs),
        requested_repeats=len(manifest['jobs']), completed_method_runs=sum(r['status'] == 'complete' for r in rows),
        blocked_method_runs=sum(r['status'] == 'not_run' for r in rows), raw_sha256=hashes,
        report_source_sha256=digest(report_source / 'constrained_sensitivity.py'), sample_sd_ddof=1,
        verification_scope='Saved pose/density/requirements/anchors/hashes; no new raycasts. Workers ran direct validation.',
        timing_scope='Final best uses actual finish; quality-time uses recorded refinement timestamps, not process end-to-end.',
        interpretation='One development geometry; conditional success-only quality; blocked rows have no numerical scores'))
    lines = ['# Constrained Sensitivity Pilot', '',
        'One development map, synthetic requirements, fixed source transfer. No detection or optimality claim.',
        f'Status: {manifest["status"]}; processed paired repeats: {len(repairs)}/{len(manifest["jobs"])}.', '',
        '| Case | Successful Repair / Processed | Already Feasible | Repaired | Not Found | Necessary-Condition Failure |',
        '|---|---:|---:|---:|---:|---:|']
    for row in result['repair_summary']:
        lines.append(f'| {row["case_id"]} | {row["repair_successes"]}/{row["processed_repeats"]} | '
            f'{row["already_feasible"]} | {row["repaired"]} | {row["no_feasible_found"]} | {row["infeasible_fixed_count"]} |')
    lines += ['', '## Conditional Quality', '',
        'Mean +/- sample SD over completed repeats only. Missing or blocked runs are not zero scores.', '',
        '| Case | Method | Completed | Coverage % | Objective J | Refinement Seconds |', '|---|---|---:|---:|---:|---:|']
    def formatted(row, key):
        avg, sd = row[key + '_mean'], row[key + '_sd']
        return 'n/a' if avg is None else f'{avg:.3f}' + (f' +/- {sd:.3f}' if sd is not None else '')
    for row in result['method_summary']:
        lines.append(f'| {row["case_id"]} | {row["method"]} | {row["completed"]} | '
            f'{formatted(row, "coverage_pct")} | {formatted(row, "objective_score")} | {formatted(row, "optimization_sec")} |')
    deadline = manifest['quality_times'][-1]
    lines += ['', f'## Recorded Quality by {deadline:g} Refinement Seconds', '',
        'n shows observed-by-deadline / completed / requested repeats. J-best deployment coverage is reported, not maximum coverage.',
        'Means with different observed counts need not describe the same paired subset; paired_quality_time.csv uses only pairs with both observations.', '',
        '| Case | Method | n | Objective J | Coverage % |', '|---|---|---:|---:|---:|']
    for row in result['quality_summary']:
        if row['refinement_deadline_sec'] == deadline:
            n = f'{row["observed_by_deadline"]}/{row["completed_repeats"]}/{row["requested_repeats"]}'
            lines.append(f'| {row["case_id"]} | {row["method"]} | {n} | '
                f'{formatted(row, "objective_score")} | {formatted(row, "coverage_pct")} |')
    lines += ['', '## Interpretation', '',
        '- Seeds vary MAPPO learning; LS repeats measure timing. These are not independent maps.',
        '- Each paired repeat has one common repair. Prefix worker cost is recorded, not hidden in refinement time.',
        '- Soft limits can overrun during final PPO/evaluation. quality_time.csv excludes improvements recorded after each deadline.',
        '- Quality-time clocks start at refinement initialization, excluding transfer, repair, imports and direct validation.',
        '- A missing early observation is blank, not zero. Initial/final arrays were directly validated by workers; trace points were not all recast.',
        '- no_feasible_found is bounded-search failure, not an infeasibility proof.',
        '- This short sequential shared-server pilot cannot establish convergence, generalization or algorithm superiority.', '',
        '[Raw Methods](raw_methods.csv) | [Raw Repairs](raw_repairs.csv) | [Paired Differences](paired.csv) | '
        '[Quality-Time](quality_time.csv) | [Quality Summary](quality_summary.csv) | '
        '[Paired Quality-Time](paired_quality_time.csv) | [Validation](validation_summary.json)', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    if plot:
        _plot(root, result, ids)
    return result


def _plot(root, result, ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    x = np.arange(len(ids))
    axes[0, 0].bar(x, [r['repair_success_rate'] if r['repair_success_rate'] is not None else np.nan
                      for r in result['repair_summary']], color='#687e41')
    axes[0, 0].set(title='Repair Success / Processed Repeats', ylim=(0, 1.05))
    for ax, metric, title in [(axes[0, 1], 'coverage_pct', 'Final Coverage (%)'),
                              (axes[1, 0], 'objective_score', 'Final Objective J'),
                              (axes[1, 1], 'optimization_sec', 'Actual Refinement Seconds')]:
        for method, color in zip(METHODS, ['#007f79', '#b94c4c']):
            values = [r for r in result['method_summary'] if r['method'] == method]
            ys = [v[metric + '_mean'] if v[metric + '_mean'] is not None else np.nan for v in values]
            errors = [v[metric + '_sd'] or 0. for v in values]
            ax.errorbar(x, ys, yerr=errors, marker='o', capsize=3, label=method, color=color)
        ax.set_title(title)
        ax.legend(fontsize=8, loc='lower left' if metric == 'coverage_pct' else 'best')
    for ax in axes.flat:
        ax.set_xticks(x, ids, rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Development Pilot: Conditional Quality, Not Independent-Map Evidence')
    fig.savefig(root / 'sensitivity.png', dpi=150)
    plt.close(fig)


def _freeze(root, plan):
    frozen = root / 'frozen'
    frozen.mkdir()
    baseline = deepcopy(plan['baseline_plan'])
    freeze_inputs(frozen, baseline)
    save(frozen / 'base_plan.json', baseline)
    shutil.copy2(plan['suite_path'], frozen / 'suite.json')
    library = frozen / 'library/library'
    library.mkdir(parents=True)
    for original in baseline['source_prototypes']:
        name = Path(original['root']).name
        shutil.copytree(frozen / 'inputs' / name, library / name)
    (frozen / 'variants').mkdir()
    for case in plan['cases']:
        save(frozen / 'variants' / f'{case["id"]}.json', case['requirements'])
    save(frozen / 'study_plan.json', {k: plan[k] for k in ('cases', 'seeds', 'limits', 'initial_repair',
        'quality_times', 'target_id', 'initialization', 'baseline_plan')})
    plan['frozen_hashes'] = {str(p.relative_to(root)): digest(p) for p in frozen.rglob('*') if p.is_file()}


def run(args):
    root = Path(args.output).resolve()
    if args.report_only:
        if args.execute:
            raise ValueError('Report-only cannot execute experiments')
        write_report(root)
        return 0
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    plan.update(started_at=now(), status='preparing', command=sys.argv)
    save(root / 'manifest.json', plan)
    try:
        _freeze(root, plan)
        (root / 'logs').mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        plan['status'] = 'running'
        for index, job in enumerate(plan['jobs']):
            save(root / 'manifest.json', plan)
            prefix = root / 'frozen'
            limits, repair = plan['limits'], plan['initial_repair']
            command = [sys.executable, '-u', str(prefix / 'source/prototype_constrained.py'),
                '--library', str(prefix / 'library'), '--suite', str(prefix / 'suite.json'),
                '--requirements', str(prefix / 'variants' / f'{job["case_id"]}.json'),
                '--target-id', str(plan['target_id']), '--initialization', plan['initialization'],
                '--steps', str(limits['steps']), '--evaluations', str(limits['max_evaluations']),
                '--seconds', str(limits['time_limit_sec']), '--radius-m', str(limits['radius_m']),
                '--seed', str(job['seed']), '--repair-initial', '--repair-evaluations', str(repair['max_evaluations']),
                '--repair-seconds', str(repair['time_limit_sec']), '--output', str(root / job['directory']), '--execute']
            print(f'[{index+1}/{len(plan["jobs"])}] {job["id"]} started', flush=True)
            started = time.perf_counter()
            with (root / 'logs' / f'{job["id"]}.log').open('w') as log:
                process = subprocess.run(command, cwd=prefix / 'source', env=env, stdout=log, stderr=subprocess.STDOUT)
            job.update(returncode=process.returncode, launcher_wall_sec=time.perf_counter() - started)
            child = json.loads((root / job['directory'] / 'manifest.json').read_text())
            job['status'] = child['status']
            if (process.returncode, child['status']) not in ((0, 'complete'), (2, 'no_feasible_initial')):
                raise RuntimeError(f'Unexpected worker failure: {job["id"]}; see logs')
            save(root / 'manifest.json', plan)
            write_report(root, plot=False)
            print(f'[{index+1}/{len(plan["jobs"])}] {job["id"]}: {job["status"]}', flush=True)
        plan.update(status='complete', finished_at=now())
        save(root / 'manifest.json', plan)
        write_report(root)
    except BaseException as error:
        plan.update(status='failed', error=f'{type(error).__name__}: {error}', finished_at=now())
        save(root / 'manifest.json', plan)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))

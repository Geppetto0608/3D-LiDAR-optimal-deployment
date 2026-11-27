"""Multi-map, literature-motivated geometric coverage study; read-only by default."""
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
import time

import numpy as np

from constrained_sensitivity import trace_at
from prototype_constrained import freeze_inputs, make_plan as child_plan, parser as child_parser
from prototype_library import load_suite, now
from prototype_offline import digest, save
from prototype_refinement_study import pose_hash
from src.core.coverage_requirements import CoverageRequirements


METHODS = ('local_search', 'mappo')
PROFILES = ('coverage', 'infrastructure', 'resilient')


def requirements_for(target, k, profile):
    """Geometry-only illustrative requirements; never read target solutions."""
    if profile not in PROFILES:
        raise ValueError('Unknown requirements profile')
    x, y = [target['geometry'][key] for key in ('junction_x_m', 'junction_y_m')]
    h = target['road_width'] / 2.
    spec = dict(schema='constrained_coverage_v2', budget_k=k,
        weights=dict(coverage_weight=1., importance_weight=.5, overlap_penalty=.05,
                     installation_cost_weight=2.),
        cost=dict(base_per_sensor=10., per_height_m=1.),
        constraints=dict(max_installation_cost=15. * k, min_xy_separation_m=3.,
            max_sensors_per_pole=1, min_coverage_pct=75., max_overlap_ratio=.9),
        regions=[dict(name='junction', bounds=[x-h-2., y-h-2., x+h+2., y+h+2.],
            importance_factor=3., min_coverage_pct=80., min_overlap_ratio=.15)])
    branches = [('west', [4., y-h, x-h-2., y+h]),
                ('east', [x+h+2., y-h, 96., y+h]),
                ('south', [x-h, 4., x+h, y-h-2.])]
    if target['topology'] == 'CROSS':
        branches.append(('north', [x-h, y+h+2., x+h, 96.]))
    spec['regions'] += [dict(name=name, bounds=bounds, min_coverage_pct=60.) for name, bounds in branches]
    if profile in ('infrastructure', 'resilient'):
        hubs = [('west', [max(4., x-25.), y+h+2.5]),
                ('east', [min(96., x+25.), y-h-2.5]),
                ('south', [x+h+2.5, max(4., y-25.)])]
        if target['topology'] == 'CROSS':
            hubs.append(('north', [x-h-2.5, min(96., y+25.)]))
        spec['constraints'].update(mounting_z_range_m=[3., 5.],
            allowed_installation_regions=[[4., 4., 96., 96.]],
            forbidden_regions=[[0., 0., 5., 100.]],
            service_hubs=[dict(name=name, xy=xy, max_horizontal_distance_m=35., capacity=2)
                          for name, xy in hubs])
    if profile == 'resilient':
        spec['constraints']['min_single_failure_coverage_pct'] = 50.
        spec['regions'][0]['min_single_failure_coverage_pct'] = 40.
    CoverageRequirements(spec)
    return spec


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--study', default='configs/rsu_coverage_study.json')
    p.add_argument('--library', default='results/prototype_library_shapes_20261008')
    p.add_argument('--target-ids', nargs='+', type=int)
    p.add_argument('--k-values', nargs='+', type=int)
    p.add_argument('--profiles', nargs='+', choices=PROFILES)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--step-budgets', nargs='+', type=int)
    p.add_argument('--output', default='results/rsu_coverage_study')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--report-only', action='store_true')
    return p


def _selection(requested, available, label):
    values = available if requested is None else requested
    if not values or len(set(values)) != len(values) or not set(values) <= set(available):
        raise ValueError(f'Invalid or duplicate {label}')
    return values


def make_plan(args):
    study = json.loads(Path(args.study).read_text())
    expected = {'schema', 'split', 'suite', 'new_targets', 'profiles', 'k_values', 'seeds', 'budgets',
                'radius_m', 'repair_evaluations', 'repair_seconds', 'ls_evaluations', 'initialization', 'numeric_status'}
    if (set(study) != expected or study['schema'] != 'rsu_coverage_study_v1'
            or study['split'] != 'development' or study['numeric_status'] !=
            'illustrative_sensitivity_values_not_operational_requirements'):
        raise ValueError('Expected a development RSU study with explicitly illustrative values')
    suite = deepcopy(load_suite(json.loads(Path(study['suite']).read_text())))
    for target in study['new_targets']:
        families = [f for f in suite['families'] if f['sources'][0]['topology'] == target['topology']]
        if len(families) != 1:
            raise ValueError('New target has no unique source family')
        families[0]['targets'].append(target)
    load_suite(suite)
    targets = {t['id']: t for f in suite['families'] for t in f['targets']}
    ids = _selection(args.target_ids, list(targets), 'targets')
    ks = _selection(args.k_values, study['k_values'], 'budgets K')
    profiles = _selection(args.profiles, study['profiles'], 'profiles')
    seeds = _selection(args.seeds, study['seeds'], 'seeds')
    budgets = {b['steps']: b for b in study['budgets']}
    if (len(budgets) != len(study['budgets']) or any(set(b) != {'steps', 'seconds'}
            or isinstance(b['steps'], bool) or not isinstance(b['steps'], int) or b['steps'] < 256
            or isinstance(b['seconds'], bool) or not isinstance(b['seconds'], (int, float))
            or not math.isfinite(b['seconds']) or b['seconds'] <= 0 for b in study['budgets'])):
        raise ValueError('Unique training budgets >=256 steps with finite time caps required')
    steps = _selection(args.step_budgets, list(budgets), 'step budgets')
    jobs = []
    for tid in ids:
        for k in ks:
            for profile in profiles:
                requirements = requirements_for(targets[tid], k, profile)
                for budget in steps:
                    for seed in seeds:
                        identifier = f's{tid}_k{k}_{profile}_n{budget}_seed{seed}'
                        flags = ['--library', args.library, '--target-id', str(tid),
                            '--initialization', study['initialization'], '--steps', str(budget),
                            '--seconds', str(budgets[budget]['seconds']), '--evaluations', str(study['ls_evaluations']),
                            '--radius-m', str(study['radius_m']), '--seed', str(seed), '--repair-initial',
                            '--repair-evaluations', str(study['repair_evaluations']),
                            '--repair-seconds', str(study['repair_seconds'])]
                        # Reuse the child planner's limit validation without file access to source results.
                        child = child_plan(child_parser().parse_args(flags), suite_data=suite,
                                           requirements_spec=requirements)
                        jobs.append(dict(id=identifier, directory=f'runs/{identifier}', target_id=tid, K=k,
                            profile=profile, seed=seed, steps=budget, status='pending', plan=child))
    return dict(schema='rsu_coverage_study_run_v1', status='planned', split='development', suite=suite,
        jobs=jobs, requested_pairs=len(jobs), requested_method_runs=2*len(jobs),
        configured_refinement_caps_sec=sum(2*j['plan']['limits']['time_limit_sec'] for j in jobs),
        numeric_status=study['numeric_status'],
        notes=['All geometries are synthetic development maps; no final holdout claim.',
               'Fresh policy per budget; actual steps/updates must be checked, not inferred from limits.',
               'Same objective, constraints, initial repair, radius and time cap for both methods.',
               'No detection, full constrained optimum, cable routing, radio QoS or policy transfer.',
               'Refinement caps exclude repair/import/validation and may overrun; not total runtime bounds.'])


def _freeze(root, plan):
    frozen = root / 'frozen'
    frozen.mkdir()
    representatives = {}
    for job in plan['jobs']:
        child = job['plan']
        key = (tuple(child['job']['source_roots']), job['K'])
        representatives.setdefault(key, child)
    for index, child in enumerate(representatives.values()):
        bundle = frozen / 'bundles' / str(index)
        bundle.mkdir(parents=True)
        copy = deepcopy(child)
        freeze_inputs(bundle, copy)
        save(bundle / 'plan.json', copy)
        if index == 0:
            shutil.copytree(bundle / 'source', frozen / 'source')
        for source in (bundle / 'inputs').iterdir():
            shutil.copytree(source, frozen / 'library/library' / source.name, dirs_exist_ok=True)
    save(frozen / 'suite.json', plan['suite'])
    (frozen / 'requirements').mkdir()
    for job in plan['jobs']:
        save(frozen / 'requirements' / f'{job["id"]}.json', job['plan']['requirements'])
    save(frozen / 'plan.json', plan)
    plan['frozen_hashes'] = {str(p.relative_to(root)): digest(p) for p in frozen.rglob('*') if p.is_file()}


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ['status']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assessment(directory, phase, expected, evaluator, anchors, radius):
    poses = np.load(directory / f'{phase}.npy', allow_pickle=False)
    with np.load(directory / f'{phase}_coverage.npz', allow_pickle=False) as data:
        if not np.array_equal(data['targets'], evaluator.targets):
            raise ValueError('Target ordering differs')
        masks = data['sensor_masks'] if 'sensor_masks' in data else None
        actual = evaluator.evaluate(poses, data['density'], sensor_masks=masks)
    if (poses.shape != anchors.shape or not np.isfinite(poses).all()
            or np.any(np.linalg.norm(poses[:, :2].astype(float) - anchors[:, :2], axis=1) > radius + 1e-5)):
        raise ValueError('Invalid sensor count or original anchor radius')
    if actual.to_dict() != dict(expected, violations=tuple(expected['violations'])):
        raise ValueError('Stored masks/density do not reproduce the assessment')
    return poses


def collect_job(root, job):
    directory = root / job['directory']
    child = json.loads((directory / 'manifest.json').read_text())
    for name, expected in child['frozen_hashes'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError('Child frozen input changed')
    for source in child['source_prototypes']:
        frozen_source = root / 'frozen/library/library' / Path(source['root']).name
        if any(digest(frozen_source / name) != expected for name, expected in source['hashes'].items()):
            raise ValueError('Source prototype differs from study snapshot')
    settings = json.loads((directory / 'settings.json').read_text())
    spec = json.loads((directory / 'requirements.json').read_text())
    if (spec != job['plan']['requirements'] or settings['limits'] != job['plan']['limits']
            or settings['initial_repair'] != job['plan']['initial_repair']
            or settings['methods'] != list(METHODS) or child['job']['scenario'] != job['plan']['job']['scenario']):
        raise ValueError('Child differs from prespecified study job')
    repair_dir = directory / 'repair'
    repair = json.loads((repair_dir / 'result.json').read_text())
    if (not repair['validated_direct_raycast'] or repair['full_visibility_cache']
            or repair['requirements_sha256'] != digest(directory / 'requirements.json')):
        raise ValueError('Missing repair validation')
    anchors = np.load(repair_dir / 'anchors.npy', allow_pickle=False)
    if pose_hash(anchors) != repair['transferred_pose_sha256']:
        raise ValueError('Transfer anchors changed')
    with np.load(repair_dir / 'initial_coverage.npz', allow_pickle=False) as data:
        evaluator = CoverageRequirements(spec).bind(data['targets'])
    _assessment(repair_dir, 'initial', repair['assessments']['initial'], evaluator, anchors, settings['limits']['radius_m'])
    repaired = _assessment(repair_dir, 'best_candidate', repair['assessments']['best_candidate'], evaluator,
                           anchors, settings['limits']['radius_m'])
    success = repair['status'] in ('already_feasible', 'repaired')
    if success != repair['assessments']['best_candidate']['feasible'] or child['status'] != job['status']:
        raise ValueError('Child repair status mismatch')
    if success:
        np.testing.assert_array_equal(repaired, np.load(repair_dir / 'feasible.npy', allow_pickle=False))
        if pose_hash(repaired) != repair['feasible_pose_sha256']:
            raise ValueError('Common repair hash mismatch')
    elif repair['status'] not in ('no_feasible_found', 'infeasible_fixed_count'):
        raise ValueError('Unexpected repair failure status')
    rows = []
    for method in METHODS:
        row = dict(target_id=job['target_id'], K=job['K'], profile=job['profile'], seed=job['seed'],
            requested_steps=job['steps'], method=method, status='complete' if success else 'not_run',
            repair_status=repair['status'], coverage_pct=None, objective_score=None, overlap_ratio=None,
            installation_cost=None, single_failure_coverage_pct=None, training_steps=None, ppo_updates=None,
            training_budget_completed=None, optimization_sec=None, rejected_fraction=None,
            final_policy_coverage_pct=None, final_policy_objective=None, matched_time_objective=None,
            repair_search_sec=repair['repair_search_sec'],
            repair_worker_wall_sec=child['initial_repair']['worker_wall_sec'],
            worker_wall_sec=None, rays_submitted=None, service_connected_sensors=None)
        for region in spec.get('regions', []):
            for metric in ('coverage_pct', 'overlap_ratio', 'single_failure_coverage_pct'):
                row[f'region_{region["name"]}_{metric}'] = None
        if not success:
            if (directory / method).exists() or (repair_dir / 'feasible.npy').exists() or child['status'] != 'no_feasible_initial':
                raise ValueError('Optimizer ran after unsuccessful common repair')
        else:
            out = directory / method
            result = json.loads((out / 'result.json').read_text())
            if (result['status'] != 'complete' or not result['validated_direct_raycast']
                    or result['full_visibility_cache'] or result['reused_checkpoint']
                    or result['initial_pose_sha256'] != pose_hash(repaired)
                    or result['requirements_sha256'] != digest(directory / 'requirements.json')
                    or result['seed'] != job['seed'] or result['method'] != method):
                raise ValueError('Invalid method identity or common start')
            np.testing.assert_array_equal(np.load(out / 'anchors.npy', allow_pickle=False), anchors)
            for phase, expected in result['assessments'].items():
                poses = _assessment(out, phase, expected, evaluator, anchors, settings['limits']['radius_m'])
                if not expected['feasible']:
                    raise ValueError('Reported infeasible deployment')
                if phase == 'initial':
                    np.testing.assert_array_equal(poses, repaired)
            a = result['assessments']['deployment']
            point = trace_at(result['objective_trace'], settings['limits']['time_limit_sec'])
            if result['objective_trace'][-1]['objective_score'] != a['objective_score']:
                raise ValueError('Final objective trace mismatch')
            trained = (result['training_steps'] == job['steps'] and
                       result['ppo_updates'] == math.ceil(job['steps'] / result['params']['ppo_update_timestep']))
            if method == 'mappo':
                with np.load(out / 'train_log.npz', allow_pickle=False) as data:
                    if (not np.array_equal(data['step'], np.arange(1, result['training_steps'] + 1))
                            or any(not np.isfinite(data[key]).all() for key in data.files)):
                        raise ValueError('Missing/nonfinite training samples')
                diagnostics = result['ppo_diagnostics']
                if (len(diagnostics) != result['ppo_updates']
                        or sum(d['rollout_steps'] for d in diagnostics) != result['training_steps']
                        or any(d['samples'] != d['rollout_steps'] * len(anchors) for d in diagnostics)
                        or any(not math.isfinite(value) for d in diagnostics for value in d.values())):
                    raise ValueError('PPO update diagnostics do not match actual training steps')
            final = result['assessments'].get('final_policy', {})
            row.update(coverage_pct=a['coverage_pct'], objective_score=a['objective_score'],
                overlap_ratio=a['overlap_ratio'], installation_cost=a['components']['installation_cost'],
                single_failure_coverage_pct=a.get('robustness', {}).get('single_failure_coverage_pct'),
                training_steps=result['training_steps'], ppo_updates=result['ppo_updates'],
                training_budget_completed=trained if method == 'mappo' else None,
                optimization_sec=result['optimization_sec'], rejected_fraction=result['rejected_proposals'] / max(result['proposals'], 1),
                final_policy_coverage_pct=final.get('coverage_pct'), final_policy_objective=final.get('objective_score'),
                matched_time_objective=point['objective_score'] if point else None,
                worker_wall_sec=child['jobs'][method]['worker_wall_sec'],
                rays_submitted=result['runtime']['counts'].get('rays_submitted', 0),
                service_connected_sensors=a.get('infrastructure', {}).get('matched_sensors'))
            for name, values in a['regions'].items():
                for metric in ('coverage_pct', 'overlap_ratio', 'single_failure_coverage_pct'):
                    row[f'region_{name}_{metric}'] = values.get(metric)
        rows.append(row)
    hashes = {str(p.relative_to(root)): digest(p) for p in directory.rglob('*')
              if p.is_file() and not any(x in p.parts for x in ('source', '__pycache__'))}
    return rows, hashes


def write_report(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['frozen_hashes'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Frozen study input changed: {name}')
    frozen = json.loads((root / 'frozen/plan.json').read_text())
    if (len(frozen['jobs']) != len(manifest['jobs']) or
            [{k: j[k] for k in j if k != 'status'} for j in frozen['jobs']] !=
            [{k: j[k] for k in frozen['jobs'][i] if k != 'status'} for i, j in enumerate(manifest['jobs'])]):
        raise ValueError('Study plan changed')
    rows, hashes = [], {}
    for job in manifest['jobs']:
        if job['status'] in ('complete', 'no_feasible_initial'):
            part, digests = collect_job(root, job)
            rows.extend(part)
            hashes.update(digests)
    _csv(root / 'raw_methods.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ('target_id', 'K', 'profile', 'requested_steps', 'method'))].append(row)
    summary = []
    for key, values in groups.items():
        entry = dict(zip(('target_id', 'K', 'profile', 'requested_steps', 'method'), key))
        entry.update(processed=len(values), completed=sum(v['status'] == 'complete' for v in values),
            blocked=sum(v['status'] == 'not_run' for v in values),
            truncated_training=sum(v['training_budget_completed'] is False for v in values))
        for metric in ('coverage_pct', 'objective_score', 'optimization_sec', 'single_failure_coverage_pct'):
            observed = [v[metric] for v in values if v[metric] is not None]
            entry[metric + '_mean'] = mean(observed) if observed else None
            entry[metric + '_sd'] = stdev(observed) if len(observed) > 1 else None
        summary.append(entry)
    _csv(root / 'summary.csv', summary)
    paired = []
    for index in range(0, len(rows), 2):
        a, b = rows[index:index+2]
        if a['status'] == b['status'] == 'complete':
            paired.append(dict(target_id=a['target_id'], K=a['K'], profile=a['profile'], seed=a['seed'],
                requested_steps=a['requested_steps'], training_budget_completed=b['training_budget_completed'],
                mappo_minus_ls_objective=b['objective_score'] - a['objective_score'],
                mappo_minus_ls_coverage_pp=b['coverage_pct'] - a['coverage_pct'],
                mappo_minus_ls_matched_time_objective=(b['matched_time_objective'] - a['matched_time_objective']
                    if a['matched_time_objective'] is not None and b['matched_time_objective'] is not None else None)))
    _csv(root / 'paired.csv', paired)
    lines = ['# RSU Geometric Coverage Study', '',
        'Synthetic development maps and illustrative engineering values, not deployment standards.',
        'Costs/coverage/redundancy: https://arxiv.org/abs/2110.01251',
        'Road-network failure motivation (not a LiDAR threshold): https://www.mdpi.com/1424-8220/16/12/2059',
        'Power/network interfaces: https://docs.ouster.com/sensor-docs/overview/quick-start-guide', '',
        f'Status: {manifest["status"]}; processed pairs: {len(rows)//2}/{manifest["requested_pairs"]}.', '',
        '| Target | K | Profile | Requested Steps | Method | Done/Processed | Coverage % | J | Actual Steps | PPO Updates |',
        '|---|---:|---|---:|---|---|---:|---:|---|---|']
    for r in rows:
        fmt = lambda key: '' if r[key] is None else f'{r[key]:.3f}'
        lines.append(f'| {r["target_id"]} | {r["K"]} | {r["profile"]} | {r["requested_steps"]} | {r["method"]} | '
            f'{r["status"]} | {fmt("coverage_pct")} | {fmt("objective_score")} | '
            f'{r["training_steps"] if r["training_steps"] is not None else ""} | '
            f'{r["ppo_updates"] if r["ppo_updates"] is not None else ""} |')
    lines += ['', '## Interpretation', '',
        'Actual training counts, final-policy metrics, rejection fractions and matched-time J are in raw_methods.csv.',
        'A time-capped run that missed its requested steps is explicitly marked as truncated training.',
        'Blocked repairs are missing quality scores, not zero scores or certified global infeasibility.',
        'LS and MAPPO share J, hard constraints and a repaired start; no automatic MAPPO advantage is assumed.',
        'Refinement time excludes common repair, transfer, import and independent direct validation.',
        'The report rechecks saved per-sensor masks; workers performed direct raycasting. Reporting does not raycast.',
        'No convergence, detection continuity, real-site validity, controlled speedup or constrained optimum is established.']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    report_source = root / 'report_source'
    report_source.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), report_source / 'rsu_coverage_study.py')
    save(root / 'validation_summary.json', dict(raw_sha256=hashes,
        report_source_sha256=digest(report_source / 'rsu_coverage_study.py'),
        completed_method_runs=sum(r['status'] == 'complete' for r in rows),
        blocked_method_runs=sum(r['status'] == 'not_run' for r in rows),
        truncated_training_runs=sum(r['training_budget_completed'] is False for r in rows),
        sample_sd_ddof=1, maps_are_synthetic_development=True))


def run(args):
    root = Path(args.output).resolve()
    if args.report_only:
        if args.execute:
            raise ValueError('Report-only cannot train')
        write_report(root)
        return 0
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    plan.update(status='preparing', started_at=now(), command=sys.argv)
    save(root / 'manifest.json', plan)
    try:
        _freeze(root, plan)
        (root / 'logs').mkdir()
        plan['status'] = 'running'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for index, job in enumerate(plan['jobs']):
            save(root / 'manifest.json', plan)
            child = job['plan']
            limits, repair = child['limits'], child['initial_repair']
            command = [sys.executable, '-u', str(root / 'frozen/source/prototype_constrained.py'),
                '--library', str(root / 'frozen/library'), '--suite', str(root / 'frozen/suite.json'),
                '--requirements', str(root / 'frozen/requirements' / f'{job["id"]}.json'),
                '--target-id', str(job['target_id']), '--initialization', child['job']['initialization'],
                '--steps', str(job['steps']), '--seconds', str(limits['time_limit_sec']),
                '--evaluations', str(limits['max_evaluations']), '--radius-m', str(limits['radius_m']),
                '--seed', str(job['seed']), '--repair-initial', '--repair-evaluations', str(repair['max_evaluations']),
                '--repair-seconds', str(repair['time_limit_sec']), '--output', str(root / job['directory']), '--execute']
            print(f'[{index+1}/{len(plan["jobs"])}] {job["id"]} started', flush=True)
            started = time.perf_counter()
            with (root / 'logs' / f'{job["id"]}.log').open('w') as stream:
                process = subprocess.run(command, cwd=root / 'frozen/source', env=env, stdout=stream, stderr=subprocess.STDOUT)
            child_manifest = json.loads((root / job['directory'] / 'manifest.json').read_text())
            job.update(status=child_manifest['status'], returncode=process.returncode,
                       launcher_wall_sec=time.perf_counter() - started)
            if (process.returncode, job['status']) not in ((0, 'complete'), (2, 'no_feasible_initial')):
                raise RuntimeError(f'Worker failed: {job["id"]}; see logs')
            collect_job(root, job)
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

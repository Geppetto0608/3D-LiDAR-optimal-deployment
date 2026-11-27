"""Frozen-library low-K cost/critical-coverage experiment; no RL or detector.

Dry-run is read-only. Execution copies the completed source library and launches
fresh workers; physical target queries occur only after all shortlists are frozen.
"""
from argparse import ArgumentParser
from importlib import metadata
from pathlib import Path
import json
import os
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

import library_rsu_stress as core
from src.methods.rsu_coverage_milp import assess, solve_frontier
from src.roadside.rsu_stress import domain, scene

BASE = Path(__file__).resolve().parent
DEFAULT_LIBRARY = BASE / 'results/library_rsu_stress_20261123'


def default_spec():
    old = core.default_spec()
    budgets = {d['name']: ([2, 3, 4] if d['track'] == 'A' else
               [6, 8] if d['length_m'] == 480 else [3, 4, 6]) for d in old['targets']}
    return dict(schema='library_lowk_cost_v1', targets=old['targets'], sampling=old['sampling'],
        target_k=budgets, query_budget=256, random_seeds=[0, 1, 2], source_count=3,
        alternatives=12, scene_levels=['mixed', 'dense'],
        requirements=old['requirements']['union'],
        cost_model=dict(base=1., per_height_m=.10, per_service_distance_m=.008,
                        reference_height_m=3., corridor_hub_spacing_m=160.),
        profiles=dict(base=dict(cost_per_k=None, critical_coverage_ratio=0.),
                      cost=dict(cost_per_k=1.25, critical_coverage_ratio=0.),
                      cost_critical60=dict(cost_per_k=1.25, critical_coverage_ratio=.60),
                      cost_critical80=dict(cost_per_k=1.25, critical_coverage_ratio=.80)),
        optimization=dict(reference_seconds=1800., arm_seconds=300., surrogate_seconds=15.,
                          gap=.01, min_seconds=300., stagnation_seconds=120.,
                          threads=1, soft_memory_gb=24.),
        execution=dict(workers=2, hard_wall_hours=168., job_wall_hours=48., min_free_ram_gib=48.,
                       min_free_disk_gib=12., max_job_rss_gib=40., max_total_rss_gib=80.))


def smoke_spec():
    s = default_spec()
    s['targets'] = [s['targets'][0], s['targets'][4]]
    s['target_k'] = {d['name']: [2, 3] for d in s['targets']}
    s.update(query_budget=8, random_seeds=[0], scene_levels=['dense'])
    # Keep the validated installation sides and all source orientation/height modes.
    s['sampling'].update(target_heights_m=[1.5], target_b_step_m=2.)
    s['optimization'].update(reference_seconds=2., arm_seconds=.5, surrogate_seconds=.2,
                             min_seconds=.1, stagnation_seconds=.1)
    s['execution'].update(hard_wall_hours=1., job_wall_hours=.5)
    return s


def read_library(library):
    manifest = json.loads((library / 'manifest.json').read_text())
    spec = json.loads((library / 'spec.json').read_text())
    if (manifest['status'] != 'complete' or spec['schema'] != 'library_rsu_stress_v1'
            or core.digest(library / 'spec.json') != manifest['spec_sha256']):
        raise ValueError('A complete, unchanged RSU source library is required')
    hashes = {}
    for d in spec['sources']:
        folder = library / 'library' / d['name']
        result = json.loads((folder / 'result.json').read_text())
        if result['status'] != 'complete' or not result['validated_direct'] or result['descriptor'] != d:
            raise ValueError('Unvalidated/mismatched source entry')
        if set(result['artifact_hashes']) != {'candidates.npz', 'layouts.json'}:
            raise ValueError('Unexpected source artifact set')
        for name in ('result.json', 'candidates.npz', 'layouts.json'):
            path = folder / name
            h = core.digest(path)
            if name != 'result.json' and h != result['artifact_hashes'][name]:
                raise ValueError('Source library hash mismatch: ' + str(path))
            hashes[str(path.relative_to(library))] = h
    return spec, hashes


def installation_costs(problem, model):
    """Synthetic Euclidean service-distance proxy, not a routed cable quotation."""
    poses, d = problem['poses'], problem['descriptor']
    sites = np.unique(poses[:, :2], axis=0)
    if d['track'] == 'A':
        g = problem['geometry']
        desired = np.array([[.15 * g.map_len, g.junction_y_m],
                            [.85 * g.map_len, g.junction_y_m]])
    else:
        length = d['length_m']
        stations = np.unique(np.r_[np.arange(0., length, model['corridor_hub_spacing_m']), length])
        desired, _ = problem['geometry'].centerline(stations)
    hubs = np.unique(np.array([sites[np.argmin(np.linalg.norm(sites - p, axis=1))]
                              for p in desired]), axis=0)
    distance = np.linalg.norm(poses[:, None, :2] - hubs[None, :, :], axis=2).min(axis=1)
    costs = np.round(model['base'] + model['per_height_m'] * np.maximum(
        0., poses[:, 2] - model['reference_height_m']) + model['per_service_distance_m'] * distance, 6)
    return costs, dict(model=model, hubs_xy=hubs.tolist(), units='synthetic_relative_cost',
                       distance='Euclidean nearest service anchor; no capacity/network guarantee')


def plan(spec, library):
    if set(spec) != set(default_spec()) or spec['schema'] != 'library_lowk_cost_v1':
        raise ValueError('Unknown low-K study schema/settings')
    source_spec, hashes = read_library(library)
    # Reuse the established geometry/schema guards without solving or raycasting.
    check = core.default_spec()
    check.update(sources=source_spec['sources'], targets=spec['targets'], sampling=spec['sampling'],
        target_k=sorted({k for ks in spec['target_k'].values() for k in ks}),
        query_budgets=[spec['query_budget']], random_seeds=spec['random_seeds'],
        source_count=spec['source_count'], alternatives=spec['alternatives'],
        scene_levels=spec['scene_levels'], execution=spec['execution'])
    check['optimization'].update(spec['optimization'])
    check['requirements'] = dict(union=spec['requirements'], redundancy=spec['requirements'])
    checked = core.plan(check)
    if not set(source_spec['sampling']['heights_m']).issubset(spec['sampling']['heights_m']):
        raise ValueError('Target height grid must support every source height')
    if set(spec['target_k']) != {d['name'] for d in spec['targets']}:
        raise ValueError('Each target needs explicit K values')
    for ks in spec['target_k'].values():
        if not ks or ks != sorted(set(ks)) or any(type(k) is not int or not 1 <= k <= 64 for k in ks):
            raise ValueError('Invalid target K frontier')
    if set(spec['optimization']) != set(default_spec()['optimization']):
        raise ValueError('Unknown optimizer settings')
    cm = spec['cost_model']
    if (set(cm) != set(default_spec()['cost_model']) or not np.isfinite(list(cm.values())).all()
            or min(cm.values()) < 0 or cm['base'] <= 0 or cm['corridor_hub_spacing_m'] <= 0):
        raise ValueError('Invalid synthetic cost model')
    if not spec['profiles']:
        raise ValueError('No constraint profiles')
    for name, profile in spec['profiles'].items():
        if (not name or not all(c.isalnum() or c == '_' for c in name)
                or set(profile) != {'cost_per_k', 'critical_coverage_ratio'}
                or not np.isfinite(profile['critical_coverage_ratio'])
                or not 0 <= profile['critical_coverage_ratio'] <= 1
                or (profile['cost_per_k'] is not None and
                    (not np.isfinite(profile['cost_per_k']) or profile['cost_per_k'] < 0))):
            raise ValueError('Invalid cost/critical profile')
    cases = len(spec['targets']) * len(spec['scene_levels'])
    references = sum(map(len, spec['target_k'].values())) * len(spec['scene_levels']) * len(spec['profiles'])
    rows = references * (4 + len(spec['random_seeds']))
    opt = spec['optimization']
    return dict(sources=len(source_spec['sources']), library_artifacts=len(hashes),
        geometry_count=len(spec['targets']), case_jobs=cases, constraint_conditions=cases * len(spec['profiles']),
        expected_rows=rows, reference_rows=references, domains=[x for x in checked['domains'] if x['role'] == 'targets'],
        solver_cap_sum_hours=(rows * opt['arm_seconds'] + references * opt['reference_seconds']) / 3600,
        workers=spec['execution']['workers'],
        scope='Development maps, fixed source library, independent static scenes, no RL/detection/robust multi-scene solve.')


def optimize(problem, masks, costs, spec, role, profile, ks, directory, alternatives=1):
    directory.mkdir(parents=True, exist_ok=True)
    opt = spec['optimization']
    limits = {k: profile['cost_per_k'] * k for k in ks} if profile['cost_per_k'] is not None else None
    f = solve_frontier(problem['poses'], masks, problem['critical'], ks, spec['requirements'],
        candidate_costs=costs, cost_limits=limits, critical_coverage_ratio=profile['critical_coverage_ratio'],
        seconds=opt[role + '_seconds'], alternatives=alternatives, threads=opt['threads'],
        soft_mem_gb=opt['soft_memory_gb'], stop_gap=opt['gap'],
        stop_min_sec=min(opt['min_seconds'], opt[role + '_seconds']),
        stop_stagnation_sec=opt['stagnation_seconds'], progress_path=directory / 'progress.jsonl')
    core.save(directory / 'frontier.json', f)
    core.write_csv(directory / 'quality_time.csv', [dict(k=row['budget_k'], **point)
        for row in f['frontier'] for point in row['quality_time_trace']])
    return f


def make_plans(root, directory, spec, problem, costs):
    library_spec = json.loads((root / 'library_spec.json').read_text())
    pool = core.library_pool(root, dict(spec, sources=library_spec['sources']), problem)
    core.save(directory / 'transfer.json', pool)
    pool_ids = pool['order']
    q = min(spec['query_budget'], len(pool_ids))
    if q == 0:
        raise ValueError('Empty transferred pool; do not silently substitute geometry candidates')
    before = time.perf_counter()
    optical = core._optical_masks(problem['poses'].astype(np.float32), problem['targets'].astype(np.float32),
                                  core._profile_values(problem['profile']))
    union = core.shortlist(optical, problem['poses'], q)
    union_time = time.perf_counter() - before
    before = time.perf_counter()
    local_optical = core._optical_masks(problem['poses'][pool_ids].astype(np.float32),
        problem['targets'].astype(np.float32), core._profile_values(problem['profile']))
    diverse = [pool_ids[i] for i in core.shortlist(local_optical, problem['poses'][pool_ids], q, double=True)]
    diverse_time = time.perf_counter() - before + pool['planning_sec']
    del local_optical
    plans = []
    for name, profile in spec['profiles'].items():
        before = time.perf_counter()
        f = optimize(problem, optical, costs, spec, 'surrogate', profile,
                     spec['target_k'][problem['descriptor']['name']], directory / ('surrogate_' + name), 6)
        portfolio = list(dict.fromkeys([cid for rank in range(6) for row in f['frontier']
            if rank < len(row['alternatives']) for cid in row['alternatives'][rank]['selected_ids']] + union))[:q]
        portfolio_time = time.perf_counter() - before + union_time
        methods = [('library_prefix', None, pool_ids[:q], pool['planning_sec']),
                   ('library_diverse', None, diverse, diverse_time),
                   ('geometry_union', None, union, union_time),
                   ('geometry_portfolio', None, portfolio, portfolio_time)]
        for seed in spec['random_seeds']:
            before = time.perf_counter()
            ids = np.random.default_rng(seed).permutation(len(problem['poses']))[:q].tolist()
            methods.append(('random', seed, ids, time.perf_counter() - before))
        for method, seed, ids, planning_sec in methods:
            if len(ids) != q or len(set(ids)) != q:
                raise RuntimeError('Unequal/duplicate query shortlist')
            plans.append(dict(profile=name, method=method, seed=seed, selected_ids=sorted(ids),
                requested_q=spec['query_budget'], actual_q=q, pool_count=len(pool_ids), planning_sec=planning_sec,
                arm_id=name + '_' + method + (f'_s{seed}' if seed is not None else '')))
    core.save(directory / 'query_plans.json', plans)
    return plans


def case_job(root, spec, descriptor, level, *, planner=None):
    start = time.perf_counter()
    name = descriptor['name'] + '_' + level
    directory = root / 'cases' / name
    directory.mkdir(parents=True, exist_ok=False)
    problem = domain(descriptor, spec['sampling'])
    costs, cost_info = installation_costs(problem, spec['cost_model'])
    mesh = scene(problem, level)
    preparation = time.perf_counter() - start
    np.save(directory / 'installation_costs.npy', costs)
    core.save(directory / 'cost_model.json', cost_info)
    plans = (planner or make_plans)(root, directory, spec, problem, costs)
    plans_hash = core.digest(directory / 'query_plans.json')
    ks = spec['target_k'][descriptor['name']]
    arms = []
    for p in plans:
        arm = directory / p['arm_id']
        arm.mkdir()
        ids = p['selected_ids']
        masks, visibility, work = core.query(mesh, problem, ids)
        subset = dict(problem, poses=problem['poses'][ids])
        frontier = optimize(subset, masks, costs[ids], spec, 'arm', spec['profiles'][p['profile']], ks, arm)
        for row in frontier['frontier']:
            row['full_candidate_ids'] = [ids[i] for i in row['selected_ids']] if row['selected_ids'] is not None else None
        np.savez_compressed(arm / 'queried.npz', candidate_ids=ids, packed_masks=np.packbits(masks, axis=1))
        value = dict(plan=p, frontier=frontier, visibility_sec=visibility, work=work,
                     online_all_k_sec=preparation + p['planning_sec'] + visibility + frontier['wall_sec'])
        core.save(arm / 'result.json', value)
        arms.append(value)
    # Offline evaluation starts only after every online arm has fixed its result.
    full, full_time, full_work = core.query(mesh, problem, range(len(problem['poses'])))
    refs = {name: optimize(problem, full, costs, spec, 'reference', profile, ks,
                          directory / ('reference_' + name)) for name, profile in spec['profiles'].items()}
    core.save_masks(directory / 'candidates.npz', problem, full)
    selected = {i for f in refs.values() for row in f['frontier'] for i in (row['selected_ids'] or [])}
    for arm in arms:
        p = arm['plan']
        with np.load(directory / p['arm_id'] / 'queried.npz', allow_pickle=False) as data:
            if (not np.array_equal(data['candidate_ids'], p['selected_ids']) or
                    not np.array_equal(np.unpackbits(data['packed_masks'], axis=1, count=len(problem['targets'])),
                                       full[p['selected_ids']])):
                raise RuntimeError('Query/reference visibility mismatch')
        for row in arm['frontier']['frontier']:
            selected.update(row['full_candidate_ids'] or [])
    ids = sorted(selected)
    direct, validation_sec, validation_work = core.query(mesh, problem, ids, direct=True)
    if not np.array_equal(direct, full[ids]) or core.digest(directory / 'query_plans.json') != plans_hash:
        raise RuntimeError('Direct raycast/frozen plan verification failed')
    # Check reference as well as online constraints against the complete target mask.
    for name, f in refs.items():
        for row in f['frontier']:
            check_row(problem, full, costs, spec, name, row, row['selected_ids'])
    rows = []
    for arm in arms:
        p = arm['plan']
        cumulative = arm['frontier']['preparation_sec']
        for row, ref in zip(arm['frontier']['frontier'], refs[p['profile']]['frontier']):
            values = check_row(problem, full, costs, spec, p['profile'], row, row['full_candidate_ids'])
            cumulative += row['selection_sec']
            lower = upper = None
            if values:
                if ref['upper_bound_count'] is None or values['covered_count'] > ref['upper_bound_count'] + 1e-5:
                    raise RuntimeError('Full-domain bound contradicts feasible online deployment')
                lower = max(0., ref['metrics']['coverage_pct'] - values['coverage_pct']) if ref['metrics'] else None
                upper = max(0., 100. * ref['upper_bound_count'] / len(problem['targets']) - values['coverage_pct'])
            extra = p.get('comparison_metadata', {})
            rows.append(dict(case=descriptor['name'] + '_' + level,
                track=descriptor['track'], scene=level, profile=p['profile'], method=p['method'], seed=p['seed'],
                budget_k=row['budget_k'], requested_q=p['requested_q'], actual_q=p['actual_q'], pool_count=p['pool_count'],
                candidate_count=len(problem['poses']), target_count=len(problem['targets']),
                critical_target_count=int(problem['critical'].sum()), status=row['status'],
                **{key: value for key, value in (values or {}).items() if key != 'violations'},
                cost_limit=spec['profiles'][p['profile']]['cost_per_k'] * row['budget_k']
                    if spec['profiles'][p['profile']]['cost_per_k'] is not None else None,
                critical_floor=spec['profiles'][p['profile']]['critical_coverage_ratio'],
                queried_pool_gap=row['certified_gap'], reference_status=ref['status'],
                reference_coverage_pct=ref['metrics']['coverage_pct'] if ref['metrics'] else None,
                reference_gap=ref['certified_gap'], loss_lower_pp=lower, loss_upper_pp=upper,
                preparation_sec=preparation, planning_sec=p['planning_sec'], visibility_sec=arm['visibility_sec'],
                solver_sec=row['solver_sec'], k_selection_sec=row['selection_sec'],
                online_through_k_sec=preparation + p['planning_sec'] + arm['visibility_sec'] + cumulative,
                online_all_k_sec=arm['online_all_k_sec'], rays_submitted=arm['work'].get('rays_submitted', 0),
                reference_visibility_sec=full_time,
                reference_all_k_sec=preparation + full_time + refs[p['profile']]['wall_sec']))
            if set(extra) & set(rows[-1]):
                raise RuntimeError('Planner metadata cannot overwrite evaluation fields')
            rows[-1].update(extra)
    core.write_csv(directory / 'comparison.csv', rows)
    result = dict(status='complete', case=descriptor['name'] + '_' + level, rows=len(rows),
        validated_direct=True, validation_sec=validation_sec, validation_work=validation_work,
        reference_work=full_work, plans_sha256=plans_hash, wall_sec=time.perf_counter() - start,
        scope='No infeasible/unknown rows replaced by zero; no online warm start in reference.')
    core.save(directory / 'result.json', result)
    return result


def check_row(problem, masks, costs, spec, profile_name, row, ids):
    if ids is None:
        if row['metrics'] is not None or row['status'] not in ('infeasible', 'no_feasible_found'):
            raise RuntimeError('Invalid missing-solution semantics')
        return None
    p = spec['profiles'][profile_name]
    result = assess(problem['poses'], masks, ids, problem['critical'], spec['requirements'], row['budget_k'],
        candidate_costs=costs, cost_budget=p['cost_per_k'] * row['budget_k'] if p['cost_per_k'] is not None else None,
        critical_coverage_ratio=p['critical_coverage_ratio'])
    if result != row['metrics'] or not result['feasible']:
        raise RuntimeError('Independent cost/critical/installation evaluation failed')
    return result


def verify_library(root):
    for name in ('library_spec.json', 'library_provenance.json'):
        if core.digest(root / name) != core.digest(root / 'source' / name):
            raise RuntimeError('Frozen library metadata changed: ' + name)
    info = json.loads((root / 'library_provenance.json').read_text())
    for rel, expected in info['hashes'].items():
        if core.digest(root / rel) != expected:
            raise RuntimeError('Frozen library mutated: ' + rel)


def worker(root, job):
    if BASE != root / 'source':
        raise ValueError('Worker must run frozen code')
    manifest, spec = core.verify(root)
    verify_library(root)
    item = next(j for j in manifest['jobs'] if j['id'] == job)
    value = case_job(root, spec, item['descriptor'], item['scene'])
    core.verify(root)
    verify_library(root)
    value['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    core.save(root / 'job_status' / (job + '.json'), value)


def execute(spec, library, output, detach, *, plan_fn=None,
            entrypoint='library_budget_study.py', snapshot_files=()):
    planned = (plan_fn or plan)(spec, library)
    root = output.resolve()
    if root.exists():
        raise FileExistsError('Use a new output root')
    source_spec, hashes = read_library(library)
    if (core.available_ram() < spec['execution']['min_free_ram_gib'] or
            shutil.disk_usage(BASE).free / 2**30 < spec['execution']['min_free_disk_gib'] + 4.):
        raise RuntimeError('RAM/disk preflight failed')
    root.mkdir(parents=True)
    for name in ('source', 'logs', 'job_status'):
        (root / name).mkdir()
    for rel, expected in hashes.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library / rel, dest)
        if core.digest(dest) != expected:
            raise RuntimeError('Library changed during snapshot')
    core.save(root / 'library_spec.json', source_spec)
    core.save(root / 'library_provenance.json', dict(root=str(library), hashes=hashes,
        original_manifest_sha256=core.digest(library / 'manifest.json'),
        scope='Only source library files copied. No prior target poses/coverage read.'))
    source_hashes = {}
    paths = list(BASE.glob('*.py')) + list((BASE / 'src').rglob('*.py'))
    paths += [BASE / p for p in ('tests/test_rsu_stress.py', 'tests/test_library_budget_study.py',
        'LIBRARY_LOWK_COST_PROTOCOL.md', 'configs/library_lowk_cost_20261124.json')]
    paths += [BASE / p for p in snapshot_files]
    for path in paths:
        rel = path.relative_to(BASE)
        dest = root / 'source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[str(rel)] = core.digest(dest)
    core.save(root / 'spec.json', spec)
    # Include non-code inputs in the same immutable hash verification mechanism.
    for name in ('library_spec.json', 'library_provenance.json'):
        shutil.copy2(root / name, root / 'source' / name)
        source_hashes[name] = core.digest(root / name)
    jobs = [dict(id='case_' + d['name'] + '_' + level, phase='case', descriptor=d, scene=level, status='pending')
            for d in spec['targets'] for level in spec['scene_levels']]
    core.save(root / 'manifest.json', dict(schema=spec['schema'], status='prepared', created_at=core.now(),
        command=sys.argv, plan=planned, spec_sha256=core.digest(root / 'spec.json'), source_hashes=source_hashes,
        jobs=jobs, completed=0, python=sys.version, hardware=core.hardware(root),
        packages={n: metadata.version(n) for n in ('numpy', 'scipy', 'open3d', 'gurobipy')}))
    (root / 'packages.txt').write_text('\n'.join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())) + '\n')
    command = [sys.executable, '-u', str(root / 'source' / entrypoint),
               '--orchestrate', str(root), '--execute']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    if detach:
        with (root / 'launcher.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=root / 'source', env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        core.save(root / 'launch.json', dict(pid=process.pid, command=command, started_at=core.now()))
        print(json.dumps(dict(status='launched', root=str(root), pid=process.pid, plan=planned), indent=2))
    else:
        subprocess.run(command, cwd=root / 'source', env=env, check=True)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/library_lowk_cost_20261124'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--orchestrate', type=Path)
    parser.add_argument('--job')
    args = parser.parse_args(argv)
    if args.worker and args.orchestrate or args.smoke and args.spec:
        parser.error('Choose one internal role and one configuration source')
    if (args.worker or args.orchestrate) and not args.execute:
        parser.error('Internal execution also requires --execute')
    if args.worker:
        worker(args.worker.resolve(), args.job)
        return
    if args.orchestrate:
        root = args.orchestrate.resolve()
        if BASE != root / 'source':
            parser.error('Supervisor must run frozen code')
        verify_library(root)
        core.orchestrate(root, entrypoint='library_budget_study.py')
        return
    spec = smoke_spec() if args.smoke else json.loads(args.spec.read_text()) if args.spec else default_spec()
    if args.execute:
        execute(spec, args.library.resolve(), args.output, args.detach)
    else:
        print(json.dumps(plan(spec, args.library.resolve()), indent=2))


if __name__ == '__main__':
    main()

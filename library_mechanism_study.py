"""Fixed-query-budget source-library ablations on development maps only.

Small pools use explicitly counted geometry-union fill. This estimates the
effect of a library-informed hybrid, not a pure-library fixed-size pool.
Reserved geometries are recorded but cannot be evaluated by this runner.
"""
from argparse import ArgumentParser
from pathlib import Path
import csv
import hashlib
import json
import resource
import time

import numpy as np

import library_budget_study as budget
import library_rsu_stress as core
from src.roadside.rsu_stress import retrieve, transfer_ids

BASE = Path(__file__).resolve().parent


def default_split():
    original = core.default_spec()
    reserved = []
    for topology in ('T_JUNCTION', 'CROSS'):
        for size, width, center in [(140., 9., [.38, .59]), (200., 13., [.61, .39])]:
            reserved.append(dict(name=f'reserved_a_{topology.lower()}_l{int(size)}', track='A',
                topology=topology, size_m=size, width_m=width, center_fraction=center,
                seed=7000 + len(reserved)))
    for length, width, turn, modules in [(200., 9., 22.5, 3), (280., 13., 37.5, 4),
                                        (360., 9., 67.5, 5), (520., 13., 82.5, 7)]:
        reserved.append(dict(name=f'reserved_b_l{int(length)}', track='B', length_m=length,
            width_m=width, turn_deg=turn, modules=modules, seed=7000 + len(reserved)))
    return dict(schema='library_mechanism_split_v1', sources=original['sources'],
                development=original['targets'], reserved=reserved)


def default_spec():
    spec = budget.default_spec()
    spec.update(schema='library_mechanism_v1', split=default_split(),
        ablation=dict(source_counts=[1, 3, 6], alternative_counts=[1, 12],
                      random_source_seeds=[0, 1, 2], fill_policy='geometry_union'))
    # Independent arms run sequentially in each case; give the supervisor room
    # for the enlarged matrix without imposing a minimum runtime.
    spec['execution'].update(hard_wall_hours=240., job_wall_hours=48.)
    return spec


def smoke_spec():
    spec = default_spec()
    old = budget.smoke_spec()
    for name in ('targets', 'target_k', 'sampling', 'optimization', 'execution'):
        spec[name] = old[name]
    spec.update(query_budget=32, random_seeds=[0], scene_levels=['dense'])
    spec['ablation']['random_source_seeds'] = [0]
    spec['profiles'] = {name: spec['profiles'][name] for name in ('base', 'cost_critical60')}
    spec['execution'].update(workers=1, min_free_disk_gib=4.)
    return spec


def geometry_key(descriptor):
    """Changing only a name or obstacle seed does not create a new geometry."""
    if descriptor['track'] == 'A':
        d = dict(track='A', topology=descriptor['topology'], size=float(descriptor['size_m']),
                 width=float(descriptor['width_m']), center=list(map(float, descriptor['center_fraction'])))
    elif descriptor['track'] == 'B':
        # Module counts define service regions/transfer intervals, not a new road.
        d = dict(track='B', length=float(descriptor['length_m']), width=float(descriptor['width_m']),
                 turn=float(descriptor['turn_deg']), bend=float(descriptor.get('bend_length_m', 48.)))
    else:
        raise ValueError('Unknown split geometry track')
    return json.dumps(d, sort_keys=True)


def validate_split(split, sources, targets):
    if (set(split) != {'schema', 'sources', 'development', 'reserved'} or
            split['schema'] != 'library_mechanism_split_v1' or split['sources'] != sources):
        raise ValueError('Split must describe the exact frozen source library')
    names, geometry_sets, seed_sets = set(), [], []
    for part in ('sources', 'development', 'reserved'):
        entries = split[part]
        if not entries:
            raise ValueError('Empty split partition')
        keys = [geometry_key(d) for d in entries]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate geometry within split (seed/name changes do not count)')
        current_seeds = set()
        for d in entries:
            name = d['name']
            if not name or name in names or not all(c.isalnum() or c in '_-' for c in name):
                raise ValueError('Duplicate/unsafe split name')
            if type(d['seed']) is not int or d['seed'] < 0:
                raise ValueError('Invalid obstacle seed')
            names.add(name)
            current_seeds.add(d['seed'])
        geometry_sets.append(set(keys))
        seed_sets.append(current_seeds)
    for i in range(3):
        for j in range(i):
            if geometry_sets[i] & geometry_sets[j] or seed_sets[i] & seed_sets[j]:
                raise ValueError('Source/development/reserved leakage')
    if not targets or any(d not in split['development'] for d in targets):
        raise ValueError('Only development descriptors may be evaluated; reserved maps stay sealed')


def variants(spec):
    a = spec['ablation']
    rows = [dict(source_count=s, alternative_count=n, retrieval='nearest', selector='diverse', seed=None,
                 method=f'library_nearest_s{s}_a{n}_diverse')
            for s in a['source_counts'] for n in a['alternative_counts']]
    s, n = spec['source_count'], spec['alternatives']
    rows.append(dict(source_count=s, alternative_count=n, retrieval='nearest', selector='prefix', seed=None,
                     method=f'library_nearest_s{s}_a{n}_prefix'))
    rows.extend(dict(source_count=s, alternative_count=n, retrieval='random', selector='diverse', seed=seed,
                     method=f'library_random_s{s}_a{n}_diverse') for seed in a['random_source_seeds'])
    return rows


def plan(spec, library):
    if set(spec) != set(default_spec()) or spec['schema'] != 'library_mechanism_v1':
        raise ValueError('Unknown mechanism schema/settings')
    a = spec['ablation']
    if set(a) != set(default_spec()['ablation']) or a['fill_policy'] != 'geometry_union':
        raise ValueError('Only explicit geometry-union fill is supported')
    for key in ('source_counts', 'alternative_counts', 'random_source_seeds'):
        values = a[key]
        low, high = (0, 2**32 - 1) if key.endswith('seeds') else (1, 10 if key == 'source_counts' else 64)
        if (not values or any(type(v) is not int or not low <= v <= high for v in values)
                or values != sorted(set(values))):
            raise ValueError('Invalid ablation axis: ' + key)
    if spec['source_count'] not in a['source_counts'] or spec['alternatives'] not in a['alternative_counts']:
        raise ValueError('Prefix/random-source anchor must appear in factorial axes')
    source_spec, _ = budget.read_library(library)
    validate_split(spec['split'], source_spec['sources'], spec['targets'])
    for target in spec['targets']:
        retrieve(source_spec['sources'], target, max(a['source_counts']))
    compatible = {key: spec[key] for key in budget.default_spec()}
    compatible.update(schema='library_lowk_cost_v1', source_count=max(a['source_counts']),
                      alternatives=max(a['alternative_counts']))
    checked = budget.plan(compatible, library)
    if any(d['candidates'] < spec['query_budget'] for d in checked['domains']):
        raise ValueError('Q exceeds target domain; do not silently shrink the fixed budget')
    arms = len(variants(spec)) + 2 + len(spec['random_seeds'])
    refs = checked['reference_rows']
    opt = spec['optimization']
    checked.update(expected_rows=refs * arms, arms_per_profile=arms, fixed_q=spec['query_budget'],
        reserved_geometry_count=len(spec['split']['reserved']),
        solver_cap_sum_hours=(refs * arms * opt['arm_seconds'] + refs * opt['reference_seconds'] +
                              refs * opt['surrogate_seconds']) / 3600.,
        scope='Development-only fixed-Q HYBRID ablation. Reserved descriptors are not built or evaluated. '
              'Q counts poses, not rays/XYZ sites or wall time. No RL/detection or new source solve.')
    return checked


def select_sources(sources, target, count, retrieval, seed):
    eligible = [d for d in sources if d['track'] == target['track'] and
                (d['track'] == 'B' or d['topology'] == target['topology'])]
    if retrieval == 'nearest':
        return retrieve(sources, target, count)
    if retrieval != 'random' or seed is None or len(eligible) < count:
        raise ValueError('Invalid random-source selection')
    eligible.sort(key=lambda d: d['name'])
    ids = np.random.default_rng(seed).permutation(len(eligible))[:count]
    return [eligible[i] for i in ids]


def source_solutions(data, limit):
    """Incumbent is included explicitly; alternatives are feasible, not exact top-N."""
    rows = [row for profile in data['profiles'].values() for row in profile['frontier']]
    layouts = []
    for row in rows:
        unique, seen = [], set()
        candidates = [row['selected_ids']] + [a['selected_ids'] for a in row['alternatives']]
        for ids in candidates:
            if ids is None:
                continue
            key = tuple(sorted(ids))
            if key not in seen:
                unique.append(ids)
                seen.add(key)
        layouts.append(unique[:limit])
    return [choices[rank] for rank in range(limit) for choices in layouts if rank < len(choices)]


def transfer_pool(root, sources, problem, variant):
    start = time.perf_counter()
    selected = select_sources(sources, problem['descriptor'], variant['source_count'],
                              variant['retrieval'], variant['seed'])
    entries = []
    for d in selected:
        folder = root / 'library' / d['name']
        result = json.loads((folder / 'result.json').read_text())
        if core.digest(folder / 'layouts.json') != result['artifact_hashes']['layouts.json']:
            raise RuntimeError('Source layouts changed during planning')
        data = json.loads((folder / 'layouts.json').read_text())
        solutions = source_solutions(data, variant['alternative_count'])
        order, records = transfer_ids(data, solutions, problem)
        entries.append(dict(source=d['name'], order=order, records=records, layout_count=len(solutions)))
    order = list(dict.fromkeys(cid for rank in range(max((len(e['order']) for e in entries), default=0))
                for e in entries for cid in e['order'][rank:rank+1]))
    return dict(order=order, sources=entries, planning_sec=time.perf_counter() - start)


def fill_shortlist(library_ids, geometry_order, q, candidate_count):
    if q < 1 or q > candidate_count or len(set(library_ids)) != len(library_ids):
        raise ValueError('Invalid fixed query budget or duplicate library candidates')
    if any(type(i) is not int or not 0 <= i < candidate_count for i in library_ids + geometry_order):
        raise ValueError('Invalid candidate ID')
    chosen = library_ids[:q]
    seen = set(chosen)
    fill = []
    for cid in geometry_order:
        if len(chosen) + len(fill) == q:
            break
        if cid not in seen:
            fill.append(cid)
            seen.add(cid)
    if len(chosen) + len(fill) != q:
        raise ValueError('Insufficient geometry fill; Q must not shrink')
    return chosen, fill


def make_plans(root, directory, spec, problem, costs):
    source_spec = json.loads((root / 'library_spec.json').read_text())
    q, count = spec['query_budget'], len(problem['poses'])
    if q > count:
        raise ValueError('Q exceeds target candidates')
    start = time.perf_counter()
    optical = core._optical_masks(problem['poses'].astype(np.float32), problem['targets'].astype(np.float32),
                                  core._profile_values(problem['profile']))
    optical_sec = time.perf_counter() - start
    start = time.perf_counter()
    union = core.shortlist(optical, problem['poses'], q)
    union_rank_sec = time.perf_counter() - start
    templates = []
    transfers = []
    for v in variants(spec):
        pool = transfer_pool(root, source_spec['sources'], problem, v)
        start = time.perf_counter()
        ids = pool['order']
        selected = ids[:q]
        if v['selector'] == 'diverse' and len(ids) > q:
            selected = [ids[i] for i in core.shortlist(optical[ids], problem['poses'][ids], q, double=True)]
        chosen, fill = fill_shortlist(selected, union, q, count)
        planning_sec = pool['planning_sec'] + time.perf_counter() - start
        if fill or (v['selector'] == 'diverse' and len(ids) > q):
            planning_sec += optical_sec
        if fill:
            planning_sec += union_rank_sec
        metadata = dict(source_count=v['source_count'], alternative_count=v['alternative_count'],
            retrieval=v['retrieval'], selector=v['selector'], library_query_count=len(chosen),
            geometry_fill_count=len(fill), empty_library_fallback=not bool(chosen),
            query_policy='fixed_q_explicit_geometry_fill')
        templates.append(dict(method=v['method'], seed=v['seed'], selected_ids=chosen + fill,
            library_ids=chosen, geometry_fill_ids=fill, pool_count=len(ids), planning_sec=planning_sec,
            comparison_metadata=metadata))
        transfers.append(dict(variant=v, **pool))
    core.save(directory / 'transfers.json', transfers)
    plans = []
    for name, profile in spec['profiles'].items():
        start = time.perf_counter()
        frontier = budget.optimize(problem, optical, costs, spec, 'surrogate', profile,
            spec['target_k'][problem['descriptor']['name']], directory / ('surrogate_' + name), alternatives=6)
        proposed = list(dict.fromkeys(cid for rank in range(6) for row in frontier['frontier']
                        for alt in row['alternatives'][rank:rank+1] for cid in alt['selected_ids']))
        portfolio, extra = fill_shortlist(proposed, union, q, count)
        controls = [('geometry_union', None, union, optical_sec + union_rank_sec),
                    ('geometry_portfolio', None, portfolio + extra,
                     optical_sec + union_rank_sec + time.perf_counter() - start)]
        for seed in spec['random_seeds']:
            start = time.perf_counter()
            ids = np.random.default_rng(seed).permutation(count)[:q].tolist()
            controls.append(('random', seed, ids, time.perf_counter() - start))
        methods = list(templates)
        for method, seed, ids, planning_sec in controls:
            methods.append(dict(method=method, seed=seed, selected_ids=ids, library_ids=[], geometry_fill_ids=[],
                pool_count=0, planning_sec=planning_sec,
                comparison_metadata=dict(source_count=0, alternative_count=0, retrieval='none',
                    selector=method, library_query_count=0, geometry_fill_count=0,
                    empty_library_fallback=False, query_policy='fixed_q_control')))
        for p in methods:
            ids = p['selected_ids']
            if len(ids) != q or len(set(ids)) != q:
                raise RuntimeError('Fixed-Q invariant failed')
            metadata = dict(p['comparison_metadata'], query_ids_sha256=hashlib.sha256(
                json.dumps(sorted(ids)).encode('ascii')).hexdigest(),
                queried_xyz_count=len(np.unique(problem['poses'][ids, :3], axis=0)))
            plans.append(dict(p, profile=name, requested_q=q, actual_q=q, selected_ids=sorted(ids),
                comparison_metadata=metadata,
                arm_id=name + '_' + p['method'] + (f"_s{p['seed']}" if p['seed'] is not None else '')))
    core.save(directory / 'query_plans.json', plans)
    return plans


def aggregate_pairs(root):
    """Seed means first; missing/infeasible conditions never become zero coverage."""
    with (root / 'comparison.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    spec = json.loads((root / 'spec.json').read_text())
    arms = [(v['method'], '' if v['seed'] is None else str(v['seed'])) for v in variants(spec)]
    arms += [('geometry_union', ''), ('geometry_portfolio', '')]
    arms += [('random', str(seed)) for seed in spec['random_seeds']]
    expected = {(d['name'] + '_' + level, profile, str(k), method, seed)
                for d in spec['targets'] for level in spec['scene_levels'] for profile in spec['profiles']
                for k in spec['target_k'][d['name']] for method, seed in arms}
    observed = [(r['case'], r['profile'], r['budget_k'], r['method'], r['seed']) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError('Incomplete or duplicate ablation conditions/seeds')
    for row in rows:
        if int(row['actual_q']) != spec['query_budget'] or int(row['requested_q']) != spec['query_budget']:
            raise RuntimeError('Unequal query budgets in ablation report')
        if row['status'] == 'feasible':
            value = float(row['coverage_pct'])
            if not np.isfinite(value) or not 0 <= value <= 100:
                raise RuntimeError('Invalid feasible coverage')
        elif row['status'] not in ('infeasible', 'no_feasible_found') or row.get('coverage_pct', '') != '':
            raise RuntimeError('Missing solutions must remain missing')
    conditions = {}
    for row in rows:
        key = row['case'], row['profile'], row['budget_k']
        conditions.setdefault(key, {}).setdefault(row['method'], []).append(row)
    anchor = f"library_nearest_s{spec['source_count']}_a{spec['alternatives']}_diverse"
    contrasts = []
    a = spec['ablation']
    for n in a['alternative_counts']:
        for lo, hi in zip(a['source_counts'], a['source_counts'][1:]):
            contrasts.append((f'sources_{hi}_vs_{lo}_a{n}', f'library_nearest_s{hi}_a{n}_diverse',
                              f'library_nearest_s{lo}_a{n}_diverse'))
    for s in a['source_counts']:
        for lo, hi in zip(a['alternative_counts'], a['alternative_counts'][1:]):
            contrasts.append((f'alternatives_{hi}_vs_{lo}_s{s}', f'library_nearest_s{s}_a{hi}_diverse',
                              f'library_nearest_s{s}_a{lo}_diverse'))
    for method in (anchor.replace('_diverse', '_prefix'), anchor.replace('_nearest_', '_random_'),
                   'geometry_union', 'geometry_portfolio', 'random'):
        contrasts.append(('anchor_vs_' + method, anchor, method))
    paired = []
    for label, left, right in contrasts:
        for (case, profile, k), methods in conditions.items():
            lhs, rhs = methods[left], methods[right]
            ok_l = all(r['status'] == 'feasible' and r.get('coverage_pct', '') != '' for r in lhs)
            ok_r = all(r['status'] == 'feasible' and r.get('coverage_pct', '') != '' for r in rhs)
            paired.append(dict(contrast=label, case=case, profile=profile, budget_k=k,
                left=left, right=right, left_all_seeds_feasible=ok_l, right_all_seeds_feasible=ok_r,
                left_seed_count=len(lhs), right_seed_count=len(rhs),
                identical_query_set_all_trials=len({r['query_ids_sha256'] for r in lhs + rhs}) == 1,
                coverage_difference_pp=float(np.mean([float(r['coverage_pct']) for r in lhs]) -
                                             np.mean([float(r['coverage_pct']) for r in rhs])) if ok_l and ok_r else None))
    core.write_csv(root / 'paired_effects.csv', paired)
    summaries = []
    for label, _, _ in contrasts:
        group = [r for r in paired if r['contrast'] == label]
        valid = [r for r in group if r['coverage_difference_pp'] is not None]
        by_map = {}
        for row in valid:
            name = row['case'].rsplit('_', 1)[0]
            by_map.setdefault(name, []).append(row['coverage_difference_pp'])
        summaries.append(dict(contrast=label, conditions=len(group), paired_feasible_conditions=len(valid),
            geometries_with_feasible_pairs=len(by_map),
            identical_query_set_conditions=sum(r['identical_query_set_all_trials'] for r in group),
            map_balanced_difference_pp=float(np.mean([np.mean(v) for v in by_map.values()])) if by_map else None,
            wins=sum(r['coverage_difference_pp'] > 1e-7 for r in valid),
            ties=sum(abs(r['coverage_difference_pp']) <= 1e-7 for r in valid),
            losses=sum(r['coverage_difference_pp'] < -1e-7 for r in valid)))
    core.write_csv(root / 'ablation_summary.csv', summaries)
    core.save(root / 'report_status.json', dict(status='complete', rows=len(rows),
        scope='Descriptive development ablations; map means use common feasible conditions only. '
              'K/profile/scene rows and random seeds are not independent road samples. No significance test.'))


def worker(root, job):
    if BASE != root / 'source':
        raise ValueError('Worker must run frozen code')
    manifest, spec = core.verify(root)
    budget.verify_library(root)
    source_spec = json.loads((root / 'library_spec.json').read_text())
    validate_split(spec['split'], source_spec['sources'], spec['targets'])
    item = next(j for j in manifest['jobs'] if j['id'] == job)
    if item['descriptor'] not in spec['targets'] or item['scene'] not in spec['scene_levels']:
        raise ValueError('Job not in frozen development protocol')
    result = budget.case_job(root, spec, item['descriptor'], item['scene'], planner=make_plans)
    core.verify(root)
    budget.verify_library(root)
    result['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    core.save(root / 'job_status' / (job + '.json'), result)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=budget.DEFAULT_LIBRARY)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/library_mechanism_20261127'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--orchestrate', type=Path)
    parser.add_argument('--job')
    args = parser.parse_args(argv)
    if args.worker and args.orchestrate or args.smoke and args.spec:
        parser.error('Choose one internal role and one configuration source')
    if (args.worker or args.orchestrate or args.detach) and not args.execute:
        parser.error('Internal execution/detach requires --execute')
    if args.worker:
        worker(args.worker.resolve(), args.job)
        return
    if args.orchestrate:
        root = args.orchestrate.resolve()
        if BASE != root / 'source':
            parser.error('Supervisor must run frozen code')
        budget.verify_library(root)
        core.orchestrate(root, entrypoint=Path(__file__).name)
        try:
            aggregate_pairs(root)
        except Exception as exc:
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest.update(status='failed', report_error=repr(exc))
            core.save(root / 'manifest.json', manifest)
            raise
        return
    spec = smoke_spec() if args.smoke else json.loads(args.spec.read_text()) if args.spec else default_spec()
    if args.execute:
        budget.execute(spec, args.library.resolve(), args.output, args.detach, plan_fn=plan,
            entrypoint=Path(__file__).name,
            snapshot_files=('LIBRARY_MECHANISM_PROTOCOL.md', 'configs/library_mechanism_20261127.json',
                            'tests/test_library_mechanism.py'))
    else:
        print(json.dumps(plan(spec, args.library.resolve()), indent=2))


if __name__ == '__main__':
    main()

"""Expanded A/B source catalogs, larger query pools and K=1..8 evaluation.

New opt-in artifact format. Default invocation only plans; --execute freezes
inputs/code and runs sequentially. Old libraries and exact-enumeration studies
are neither modified nor consumed. No policy training is performed.
"""
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import library_components as components
from library_component_transfer import _module_metadata, _transfer
from library_occlusion_study import installation_spec, screen_spec
from library_topology import _domain as topology_domain
from prototype_offline import explicit_scenario
from prototype_segments import installation_segments, segment_coordinates, segment_to_target
from prototype_transfer import project_poses
from src.core.library_installation import allowed_candidates
from src.core.library_transfer_evaluation import Work
from src.core.scene_occlusion import with_occlusion
from src.core.visibility import MeshVisibility
from src.environment import ExperimentConfig, get_lidar_profile
from src.methods.library_scalable import coverage_metrics, geometry_plan, optimize_frontier


BASE = Path(__file__).resolve().parent
SCHEMA = 'library_expanded_study_v1'


def save(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    """Average random seeds within a case before averaging maps/conditions."""
    metrics = ['coverage_pct', 'overlap_pct', 'num_selected', 'actual_q', 'loss_lower_pp', 'loss_upper_pp',
               'online_through_k_sec', 'online_all_k_sec', 'reference_coverage_pct',
               'reference_online_all_k_sec']
    cells = defaultdict(list)
    for row in rows:
        key = (row['track'], row['source_count'], row['requested_q'], row['budget_k'], row['method'], row['case'])
        cells[key].append(row)
    groups = defaultdict(list)
    for key, trials in cells.items():
        groups[key[:-1]].append({metric: float(np.mean([r[metric] for r in trials])) for metric in metrics})
    output = []
    for key, values in sorted(groups.items()):
        output.append(dict(zip(['track', 'source_count', 'requested_q', 'budget_k', 'method'], key),
            condition_count=len(values), **{metric + '_mean': float(np.mean([v[metric] for v in values]))
                                          for metric in metrics}))
    return output


def summary_curves(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in rows:
        groups[(row['track'], row['source_count'], row['requested_q'])].append(row)
    for (track, count, q), values in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for method in sorted({r['method'] for r in values}):
            subset = sorted((r for r in values if r['method'] == method), key=lambda r: r['budget_k'])
            for ax, field, label in zip(axes.flat,
                    ('coverage_pct_mean', 'overlap_pct_mean', 'online_through_k_sec_mean', 'loss_upper_pp_mean'),
                    ('Coverage %', 'Targets seen by >=2 sensors %', 'Time through K (s)', 'Certified loss upper bound (pp)')):
                ax.plot([r['budget_k'] for r in subset], [r[field] for r in subset], marker='.', label=method)
                ax.set(xlabel='Sensor budget K', ylabel=label)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'Track {track}; {count} source(s); requested q={q}; actual q recorded separately')
        fig.savefig(root / f'comparison_{track}_s{count}_q{q}.png', dpi=130)
        plt.close(fig)


def default_spec():
    return dict(schema=SCHEMA, k_values=list(range(1, 9)), query_budgets=[32, 64, 128],
        source_counts=[1, 3], alternatives_per_k=12, random_seeds=[0, 1, 2],
        sampling=dict(xy_step_m=5., station_step_m=5., relative_yaws_deg=[-30., 0., 30.],
                      heights_m=[3.], pitches_deg=[-10.], target_b_step_m=2.),
        source_a=dict(topologies=['T_JUNCTION', 'CROSS'], widths_m=[8., 10., 12.], center_m=[50., 50.]),
        source_b=dict(widths_m=[8., 12.], turns_deg=[0., 30., 60.], length_m=80., bend_length_m=32.),
        targets=[dict(name='a_t_w9', track='A', topology='T_JUNCTION', width_m=9., center_m=[40., 54.]),
                 dict(name='a_cross_w11', track='A', topology='CROSS', width_m=11., center_m=[60., 46.]),
                 dict(name='b_l160_w8_t15', track='B', width_m=8., length_m=160., turn_deg=15., modules=2),
                 dict(name='b_l240_w12_t45', track='B', width_m=12., length_m=240., turn_deg=45., modules=3)],
        installation_levels=['open', 'restricted'], occlusion_levels=['clear', 'occluded'],
        screens=dict(thickness_m=.2, height_m=4.5, road_width_fraction=.7,
                     a_branch_station_m=20., b_station_fractions=[.25, .75]),
        installation=dict(band_length_m=20., a_east_station_m=22., b_station_fraction=.25),
        optimization=dict(source_seconds=1800., reference_seconds=1800., arm_seconds=300.,
            surrogate_seconds=10., swap_seconds=.1, certificate_gap=.01,
            stop_policy=dict(minimum_seconds=300., stagnation_seconds=120., relative_gap=.01,
                             interval_seconds=10.)),
        worker_timeout_sec=172800)


def source_specs(spec):
    sources = []
    for topology in spec['source_a']['topologies']:
        for width in spec['source_a']['widths_m']:
            sources.append(dict(name=f'a_{topology.lower()}_w{width:g}', track='A', topology=topology,
                width_m=width, center_m=spec['source_a']['center_m']))
    for width in spec['source_b']['widths_m']:
        for turn in spec['source_b']['turns_deg']:
            sources.append(dict(name=f'b_w{width:g}_t{turn:g}', track='B', width_m=width,
                length_m=spec['source_b']['length_m'], turn_deg=turn,
                bend_length_m=spec['source_b']['bend_length_m']))
    return sources


def smoke_spec():
    spec = default_spec()
    spec['source_a'].update(topologies=['T_JUNCTION'], widths_m=[8.])
    spec['source_b'].update(widths_m=[8.], turns_deg=[0.])
    spec['targets'] = [spec['targets'][0], spec['targets'][2]]
    spec['sampling'].update(xy_step_m=20., station_step_m=20., relative_yaws_deg=[0.])
    spec.update(query_budgets=[8, 16, 32], source_counts=[1], alternatives_per_k=3, random_seeds=[0, 1])
    spec['optimization'].update(source_seconds=1., reference_seconds=1., arm_seconds=.2,
        surrogate_seconds=.1, swap_seconds=.01,
        stop_policy=dict(minimum_seconds=.1, stagnation_seconds=.1, relative_gap=.01, interval_seconds=.05))
    spec['worker_timeout_sec'] = 1800
    return spec


def topology_config(spec):
    sampling = spec['sampling']
    yaws, heights, pitches = [sampling[key] for key in ('relative_yaws_deg', 'heights_m', 'pitches_deg')]
    def step(values):
        if not values or values != sorted(set(values)) or not np.isfinite(values).all():
            raise ValueError('Sampling axes must be finite, increasing and unique')
        if len(values) > 2 and not np.allclose(np.diff(values), values[1] - values[0]):
            raise ValueError('Track A requires uniformly spaced axes')
        return values[1] - values[0] if len(values) > 1 else 1.
    raw = dict(candidate=dict(xy_domain='bin_representatives', xy_step_m=sampling['xy_step_m'],
        z_min=min(heights), z_max=max(heights), z_step=step(heights),
        yaw_axis_min_deg=90 + min(yaws), yaw_axis_max_deg=90 + max(yaws), yaw_step_deg=step(yaws),
        pitch_min_deg=min(pitches), pitch_max_deg=max(pitches), pitch_step_deg=step(pitches),
        max_orientations_per_position=0), target=dict(coverage_z_slices=[1.5], road_detection=1),
        lidar_profiles=dict(VELODYNE64_R30=dict(h_fov_deg=360., v_fov_deg=26.9, max_range_m=30.)))
    return ExperimentConfig(raw=raw)


def make_domain(spec, descriptor):
    if descriptor.get('track') not in ('A', 'B'):
        raise ValueError('Track must be A or B')
    if descriptor['track'] == 'A':
        config = topology_config(spec)
        scenario = explicit_scenario(dict(id=450001, topology=descriptor['topology'],
            parent_scenario_id={'T_JUNCTION': 1, 'CROSS': 2}[descriptor['topology']],
            provenance='synthetic_junction_offset',
            road_width=descriptor['width_m'], num_lanes=2,
            geometry=dict(junction_x_m=descriptor['center_m'][0], junction_y_m=descriptor['center_m'][1])))
        geometry, poses, targets = topology_domain(config, scenario)
        relative = segment_coordinates(poses, geometry, installation_segments(geometry))
        archive = dict(poses=poses, targets=targets, canonical=relative)
    else:
        config = components.default_spec()
        sampling = spec['sampling']
        config['geometry'].update(length_m=descriptor['length_m'], road_width_m=descriptor['width_m'],
            turn_deg=descriptor['turn_deg'], bend_length_m=descriptor.get('bend_length_m', 32.), mesh_step_m=2.)
        config['candidate'] = dict(station_step_m=sampling['station_step_m'],
            heights_m=sampling['heights_m'], relative_yaws_deg=sampling['relative_yaws_deg'],
            pitches_deg=sampling['pitches_deg'])
        config['target'] = dict(step_m=sampling['target_b_step_m'], heights_m=[1.5])
        config['max_k'] = max(spec['k_values'])
        config['alternatives_per_k'] = spec['alternatives_per_k']
        config = components.validate_spec(config)
        geometry, archive = components._domain(config)
    profile = get_lidar_profile('VELODYNE64_R30', topology_config(spec))
    return dict(geometry=geometry, config=config, profile=profile, context=dict(track=descriptor['track']),
                **archive)


def _distance(source, target):
    if source['track'] != target['track']:
        return float('inf')
    width = abs(source['width_m'] - target['width_m']) / 4.
    if target['track'] == 'A':
        if source['topology'] != target['topology']:
            return float('inf')
        return width + float(np.linalg.norm(np.asarray(source['center_m']) - target['center_m'])) / 100.
    return (width + abs(source['turn_deg'] - target['turn_deg']) / 60.
            + abs(source['length_m'] - target['length_m'] / target['modules']) / 100.)


def select_sources(sources, target, count):
    ordered = sorted(sources, key=lambda row: (_distance(row, target), row['name']))
    selected = [row for row in ordered if np.isfinite(_distance(row, target))][:count]
    if len(selected) != count:
        raise ValueError('Insufficient same-track/topology sources for requested library size')
    return selected


def plan(spec):
    if not isinstance(spec, dict) or set(spec) != set(default_spec()) or spec.get('schema') != SCHEMA:
        raise ValueError('Invalid expansion schema or keys')
    for key, low, high in [('k_values', 1, 8), ('query_budgets', 1, 2000),
                           ('source_counts', 1, 20), ('random_seeds', 0, 2**32 - 1)]:
        values = spec[key]
        if not values or values != sorted(set(values)) or any(type(v) is not int or not low <= v <= high for v in values):
            raise ValueError(f'Invalid {key}')
    if type(spec['alternatives_per_k']) is not int or not 1 <= spec['alternatives_per_k'] <= 64:
        raise ValueError('Invalid alternatives per K')
    if (not spec['installation_levels'] or not spec['occlusion_levels']
            or len(set(spec['installation_levels'])) != len(spec['installation_levels'])
            or len(set(spec['occlusion_levels'])) != len(spec['occlusion_levels'])
            or set(spec['installation_levels']) - {'open', 'restricted'}
            or set(spec['occlusion_levels']) - {'clear', 'occluded'}):
        raise ValueError('Invalid condition levels')
    opt = spec['optimization']
    if set(opt) != set(default_spec()['optimization']):
        raise ValueError('Explicit optimization settings required')
    if any(type(opt[key]) not in (int, float) or not np.isfinite(opt[key]) or opt[key] <= 0
           for key in ('source_seconds', 'reference_seconds', 'arm_seconds', 'surrogate_seconds')):
        raise ValueError('Invalid solver time')
    policy = opt['stop_policy']
    if (set(policy) != set(default_spec()['optimization']['stop_policy'])
            or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in policy.values())
            or policy['interval_seconds'] <= 0 or not 0 < opt['certificate_gap'] <= 1
            or policy['relative_gap'] != opt['certificate_gap']
            or policy['minimum_seconds'] > min(opt['source_seconds'], opt['reference_seconds'])
            or not np.isfinite(opt['swap_seconds']) or opt['swap_seconds'] < 0):
        raise ValueError('Invalid certificate/stop policy; plateau alone is not a certificate')
    if type(spec['worker_timeout_sec']) is not int or spec['worker_timeout_sec'] < 1:
        raise ValueError('Positive hard worker timeout required')
    sources = source_specs(spec)
    if not sources or len(sources) > 100 or not spec['targets'] or len(spec['targets']) > 100:
        raise ValueError('Source/target study-size guard exceeded')
    names = [row['name'] for row in [*sources, *spec['targets']]]
    if len(set(names)) != len(names) or any(not name or not all(c.isalnum() or c in '_-' for c in name) for name in names):
        raise ValueError('Names must be distinct safe identifiers')
    source_geometry = []
    domains = []
    for row in [*sources, *spec['targets']]:
        domain = make_domain(spec, row)
        geometry = domain['geometry']
        identity = (row['track'], json.dumps(asdict(geometry) if row['track'] == 'B' else
            dict(topology=row['topology'], width=row['width_m'], center=row['center_m']), sort_keys=True))
        if row in sources:
            if identity in source_geometry:
                raise ValueError('Duplicate source geometry')
            source_geometry.append(identity)
        else:
            if identity in source_geometry:
                raise ValueError('Target geometry duplicates a source')
            select_sources(sources, row, max(spec['source_counts']))
            installation_spec(domain, spec['installation'])
            with_occlusion(domain, screen_spec(domain, spec['screens']))
        domains.append(dict(name=row['name'], track=row['track'], candidate_count=len(domain['poses']),
                            target_count=len(domain['targets'])))
    cases = len(spec['targets']) * len(spec['installation_levels']) * len(spec['occlusion_levels'])
    arms = 4 + len(spec['random_seeds'])
    settings = len(spec['source_counts']) * len(spec['query_budgets'])
    return dict(schema=SCHEMA, execute=False, sources=len(sources), target_cases=cases,
        k_values=spec['k_values'], requested_query_budgets=spec['query_budgets'],
        source_counts=spec['source_counts'], expected_comparison_rows=cases * settings * arms * len(spec['k_values']),
        domains=domains, solver_call_time_caps_sec=opt,
        solver_time_cap_sum_sec=len(spec['k_values']) * (len(sources) * opt['source_seconds']
            + cases * (opt['reference_seconds'] + settings * (arms * opt['arm_seconds'] + opt['surrogate_seconds']))),
        timing_note='Cap sum is not a runtime estimate; analytic certificates and early solver proofs can finish sooner. '
                    'Model construction, mapping, raycasts, validation and export are outside MILP time caps.',
        scope='Synthetic development extension; source size, query budget and K vary explicitly. No RL.')


def _optimize(domain, masks, spec, role, alternatives=1):
    opt = spec['optimization']
    return optimize_frontier(domain['poses'], masks, domain['targets'], spec['k_values'],
        seconds=opt[f'{role}_seconds'], alternatives=alternatives, swap_seconds=opt['swap_seconds'],
        stop_policy=opt['stop_policy'] if role in ('source', 'reference') else
            dict(opt['stop_policy'], minimum_seconds=min(opt['arm_seconds'], opt['stop_policy']['minimum_seconds'])),
        certificate_gap=opt['certificate_gap'])


def query(mesh, domain, ids):
    work = Work()
    started = time.perf_counter()
    provider = MeshVisibility(mesh, domain['targets'], domain['profile'], work=work)
    masks = np.asarray([provider.mask(domain['poses'][i]) for i in ids], dtype=bool)
    masks = masks.reshape(len(ids), len(domain['targets']))
    return masks, time.perf_counter() - started, dict(work.counts)


def save_curves(root, frontier):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    traces = []
    for row in frontier['frontier']:
        trace = row['solver'].get('quality_time_trace') or [dict(solver_elapsed_sec=0.,
            covered_count=row['covered_count'], coverage_pct=row['coverage_pct'],
            upper_bound_count=row['upper_bound_count'],
            upper_bound_coverage_pct=100 * row['upper_bound_count'] / frontier['target_count'],
            certified_gap=row['certified_gap'], event='analytic_certificate')]
        traces.extend(dict(budget_k=row['budget_k'], **point) for point in trace)
    write_csv(root / 'quality_time.csv', traces)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for k in frontier['k_values']:
        points = [p for p in traces if p['budget_k'] == k]
        x = [p['solver_elapsed_sec'] for p in points]
        line, = axes[0].step(x, [p['coverage_pct'] for p in points], where='post', marker='.', label=f'K={k}')
        axes[0].step(x, [p['upper_bound_coverage_pct'] for p in points], where='post',
                     color=line.get_color(), linestyle='--', marker='.', alpha=.7)
        axes[1].step(x, [100 * p['certified_gap'] if p['certified_gap'] is not None else np.nan for p in points],
                     where='post', marker='.', color=line.get_color(), label=f'K={k}')
    axes[0].set(ylabel='Coverage %', title='Feasible coverage (solid) and certified upper bound (dashed)')
    axes[0].legend(ncol=4)
    axes[1].axhline(100 * frontier['frontier'][0]['certificate_gap_threshold'], color='black', linestyle=':')
    axes[1].set(xlabel='Solver time (s), excludes preprocessing', ylabel='Certified relative gap %')
    fig.savefig(root / 'quality_time.png', dpi=130)
    plt.close(fig)


def build_source(root, spec, descriptor):
    begin = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    domain = make_domain(spec, descriptor)
    mesh = domain['geometry'].create_environment_mesh()
    preparation = time.perf_counter() - begin
    masks, visibility, work = query(mesh, domain, list(range(len(domain['poses']))))
    frontier = _optimize(domain, masks, spec, 'source', spec['alternatives_per_k'])
    for row in frontier['frontier']:
        for solution in [row, *row['alternatives']]:
            solution['canonical'] = ([domain['canonical'][i] for i in solution['selected_ids']]
                if descriptor['track'] == 'A' else domain['canonical'][solution['selected_ids']].tolist())
    selected = sorted({i for row in frontier['frontier'] for sol in row['alternatives'] for i in sol['selected_ids']})
    checked, validation, validation_work = query(mesh, domain, selected)
    if not np.array_equal(checked, masks[selected]):
        raise RuntimeError('Independent source visibility validation failed')
    np.savez_compressed(root / 'candidates.npz', poses=domain['poses'], targets=domain['targets'], masks=masks)
    save(root / 'frontier.json', frontier)
    save_curves(root, frontier)
    result = dict(descriptor=descriptor, schema='expanded_library_entry_v1', status='complete',
        k_values=spec['k_values'], sampling=spec['sampling'], profile=asdict(domain['profile']),
        preparation_sec=preparation, visibility_sec=visibility,
        selection_sec=frontier['phase_wall_sec']['all_k_selection'], validation_sec=validation,
        wall_sec=time.perf_counter() - begin, work=work, validation_work=validation_work,
        validated_direct_raycast=True, artifact_hashes={name: digest(root / name) for name in
            ('candidates.npz', 'frontier.json', 'quality_time.csv', 'quality_time.png')})
    save(root / 'entry.json', result)
    return result


def transfer_pool(root, spec, target, domain, count):
    """Rank sources without target masks, then round-robin feasible stored modes."""
    begin = time.perf_counter()
    sources = select_sources(source_specs(spec), target, count)
    entries = []
    for descriptor in sources:
        path = root / 'library' / descriptor['name']
        entry = json.loads((path / 'entry.json').read_text())
        for name, expected in entry['artifact_hashes'].items():
            if digest(path / name) != expected:
                raise RuntimeError('Source library changed')
        if (entry['sampling'] != spec['sampling'] or entry['k_values'] != spec['k_values']
                or entry['descriptor'] != descriptor or entry['profile'] != asdict(domain['profile'])
                or entry['status'] != 'complete'):
            raise ValueError('Source contract mismatch')
        entries.append((descriptor, json.loads((path / 'frontier.json').read_text()),
                        make_domain(spec, descriptor)['geometry'] if target['track'] == 'B' else None))
    if target['track'] == 'A':
        segments = installation_segments(domain['geometry'])
        lookup = {tuple(p): i for i, p in enumerate(domain['poses'])}
    else:
        modules = _module_metadata(domain['geometry'], target['modules'],
                                   domain['canonical'], domain['target_coordinates'])
    order, seen, records = [], set(), []
    for rank in range(spec['alternatives_per_k']):
        for k in spec['k_values']:
            for descriptor, frontier, source_geometry in entries:
                row = next(r for r in frontier['frontier'] if r['budget_k'] == k)
                if rank >= len(row['alternatives']):
                    continue
                solution = row['alternatives'][rank]
                record = dict(source=descriptor['name'], budget_k=k, alternative=rank)
                if target['track'] == 'A':
                    try:
                        raw = segment_to_target(solution['canonical'], domain['geometry'], segments)
                        projected = project_poses(raw, domain['geometry'], domain['config'], solution['canonical'])
                        ids = [lookup[tuple(p)] for p in projected]
                        records.append(dict(record, status='complete', selected_ids=ids))
                    except (ValueError, KeyError) as exc:
                        records.append(dict(record, status='unavailable', reason=str(exc)))
                        continue
                else:
                    ids = []
                    for module in modules:
                        mapped = _transfer(solution, k, rank, module, source_geometry, domain['geometry'], domain)
                        records.append(dict(record, **{key: value for key, value in mapped.items() if key not in record}))
                        if mapped['status'] == 'complete':
                            ids.extend(mapped['selected_ids'])
                for cid in ids:
                    if cid not in seen:
                        order.append(cid)
                        seen.add(cid)
    return dict(query_order=order, sources=[s['name'] for s in sources], transfers=records,
                planning_sec=time.perf_counter() - begin,
                selection_rule='geometry distance, then name; alternative rank, K, source, module order; no target coverage')


def evaluate_case(root, case_root, spec, target, installation, occlusion):
    case_begin = time.perf_counter()
    case_root.mkdir(parents=True, exist_ok=False)
    domain = make_domain(spec, target)
    unrestricted = domain
    mask = (allowed_candidates(domain['poses'], installation_spec(domain, spec['installation']))
            if installation == 'restricted' else np.ones(len(domain['poses']), dtype=bool))
    kept = np.flatnonzero(mask)
    remap = {int(old): new for new, old in enumerate(kept)}
    scene = with_occlusion(domain, screen_spec(domain, spec['screens'])) if occlusion == 'occluded' else domain
    mesh = scene['geometry'].create_environment_mesh()
    domain = dict(domain, poses=domain['poses'][kept])
    preparation = time.perf_counter() - case_begin
    plans = []
    for count in spec['source_counts']:
        transferred = transfer_pool(root, spec, target, unrestricted, count)
        order = [remap[i] for i in transferred['query_order'] if i in remap]
        save(case_root / f'transfer_s{count}.json', dict(transferred, eligible_query_order=order))
        for requested in spec['query_budgets']:
            q = min(requested, len(order))
            prefix = f's{count}_q{requested}'
            plans.append(dict(arm_id=prefix + '_library', method='library', seed=None,
                source_count=count, requested_q=requested, q=q, selected_ids=sorted(order[:q]),
                planning_sec=transferred['planning_sec'], library_pool_count=len(order)))
            for method, seed in [(m, None) for m in ('geometry_single', 'geometry_union', 'geometry_portfolio_milp')] + [
                    ('random', s) for s in spec['random_seeds']]:
                proposal = geometry_plan(domain['poses'], domain['targets'], domain['profile'], q,
                    spec['k_values'], method, seconds=spec['optimization']['surrogate_seconds'], seed=seed)
                plans.append(dict(proposal, arm_id=prefix + '_' + method + (f'_{seed}' if seed is not None else ''),
                    source_count=count, requested_q=requested, q=q, library_pool_count=len(order)))
    save(case_root / 'query_plans.json', plans)
    frozen_plans = digest(case_root / 'query_plans.json')
    results = []
    # Physical providers are fresh per arm. All arm decisions precede the reference.
    for plan_row in plans:
        arm_root = case_root / plan_row['arm_id']
        arm_root.mkdir()
        ids = plan_row['selected_ids']
        if len(ids) != plan_row['q'] or ids != sorted(set(ids)) or any(not 0 <= i < len(domain['poses']) for i in ids):
            raise RuntimeError('Invalid or non-matched candidate query plan')
        masks, visibility, work = query(mesh, domain, ids)
        if work.get('direct_pose_evaluations', 0) != len(ids):
            raise RuntimeError('Physical query counter disagrees with selected pool')
        pool = dict(domain, poses=domain['poses'][ids])
        frontier = _optimize(pool, masks, spec, 'arm')
        for row in frontier['frontier']:
            row['full_candidate_ids'] = [ids[i] for i in row['selected_ids']]
        result = dict(plan=plan_row, frontier=frontier, visibility_sec=visibility, work=work,
            preparation_sec=preparation, online_all_k_sec=preparation + plan_row['planning_sec'] + visibility
                + frontier['phase_wall_sec']['all_k_selection'])
        save(arm_root / 'result.json', result)
        np.savez_compressed(arm_root / 'queried.npz', candidate_ids=ids, masks=masks)
        save_curves(arm_root, frontier)
        results.append(result)
    ref_root = case_root / 'reference'
    ref_root.mkdir()
    full_masks, full_visibility, full_work = query(mesh, domain, list(range(len(domain['poses']))))
    reference = _optimize(domain, full_masks, spec, 'reference')
    save(ref_root / 'frontier.json', reference)
    save_curves(ref_root, reference)
    np.savez_compressed(case_root / 'candidates.npz', poses=domain['poses'], targets=domain['targets'],
                        masks=full_masks, original_candidate_ids=kept)
    for result in results:
        ids = result['plan']['selected_ids']
        with np.load(case_root / result['plan']['arm_id'] / 'queried.npz', allow_pickle=False) as data:
            if not np.array_equal(data['masks'], full_masks[ids]):
                raise RuntimeError('Arm/reference masks disagree')
    selected = sorted({i for r in results for row in r['frontier']['frontier'] for i in row['full_candidate_ids']}
                      | {i for row in reference['frontier'] for i in row['selected_ids']})
    checked, validation, validation_work = query(mesh, domain, selected)
    if not np.array_equal(checked, full_masks[selected]) or digest(case_root / 'query_plans.json') != frozen_plans:
        raise RuntimeError('Frozen plan/independent raycast validation failed')
    rows = []
    for result in results:
        p = result['plan']
        cumulative = 0.
        for row, ref in zip(result['frontier']['frontier'], reference['frontier']):
            metrics = coverage_metrics(full_masks, row['full_candidate_ids'])
            if metrics != {key: row[key] for key in metrics}:
                raise RuntimeError('Saved union/overlap metrics disagree with density')
            if row['covered_count'] > ref['upper_bound_count'] + 1e-5:
                raise RuntimeError('Online feasible value exceeds reference bound')
            cumulative += row['selection_sec']
            n = len(domain['targets'])
            rows.append(dict(case=case_root.name, track=target['track'], installation=installation,
                occlusion=occlusion, method=p['method'], seed=p['seed'], source_count=p['source_count'],
                requested_q=p['requested_q'], actual_q=p['q'], pool_count=p['library_pool_count'],
                candidate_count=len(domain['poses']), budget_k=row['budget_k'], **metrics,
                certified_gap=row['certified_gap'], near_optimal_certified=row['near_optimal_certified'],
                certificate_domain='queried_pool_only', reference_coverage_pct=ref['coverage_pct'],
                reference_upper_coverage_pct=100 * ref['upper_bound_count'] / n,
                reference_certified_gap=ref['certified_gap'],
                full_domain_near_optimal_certified=ref['upper_bound_count'] - row['covered_count']
                    <= spec['optimization']['certificate_gap'] * row['covered_count'] + 1e-5,
                reference_num_selected=ref['num_selected'], reference_overlap_pct=ref['overlap_pct'],
                reference_minus_method_pp=ref['coverage_pct'] - row['coverage_pct'],
                loss_lower_pp=max(0., ref['coverage_pct'] - row['coverage_pct']),
                loss_upper_pp=max(0., 100 * ref['upper_bound_count'] / n - row['coverage_pct']),
                preparation_sec=preparation, planning_sec=p['planning_sec'], visibility_sec=result['visibility_sec'],
                k_selection_sec=row['selection_sec'], k_solver_sec=row['solver']['solver_sec'],
                reference_k_selection_sec=ref['selection_sec'], reference_k_solver_sec=ref['solver']['solver_sec'],
                reference_visibility_sec=full_visibility,
                reference_online_all_k_sec=preparation + full_visibility + reference['phase_wall_sec']['all_k_selection'],
                online_through_k_sec=preparation + p['planning_sec'] + result['visibility_sec'] + cumulative,
                online_all_k_sec=result['online_all_k_sec'], rays_submitted=result['work'].get('rays_submitted', 0)))
    write_csv(case_root / 'comparison.csv', rows)
    reference_time = dict(preparation_sec=preparation, visibility_sec=full_visibility,
        selection_sec=reference['phase_wall_sec']['all_k_selection'],
        online_all_k_sec=preparation + full_visibility + reference['phase_wall_sec']['all_k_selection'])
    save(case_root / 'validation.json', dict(passed=True, source_masks_transferred=False,
        query_plans_frozen_before_raycast=True, reference_after_arm_decisions=True,
        independent_raycast=True, exact_density_overlap=True, selected_ids=selected,
        validation_sec=validation, validation_work=validation_work, reference_work=full_work,
        reference_time=reference_time, case_wall_sec=time.perf_counter() - case_begin,
        timing_scope='Reported online time excludes validation, figures, artifact IO and imports. '
                     'Shared preparation is charged once per hypothetical arm; K rows share preprocessing. '
                     'Single sequential measurements with warm OS caches, not controlled speedup evidence.'))
    return rows


def run(root):
    if BASE != root / 'source':
        raise ValueError('Worker requires frozen code')
    manifest = json.loads((root / 'manifest.json').read_text())
    spec = json.loads((root / 'spec.json').read_text())
    if manifest['status'] != 'prepared' or digest(root / 'spec.json') != manifest['spec_sha256']:
        raise ValueError('Prepared immutable run required')
    for name, expected in manifest['source_hashes'].items():
        if digest(BASE / name) != expected:
            raise RuntimeError('Frozen source changed')
    begin = time.perf_counter()
    rows = []
    try:
        manifest.update(status='running')
        save(root / 'manifest.json', manifest)
        entries = []
        for source in source_specs(spec):
            entries.append(build_source(root / 'library' / source['name'], spec, source))
            manifest['completed_sources'].append(source['name'])
            save(root / 'manifest.json', manifest)
            print('source complete:', source['name'], flush=True)
        save(root / 'catalog.json', dict(schema='expanded_ab_catalog_v1', entries=entries,
            offline_wall_sec=sum(e['wall_sec'] for e in entries),
            sample_scope='T/C whole topologies and finite corridor components; synthetic sources'))
        for target in spec['targets']:
            for installation in spec['installation_levels']:
                for occlusion in spec['occlusion_levels']:
                    name = f'{target["name"]}_{installation}_{occlusion}'
                    rows.extend(evaluate_case(root, root / 'cases' / name, spec, target, installation, occlusion))
                    manifest['completed_cases'].append(name)
                    write_csv(root / 'comparison.csv', rows)
                    save(root / 'manifest.json', manifest)
                    print('case complete:', name, flush=True)
        if len(rows) != manifest['plan']['expected_comparison_rows']:
            raise RuntimeError('Incomplete comparison rows')
        grouped = summarize(rows)
        write_csv(root / 'summary.csv', grouped)
        summary_curves(root, grouped)
        for entry in entries:
            path = root / 'library' / entry['descriptor']['name']
            for name, expected in entry['artifact_hashes'].items():
                if digest(path / name) != expected:
                    raise RuntimeError('Library changed during evaluation')
        for name, expected in manifest['source_hashes'].items():
            if digest(BASE / name) != expected:
                raise RuntimeError('Frozen source changed during evaluation')
        if digest(root / 'spec.json') != manifest['spec_sha256']:
            raise RuntimeError('Frozen specification changed during evaluation')
        manifest.update(status='complete', row_count=len(rows), source_libraries_unchanged=True)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest.update(worker_wall_sec=time.perf_counter() - begin, finished_at=datetime.now(timezone.utc).isoformat())
        save(root / 'manifest.json', manifest)


def execute(spec, output):
    planned = plan(spec)
    root = Path(output).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError('New output directory required')
    names = [*sorted(p.name for p in BASE.glob('*.py')),
             *sorted(str(p.relative_to(BASE)) for p in (BASE / 'src').rglob('*.py'))]
    contents = {name: (BASE / name).read_bytes() for name in names}
    root.mkdir(parents=True)
    (root / 'cases').mkdir()
    for name, content in contents.items():
        path = root / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    save(root / 'spec.json', spec)
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    manifest = dict(schema=SCHEMA, status='prepared', plan=planned, completed_sources=[], completed_cases=[],
        created_at=datetime.now(timezone.utc).isoformat(), command=sys.argv, spec_sha256=digest(root / 'spec.json'),
        python=sys.version, packages={name: package_metadata.version(name) for name in
            ('numpy', 'scipy', 'open3d', 'gurobipy', 'matplotlib', 'PyYAML')},
        thread_environment={key: environment[key] for key in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
        cpu=os.uname().machine, cpu_count=os.cpu_count(),
        source_hashes={name: digest(root / 'source' / name) for name in names})
    save(root / 'manifest.json', manifest)
    with (root / 'worker.log').open('x') as stream:
        try:
            result = subprocess.run([sys.executable, '-B', str(root / 'source/library_expanded_study.py'),
                '--worker', str(root), '--execute'], cwd=root / 'source', stdout=stream, stderr=subprocess.STDOUT,
                timeout=spec['worker_timeout_sec'], env=environment)
            if result.returncode:
                raise RuntimeError(f'Worker failed; see {root / "worker.log"}')
        except subprocess.TimeoutExpired:
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest.update(status='failed', error='explicit_hard_worker_timeout')
            save(root / 'manifest.json', manifest)
            raise
    return root


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--spec', default=str(BASE / 'configs/library_expanded_20261123.json'))
    parser.add_argument('--output')
    parser.add_argument('--smoke', action='store_true', help='Tiny functional run, not performance evidence')
    parser.add_argument('--worker')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.worker:
        if not args.execute:
            parser.error('--execute required for workers')
        run(Path(args.worker).resolve())
        return
    spec = smoke_spec() if args.smoke else json.loads(Path(args.spec).read_text())
    if args.execute:
        output = args.output or ('results/library_expanded_smoke_20261123' if args.smoke
                                 else 'results/library_expanded_20261123')
        print(execute(spec, output))
    else:
        print(json.dumps(plan(spec), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

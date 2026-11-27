"""Sequential cold-cache runtime ablation. Plans only unless --execute is supplied."""
from argparse import ArgumentParser, Namespace, SUPPRESS
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import itertools
import html
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from prototype_offline import describe, digest, save, width_scenario
from prototype_refine import SOURCES, make_plan
from prototype_transfer import export_view, load_prototype
from src.core.runtime import RuntimeMeter
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, Scenario, get_lidar_profile
from src.methods.local_mappo import refine_placement, validate_poses
from src.methods.local_search import refine_local_search


def now():
    return datetime.now(timezone.utc).isoformat()


def plan_args(args, sid, width, mode, target_scenario=None):
    return Namespace(scenarios=[sid], width=width, k=args.k, steps=args.steps,
                     radius=args.radius, auxiliary_root=args.auxiliary_root, coordinate_mode=mode,
                     target_scenario=target_scenario)


def map_specs(args):
    if not args.shape_config:
        return [dict(key=f's{sid}_w{width:g}', scenario=sid, width=width)
                for sid, width in itertools.product(args.scenarios, args.widths)]
    data = json.loads(Path(args.shape_config).read_text())
    if data.get('schema') != 'axis_aligned_shape_pilot_v1' or data.get('split') != 'development':
        raise ValueError('Expected axis-aligned development shape suite')
    specs, ids = [], set()
    for row in data['scenarios']:
        sid = row['parent_scenario_id']
        if sid not in (1, 2) or row['topology'] != {1: 'T_JUNCTION', 2: 'CROSS'}[sid]:
            raise ValueError('Shape parent/topology mismatch')
        if type(row['id']) is not int or row['id'] < 400000 or row['id'] in ids:
            raise ValueError('Shape IDs must be unique integers >= 400000')
        if not np.isfinite(row['road_width']) or not 7 < row['road_width'] < 14:
            raise ValueError('Shape width must lie between current 7m/14m sources')
        geometry = MapGeometry(Scenario(row['id'], row['topology'], row['road_width'], row['num_lanes'], row))
        if not row.get('geometry'):
            raise ValueError('Explicit shape geometry required')
        ids.add(row['id'])
        specs.append(dict(key=f'shape{row["id"]}', scenario=sid, width=geometry.scenario.road_width,
                          target_scenario_id=row['id'], shape_name=row.get('name', str(row['id'])),
                          target_scenario=row))
    if not specs:
        raise ValueError('Empty shape suite')
    return specs


def jobs(args):
    for spec, repeat in itertools.product(map_specs(args), range(args.repeats)):
        for method in args.methods:
            modes = args.modes if method in ('select', 'local_search', 'mappo') else ['baseline']
            seeds = args.seeds if method == 'mappo' else [None]
            for mode, seed in itertools.product(modes, seeds):
                key = f'{spec["key"]}_{mode}_{method}_r{repeat}'
                if seed is not None:
                    key += f'_seed{seed}'
                yield dict(spec, key=key, repeat=repeat, method=method, mode=mode, seed=seed)


def validate_arguments(args):
    for values in (args.scenarios, args.widths, args.methods, args.modes, args.seeds):
        if not values or len(set(values)) != len(values):
            raise ValueError('Dimensions must be nonempty and unique')
    if args.k != 3:
        raise ValueError('The current frozen library supports this study at K=3 only')
    if any(not math.isfinite(w) or not 7 < w < 14 for w in args.widths):
        raise ValueError('Interpolation widths must lie strictly inside (7,14)')
    for value in (args.radius, args.refine_seconds, args.solver_seconds, args.memory_gb):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Finite positive radius/time/memory limits required')
    if min(args.steps, args.evaluations, args.repeats) < 1 or args.topm < 0:
        raise ValueError('Positive work limits/repeats and nonnegative topM required')
    if not 0 <= args.gap < 1 or any(not 0 <= s < 2**32 for s in args.seeds):
        raise ValueError('Invalid gap or seed')
    if args.gpu != 'cpu' and not args.gpu.isdecimal():
        raise ValueError('Use cpu or a single nonnegative CUDA device index')


def select_initial(placements, provider, work):
    """Evaluate all available controls; deterministic tie order, no oracle baseline."""
    scores = {}
    for name in ('transfer_low', 'transfer_high', 'interpolation'):
        if name not in placements:
            continue
        work.add('deployment_evaluations')
        covered = np.stack([provider.mask(p) for p in placements[name]]).any(axis=0)
        scores[name] = 100 * float(covered.mean())
    if not scores:
        raise ValueError('No feasible initial placement')
    name = max(scores, key=scores.get)
    work.observe(scores[name], 'initial_selection_complete')
    return placements[name].copy(), dict(selected=name, coverage_pct=scores[name], alternatives=scores,
        tie_order=['transfer_low', 'transfer_high', 'interpolation'])


def full_baseline(job, args, root, cfg, geometry, work):
    from src.methods.greedy import greedy_select
    from src.methods.milp import _solve_with_gurobi
    from src.roadside.candidate_generator import build_oriented_candidates
    cfg.candidate['max_orientations_per_position'] = args.topm
    cfg.raw['output'] = dict(results_dir=str(root), cache_dir=str(root/'cache'))
    cfg.solver.update(time_limit_sec=args.solver_seconds, mip_gap=args.gap, threads=1,
        soft_mem_limit_gb=args.memory_gb, sparse_target_batch_size=128,
        nodefile_dir=str(root/'nodefiles'), log_file=str(root/'gurobi.log'))
    with work.stage('full_visibility_and_candidates'):
        candidates = build_oriented_candidates(geometry.scenario, cfg.lidar_profile, cfg, work=work)
    if candidates.stats['num_coverage_edges'] > 1_500_000_000:
        raise RuntimeError('Candidate edge safety limit exceeded; no silent pruning')
    with work.stage('greedy_selection'):
        ids = greedy_select(candidates, args.k).selected_ids
    def coverage(chosen):
        union = np.zeros(len(candidates.target_points), dtype=bool)
        for i in chosen:
            union |= candidates.candidates[i].covered
        return 100 * float(union.mean())
    work.observe(coverage(ids), 'greedy_incumbent')
    meta = dict(candidate_stats=candidates.stats, topm=args.topm, full_visibility_cache=True,
                cache_state='fresh per worker, no shared persistent cache', solver=None)
    if job['method'] == 'milp':
        with work.stage('milp_assembly_and_solve'):
            solved = _solve_with_gurobi(candidates, args.k, cfg, ids)
        if coverage(solved['selected_ids']) >= coverage(ids):
            ids = solved['selected_ids']
            meta['selection_source'] = 'solver'
        else:
            meta['selection_source'] = 'greedy_incumbent'
        meta['solver'] = solved
        work.observe(coverage(ids), 'milp_return')
    meta['coverage_pct'] = coverage(ids)
    meta['optimality_domain'] = 'declared discrete candidate set only; topM may prune orientations'
    return candidates.positions_array(ids), meta


def run_worker(args):
    import torch
    torch.set_num_threads(1)
    root = Path(args.output)
    manifest = json.loads((root/'manifest.json').read_text())
    if {k: v for k, v in vars(args).items() if k != 'worker'} != {
            k: v for k, v in manifest['arguments'].items() if k != 'worker'}:
        raise ValueError('Worker arguments differ from the frozen manifest')
    if args.worker < 0 or args.worker >= len(manifest['jobs']):
        raise ValueError('Invalid worker index')
    if args.gpu != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device unavailable; refusing silent CPU fallback')
    job = manifest['jobs'][args.worker]
    for path, expected in {**manifest['source_hashes'], **manifest['prototype_hashes']}.items():
        if digest(path) != expected:
            raise RuntimeError(f'Input/source changed since plan: {path}')
    directory = root/job['key']
    directory.mkdir(exist_ok=False)
    record = dict(job, status='running', started_at=now())
    save(directory/'result.json', record)
    work = RuntimeMeter()
    try:
        with work.stage('input_geometry_transfer'):
            if job['mode'] != 'baseline':
                _, low, high, geometry, placements, transfer, _ = make_plan(
                    plan_args(args, job['scenario'], job['width'], job['mode'], job.get('target_scenario')))[0]
                cfg = deepcopy(low[5])
            else:
                cfg = deepcopy(load_prototype(SOURCES[job['scenario']][0], args.k)[5])
                spec = job.get('target_scenario')
                geometry = MapGeometry(Scenario(spec['id'], spec['topology'], spec['road_width'],
                    spec['num_lanes'], spec) if spec else width_scenario(job['width'], job['scenario']))
                transfer = None
        with work.stage('direct_scene_setup'):
            provider = MeshVisibility(geometry.create_environment_mesh(),
                geometry.target_points(cfg.target['coverage_z_slices']),
                get_lidar_profile(cfg.lidar_profile, cfg), work=work)
        selection = None
        if job['mode'] != 'baseline':
            with work.stage('initial_selection'):
                poses, selection = select_initial(placements, provider, work)
            np.save(directory/'initial.npy', poses)
            save(directory/'initial_candidates.json', {name: p.tolist() for name, p in placements.items()})
            if job['method'] == 'mappo':
                with work.stage('refinement_including_checkpoint_save'):
                    poses, meta = refine_placement(poses, geometry.scenario, cfg, directory/'refinement',
                        steps=args.steps, seed=job['seed'], radius_m=args.radius, work=work,
                        time_limit_sec=args.refine_seconds)
            elif job['method'] == 'local_search':
                with work.stage('refinement'):
                    poses, meta = refine_local_search(poses, geometry.scenario, cfg, radius_m=args.radius,
                        max_evaluations=args.evaluations, time_limit_sec=args.refine_seconds, work=work)
            else:
                meta = dict(coverage_pct=selection['coverage_pct'], full_visibility_cache=False)
        else:
            poses, meta = full_baseline(job, args, directory, cfg, geometry, work)
        with work.stage('final_direct_validation'):
            validate_poses(poses, geometry, cfg)
            if len(poses) > args.k:
                raise RuntimeError('Sensor budget exceeded')
            work.add('deployment_evaluations')
            density = np.stack([provider.mask(p) for p in poses]).sum(axis=0)
            cov = 100 * float((density > 0).mean())
            if not np.isclose(cov, meta['coverage_pct'], atol=1e-6, rtol=0):
                raise RuntimeError('Internal and direct coverage disagree')
            if selection is not None and cov + 1e-6 < selection['coverage_pct']:
                raise RuntimeError('Refinement lost the retained initial deployment')
        work.observe(cov, 'validated_deployment')
        record.update(status='complete', coverage_pct=cov, covered_count=int((density > 0).sum()),
            target_count=len(density), num_selected=len(poses), selection=selection, transfer=transfer,
            method_metadata=meta, runtime=work.report())
        record['hardware'] = dict(platform=platform.platform(), cpu=platform.processor(),
            logical_cpu_count=os.cpu_count(), total_ram_bytes=os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
            torch_threads=torch.get_num_threads(),
            gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_initialized() else None))
        # Report/figure export is deliberately outside the measured online interval.
        np.save(directory/'deployment.npy', poses)
        np.savez_compressed(directory/'coverage.npz', targets=provider.targets, density=density)
        (directory/'config.yaml').write_text(yaml.safe_dump(cfg.raw, sort_keys=False), encoding='utf-8')
        save(directory/'geometry.json', describe(geometry, cfg))
        if args.save_3d:
            export_view(directory/'deployment.html', geometry.create_environment_mesh(), provider.targets,
                        density, poses, job['key'], cov)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), runtime=work.report())
        raise
    finally:
        record['finished_at'] = now()
        save(directory/'result.json', record)


def summarize(root, manifest):
    rows = []
    for job in manifest['jobs']:
        path = root/job['key']/'result.json'
        record = json.loads(path.read_text()) if path.exists() else dict(status=job.get('status', 'pending'))
        rt = record.get('runtime', {})
        solver = record.get('method_metadata', {}).get('solver') or {}
        status = 'failed' if job.get('status') == 'failed' else record['status']
        rows.append(dict(job, status=status, coverage_pct=record.get('coverage_pct'),
            online_wall_sec=rt.get('elapsed_sec'), process_peak_rss_bytes=rt.get('process_peak_rss_bytes'),
            rays_submitted=rt.get('counts', {}).get('rays_submitted'),
            selected_initial=(record.get('selection') or {}).get('selected'),
            mip_gap=solver.get('mip_gap'), result_file=str(path), error=record.get('error', '')))
    with (root/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    body = []
    for row in rows:
        key = html.escape(row['key'])
        view = root/row['key']/'deployment.html'
        label = f'<a href="{key}/deployment.html">{key}</a>' if view.exists() else key
        cov = '-' if row['coverage_pct'] is None else f'{row["coverage_pct"]:.3f}'
        seconds = '-' if row['online_wall_sec'] is None else f'{row["online_wall_sec"]:.2f}'
        body.append(f'<tr><td>{label}</td><td>{html.escape(row["status"])}</td><td>{cov}</td><td>{seconds}</td></tr>')
    overview = '<img src="maps.png" alt="Road shapes and installation lines" style="max-width:100%">' if (root/'maps.png').exists() else ''
    (root/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Placement pilot results</title><style>body{font-family:sans-serif;margin:24px} '
        'td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
        '<h1>Placement pilot results</h1><p>Development pilot, not a convergence or optimality claim.</p>'
        '<p><a href="comparison.csv">CSV</a> | <a href="manifest.json">Manifest</a></p>'+overview+
        '<div style="overflow-x:auto"><table><tr><th>Run / interactive 3D</th><th>Status</th>'
        '<th>Coverage (%)</th><th>Online wall time (s)</th></tr>'+''.join(body)+'</table></div></html>', encoding='utf-8')


def export_shape_overview(root, specs):
    """Plot actual shared road/installation predicates; no placement optimization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    shaped = [s['target_scenario'] for s in specs if 'target_scenario' in s]
    if not shaped:
        return
    fig, axes = plt.subplots(math.ceil(len(shaped)/2), 2, figsize=(11, 5*math.ceil(len(shaped)/2)), squeeze=False)
    for axis, row in zip(axes.flat, shaped):
        g = MapGeometry(Scenario(row['id'], row['topology'], row['road_width'], row['num_lanes'], row))
        raster = np.zeros((int(g.map_width/g.grid_size), int(g.map_len/g.grid_size)), dtype=np.uint8)
        for gy, gx in np.ndindex(raster.shape):
            x, y = (gx+.5)*g.grid_size, (gy+.5)*g.grid_size
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0,y0,x1,y1 in g.negative_zones):
                raster[gy, gx] = 1
            if g.is_road(x, y):
                raster[gy, gx] = 2
            if g.is_installable(x, y):
                raster[gy, gx] = 3
        axis.imshow(raster, origin='lower', extent=(0,g.map_len,0,g.map_width),
                    cmap=ListedColormap(['#f6f6f6','#999999','#303030','#d79500']), vmin=0, vmax=3)
        axis.set(title=f'{row["name"]}: junction ({g.junction_x_m:g}, {g.junction_y_m:g})', xlabel='x (m)', ylabel='y (m)')
        axis.set_aspect('equal')
    for axis in list(axes.flat)[len(shaped):]:
        axis.set_visible(False)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [('#999999','Buildings'),('#303030','Road targets'),('#d79500','Installable cells')]],
               loc='lower center', ncol=3)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(root/'maps.png', dpi=150)
    plt.close(fig)


def run(args):
    validate_arguments(args)
    if args.worker is not None:
        if not args.execute:
            raise ValueError('Worker execution also requires --execute')
        run_worker(args)
        return
    if args.execute and Path(args.output).exists():
        raise FileExistsError(args.output)
    planned = list(jobs(args))
    prototypes = {}
    for spec, mode in itertools.product(map_specs(args), args.modes):
        _, low, high, _, placements, meta, _ = make_plan(plan_args(args, spec['scenario'], spec['width'], mode,
            spec.get('target_scenario')))[0]
        print(f'{spec["key"]} {mode}: {list(placements)}; interpolation={meta["status"]}', flush=True)
        for source in (low, high):
            for path in (source[0]/'config.yaml', source[0]/'geometry.json',
                         source[0]/f'k{args.k}/result.json', source[0]/source[1]['positions_npy']):
                prototypes[str(path)] = digest(path)
    if args.shape_config:
        prototypes[str(Path(args.shape_config))] = digest(args.shape_config)
    print(f'{len(planned)} sequential fresh-process jobs; execute={args.execute}', flush=True)
    if not args.execute:
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    paths = sorted(set([*Path('src').rglob('*.py'), *Path('.').glob('prototype*.py')]))
    source_hashes = {str(p): digest(p) for p in paths}
    for path in [*paths, *(Path(p) for p in prototypes)]:
        try:
            relative = path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            relative = Path('external')/digest(path)[:12]/path.name
        dest = root/'inputs'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    manifest = dict(schema='prototype_runtime_v1', status='running', started_at=now(),
        command=[sys.executable, *sys.argv], arguments=vars(args), jobs=planned,
        source_hashes=source_hashes, prototype_hashes=prototypes,
        timing_scope='online input/geometry/transfer/selection/preprocessing/refinement/final validation; '
            'includes baseline cache writes, MAPPO checkpoint writes and lazy imports inside stages; '
            'excludes eager module imports, worker startup, report/3D export',
        resource_scope='fresh subprocess per job, sequential, GPU fixed; external host contention not controlled',
        cache_scope='algorithm caches cold per job; OS filesystem cache uncontrolled; on-demand memoization within a job allowed',
        limitations=['No new held-out claim for development widths', 'Soft refinement time budget can overrun at safe boundaries',
            'MILP time limit applies to solver only, not full pipeline', 'Offline library build cost not amortized here',
            'Shape suite: width interpolation then segment transfer to shifted junction; not interpolation between junction centers',
            'Different candidate/reachable spaces; not a proof of speedup at equal optimality'],
        environment=dict(python=sys.version, platform=sys.platform, gpu=args.gpu))
    save(root/'manifest.json', manifest)
    with (root/'packages.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if args.gpu == 'cpu' else args.gpu,
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        export_shape_overview(root, map_specs(args))
        summarize(root, manifest)
        for index, job in enumerate(planned):
            job['status'] = 'running'
            save(root/'manifest.json', manifest)
            start = time.perf_counter()
            with (root/f'{job["key"]}.log').open('w') as stream:
                result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    *sys.argv[1:], '--worker', str(index)], env=env, stdout=stream, stderr=subprocess.STDOUT)
            job.update(status='complete' if result.returncode == 0 else 'failed',
                       worker_process_wall_sec=time.perf_counter() - start)
            summarize(root, manifest)
            save(root/'manifest.json', manifest)
            if result.returncode:
                raise RuntimeError(f'Worker failed: {job["key"]}; see its log')
        manifest['status'] = 'complete'
    except BaseException as exc:
        for job in planned:
            if job.get('status') == 'running':
                job['status'] = 'failed'
        manifest.update(status='failed', error=repr(exc))
        summarize(root, manifest)
        raise
    finally:
        manifest['finished_at'] = now()
        save(root/'manifest.json', manifest)


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--shape-config', help='Explicit synthetic shape suite; replaces scenario/width lists')
    p.add_argument('--scenarios', type=int, nargs='+', choices=[0, 1, 2], default=[0, 1, 2])
    p.add_argument('--widths', type=float, nargs='+', default=[10.])
    p.add_argument('--modes', nargs='+', choices=['distance', 'fraction'], default=['distance', 'fraction'])
    p.add_argument('--methods', nargs='+', choices=['select', 'local_search', 'mappo', 'greedy', 'milp'],
                   default=['select', 'local_search', 'mappo', 'greedy', 'milp'])
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--evaluations', type=int, default=2000)
    p.add_argument('--refine-seconds', type=float, default=60.)
    p.add_argument('--radius', type=float, default=10.)
    p.add_argument('--solver-seconds', type=float, default=300.)
    p.add_argument('--gap', type=float, default=.05)
    p.add_argument('--topm', type=int, default=10)
    p.add_argument('--memory-gb', type=float, default=26.)
    p.add_argument('--gpu', default='cpu', help='cpu or one physical CUDA device index')
    p.add_argument('--auxiliary-root', default='results/prototype_compatible_20261002')
    p.add_argument('--save-3d', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', type=int, help=SUPPRESS)
    return p


if __name__ == '__main__':
    run(parser().parse_args())

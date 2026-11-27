"""Held-out width sweep with frozen prototypes, paired seeds and solver baselines.

Without --execute, only validates plans. Existing result directories are refused.
"""
from argparse import ArgumentParser, Namespace, SUPPRESS
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from prototype_offline import describe, digest, save, width_scenario
from prototype_refine import BASE_METHODS, METHODS, make_plan, record_result, run as refine_run
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, get_lidar_profile
from src.methods.greedy import greedy_select
from src.methods.milp import _solve_with_gurobi
from src.roadside.candidate_generator import build_oriented_candidates


def now():
    return datetime.now(timezone.utc).isoformat()


def map_key(scenario, width):
    return f's{scenario}_w{float(width):g}'


def refinement_args(args, scenario, width, seed, output):
    return Namespace(output=str(output), scenarios=[scenario], width=width, k=args.k,
                     steps=args.steps, seed=seed, radius=args.radius,
                     auxiliary_root=args.auxiliary_root, execute=True, report_only=False)


def validate_arguments(args):
    for values in (args.widths, args.scenarios, args.seeds, args.gpus):
        if not values or len(set(values)) != len(values):
            raise ValueError('Sweep dimensions must be nonempty and unique')
    if any(not np.isfinite(w) or not 7 < w < 14 or w == 10 for w in args.widths):
        raise ValueError('Held-out interpolation widths must be inside (7,14), excluding pilot width 10')
    if any(s not in (0, 1, 2) for s in args.scenarios):
        raise ValueError('Only parent scenarios 0,1,2 are supported')
    if any(s < 0 or s > 2**32-1 for s in args.seeds):
        raise ValueError('Seeds must fit unsigned 32-bit integers')
    if args.k != 3 or args.steps < 1 or not np.isfinite(args.radius) or args.radius <= 0:
        raise ValueError('This frozen-library study requires K=3 and positive steps/radius')
    if args.baseline_topm < 0 or not np.isfinite(args.time_limit) or args.time_limit <= 0:
        raise ValueError('Invalid baseline settings')
    if args.baseline_topm == 0 and len(args.gpus) != 1:
        raise ValueError('Full-orientation baselines require one worker for memory safety')
    if not 0 <= args.gap < 1:
        raise ValueError('Relative gap must be in [0,1)')


def preflight(args):
    validate_arguments(args)
    plans, source_hashes = [], {}
    for sid in args.scenarios:
        for width in args.widths:
            plan = make_plan(refinement_args(args, sid, width, args.seeds[0], 'unused'))[0]
            _, low, high, geometry, placements, meta, _ = plan
            if set(placements) != set(BASE_METHODS) or meta['status'] != 'available':
                raise ValueError(f'Interpolation unavailable: {sid}/{width}: {meta}')
            for source in (low, high):
                for path in (source[0]/'config.yaml', source[0]/'geometry.json',
                             source[0]/f'k{args.k}/result.json', source[0]/source[1]['positions_npy']):
                    source_hashes[str(path)] = digest(path)
            plans.append(dict(key=map_key(sid,width), scenario=sid, width=width,
                              topology=geometry.scenario.topology, seeds=args.seeds))
    return plans, source_hashes


def run_baselines(args, sid, width, directory):
    root = directory/'baselines'
    root.mkdir(exist_ok=False)
    started = time.perf_counter()
    plan = make_plan(refinement_args(args,sid,width,args.seeds[0],'unused'))[0]
    cfg = copy.deepcopy(plan[1][5])
    scenario = width_scenario(width,sid)
    geometry = MapGeometry(scenario)
    cfg.candidate['max_orientations_per_position'] = args.baseline_topm
    cfg.raw['output'] = dict(results_dir=str(root),cache_dir=str(root/'cache'))
    cfg.solver.update(time_limit_sec=args.time_limit,mip_gap=args.gap,threads=1,
        soft_mem_limit_gb=90 if args.baseline_topm == 0 else 26,
        sparse_target_batch_size=128,nodefile_dir=str(root/'nodefiles'),log_file=str(root/'gurobi.log'))
    (root/'config.yaml').write_text(yaml.safe_dump(cfg.raw,sort_keys=False),encoding='utf-8')
    save(root/'geometry.json',describe(geometry,cfg))
    candidates = build_oriented_candidates(scenario,cfg.lidar_profile,cfg)
    preprocessing = time.perf_counter()-started
    if candidates.stats['num_coverage_edges'] > 1_500_000_000:
        raise RuntimeError('Baseline edge safety limit exceeded; candidates were not silently pruned')
    save(root/'candidate_stats.json',candidates.stats)
    stage = time.perf_counter()
    warm = greedy_select(candidates,args.k).selected_ids
    greedy_sec = time.perf_counter()-stage
    provider = MeshVisibility(geometry.create_environment_mesh(),candidates.target_points,
                              get_lidar_profile(cfg.lidar_profile,cfg))
    records = {}
    def store(name, ids, selection_sec, metadata):
        poses = candidates.positions_array(ids)
        rec = record_result(root,name,poses,geometry,cfg,provider,dict(
            baseline_topm=args.baseline_topm,candidate_count=len(candidates.candidates),
            cache_key=candidates.cache_key,config_hash=candidates.config_hash,
            preprocessing_sec=preprocessing,selection_sec=selection_sec,solver=metadata,
            optimality_domain='baseline discrete candidates only',budget_mode='at_most'))
        expected = np.any([candidates.candidates[i].covered for i in ids],axis=0)
        if rec['covered_count'] != int(expected.sum()):
            raise RuntimeError('Baseline mask/direct raycasting mismatch')
        with np.load(root/f'{name}_coverage.npz') as saved:
            np.testing.assert_array_equal(saved['density']>0,expected)
        rec['compute_component_sum_sec'] = preprocessing+selection_sec+rec['evaluation_sec']
        save(root/f'{name}.json',rec)
        return rec
    suffix = f'top{args.baseline_topm}' if args.baseline_topm else 'all'
    records['greedy'] = store(f'greedy_{suffix}',warm,greedy_sec,{})
    stage = time.perf_counter()
    solved = _solve_with_gurobi(candidates,args.k,cfg,warm)
    solve_wall = time.perf_counter()-stage
    ids = solved['selected_ids']
    count = lambda chosen: int(np.any([candidates.candidates[i].covered for i in chosen],axis=0).sum())
    if count(ids) < count(warm):
        ids = warm
        solved['selection_source'] = 'greedy_warm_start'
    else:
        solved['selection_source'] = 'solver_or_accepted_start'
    records['milp'] = store(f'milp_{suffix}',ids,greedy_sec+solve_wall,solved)
    save(root/'report.json',dict(status='complete',results=records,candidate_stats=candidates.stats,
        wall_time_sec=time.perf_counter()-started,
        timing_note='Cold candidate/cache build charged to both baselines; selection includes MILP assembly. '
                    'Component sums exclude plot/report export; shared host contention is not isolated latency.'))


def worker(args):
    import torch
    torch.set_num_threads(1)
    sid, width = args.worker_scenario, args.worker_width
    directory = Path(args.output)/map_key(sid,width)
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    manifest = dict(status='running',scenario=sid,width=width,started_at=now(),completed_seeds=[],
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),command=[sys.executable,*sys.argv])
    save(directory/'manifest.json',manifest)
    try:
        for seed in args.seeds:
            frozen = json.loads((Path(args.output)/'manifest.json').read_text())['frozen_prototype_hashes']
            if any(digest(path) != expected for path,expected in frozen.items()):
                raise RuntimeError('Source prototype changed after benchmark planning')
            output = directory/f'seed{seed}'
            refine_run(refinement_args(args,sid,width,seed,output))
            subprocess.run([sys.executable,'tests/validate_prototype_refinement.py',str(output)],check=True)
            manifest['completed_seeds'].append(seed)
            save(directory/'manifest.json',manifest)
        manifest['phase'] = 'baselines'
        save(directory/'manifest.json',manifest)
        print(f'BASELINES {map_key(sid,width)}: building fresh visibility and candidates',flush=True)
        run_baselines(args,sid,width,directory)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc))
        raise
    finally:
        manifest.update(finished_at=now(),wall_time_sec=time.perf_counter()-started)
        save(directory/'manifest.json',manifest)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['scenario'],row['width'],row['method'])].append(row)
    summary = []
    for (sid,width,method), items in sorted(grouped.items()):
        values = np.asarray([i['coverage_pct'] for i in items],dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite benchmark score')
        if len({i['seed'] for i in items}) != len(items):
            raise ValueError('Duplicate trial')
        gains = [i['gain_pp'] for i in items if i['gain_pp'] is not None]
        policies = [i['final_policy_coverage_pct'] for i in items if i['final_policy_coverage_pct'] is not None]
        summary.append(dict(scenario=sid,width=width,method=method,n=len(items),
            coverage_mean=float(values.mean()),coverage_sd=float(values.std(ddof=1)) if len(items)>1 else None,
            gain_mean_pp=float(np.mean(gains)) if gains else None,
            final_policy_mean=float(np.mean(policies)) if policies else None,
            compute_component_mean_sec=float(np.mean([i['compute_component_sum_sec'] for i in items]))))
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root, manifest):
    rows, links = [], []
    seeds = manifest['arguments']['seeds']
    for plan in manifest['plans']:
        directory = root/plan['key']
        status_path = directory/'manifest.json'
        if not status_path.exists() or json.loads(status_path.read_text())['status'] != 'complete':
            continue
        def row(method, record, seed, path, components, gain=None, policy=None):
            return dict(scenario=plan['scenario'],width=plan['width'],topology=plan['topology'],
                method=method,seed=seed,coverage_pct=record['coverage_pct'],
                overlap_target_count=record['overlap_target_count'],uncovered_count=record['uncovered_count'],
                target_count=record['target_count'],gain_pp=gain,final_policy_coverage_pct=policy,
                compute_component_sum_sec=components,solver_status=record.get('solver',{}).get('solver_status'),
                mip_gap=record.get('solver',{}).get('mip_gap'),best_bound=record.get('solver',{}).get('best_bound'),
                result_file=str(path.relative_to(root)))
        for seed in seeds:
            runroot = directory/f'seed{seed}'/f's{plan["scenario"]}'
            r = json.loads((runroot/'report.json').read_text())
            validation = json.loads((directory/f'seed{seed}'/'validation.json').read_text())
            if validation['status'] != 'passed' or validation['checked'] != 6:
                raise ValueError('Unvalidated refinement results')
            for method in METHODS:
                rec = r['results'][method]
                if rec['status'] != 'complete':
                    raise ValueError('Missing result; never impute zero')
                if method in BASE_METHODS and seed != seeds[0]:
                    first = directory/f'seed{seeds[0]}'/f's{plan["scenario"]}'/f'{method}.npy'
                    np.testing.assert_array_equal(np.load(first),np.load(runroot/f'{method}.npy'))
                    continue
                meta = rec.get('refinement',{})
                base = r['results'][method.removesuffix('_mappo')]
                components = r['placement_generation_sec']+r['base_scene_setup_sec']+base['evaluation_sec']
                if meta:
                    components += meta['elapsed_sec']+rec['evaluation_sec']
                rows.append(row(method,rec,seed if meta else None,runroot/f'{method}.json',components,
                    rec['coverage_pct']-base['coverage_pct'] if meta else None,meta.get('final_policy_coverage_pct')))
            links.append(f'<li>{plan["key"]} seed {seed}: <a href="{plan["key"]}/seed{seed}/index.html">six placements / 3D</a></li>')
        baseline = json.loads((directory/'baselines/report.json').read_text())
        for rec in baseline['results'].values():
            rows.append(row(rec['method'],rec,None,directory/'baselines'/f'{rec["method"]}.json',rec['compute_component_sum_sec']))
            links.append(f'<li>{plan["key"]}: <a href="{plan["key"]}/baselines/{rec["view_html"]}">{rec["method"]} 3D</a></li>')
    return rows,links


def report(root):
    import plotly.graph_objects as go
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    rows,links = collect(root,manifest)
    summary = aggregate(rows)
    write_csv(root/'trials.csv',rows)
    write_csv(root/'summary.csv',summary)
    comparisons = []
    for plan in manifest['plans']:
        selected = [r for r in rows if r['scenario']==plan['scenario'] and r['width']==plan['width']]
        if not selected:
            continue
        initial = {r['method']:r['coverage_pct'] for r in selected if r['method'] in BASE_METHODS}
        for seed in manifest['arguments']['seeds']:
            refined = {r['method']:r['coverage_pct'] for r in selected if r['seed']==seed}
            comparisons.append(dict(scenario=plan['scenario'],width=plan['width'],seed=seed,
                interpolation_minus_best_single_initial_pp=initial['interpolation']-max(initial['transfer_low'],initial['transfer_high']),
                interpolation_minus_best_single_refined_pp=refined['interpolation_mappo']-max(refined['transfer_low_mappo'],refined['transfer_high_mappo']),
                comparison_note='Best single is a posteriori diagnostic, not a free deployable selector'))
    write_csv(root/'paired_differences.csv',comparisons)
    plots=[]
    for sid in manifest['arguments']['scenarios']:
        fig = go.Figure()
        subset = [r for r in summary if r['scenario']==sid]
        for method in sorted({r['method'] for r in subset}):
            values = sorted([r for r in subset if r['method']==method],key=lambda r:r['width'])
            fig.add_scatter(name=method,x=[r['width'] for r in values],y=[r['coverage_mean'] for r in values],
                mode='lines+markers',error_y=dict(type='data',array=[r['coverage_sd'] or 0 for r in values],visible=True))
        fig.update_layout(title=f'Scenario {sid}: held-out widths',yaxis_title='Coverage (%)',
            xaxis_title='Road width (m)',yaxis_range=[0,100],font=dict(size=11),
            margin=dict(l=55,r=15,t=55,b=230),legend=dict(orientation='h',x=0,y=-.25))
        name=f'coverage_s{sid}.html'
        fig.write_html(root/name,include_plotlyjs=True,default_height='700px',config=dict(responsive=True))
        plots.append(f'<li><a href="{name}">Scenario {sid} width comparison</a></li>')
    note=('Frozen 7m/14m prototypes; 10m was a development pilot. Width-only parametric maps. '
          'MAPPO values are best-seen deployment, not final policy. Error bars are sample SD across paired seeds. '
          'Baselines use their recorded discrete candidate set and are not global upper bounds. '
          'Timing component sums include visibility work but are not isolated latency or speedup measurements. '
          'T/CROSS high prototypes use low-prototype installation allocations. Incomplete maps are omitted, never zero-filled.')
    (root/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Width benchmark</title>'
        '<h1>Held-out width benchmark</h1><p>'+note+'</p><p><a href="summary.csv">Summary CSV</a> | '
        '<a href="trials.csv">Trials CSV</a> | <a href="paired_differences.csv">Paired differences</a></p>'
        '<ul>'+''.join(plots)+'</ul><h2>Saved placements</h2><ul>'+''.join(links)+'</ul></html>',encoding='utf-8')
    validation=dict(status='passed' if manifest['status']=='complete' else 'partial',
        maps=len({(r['scenario'],r['width']) for r in rows}),raw_rows=len(rows),summary_rows=len(summary),
        command=[sys.executable,*sys.argv],report_source_sha256=digest(__file__))
    if manifest['status']=='complete':
        expected=len(manifest['plans'])*(5+3*len(manifest['arguments']['seeds']))
        if len(rows)!=expected or len(summary)!=8*len(manifest['plans']):
            raise ValueError('Benchmark completeness mismatch')
    save(root/'validation.json',validation)
    return validation


def launch(args):
    plans, hashes = preflight(args)
    print(json.dumps(dict(maps=len(plans),mappo_runs=len(plans)*3*len(args.seeds),
        widths=args.widths,seeds=args.seeds,baseline_topm=args.baseline_topm,
        time_limit=args.time_limit,plans=plans),indent=2),flush=True)
    if not args.execute:
        return
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    source={}
    for path in [Path(__file__),*Path('.').glob('prototype_*.py'),*Path('src').rglob('*.py'),
                 *Path('tests').glob('*prototype*.py'),Path('tests/test_local_mappo.py'),
                 Path('data/scenarios/yeongjong_final.json')]:
        dest=root/'source'/(path.name if path.is_absolute() else path)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest); source[str(path)]=digest(path)
    manifest=dict(status='running',started_at=now(),arguments=vars(args),plans=plans,
        frozen_prototype_hashes=hashes,source_hashes=source,command=[sys.executable,*sys.argv],
        expected_maps=len(plans),completed_maps=[],failed_maps=[],
        timing_note='Concurrent workers: runtime is not an isolated cross-method speed comparison')
    save(root/'manifest.json',manifest)
    pending=list(plans); active={}
    try:
        while pending or active:
            for gpu in args.gpus:
                if gpu in active or not pending:
                    continue
                plan=pending.pop(0)
                log=(root/f'{plan["key"]}.log').open('w')
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(root),
                     '--worker-scenario',str(plan['scenario']),'--worker-width',str(plan['width']),
                     '--auxiliary-root',args.auxiliary_root,'--steps',str(args.steps),'--radius',str(args.radius),
                     '--k',str(args.k),'--baseline-topm',str(args.baseline_topm),'--time-limit',str(args.time_limit),
                     '--gap',str(args.gap),'--seeds',*map(str,args.seeds),'--execute']
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(proc,log,plan)
                print(f'START {plan["key"]} GPU={gpu} PID={proc.pid}',flush=True)
            for gpu,(proc,log,plan) in list(active.items()):
                code=proc.poll()
                if code is None:
                    continue
                log.close(); del active[gpu]
                key='completed_maps' if code==0 else 'failed_maps'
                manifest[key].append(plan['key'])
                print(f'FINISH {plan["key"]} exit={code}',flush=True)
                save(root/'manifest.json',manifest)
                report(root)
            if active:
                time.sleep(2)
        if any(digest(path)!=value for path,value in {**hashes,**source}.items()):
            raise RuntimeError('Frozen source prototype changed during benchmark')
        manifest['status']='complete' if not manifest['failed_maps'] else 'partial'
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc))
        raise
    finally:
        for proc,log,plan in active.values():
            proc.terminate()
        for proc,log,plan in active.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            log.close()
        manifest['finished_at']=now()
        save(root/'manifest.json',manifest)
    print(json.dumps(report(root),indent=2),flush=True)


if __name__ == '__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--widths',nargs='+',type=float,default=[8,9,11,12,13])
    parser.add_argument('--scenarios',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--gpus',nargs='+',type=int,default=[1,2,3])
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--radius',type=float,default=10.)
    parser.add_argument('--k',type=int,default=3)
    parser.add_argument('--baseline-topm',type=int,default=10)
    parser.add_argument('--time-limit',type=float,default=300.)
    parser.add_argument('--gap',type=float,default=.05)
    parser.add_argument('--auxiliary-root',default='results/prototype_compatible_20261002')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--worker-scenario',type=int,help=SUPPRESS)
    parser.add_argument('--worker-width',type=float,help=SUPPRESS)
    args=parser.parse_args()
    if args.worker_scenario is not None:
        if not args.execute or args.worker_width is None:
            parser.error('Internal worker execution requires --execute and --worker-width')
        worker(args)
    elif args.report_only:
        print(report(args.output))
    else:
        launch(args)

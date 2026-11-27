"""Three road templates: two independent transfers, interpolation, and local MAPPO.

Run without --execute to inspect all six result slots per scenario. Incompatible
installation-line assignments remain unavailable; they are not fabricated runs.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import shutil
import sys
import time

import numpy as np

from prototype_offline import describe, digest, save, width_scenario
from prototype_transfer import export_view, load_prototype, prepare_relative_placements
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, Scenario, get_lidar_profile
from src.methods.local_mappo import refine_placement, validate_poses


SOURCES = {
    0: ('results/prototype_s0_k3_90gb_20261002', 'results/prototype_straight_w14_k3_20261002'),
    1: ('results/prototype_junctions_20261002/s1_w7', 'results/prototype_junctions_20261002/s1_w14'),
    2: ('results/prototype_junctions_20261002/s2_w7', 'results/prototype_junctions_20261002/s2_w14'),
}
BASE_METHODS = ('transfer_low', 'transfer_high', 'interpolation')
METHODS = (*BASE_METHODS, *(f'{name}_mappo' for name in BASE_METHODS))


def make_plan(args):
    if args.steps < 1 or args.k < 1 or not np.isfinite(args.radius) or args.radius <= 0:
        raise ValueError('Positive K, steps and radius required')
    plans = []
    for sid in args.scenarios:
        sources = list(SOURCES[sid])
        if sid in (1,2) and getattr(args, 'auxiliary_root', None):
            sources[1] = str(Path(args.auxiliary_root)/f's{sid}_w14')
        low, high = [load_prototype(p,args.k) for p in sources]
        cfg = low[5]
        for field in ('candidate','target','mappo'):
            if getattr(cfg,field) != getattr(high[5],field):
                raise ValueError(f'Source {field} mismatch for scenario {sid}')
        profile = get_lidar_profile(cfg.lidar_profile,cfg)
        if asdict(profile) != asdict(get_lidar_profile(high[5].lidar_profile,high[5])):
            raise ValueError('Source LiDAR profiles differ')
        target_spec = getattr(args, 'target_scenario', None)
        target = MapGeometry(Scenario(target_spec['id'], target_spec['topology'],
            target_spec['road_width'], target_spec['num_lanes'], target_spec) if target_spec else width_scenario(args.width,sid))
        if target.scenario.topology != low[3].scenario.topology or target.scenario.road_width != args.width:
            raise ValueError('Target topology/width must match the declared transfer plan')
        for g in (low[3],high[3]):
            for field in ('map_len','map_width','grid_size','sidewalk_width','building_height','median_width','road_centers_y','num_lanes'):
                if target_spec is not None and field == 'road_centers_y':
                    continue
                if getattr(g,field) != getattr(target,field):
                    raise ValueError(f'Only width changes supported: {field}')
        started = time.perf_counter()
        prepare = prepare_relative_placements
        if getattr(args, 'coordinate_mode', 'distance') == 'fraction':
            from prototype_segments import prepare_segment_placements
            prepare = prepare_segment_placements
        placements, metadata = prepare(low[4],high[4],low[3],high[3],target,cfg)
        elapsed = time.perf_counter()-started
        for poses in placements.values():
            validate_poses(poses,target,cfg)
            if len(poses) != args.k:
                raise ValueError('This paired pilot requires K actual sensors in every source')
        plans.append((sid,low,high,target,placements,metadata,elapsed))
    return plans


def record_result(root, name, poses, geometry, cfg, provider, extra=None):
    validate_poses(poses,geometry,cfg)
    started = time.perf_counter()
    density = np.stack([provider.mask(p) for p in poses]).sum(axis=0)
    rec = dict(status='complete', method=name, coverage_pct=100*float((density>0).mean()),
               covered_count=int((density>0).sum()), target_count=len(density),
               overlap_target_count=int((density>1).sum()), uncovered_count=int((density==0).sum()),
               num_selected=len(poses), evaluation_sec=time.perf_counter()-started,
               poses_npy=f'{name}.npy', view_html=f'{name}.html', **(extra or {}))
    np.save(root/f'{name}.npy',poses)
    np.savez_compressed(root/f'{name}_coverage.npz',target_points=provider.targets,density=density)
    export_view(root/rec['view_html'],geometry.create_environment_mesh(),provider.targets,density,
                poses,name,rec['coverage_pct'])
    save(root/f'{name}.json',rec)
    return rec


def summarize(output, reports):
    import plotly.graph_objects as go
    rows, links = [], []
    for report in reports:
        sid = report['parent_scenario_id']
        for name in METHODS:
            rec = report['results'][name]
            rows.append(dict(scenario=sid, topology=report['topology'], method=name,
                             status=rec['status'], coverage_pct=rec.get('coverage_pct'),
                             overlap_target_count=rec.get('overlap_target_count'),
                             uncovered_count=rec.get('uncovered_count'),
                             refinement_sec=rec.get('refinement',{}).get('elapsed_sec'),
                             reason=rec.get('reason','')))
            if rec['status'] == 'complete':
                links.append(f'<li>s{sid} <a href="s{sid}/{name}.html">{name}</a>: {rec["coverage_pct"]:.3f}%</li>')
            else:
                links.append(f'<li>s{sid} {name}: {rec["status"]}</li>')
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig = go.Figure()
    labels = dict(transfer_low='Low',transfer_high='High',interpolation='Interpolation',
                  transfer_low_mappo='Low + MAPPO',transfer_high_mappo='High + MAPPO',
                  interpolation_mappo='Interp. + MAPPO')
    for name in METHODS:
        selected = [r for r in rows if r['method'] == name and r['status'] == 'complete']
        fig.add_bar(name=labels[name], x=[f'S{r["scenario"]}' for r in selected],
                    y=[r['coverage_pct'] for r in selected])
    fig.update_layout(barmode='group',yaxis_title='Coverage (%)',yaxis_range=[0,100],
                      title=dict(text='Coverage comparison',font=dict(size=16)),xaxis_title='Scenario',
                      margin=dict(l=55,r=15,t=60,b=150),
                      legend=dict(orientation='h',x=0,y=-.2,font=dict(size=11)))
    fig.write_html(output/'comparison.html',include_plotlyjs=True,default_height='560px',config=dict(responsive=True))
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Prototype refinement</title><h1>Prototype refinement</h1>'
        '<p><a href="comparison.html">Coverage comparison</a> | <a href="comparison.csv">CSV</a></p>'
        '<p>Best deployment includes the initial pose. This is not final policy performance. '
        'Unavailable interpolation is not a zero coverage result.</p><ul>'+''.join(links)+'</ul></html>',encoding='utf-8')
    save(output/'report_source_hashes.json',dict(command=[sys.executable,*sys.argv],
         generated_at=datetime.now(timezone.utc).isoformat(),source_hashes={str(Path(__file__)):digest(__file__)},
         scope='CSV, comparison plot, index only; training source remains in manifest/source'))
    return rows


def run(args):
    if args.report_only:
        output = Path(args.output)
        manifest = json.loads((output/'manifest.json').read_text())
        if manifest['status'] not in ('complete','partial'):
            raise ValueError('Report-only requires a finished experiment')
        reports = [json.loads((output/f's{item["scenario"]}'/'report.json').read_text())
                   for item in manifest['scenarios']]
        summarize(output,reports)
        return
    plans = make_plan(args)
    for sid,low,high,target,placements,meta,elapsed in plans:
        print(json.dumps(dict(scenario=sid,methods=list(placements),interpolation=meta['status'],
                             reason=meta.get('reason'),steps=args.steps,seed=args.seed,
                             radius_m=args.radius,full_target_cache=False)),flush=True)
    if not args.execute:
        return
    import torch
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    source_paths = [Path(__file__),Path('prototype_transfer.py'),Path('prototype_offline.py'),
                    Path('prototype_compatible.py'),
                    *Path('src').rglob('*.py'),*Path('tests').glob('test*prototype*.py'),
                    Path('tests/test_local_mappo.py')]
    hashes = {}
    for path in source_paths:
        hashes[str(path)] = digest(path)
        dest = output/'source'/(path.name if path.is_absolute() else path)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    manifest = dict(status='running',schema_version='prototype_refinement_v1',
        started_at=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        arguments=vars(args),source_hashes=hashes,expected_result_slots=6*len(plans),
        steps_per_refinement=args.steps,seed=args.seed,local_radius_m=args.radius,
        interpolation='road-relative same branch/side only',
        limitations=['Width-only parametric maps, not actual OSM polylines',
            'Single seed pilot, no convergence or generalization claim',
            'No held-out MILP optimum; source results may be best-known only',
            '2m XY moves cannot connect every disconnected installation line',
            'Best deployment retains initial placement; improvement is not policy convergence'],
        full_target_cache=False,software=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),
        scenarios=[])
    if args.auxiliary_root:
        manifest['limitations'].append(
            'Auxiliary high-width prototypes are constrained by low-width installation allocation; '
            'single-high transfer is independent online, not independent offline construction')
    save(output/'manifest.json',manifest)
    reports = []
    try:
        for sid,low,high,target,placements,meta,elapsed in plans:
            root = output/f's{sid}'; root.mkdir()
            cfg = low[5]
            shutil.copy2(low[0]/'config.yaml',root/'config.yaml')
            save(root/'geometry.json',describe(target,cfg))
            report = dict(parent_scenario_id=sid,topology=target.scenario.topology,status='running',
                          budget_k=args.k,held_out_width_m=args.width,interpolation=meta,
                          placement_generation_sec=elapsed,results={},sources=[])
            for label,source in [('low',low),('high',high)]:
                directory = root/'library'/label; directory.mkdir(parents=True)
                for name in ('geometry.json','config.yaml'):
                    shutil.copy2(source[0]/name,directory/name)
                for name in ('manifest.json','allocation.json'):
                    if (source[0]/name).exists():
                        shutil.copy2(source[0]/name,directory/name)
                np.save(directory/'poses.npy',source[4])
                save(directory/'result.json',dict(source[1],positions_npy='poses.npy'))
                report['sources'].append(dict(root=str(source[0]),certification=source[1]['certification'],
                    installation_allocation=source[1].get('installation_allocation'),
                    optimality_domain=source[1].get('optimality_domain','declared discrete grid'),
                    pose_sha256=digest(source[0]/source[1]['positions_npy']),
                    result_sha256=digest(source[0]/f'k{args.k}/result.json')))
            for name in METHODS:
                base = name.removesuffix('_mappo')
                report['results'][name] = dict(status='pending' if base in placements else 'unavailable',
                                               reason='' if base in placements else meta['reason'])
            reports.append(report)
            save(root/'report.json',report)
            scene_started = time.perf_counter()
            provider = MeshVisibility(target.create_environment_mesh(),
                target.target_points(cfg.target['coverage_z_slices']),get_lidar_profile(cfg.lidar_profile,cfg))
            report['base_scene_setup_sec'] = time.perf_counter()-scene_started
            for name,poses in placements.items():
                report['results'][name] = record_result(root,name,poses,target,cfg,provider)
            save(root/'report.json',report)
            for name in BASE_METHODS:
                if name not in placements:
                    continue
                refined_name = name+'_mappo'
                report['results'][refined_name]['status'] = 'running'
                save(root/'report.json',report)
                best,metadata = refine_placement(placements[name],target.scenario,cfg,root/refined_name,
                    steps=args.steps,seed=args.seed,radius_m=args.radius)
                rec = record_result(root,refined_name,best,target,cfg,provider,dict(refinement=metadata))
                if not np.isclose(rec['coverage_pct'],metadata['coverage_pct'],rtol=0,atol=1e-6):
                    raise RuntimeError('Saved deployment coverage mismatch')
                report['results'][refined_name] = rec
                save(root/'report.json',report)
            report['status'] = 'complete' if meta['status'] == 'available' else 'partial'
            save(root/'report.json',report)
            manifest['scenarios'].append(dict(scenario=sid,status=report['status']))
            save(output/'manifest.json',manifest)
            summarize(output,reports)
        rows = summarize(output,reports)
        manifest['completed_results'] = sum(r['status']=='complete' for r in rows)
        manifest['unavailable_results'] = sum(r['status']=='unavailable' for r in rows)
        manifest['status'] = 'complete' if manifest['unavailable_results']==0 else 'partial'
    except BaseException as exc:
        manifest.update(status='failed',error=str(exc))
        if reports:
            reports[-1].update(status='failed',error=str(exc))
            save(output/f's{reports[-1]["parent_scenario_id"]}'/'report.json',reports[-1])
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(output/'manifest.json',manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--scenarios',type=int,nargs='+',choices=[0,1,2],default=[0,1,2])
    parser.add_argument('--width',type=float,default=10.)
    parser.add_argument('--k',type=int,default=3)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--radius',type=float,default=10.)
    parser.add_argument('--auxiliary-root',help='Use constrained s1_w14/s2_w14 prototypes from this root')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--report-only',action='store_true',help='Regenerate CSV/HTML without training')
    run(parser.parse_args())

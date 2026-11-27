"""Long-running, frozen A/B library stress benchmark. No RL training.

Default is a geometry-only dry-run. --execute --detach starts the supervisor;
only it monitors lifecycle/resources. Saved results are analyzed after completion.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import csv
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from src.core.library_transfer_evaluation import Work
from src.core.visibility import MeshVisibility
from src.environment import angular_visibility_mask, compute_view_angles_numpy
from src.methods.coverage_cut_milp import PackedCoverage
from src.methods.library_query_controls import _optical_masks, _profile_values
from src.methods.rsu_coverage_milp import assess, solve_frontier
from src.roadside.rsu_stress import domain, retrieve, scene, transfer_ids


BASE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def default_spec():
    sources = []
    for topology in ['T_JUNCTION', 'CROSS']:
        for size in [100.,160.]:
            for width in [8.,12.,16.]:
                sources.append(dict(name=f'a_{topology.lower()}_l{int(size)}_w{int(width)}', track='A',
                    topology=topology, size_m=size, width_m=width, center_fraction=[.5,.5], seed=1000+len(sources)))
    for length in [80.,120.]:
        for width in [8.,12.,16.]:
            for turn in [0.,45.]:
                sources.append(dict(name=f'b_l{int(length)}_w{int(width)}_t{int(turn)}', track='B',
                    length_m=length, width_m=width, turn_deg=turn, modules=2, seed=1000+len(sources)))
    targets = []
    for topology, size, width, center in [('T_JUNCTION',120.,10.,[.42,.55]),
            ('T_JUNCTION',180.,14.,[.57,.43]),('CROSS',120.,10.,[.55,.42]),('CROSS',180.,14.,[.43,.57])]:
        targets.append(dict(name=f'target_a_{topology.lower()}_l{int(size)}',track='A',topology=topology,
            size_m=size,width_m=width,center_fraction=center,seed=2000+len(targets)))
    for length, width, turn in [(160.,10.,15.),(240.,14.,30.),(320.,10.,60.),(480.,14.,75.)]:
        targets.append(dict(name=f'target_b_l{int(length)}',track='B',length_m=length,width_m=width,
            turn_deg=turn,modules=int(length/80),seed=2000+len(targets)))
    return dict(schema='library_rsu_stress_v1', sources=sources, targets=targets,
        sampling=dict(installation_step_m=4., heights_m=[3.,5.], relative_yaws_deg=[-45.,-22.5,0.,22.5,45.],
                      pitches_deg=[-20.,-10.], target_heights_m=[1.,1.5,2.], target_b_step_m=1.),
        source_k=list(range(1,9)), source_redundancy_k=[4,8], target_k=[4,8,12],
        source_count=3, alternatives=12, query_budgets=[128,256], random_seeds=[0,1,2],
        scene_levels=['clear','mixed','dense'], source_scene='mixed',
        requirements=dict(union=dict(overlap_ratio=0.,critical_overlap_ratio=0.,min_separation_m=4.),
                          redundancy=dict(overlap_ratio=.10,critical_overlap_ratio=.30,min_separation_m=4.)),
        optimization=dict(source_seconds=1800., reference_seconds=1800., arm_seconds=300.,
                          surrogate_seconds=15., gap=.01, min_seconds=300., stagnation_seconds=120.,
                          threads=1, soft_memory_gb=12.),
        execution=dict(workers=4, hard_wall_hours=120., job_wall_hours=18., min_free_ram_gib=32.,
                       min_free_disk_gib=8., max_job_rss_gib=24., max_total_rss_gib=80.))


def smoke_spec():
    s = default_spec()
    s['sources'] = [s['sources'][0],s['sources'][1],s['sources'][12],s['sources'][14]]
    s['targets'] = [s['targets'][0],s['targets'][4]]
    s.update(source_k=[1,2,4], source_redundancy_k=[4], target_k=[2,4], source_count=1,
             alternatives=3, query_budgets=[8,16], random_seeds=[0], scene_levels=['clear','dense'])
    s['sampling'].update(installation_step_m=20.,heights_m=[3.],relative_yaws_deg=[0.],
                        pitches_deg=[-10.],target_heights_m=[1.5],target_b_step_m=2.)
    s['optimization'].update(source_seconds=2.,reference_seconds=2.,arm_seconds=.5,
                            surrogate_seconds=.2,min_seconds=.1,stagnation_seconds=.1)
    s['execution'].update(workers=2,hard_wall_hours=1.,job_wall_hours=.5)
    return s


def plan(spec):
    if set(spec)!=set(default_spec()) or spec['schema']!='library_rsu_stress_v1':
        raise ValueError('Unknown study schema or settings')
    for name in ['source_k','source_redundancy_k','target_k','query_budgets','random_seeds']:
        x = spec[name]
        low = 0 if name=='random_seeds' else 1
        if not x or x!=sorted(set(x)) or any(type(v) is not int or v<low for v in x):
            raise ValueError('Invalid integer axis '+name)
    if max(spec['source_k']+spec['source_redundancy_k']+spec['target_k'])>64:
        raise ValueError('K safety limit exceeded')
    sampling = spec['sampling']
    for name in ['heights_m','relative_yaws_deg','pitches_deg','target_heights_m']:
        x = sampling[name]
        if not x or x!=sorted(set(x)) or not np.isfinite(x).all():
            raise ValueError('Invalid sampling axis')
        if name!='target_heights_m' and len(x)>2 and not np.allclose(np.diff(x),x[1]-x[0]):
            raise ValueError('Uniform A sampling axes required')
    if sampling['installation_step_m']<=0 or sampling['target_b_step_m']<=0:
        raise ValueError('Positive spatial spacing required')
    if not 1<=spec['source_count']<=10 or not 1<=spec['alternatives']<=64:
        raise ValueError('Invalid library size')
    opt, exe = spec['optimization'], spec['execution']
    if (set(opt)!=set(default_spec()['optimization']) or set(exe)!=set(default_spec()['execution'])
            or not np.isfinite(list(opt.values())+list(exe.values())).all()
            or any(v<=0 for v in opt.values()) or any(v<=0 for v in exe.values())
            or type(exe['workers']) is not int or not 1<=exe['workers']<=4
            or type(opt['threads']) is not int or not 1<=opt['threads']<=4
            or not 0<opt['gap']<=.05):
        raise ValueError('Invalid execution/resource limits')
    if set(spec['requirements'])!={'union','redundancy'}:
        raise ValueError('Explicit union and redundancy profiles required')
    for req in spec['requirements'].values():
        if (set(req)!={'overlap_ratio','critical_overlap_ratio','min_separation_m'}
                or not np.isfinite(list(req.values())).all() or req['min_separation_m']<0
                or not 0<=req['overlap_ratio']<=1 or not 0<=req['critical_overlap_ratio']<=1):
            raise ValueError('Invalid constraints')
    if (not spec['scene_levels'] or len(set(spec['scene_levels']))!=len(spec['scene_levels'])
            or set(spec['scene_levels'])-{'clear','mixed','dense'} or spec['source_scene'] not in ('clear','mixed','dense')):
        raise ValueError('Invalid scenes')
    names, identities, domains = set(), set(), []
    for label in ['sources','targets']:
        for d in spec[label]:
            if d['name'] in names or not all(c.isalnum() or c in '_-' for c in d['name']):
                raise ValueError('Duplicate/unsafe name')
            names.add(d['name'])
            identity=json.dumps({k:v for k,v in d.items() if k not in ('name','seed')},sort_keys=True)
            if identity in identities:
                raise ValueError('Duplicate source/target geometry')
            identities.add(identity)
            problem=domain(d,sampling)
            if label=='targets':
                retrieve(spec['sources'],d,spec['source_count'])
            domains.append(dict(name=d['name'],track=d['track'],role=label,
                candidates=len(problem['poses']),targets=len(problem['targets']),
                dense_mask_gib=len(problem['poses'])*len(problem['targets'])/2**30,
                removed_targets=problem['removed_targets'],objects=len(problem['boxes'])))
    cases=len(spec['targets'])*len(spec['scene_levels'])
    arms=4+len(spec['random_seeds'])
    rows=cases*len(spec['requirements'])*len(spec['query_budgets'])*arms*len(spec['target_k'])
    source_solves=len(spec['sources'])*(len(spec['source_k'])+len(spec['source_redundancy_k']))
    reference_solves=cases*len(spec['requirements'])*len(spec['target_k'])
    cap=(source_solves*opt['source_seconds']+reference_solves*opt['reference_seconds']+rows*opt['arm_seconds'])
    return dict(sources=len(spec['sources']),geometry_count=len(spec['targets']),case_jobs=cases,
        constraint_conditions=cases*len(spec['requirements']),expected_rows=rows,domains=domains,
        solver_cap_sum_hours=cap/3600,workers=exe['workers'],
        note='Cap sum is not forecast; no artificial minimum experiment duration. Synthetic static coverage, no detector or RL.')


def query(mesh, problem, ids, *, direct=False):
    start=time.perf_counter()
    work=Work()
    provider=MeshVisibility(mesh,problem['targets'],problem['profile'],work=work)
    masks=np.zeros((len(ids),len(problem['targets'])),dtype=bool)
    los={}
    profile=problem['profile']
    targets=problem['targets'].astype(np.float32)
    for j,i in enumerate(ids):
        pose=problem['poses'][i].astype(np.float32)
        if direct:
            masks[j]=provider.mask(pose)
        else:
            xyz=tuple(pose[:3])
            if xyz not in los:
                los[xyz]=provider.line_of_sight_mask(pose[:3])
            vectors=targets-pose[:3]
            az,el,_,distance=compute_view_angles_numpy(*vectors.T,pose[3],pose[4])
            masks[j]=los[xyz]&angular_visibility_mask(az,el,distance,np.deg2rad(profile.h_fov_deg)/2,
                                                    np.deg2rad(profile.v_fov_deg)/2,profile.max_range_m)
    return masks,time.perf_counter()-start,dict(work.counts)


def optimize(problem,masks,spec,role,req,k_values,directory,alternatives=1):
    directory.mkdir(parents=True,exist_ok=True)
    opt=spec['optimization']
    f=solve_frontier(problem['poses'],masks,problem['critical'],k_values,req,
        seconds=opt[role+'_seconds'],alternatives=alternatives,threads=opt['threads'],
        soft_mem_gb=opt['soft_memory_gb'],stop_gap=opt['gap'],
        stop_min_sec=min(opt['min_seconds'],opt[role+'_seconds']),stop_stagnation_sec=opt['stagnation_seconds'],
        progress_path=directory/'progress.jsonl')
    save(directory/'frontier.json',f)
    write_csv(directory/'quality_time.csv',[dict(k=row['budget_k'],**point)
        for row in f['frontier'] for point in row['quality_time_trace']])
    return f


def save_masks(path,problem,masks):
    np.savez_compressed(path,poses=problem['poses'],targets=problem['targets'],critical=problem['critical'],
                        packed_masks=np.packbits(masks,axis=1))


def source_job(root,spec,d):
    start=time.perf_counter()
    directory=root/'library'/d['name']
    directory.mkdir(parents=True,exist_ok=False)
    problem=domain(d,spec['sampling'])
    mesh=scene(problem,spec['source_scene'])
    preparation=time.perf_counter()-start
    masks,visibility,work=query(mesh,problem,range(len(problem['poses'])))
    profiles={}
    for name,req in spec['requirements'].items():
        ks=spec['source_k'] if name=='union' else spec['source_redundancy_k']
        profiles[name]=optimize(problem,masks,spec,'source',req,ks,directory/name,spec['alternatives'])
    selected=sorted({i for f in profiles.values() for row in f['frontier'] for a in row['alternatives'] for i in a['selected_ids']})
    checked,validation,vwork=query(mesh,problem,selected,direct=True)
    if not np.array_equal(checked,masks[selected]):
        raise RuntimeError('Fresh direct source mask validation failed')
    save_masks(directory/'candidates.npz',problem,masks)
    canonical=problem['canonical'] if d['track']=='A' else problem['canonical'].tolist()
    save(directory/'layouts.json',dict(descriptor=d,canonical=canonical,profiles=profiles))
    result=dict(status='complete',descriptor=d,preparation_sec=preparation,visibility_sec=visibility,
        validation_sec=validation,work=work,validation_work=vwork,wall_sec=time.perf_counter()-start,
        validated_direct=True,artifact_hashes={name:digest(directory/name) for name in ('candidates.npz','layouts.json')})
    save(directory/'result.json',result)
    return result


def library_pool(root,spec,problem):
    start=time.perf_counter()
    entries=[]
    for d in retrieve(spec['sources'],problem['descriptor'],spec['source_count']):
        directory=root/'library'/d['name']
        result=json.loads((directory/'result.json').read_text())
        for name,expected in result['artifact_hashes'].items():
            if digest(directory/name)!=expected:
                raise RuntimeError('Source archive mutated')
        data=json.loads((directory/'layouts.json').read_text())
        rows=[r for f in data['profiles'].values() for r in f['frontier']]
        solutions=[r['alternatives'][rank]['selected_ids'] for rank in range(spec['alternatives'])
                   for r in rows if rank<len(r['alternatives'])]
        order,records=transfer_ids(data,solutions,problem)
        entries.append(dict(source=d['name'],order=order,records=records))
    # Preserve the historical round-robin prefix as a disclosed ablation.
    order,seen=[],set()
    for rank in range(max((len(e['order']) for e in entries),default=0)):
        for e in entries:
            if rank<len(e['order']) and e['order'][rank] not in seen:
                order.append(e['order'][rank]); seen.add(e['order'][rank])
    return dict(order=order,sources=entries,planning_sec=time.perf_counter()-start)


def shortlist(masks,poses,q,*,double=False):
    """Geometry-only union/double response ranking, with spatial fill after saturation."""
    if q==0:
        return []
    packed=PackedCoverage(np.packbits(masks,axis=1),masks.shape[1])
    density=np.zeros(masks.shape[1],dtype=np.int32)
    used=np.zeros(len(masks),dtype=bool)
    distances=np.full(len(masks),np.inf)
    order=[]
    for _ in range(q):
        gains=packed.gains(np.packbits(density>=1)).astype(float)
        if double:
            gains=.5*gains+.5*packed.gains(np.packbits(density>=2))
        gains[used]=-np.inf
        if np.max(gains)<=0:
            gains=distances.copy();gains[used]=-np.inf
        cid=int(np.argmax(gains))
        order.append(cid);used[cid]=True;density+=masks[cid]
        distances=np.minimum(distances,np.linalg.norm(poses[:,:2]-poses[cid,:2],axis=1))
    return order


def case_job(root,spec,d,level):
    start=time.perf_counter()
    name=d['name']+'_'+level
    directory=root/'cases'/name
    directory.mkdir(parents=True,exist_ok=False)
    problem=domain(d,spec['sampling'])
    mesh=scene(problem,level)
    preparation=time.perf_counter()-start
    pool=library_pool(root,spec,problem)
    save(directory/'transfer.json',pool)
    pool_ids=pool['order']
    qmax=min(max(spec['query_budgets']),len(pool_ids))
    optical_start=time.perf_counter()
    optical=_optical_masks(problem['poses'].astype(np.float32),problem['targets'].astype(np.float32),_profile_values(problem['profile']))
    optical_time=time.perf_counter()-optical_start
    union_start=time.perf_counter()
    union_order=shortlist(optical,problem['poses'],qmax)
    union_time=time.perf_counter()-union_start+optical_time
    library_start=time.perf_counter()
    # Charge the library only its own optical calculation, not the full-domain geometry cache.
    library_optical=_optical_masks(problem['poses'][pool_ids].astype(np.float32),problem['targets'].astype(np.float32),_profile_values(problem['profile']))
    diverse_order=[pool_ids[i] for i in shortlist(library_optical,problem['poses'][pool_ids],qmax,double=True)]
    diverse_time=time.perf_counter()-library_start+pool['planning_sec']
    del library_optical
    plans=[]
    for profile,req in spec['requirements'].items():
        surrogate_start=time.perf_counter()
        surrogate=optimize(problem,optical,spec,'surrogate',req,spec['target_k'],directory/('surrogate_'+profile),6)
        candidates=[]
        for rank in range(6):
            for row in surrogate['frontier']:
                if rank<len(row['alternatives']):
                    candidates.extend(row['alternatives'][rank]['selected_ids'])
        portfolio=list(dict.fromkeys(candidates+union_order))[:qmax]
        if len(portfolio)<qmax:
            raise RuntimeError('Portfolio failed to fill query budget')
        portfolio_time=time.perf_counter()-surrogate_start+union_time
        for requested in spec['query_budgets']:
            q=min(requested,len(pool_ids))
            methods=[('library_prefix',None,pool_ids[:q],pool['planning_sec']),
                     ('library_diverse',None,diverse_order[:q],diverse_time),
                     ('geometry_union',None,union_order[:q],union_time),
                     ('geometry_portfolio',None,portfolio[:q],portfolio_time)]
            for seed in spec['random_seeds']:
                before=time.perf_counter()
                ids=np.random.default_rng(seed).permutation(len(problem['poses']))[:q].tolist()
                methods.append(('random',seed,ids,time.perf_counter()-before))
            for method,seed,ids,planning in methods:
                plans.append(dict(profile=profile,method=method,seed=seed,requested_q=requested,actual_q=q,
                    pool_count=len(pool_ids),selected_ids=sorted(ids),planning_sec=planning,
                    arm_id=f'{profile}_q{requested}_{method}'+(f'_s{seed}' if seed is not None else '')))
    del optical
    save(directory/'query_plans.json',plans)
    plans_hash=digest(directory/'query_plans.json')
    arm_results=[]
    for p in plans:
        arm=directory/p['arm_id']
        arm.mkdir()
        ids=p['selected_ids']
        if len(ids)!=p['actual_q'] or len(set(ids))!=len(ids):
            raise RuntimeError('Query matching/uniqueness failed')
        masks,visibility,work=query(mesh,problem,ids)
        subset=dict(problem,poses=problem['poses'][ids])
        frontier=optimize(subset,masks,spec,'arm',spec['requirements'][p['profile']],spec['target_k'],arm)
        for row in frontier['frontier']:
            row['full_candidate_ids']=[ids[i] for i in row['selected_ids']] if row['selected_ids'] is not None else None
        np.savez_compressed(arm/'queried.npz',candidate_ids=ids,packed_masks=np.packbits(masks,axis=1))
        result=dict(plan=p,frontier=frontier,visibility_sec=visibility,work=work,
                    online_all_k_sec=preparation+p['planning_sec']+visibility+frontier['wall_sec'])
        save(arm/'result.json',result)
        arm_results.append(result)
    full_masks,full_visibility,full_work=query(mesh,problem,range(len(problem['poses'])))
    references={}
    for profile,req in spec['requirements'].items():
        references[profile]=optimize(problem,full_masks,spec,'reference',req,spec['target_k'],directory/('reference_'+profile))
    save_masks(directory/'candidates.npz',problem,full_masks)
    selected=set()
    for f in references.values():
        for row in f['frontier']:
            selected.update(row['selected_ids'] or [])
    for result in arm_results:
        p=result['plan']
        with np.load(directory/p['arm_id']/'queried.npz',allow_pickle=False) as data:
            queried=np.unpackbits(data['packed_masks'],axis=1,count=len(problem['targets'])).astype(bool)
            if not np.array_equal(queried,full_masks[p['selected_ids']]):
                raise RuntimeError('Arm/reference visibility mismatch')
        for row in result['frontier']['frontier']:
            selected.update(row['full_candidate_ids'] or [])
    ids=sorted(selected)
    checked,validation,vwork=query(mesh,problem,ids,direct=True)
    if not np.array_equal(checked,full_masks[ids]) or digest(directory/'query_plans.json')!=plans_hash:
        raise RuntimeError('Independent visibility/immutable plan check failed')
    rows=[]
    for result in arm_results:
        p=result['plan'];cumulative=result['frontier']['preparation_sec']
        for row,ref in zip(result['frontier']['frontier'],references[p['profile']]['frontier']):
            cumulative+=row['selection_sec']
            values={}
            loss_lower=loss_upper=None
            if row['selected_ids'] is not None:
                values=assess(problem['poses'],full_masks,row['full_candidate_ids'],problem['critical'],
                              spec['requirements'][p['profile']],row['budget_k'])
                if values!=row['metrics'] or not values['feasible']:
                    raise RuntimeError('Independent constraint/density check failed')
                if ref['upper_bound_count'] is None or values['covered_count']>ref['upper_bound_count']+1e-5:
                    raise RuntimeError('Reference certificate contradicts feasible arm')
                loss_upper=max(0.,100*ref['upper_bound_count']/len(problem['targets'])-values['coverage_pct'])
                if ref['metrics'] is not None:
                    loss_lower=max(0.,ref['metrics']['coverage_pct']-values['coverage_pct'])
            rows.append(dict(case=name,track=d['track'],scene=level,profile=p['profile'],method=p['method'],
                seed=p['seed'],requested_q=p['requested_q'],actual_q=p['actual_q'],pool_count=p['pool_count'],
                candidate_count=len(problem['poses']),target_count=len(problem['targets']),budget_k=row['budget_k'],
                status=row['status'],**{k:v for k,v in values.items() if k!='violations'},
                queried_pool_gap=row['certified_gap'],reference_status=ref['status'],reference_gap=ref['certified_gap'],
                reference_coverage_pct=ref['metrics']['coverage_pct'] if ref['metrics'] else None,
                loss_lower_pp=loss_lower,loss_upper_pp=loss_upper,preparation_sec=preparation,
                planning_sec=p['planning_sec'],visibility_sec=result['visibility_sec'],k_selection_sec=row['selection_sec'],
                solver_sec=row['solver_sec'],online_through_k_sec=preparation+p['planning_sec']+result['visibility_sec']+cumulative,
                online_all_k_sec=result['online_all_k_sec'],reference_visibility_sec=full_visibility,
                reference_all_k_sec=preparation+full_visibility+references[p['profile']]['wall_sec'],
                rays_submitted=result['work'].get('rays_submitted',0)))
    write_csv(directory/'comparison.csv',rows)
    result=dict(status='complete',case=name,rows=len(rows),validated_direct=True,validation_sec=validation,
        validation_work=vwork,reference_work=full_work,plans_sha256=plans_hash,wall_sec=time.perf_counter()-start,
        scope='No failures converted to zero coverage. Infeasible and no_feasible_found are distinct.')
    save(directory/'result.json',result)
    return result


def verify(root):
    m=json.loads((root/'manifest.json').read_text())
    if digest(root/'spec.json')!=m['spec_sha256']:
        raise RuntimeError('Frozen spec changed')
    for name,h in m['source_hashes'].items():
        if digest(root/'source'/name)!=h:
            raise RuntimeError('Frozen code changed: '+name)
    return m,json.loads((root/'spec.json').read_text())


def worker(root,job):
    if BASE!=root/'source':
        raise ValueError('Worker must run frozen code')
    manifest,spec=verify(root)
    item=next(x for x in manifest['jobs'] if x['id']==job)
    value=source_job(root,spec,item['descriptor']) if item['phase']=='source' else case_job(root,spec,item['descriptor'],item['scene'])
    verify(root)
    value['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    save(root/'job_status'/f'{job}.json',value)


def available_ram():
    fields=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0])*1024/2**30


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])/2**20
    except FileNotFoundError:
        pass
    return 0.


def hardware(root):
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'unknown')
    return dict(cpu_model=cpu, logical_cpus=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
                available_ram_gib=available_ram(), free_disk_gib=shutil.disk_usage(root).free/2**30,
                platform=sys.platform, note='Shared server; no exclusive CPU or OS-cache isolation.')


def aggregate(root):
    rows=[]
    for path in sorted((root/'cases').glob('*/comparison.csv')):
        with path.open(newline='') as stream:
            rows.extend(csv.DictReader(stream))
    write_csv(root/'comparison.csv',rows)
    return len(rows)


def orchestrate(root, *, entrypoint='library_rsu_stress.py'):
    if BASE!=root/'source':
        raise ValueError('Supervisor must run frozen code')
    manifest,spec=verify(root)
    exe=spec['execution'];active={};started=time.monotonic()
    def stop_all():
        for process,_,stream in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process,_,stream in active.values():
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            stream.close()
    def interrupted(signum,frame):
        raise RuntimeError(f'Supervisor interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        manifest.update(status='running',started_at=now(),supervisor_pid=os.getpid())
        save(root/'manifest.json',manifest)
        for phase in ['source','case']:
            pending=[j for j in manifest['jobs'] if j['phase']==phase]
            while pending or active:
                if time.monotonic()-started>exe['hard_wall_hours']*3600:
                    raise TimeoutError('Total hard wall limit; completed artifacts preserved')
                total=sum(rss(p.pid) for p,_,_ in active.values())
                if total>exe['max_total_rss_gib'] or available_ram()<8.:
                    raise MemoryError('Worker total RSS/system available RAM safety guard')
                changed=False
                for jid,(process,began,stream) in list(active.items()):
                    if rss(process.pid)>exe['max_job_rss_gib'] or time.monotonic()-began>exe['job_wall_hours']*3600:
                        raise RuntimeError('Worker resource/time guard: '+jid)
                    code=process.poll()
                    if code is not None:
                        stream.close();del active[jid]
                        item=next(j for j in manifest['jobs'] if j['id']==jid)
                        item.update(status='complete' if code==0 else 'failed',finished_at=now(),returncode=code)
                        if code or not (root/'job_status'/f'{jid}.json').exists():
                            raise RuntimeError('Worker failed: '+jid+'; see its saved log')
                        changed=True
                        print('complete:',jid,flush=True)
                while pending and len(active)<exe['workers']:
                    if shutil.disk_usage(root).free/2**30<exe['min_free_disk_gib']:
                        raise OSError('Insufficient disk safety reserve')
                    if available_ram()<exe['min_free_ram_gib']:
                        if active:break
                        raise MemoryError('Insufficient available RAM to launch worker')
                    item=pending.pop(0);jid=item['id']
                    stream=(root/'logs'/f'{jid}.log').open('w')
                    command=[sys.executable,'-u',str(BASE/entrypoint),'--worker',str(root),'--job',jid,'--execute']
                    process=subprocess.Popen(command,cwd=BASE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    active[jid]=(process,time.monotonic(),stream)
                    item.update(status='running',pid=process.pid,started_at=now());changed=True
                if changed:
                    manifest.update(updated_at=now(),completed=sum(j['status']=='complete' for j in manifest['jobs']))
                    save(root/'manifest.json',manifest)
                time.sleep(3.)
        rows=aggregate(root)
        if rows!=manifest['plan']['expected_rows']:
            raise RuntimeError('Incomplete final CSV')
        verify(root)
        manifest.update(status='complete',row_count=rows,finished_at=now())
    except BaseException as exc:
        stop_all()
        for item in manifest['jobs']:
            if item['status']=='running':item['status']='interrupted'
        manifest.update(status='failed',error=repr(exc),finished_at=now())
        raise
    finally:
        manifest['wall_sec']=time.monotonic()-started
        save(root/'manifest.json',manifest)


def execute(spec,output,detach):
    planned=plan(spec)
    root=output.resolve()
    if root.exists():
        raise FileExistsError('Use a new output root; existing results are never overwritten')
    if available_ram()<spec['execution']['min_free_ram_gib']:
        raise MemoryError('Insufficient RAM for preflight')
    if shutil.disk_usage(root.parent if root.parent.exists() else BASE).free/2**30<spec['execution']['min_free_disk_gib']+4:
        raise OSError('Insufficient disk reserve for this run')
    root.mkdir(parents=True)
    for name in ['source','logs','job_status']:(root/name).mkdir()
    source_hashes={}
    snapshot_paths = list(BASE.glob('*.py'))+list((BASE/'src').rglob('*.py'))
    snapshot_paths += [BASE/'tests/test_rsu_stress.py', BASE/'LIBRARY_RSU_STRESS_PROTOCOL.md',
                       BASE/'configs/library_rsu_stress_20261123.json']
    for path in snapshot_paths:
        rel=path.relative_to(BASE);dest=root/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);source_hashes[str(rel)]=digest(dest)
    save(root/'spec.json',spec)
    jobs=[dict(id='source_'+d['name'],phase='source',descriptor=d,status='pending') for d in spec['sources']]
    jobs.extend(dict(id='case_'+d['name']+'_'+level,phase='case',descriptor=d,scene=level,status='pending')
                for d in spec['targets'] for level in spec['scene_levels'])
    packages={n:metadata.version(n) for n in ['numpy','scipy','open3d','gurobipy']}
    save(root/'manifest.json',dict(schema=spec['schema'],status='prepared',created_at=now(),plan=planned,
        command=sys.argv,source_hashes=source_hashes,spec_sha256=digest(root/'spec.json'),jobs=jobs,
        packages=packages,python=sys.version,hardware=hardware(root),completed=0))
    (root/'packages.txt').write_text('\n'.join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()))+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    command=[sys.executable,'-u',str(root/'source/library_rsu_stress.py'),'--orchestrate',str(root),'--execute']
    if detach:
        with (root/'launcher.log').open('w') as stream:
            process=subprocess.Popen(command,cwd=root/'source',env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'launch.json',dict(pid=process.pid,command=command,started_at=now()))
        print(json.dumps(dict(status='launched',pid=process.pid,root=str(root),plan=planned),indent=2))
    else:
        subprocess.run(command,cwd=root/'source',env=env,check=True)


def main(argv=None):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/library_rsu_stress_20261123'))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--orchestrate',type=Path)
    parser.add_argument('--job')
    args=parser.parse_args(argv)
    if args.worker and args.orchestrate:
        parser.error('Choose worker or supervisor, not both')
    if (args.worker or args.orchestrate) and not args.execute:
        parser.error('Internal execution also requires --execute')
    if args.worker:
        worker(args.worker.resolve(),args.job);return
    if args.orchestrate:
        orchestrate(args.orchestrate.resolve());return
    if args.smoke and args.spec:
        parser.error('--smoke and --spec are mutually exclusive')
    spec=smoke_spec() if args.smoke else json.loads(args.spec.read_text()) if args.spec else default_spec()
    if args.execute:execute(spec,args.output,args.detach)
    else:print(json.dumps(plan(spec),indent=2))


if __name__=='__main__':
    main()

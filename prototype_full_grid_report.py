"""Validate saved poses against full-grid masks and report without training/solving."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from prototype_offline import digest, save


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Archive:
    def __init__(self, root):
        self.poses = np.load(root/'poses.npy', mmap_mode='r', allow_pickle=False)
        self.masks = np.load(root/'packed_masks.npy', mmap_mode='r', allow_pickle=False)
        self.targets = np.load(root/'targets.npy', allow_pickle=False)
        changed = np.ones(len(self.poses), dtype=bool)
        changed[1:] = np.any(self.poses[1:, :3] != self.poses[:-1, :3], axis=1)
        starts = np.flatnonzero(changed)
        ends = np.r_[starts[1:], len(self.poses)]
        self.blocks = {tuple(self.poses[start, :3]): (start, end) for start, end in zip(starts, ends)}
        if len(self.blocks) != len(starts):
            raise ValueError('Archive XYZ blocks must be contiguous and unique')

    def check(self, poses, targets, density, coverage, k):
        if poses.ndim != 2 or poses.shape[1] != 5 or not 0 < len(poses) <= k or not np.isfinite(poses).all():
            raise ValueError('Invalid deployment')
        if len(np.unique(poses, axis=0)) != len(poses):
            raise ValueError('Duplicate deployment poses')
        if not np.array_equal(self.targets, targets):
            raise ValueError('Target coordinates differ')
        reconstructed = np.zeros(len(targets), dtype=np.int32)
        for pose in poses:
            start, end = self.blocks[tuple(pose[:3])]
            block = self.poses[start:end]
            angular_error = np.abs((block[:, 3]-pose[3]+180) % 360-180)
            found = np.flatnonzero((angular_error < 1e-4) & np.isclose(block[:, 4], pose[4], atol=1e-5, rtol=0))
            if len(found) != 1:
                raise ValueError('Pose missing or ambiguous in full archive')
            reconstructed += np.unpackbits(self.masks[start+found[0]])[:len(targets)]
        if not np.array_equal(reconstructed, density):
            raise ValueError('Stored density differs from full-grid archive')
        value = 100*int((reconstructed>0).sum())/len(targets)
        if abs(value-coverage) > 1e-8:
            raise ValueError('Stored coverage differs from archive')


def report(root, study):
    root, study = Path(root), Path(study)
    full = json.loads((root/'manifest.json').read_text())
    online = json.loads((study/'manifest.json').read_text())
    for manifest in (full, online):
        if manifest['status'] != 'complete' or any(j['status'] != 'complete' for j in manifest['jobs']):
            raise ValueError('Both experiments must be complete')
    raw, references, hashes = [], {}, {}
    for item in full['jobs']:
        sid = item['scenario']['id']
        directory = root/f's{sid}'
        archive = Archive(directory/'candidates')
        for k in item['k_values']:
            path = directory/f'k{k}'
            r = json.loads((path/'result.json').read_text())
            if not r['validated_direct_raycast']:
                raise ValueError('Reference direct validation absent')
            with np.load(path/'coverage.npz', allow_pickle=False) as saved:
                archive.check(np.load(path/'poses.npy', allow_pickle=False), saved['targets'], saved['density'], r['coverage_pct'], k)
            references[(sid,k)] = r
            hashes[str(path/'result.json')] = digest(path/'result.json')
        for job in online['jobs']:
            if job['scenario']['id'] != sid:
                continue
            if job['scenario'] != item['scenario']:
                raise ValueError('Scenario metadata differ')
            path = study/'jobs'/job['id']
            r = json.loads((path/'result.json').read_text())
            if r['full_target_cache'] or r['checkpoint_reused']:
                raise ValueError('Unexpected online cache/checkpoint reuse')
            k = job['budget_k']
            with np.load(path/'coverage.npz', allow_pickle=False) as saved:
                archive.check(np.load(path/'deployment.npy', allow_pickle=False), saved['targets'], saved['density'], r['coverage_pct'], k)
            ref = references[(sid,k)]
            raw.append(dict(scenario_id=sid,K=k,initialization=job['initialization'],method=job['method'],seed=job['seed'],
                initial_pose_sha256=r['initial_pose_sha256'],initial_coverage_pct=r['initial_coverage_pct'],
                coverage_pct=r['coverage_pct'],reference_coverage_pct=ref['coverage_pct'],
                minus_reference_pp=r['coverage_pct']-ref['coverage_pct'],
                online_sec=r['online_runtime']['elapsed_sec'],archive_density_equal=True))
            hashes[str(path/'result.json')] = digest(path/'result.json')
    if len(raw) != len(online['jobs']):
        raise ValueError('Some online jobs have no reference')
    keys = sorted({(r['scenario_id'],r['K'],r['initialization']) for r in raw})
    pairs = []
    for sid,k,initialization in keys:
        group = [r for r in raw if (r['scenario_id'],r['K'],r['initialization'])==(sid,k,initialization)]
        ls = [r for r in group if r['method']=='local_search']
        rl = [r for r in group if r['method']=='mappo']
        if len(ls)!=1 or len(rl)!=len(online['seeds']) or {r['seed'] for r in rl}!=set(online['seeds']):
            raise ValueError('Incomplete or duplicate comparison group')
        if len({r['initial_pose_sha256'] for r in group})!=1:
            raise ValueError('Initial poses differ')
        ref = references[(sid,k)]
        average = mean(r['coverage_pct'] for r in rl)
        pairs.append(dict(scenario_id=sid,K=k,initialization=initialization,
            reference_coverage_pct=ref['coverage_pct'],reference_relative_gap=ref['certified_gap'],
            initial_coverage_pct=ls[0]['initial_coverage_pct'],local_search_coverage_pct=ls[0]['coverage_pct'],
            mappo_mean_pct=average,mappo_sample_sd_pct=stdev(r['coverage_pct'] for r in rl),
            local_search_minus_reference_pp=ls[0]['minus_reference_pp'],mappo_minus_reference_pp=average-ref['coverage_pct'],
            local_search_online_sec=ls[0]['online_sec'],mappo_mean_online_sec=mean(r['online_sec'] for r in rl),
            all_poses_in_archive=True,all_density_equal=True))
    output = root/'comparison_report'
    output.mkdir(exist_ok=True)
    write_csv(output/'raw_comparison.csv',raw)
    write_csv(output/'comparison.csv',pairs)
    summary = dict(reference_count=len(references),online_count=len(raw),paired_count=len(pairs),
        mean_initial_coverage_pct=mean(r['initial_coverage_pct'] for r in pairs),
        mean_local_search_coverage_pct=mean(r['local_search_coverage_pct'] for r in pairs),
        mean_mappo_coverage_pct=mean(r['mappo_mean_pct'] for r in pairs),
        mean_local_search_minus_reference_pp=mean(r['local_search_minus_reference_pp'] for r in pairs),
        mean_mappo_minus_reference_pp=mean(r['mappo_minus_reference_pp'] for r in pairs),
        ls_above_reference=sum(r['local_search_minus_reference_pp']>1e-8 for r in pairs),
        mappo_mean_above_reference=sum(r['mappo_minus_reference_pp']>1e-8 for r in pairs),
        source_hashes=hashes,report_source_sha256=digest(__file__),
        note='Saved-array and full archive membership/coverage check; no new direct raycast, training or solve. '
             'Reference is a time-limited incumbent, not the optimum. Equal weight per initialization; '
             'map/K references repeat. No controlled speedup claim.')
    save(output/'validation_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='results/prototype_full_grid_20261012')
    parser.add_argument('--study',default='results/prototype_refinement_paired_20261011')
    args=parser.parse_args()
    report(args.root,args.study)

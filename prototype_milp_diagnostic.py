"""Opt-in packed-coverage MILP diagnostic on an existing candidate archive."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from prototype_offline import digest, save
from prototype_transfer import load_prototype, export_view
from src.core.visibility import MeshVisibility
from src.environment import get_lidar_profile
from src.methods.coverage_cut_milp import PackedCoverage, solve_cut_milp
from src.methods.local_mappo import validate_poses


def run(args):
    baseline, root = Path(args.baseline).resolve(), Path(args.output).resolve()
    previous = json.loads((baseline/'manifest.json').read_text())
    prior = json.loads((baseline/f'k{args.k}/result.json').read_text())
    if previous['status'] != 'complete' or previous['plan']['topM'] != 0:
        raise ValueError('A complete full-orientation archive is required')
    if args.k not in previous['k_values']:
        raise ValueError('Budget absent from archive')
    print(json.dumps(dict(scenario_id=previous['scenario_id'], k=args.k,
        candidates=previous['candidate_stats']['num_candidates'],
        old_objective=prior['covered_count'], solver_seconds=args.seconds,
        swap_seconds=args.swap_seconds, new_target_cache=False, execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return
    for name, expected in previous['source_hashes'].items():
        if name.startswith(('src/environment', 'src/core/visibility', 'src/roadside/')):
            if digest(name) != expected:
                raise ValueError(f'Physics/candidate source changed: {name}')
    root.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path('src/methods/coverage_cut_milp.py'), Path('prototype_transfer.py'),
               Path('prototype_offline.py'), *Path('src').rglob('*.py')]
    hashes = {}
    for p in sources:
        dest = root/'source'/(p.name if p.is_absolute() else p)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        hashes[str(p)] = digest(p)
    manifest = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], baseline=str(baseline), source_hashes=hashes,
        archive_sha256=digest(baseline/'candidates.npz'), prior_result_sha256=digest(baseline/f'k{args.k}/result.json'),
        k=args.k, scenario_id=previous['scenario_id'])
    save(root/'manifest.json', manifest)
    try:
        started = time.perf_counter()
        with np.load(baseline/'candidates.npz', allow_pickle=False) as archive:
            poses, masks, targets = archive['poses'], archive['packed_masks'], archive['target_points']
        oracle = PackedCoverage(masks, len(targets))
        if len(poses) != len(masks) or len(poses) != previous['candidate_stats']['num_candidates']:
            raise ValueError('Archive candidate count mismatch')
        start_ids = prior['solver']['selected_ids']
        if oracle.count(start_ids) != prior['covered_count']:
            raise ValueError('Previous incumbent and archive disagree')
        load_sec = time.perf_counter()-started
        print(f'Archive loaded: {len(poses)} candidates; starting swap + lazy cuts', flush=True)
        result = solve_cut_milp(masks, len(targets), args.k, start_ids,
            seconds=args.seconds, swap_seconds=args.swap_seconds, log_file=root/'gurobi.log')
        selected = poses[result['selected_ids']]
        _, _, _, geometry, _, cfg = load_prototype(baseline, args.k)
        validate_poses(selected, geometry, cfg)
        if not np.array_equal(targets, geometry.target_points(cfg.target['coverage_z_slices'])):
            raise ValueError('Target geometry does not match archive')
        mesh = geometry.create_environment_mesh()
        provider = MeshVisibility(mesh, targets, get_lidar_profile(cfg.lidar_profile, cfg))
        density = np.stack([provider.mask(p) for p in selected]).sum(axis=0)
        if int((density > 0).sum()) != result['objective']:
            raise RuntimeError('Lazy MILP result fails independent direct raycasting')
        np.save(root/'poses.npy', selected)
        np.savez_compressed(root/'coverage.npz', targets=targets, density=density)
        result.update(coverage_pct=100*result['objective']/len(targets), archive_load_sec=load_sec,
            prior_coverage_pct=prior['coverage_pct'], validated_direct_raycast=True,
            archive_reused=True, online_pose_warm_start=False)
        save(root/'result.json', result)
        export_view(root/'view.html', mesh, targets, density, selected,
                    f'Coverage-cut MILP S{previous["scenario_id"]} K={args.k}', result['coverage_pct'])
        manifest.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(), result=result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        save(root/'manifest.json', manifest)


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='results/prototype_target_full_20261012/baselines/s420002')
    p.add_argument('--output', required=True)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--seconds', type=float, default=300.)
    p.add_argument('--swap-seconds', type=float, default=30.)
    p.add_argument('--execute', action='store_true')
    run(p.parse_args())

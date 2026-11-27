"""Geometry-only full-grid planning and optional inspection of frozen online poses.

Does not build visibility, train, or solve. Pose files never define the domain.
"""
from argparse import ArgumentParser
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import yaml

from prototype_offline import digest, explicit_scenario, save
from src.environment import MapGeometry, installation_base_positions, load_config, validate_installation_pose
from src.roadside.candidate_generator import planned_candidate_signature


def audit(config_path, suite_path, study=None):
    config = load_config(config_path)
    config.candidate.update(xy_domain='installable_grid', max_orientations_per_position=0)
    suite = json.loads(Path(suite_path).read_text())
    rows = [row for family in suite['families'] for row in family['targets']]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate target IDs')
    maps, domains, geometries, legacy_domains = [], {}, {}, {}
    for row in rows:
        scenario = explicit_scenario(row)
        geometry = MapGeometry(scenario)
        legacy = copy.deepcopy(config)
        legacy.candidate['xy_domain'] = 'bin_representatives'
        old = set(installation_base_positions(geometry, legacy))
        full = set(installation_base_positions(geometry, config))
        if not old <= full:
            raise RuntimeError('Legacy positions not contained in full grid')
        plan = planned_candidate_signature(scenario, config.lidar_profile, config)
        targets = len(geometry.target_points(config.target['coverage_z_slices']))
        maps.append(dict(scenario=row, plan=plan, legacy_xyz_count=len(old),
                         full_xyz_count=len(full), target_count=targets,
                         raw_dense_mask_gib=plan['num_candidates_planned'] * targets / 1024**3))
        domains[row['id']], legacy_domains[row['id']] = full, old
        geometries[row['id']] = geometry
    checks = []
    if study is not None:
        study = Path(study)
        manifest = json.loads((study/'manifest.json').read_text())
        if manifest['status'] != 'complete' or any(j['status'] != 'complete' for j in manifest['jobs']):
            raise ValueError('Audit requires a complete online study')
        expected = {row['id']: row for row in rows}
        for job in manifest['jobs']:
            sid = job['scenario']['id']
            if sid not in expected or job['scenario'] != expected[sid]:
                raise ValueError('Online geometry differs from geometry-only plan')
            path = study/'jobs'/job['id']/'deployment.npy'
            poses = np.load(path, allow_pickle=False)
            if poses.ndim != 2 or poses.shape[1] != 5 or not 0 < len(poses) <= job['budget_k']:
                raise ValueError(f'Invalid pose shape/budget: {path}')
            if len(np.unique(poses, axis=0)) != len(poses):
                raise ValueError(f'Duplicate poses: {path}')
            outside, invalid = [], []
            for index, pose in enumerate(poses):
                try:
                    validate_installation_pose(pose, geometries[sid], config)
                except ValueError as exc:
                    invalid.append(dict(index=index, error=str(exc)))
                if tuple(map(float, pose[:3])) not in domains[sid]:
                    outside.append(index)
            checks.append(dict(job=job['id'], scenario_id=sid, pose_sha256=digest(path),
                outside_full_xyz=outside, invalid_poses=invalid,
                inside_full_domain=not outside and not invalid,
                inside_legacy_xyz=all(tuple(map(float, p[:3])) in legacy_domains[sid] for p in poses)))
    result = dict(schema='installation_domain_audit_v1', created_at=datetime.now(timezone.utc).isoformat(),
        config_path=str(config_path), suite_path=str(suite_path), online_study=str(study) if study else None,
        maps=maps, checks=checks, checked_runs=len(checks),
        outside_full_runs=sum(not r['inside_full_domain'] for r in checks),
        outside_legacy_xyz_runs=sum(not r['inside_legacy_xyz'] for r in checks),
        full_visibility_built=False, optimization_executed=False,
        note='Full discrete physical domain, before union-equivalent mask dedup. '
             'Online 2m steps and 10m anchors are search restrictions, not exhaustive reachability. '
             'Old bounds are not certificates for this larger candidate domain.')
    return result, config


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.yaml')
    parser.add_argument('--suite', default='configs/library_expansion.json')
    parser.add_argument('--study', help='Optional completed online study to audit after domain construction')
    parser.add_argument('--output', help='Fresh directory for report and proposed config snapshots')
    args = parser.parse_args()
    result, config = audit(args.config, args.suite, args.study)
    if args.output:
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=False)
        config.output.update(cache_dir=str(root/'future_cache'), results_dir=str(root/'future_results'))
        (root/'config.yaml').write_text(yaml.safe_dump(config.raw, sort_keys=False))
        for item in result['maps']:
            save(root/f"s{item['scenario']['id']}.json", item['scenario'])
        result['source_hashes'] = {name: digest(name) for name in (
            'prototype_domain_audit.py', 'src/environment.py', 'src/roadside/candidate_generator.py')}
        save(root/'report.json', result)
    print(json.dumps(result, indent=2))
    if result['outside_full_runs']:
        raise SystemExit('Some online poses remain outside the full discrete domain')


if __name__ == '__main__':
    main()

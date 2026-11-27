"""Read-only diagnostics for the frozen A/B query study, not a new benchmark.

The integer-bitset oracle does not call the production frontier optimizer.
Known input/FOV counterexamples are reported as findings, not passing tests.
"""
import argparse
import copy
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from library_topology_transfer import prepare_problem
from src.core.visibility import MeshVisibility
from src.environment import LidarProfile
from src.methods.library_query_controls import _optical_masks


def digest(path):
    with path.open('rb') as stream:
        result = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def union_count(bits, ids):
    value = 0
    for index in ids:
        value |= bits[index]
    return value.bit_count()


def check_frontier(frontier, domain_ids, poses, masks):
    bits = [int.from_bytes(np.packbits(mask).tobytes(), 'little') for mask in masks]
    max_k = max(frontier['k_values'])
    if sum(comb(len(domain_ids), size) for size in range(min(max_k, len(domain_ids)) + 1)) > 500000:
        raise ValueError('Audit oracle limited to 500,000 subsets per domain')
    best_by_size = [0] * (max_k + 1)
    for size in range(1, min(max_k, len(domain_ids)) + 1):
        best_by_size[size] = max(union_count(bits, ids) for ids in combinations(domain_ids, size))
    for row in frontier['frontier']:
        k = row['budget_k']
        optimum = max(best_by_size[:k + 1])
        assert row['covered_count'] == row['upper_bound_count'] == optimum
        assert row['certification'] == 'discrete_optimal'
        for solution in [row, *row['alternatives']]:
            ids = solution['full_candidate_ids']
            assert len(ids) == len(set(ids)) == solution['num_selected'] <= k
            assert set(ids) <= set(domain_ids)
            assert [domain_ids[i] for i in solution['selected_ids']] == ids
            assert solution['covered_count'] == union_count(bits, ids)
            assert abs(solution['coverage_pct'] - 100 * union_count(bits, ids) / masks.shape[1]) < 1e-9
            np.testing.assert_array_equal(np.asarray(solution['poses']).reshape(-1, 5), poses[ids])
    return len(frontier['frontier'])


def audit_archives(root, *, recast=False):
    observations, hashes = [], {}
    for case in ('a_w9', 'a_w12', 'b_turn45', 'b_turn60'):
        report_path, archive_path = root / case / 'report.json', root / case / 'target.npz'
        report = json.loads(report_path.read_text())
        for path in (report_path, archive_path):
            hashes[str(path)] = digest(path)
        with np.load(archive_path, allow_pickle=False) as archive:
            poses, targets, masks = (archive[key] for key in ('poses', 'targets', 'masks'))
        assert masks.dtype == bool and masks.shape == (len(poses), len(targets))
        checked = check_frontier(report['full_frontier'], list(range(len(poses))), poses, masks)
        for arm in report['arms']:
            domain_ids = arm['query_plan']['selected_ids']
            assert sorted(arm['physical_query_ids']) == domain_ids
            assert arm['physical_query_ids'] == arm['query_plan']['query_order']
            checked += check_frontier(arm['frontier'], domain_ids, poses, masks)
        sensor = report['plan']['context']['sensor']
        optical = _optical_masks(poses.astype(np.float32), targets.astype(np.float32),
            (sensor['h_fov_deg'], sensor['v_fov_deg'], sensor['max_range_m']))
        assert not np.any(masks & ~optical)
        observations.append(dict(case=case, checked_frontier_rows=checked,
            candidate_count=len(poses), target_count=len(targets),
            optical_edges=int(optical.sum()), occluded_edges=int((optical & ~masks).sum()),
            poses_changed_by_occlusion=int(np.any(optical != masks, axis=1).sum())))
        if recast:
            import open3d as o3d

            mesh_path = root / case / 'mesh.npz'
            hashes[str(mesh_path)] = digest(mesh_path)
            with np.load(mesh_path, allow_pickle=False) as saved:
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(saved['vertices']),
                    o3d.utility.Vector3iVector(saved['triangles']))
            provider = MeshVisibility(mesh, targets, LidarProfile(**sensor))
            measured = np.asarray([provider.mask(pose) for pose in poses])
            np.testing.assert_array_equal(measured, masks)
            observations[-1]['fresh_recast_candidates'] = len(poses)
    return dict(cases=observations, input_hashes=hashes,
        scope=('Independent exact selection from archived masks; '
               + ('all candidate masks recast with current implementation and archived mesh'
                  if recast else 'NOT independent physical raycasting')))


def audit_unsupported_inputs(source):
    spec_path = BASE / 'configs/library_target_a_w12.json'
    spec = json.loads(spec_path.read_text())
    baseline = prepare_problem(source, spec, [1, 2, 3])
    baseline_mesh = baseline['geometry'].create_environment_mesh()
    cases = {
        'constraints': {'constraints': {'max_sensors': 0, 'forbidden_xy_boxes': [[0, 0, 100, 100]]}},
        'obstacles': {'obstacles': [{'min': [0, 0, 0], 'max': [100, 100, 20]}]},
    }
    observations = []
    for name, extra in cases.items():
        target = copy.deepcopy(spec)
        target.update(extra)
        try:
            problem = prepare_problem(source, target, [1, 2, 3])
        except ValueError as exc:
            observations.append(dict(field=name, rejected=True, error=str(exc)))
            continue
        mesh = problem['geometry'].create_environment_mesh()
        unchanged = (all(np.array_equal(problem[key], baseline[key])
                         for key in ('poses', 'targets', 'pool_ids'))
                     and np.array_equal(np.asarray(mesh.vertices), np.asarray(baseline_mesh.vertices))
                     and np.array_equal(np.asarray(mesh.triangles), np.asarray(baseline_mesh.triangles)))
        observations.append(dict(field=name, rejected=False, domain_and_mesh_unchanged=unchanged,
            reported_operational_constraints=problem['context']['operational_constraints']))
    return dict(cases=observations, baseline_spec_sha256=digest(spec_path),
                source_manifest_sha256=digest(source / 'manifest.json'))


def audit_fov():
    import open3d as o3d

    clear = o3d.geometry.TriangleMesh.create_box().translate([100., 100., 100.])
    pose = np.array([0., 0., 1., 0., 0.], dtype=np.float32)
    examples = [(90., [[.25, 0., 1.], [.5001, 0., 1.]]),
                (270., [[-1., np.sqrt(3), 1.], [1., 0., 1.]]),
                (360., [[.25, 0., 1.], [-1., np.sqrt(3), 1.]])]
    observations = []
    for h, points in examples:
        targets = np.asarray(points)
        vectors = targets - pose[:3]
        az = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
        el = np.rad2deg(np.arctan2(vectors[:, 2], np.hypot(vectors[:, 0], vectors[:, 1])))
        distance = np.linalg.norm(vectors, axis=1)
        expected = (abs(az) <= h / 2) & (abs(el) <= 45) & (distance <= 10) & (distance > 1e-6)
        actual = MeshVisibility(clear, targets, LidarProfile('audit', h, 90., 10.)).mask(pose)
        observations.append(dict(h_fov_deg=h, targets=targets.tolist(),
            angle_range_expected=expected.tolist(), actual_clear_scene=actual.tolist(),
            mismatch=bool(np.any(actual != expected))))
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-a', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recast-archives', action='store_true',
                        help='Also recast all saved candidates against the archived meshes')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(archived_selection=audit_archives(args.root, recast=args.recast_archives),
                  unsupported_inputs=audit_unsupported_inputs(args.source_a),
                  fov_counterexamples=audit_fov())
    source_paths = [Path(__file__), BASE / 'src/core/visibility.py', BASE / 'src/environment.py',
                    BASE / 'src/methods/library_query_controls.py', BASE / 'src/core/deployment_library.py',
                    BASE / 'library_topology_transfer.py', BASE / 'library_topology.py',
                    BASE / 'prototype_offline.py']
    report['audit_source_hashes'] = {str(path.relative_to(BASE)): digest(path) for path in source_paths}
    report['command'] = [sys.executable, *sys.argv]
    report['semantics'] = 'Audit completed; counterexamples are findings, NOT passing regression tests'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

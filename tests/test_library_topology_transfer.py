"""Track A transfer contract using a real, temporary frozen source library."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_topology as builder
import library_topology_transfer as transfer
from prototype_offline import canonical, digest
from prototype_segments import installation_segments, segment_coordinates, segment_key, segment_to_target
from prototype_transfer import project_poses
from src.environment import MapGeometry, get_scenario, load_config


BASE = Path(__file__).resolve().parents[1]


def target_spec(**changes):
    return dict(dict(id=430001, parent_scenario_id=1, topology='T_JUNCTION',
        road_width=9., num_lanes=2, geometry=dict(junction_x_m=45., junction_y_m=55.),
        provenance='synthetic_junction_offset'), **changes)


class TopologyTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name) / 'source_run'
        cls.config = load_config(BASE / 'configs/library_topology_smoke.yaml')
        builder.execute_library(cls.config, get_scenario(1, BASE / 'data/scenarios/yeongjong_final.json'),
                                cls.root, [1, 2], alternatives_per_k=4)
        cls.initial_hashes = cls.hashes(cls.root)

    @staticmethod
    def hashes(root):
        return {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}

    def no_execution(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name in ('library_topology.MeshVisibility', 'prototype_transfer.MeshVisibility',
                     'src.core.visibility.MeshVisibility', 'library_topology.build_frontier',
                     'src.core.deployment_library.build_frontier', 'library_topology.subprocess.run'):
            stack.enter_context(patch(name, side_effect=AssertionError('no rays/solve/worker')))
        stack.enter_context(patch.object(MapGeometry, 'create_environment_mesh',
                                        side_effect=AssertionError('no mesh')))
        return stack

    def clone(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / 'copy'
        shutil.copytree(self.root, root)
        return root

    @staticmethod
    def read(root, name):
        return json.loads((root / name).read_text())

    @staticmethod
    def rewrite(root, name, value):
        (root / name).write_text(json.dumps(value, allow_nan=False))
        if name != 'manifest.json':
            manifest = TopologyTransferTests.read(root, 'manifest.json')
            manifest['artifact_hashes'][name] = digest(root / name)
            TopologyTransferTests.rewrite(root, 'manifest.json', manifest)

    def layouts(self, root, selections):
        report = self.read(root, 'frontier.json')
        canonical_records = self.read(root, 'canonical_candidates.json')
        with np.load(root / 'candidates.npz', allow_pickle=False) as archive:
            for row, ids in zip(report['frontier'], selections):
                count = int(np.any(archive['masks'][ids], axis=0).sum())
                row.update(selected_ids=ids, num_selected=len(ids), poses=archive['poses'][ids].tolist(),
                    canonical=[canonical_records[i] for i in ids], covered_count=count,
                    coverage_pct=100. * count / len(archive['targets']), alternatives=[])
        validation = report['physical_validation']
        validation['checked_solution_records'] = len(selections)
        validation['unique_candidate_ids'] = sorted({i for ids in selections for i in ids})
        self.rewrite(root, 'frontier.json', report)
        self.rewrite(root, 'validation.json', validation)

    def test_geometry_only_full_domain_membership_and_all_layouts(self):
        self.no_execution()
        source = transfer.load_topology_library(self.root)
        problem = transfer.prepare_problem(self.root, target_spec(), [1, 2])
        repeated = transfer.prepare_problem(self.root, target_spec(), [1, 2])
        expected_geometry, expected_poses, expected_targets = builder._domain(
            self.config, transfer.explicit_scenario(target_spec()))
        self.assertIsInstance(problem['geometry'], MapGeometry)
        np.testing.assert_array_equal(problem['poses'], expected_poses)
        np.testing.assert_array_equal(problem['targets'], expected_targets)
        self.assertEqual(problem['profile'], source['profile'])
        records = problem['transfers']
        self.assertEqual(len(records), sum(1 + len(row['alternatives']) for row in source['frontier']['frontier']))
        self.assertTrue(all(record['status'] == 'complete' for record in records))
        all_ids = set()
        for record in records:
            ids = record['selected_ids']
            self.assertEqual(record['num_selected'], len(ids))
            self.assertEqual(len(ids), len(record['source_selected_ids']))
            self.assertLessEqual(len(ids), record['budget_k'])
            self.assertTrue(all(type(i) is int and 0 <= i < len(expected_poses) for i in ids))
            self.assertEqual([segment_key(p) for p in record['source_coordinates']],
                             [segment_key(p) for p in canonical(expected_poses[ids], expected_geometry)])
            self.assertTrue(all(0 <= p['u'] <= 1 for p in record['source_coordinates']))
            self.assertNotIn('covered_count', record)
            all_ids.update(ids)
        self.assertEqual(problem['pool_ids'], sorted(all_ids))
        self.assertLess(len(all_ids), len(expected_poses))
        for proposal, record in zip(problem['proposals'], records):
            self.assertEqual(proposal['selected_ids'], record['selected_ids'])
            self.assertEqual(set(proposal), {'selected_ids', 'budget_k', 'origin'})
        for key in ('context', 'transfers', 'proposals', 'pool_ids'):
            self.assertEqual(problem[key], repeated[key])
            json.dumps(problem[key], allow_nan=False)
        self.assertEqual(self.hashes(self.root), self.initial_hashes)

    def test_cwd_independent_and_requested_budgets_only(self):
        self.no_execution()
        previous = Path.cwd()
        try:
            os.chdir(self.temporary.name)
            problem = transfer.prepare_problem(self.root, target_spec(), [2])
        finally:
            os.chdir(previous)
        self.assertEqual(problem['k_values'], [2])
        self.assertTrue(all(record['budget_k'] == 2 for record in problem['transfers']))

    def test_missing_k_and_invalid_budgets(self):
        self.no_execution()
        for ks in ([], [True], [2, 1], [1, 1], [0], [65], [3]):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                transfer.prepare_problem(self.root, target_spec(), ks)

    def test_unsupported_inputs_rejected_before_source_load(self):
        with patch.object(transfer, 'load_topology_library', side_effect=AssertionError('no source read')):
            for extra in ({'constraints': {}}, {'obstacles': []}, {'installation_costs': [1]},
                          {'budget_k': 0}, {'geometery': {}}, {'unknown': None}):
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'Unsupported Track A'):
                    transfer.prepare_problem(self.root, target_spec(**extra), [1])
            for value in (None, [], 'not an object'):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be an object'):
                    transfer.prepare_problem(self.root, value, [1])

    def test_legacy_limited_fov_source_requires_rebuild(self):
        config = deepcopy(self.config)
        config.raw['lidar_profiles'][config.lidar_profile]['h_fov_deg'] = 270.
        with patch.object(transfer, 'load_config', return_value=config):
            with self.assertRaisesRegex(ValueError, 'optical model mismatch'):
                transfer.load_topology_library(self.root)

    def test_same_topology_and_nonidentical_geometry_required(self):
        self.no_execution()
        with self.assertRaisesRegex(ValueError, 'same topology'):
            transfer.prepare_problem(self.root, target_spec(parent_scenario_id=2, topology='CROSS'), [1])
        same = target_spec(id=499999, road_width=7., geometry=dict(junction_x_m=50., junction_y_m=50.))
        with self.assertRaisesRegex(ValueError, 'identical geometry'):
            transfer.prepare_problem(self.root, same, [1])

    def test_corrupt_files_and_missing_hashes_rejected(self):
        self.no_execution()
        for name in ('inputs/config.yaml', 'inputs/scenario.json', 'source/library_topology.py',
                     'candidates.npz', 'mesh.npz', 'frontier.json'):
            with self.subTest(name=name):
                root = self.clone()
                with (root / name).open('ab') as stream:
                    stream.write(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    transfer.load_topology_library(root)
        for field in ('artifact_hashes', 'input_hashes', 'source_hashes'):
            with self.subTest(field=field):
                root = self.clone()
                manifest = self.read(root, 'manifest.json')
                manifest[field].pop(next(iter(manifest[field])))
                self.rewrite(root, 'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    transfer.load_topology_library(root)

    def test_manifest_schema_status_and_track_required(self):
        for field, value in (('schema', 'legacy'), ('status', 'running'), ('track', 'B')):
            with self.subTest(field=field):
                root = self.clone()
                manifest = self.read(root, 'manifest.json')
                manifest[field] = value
                self.rewrite(root, 'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'complete Track A'):
                    transfer.load_topology_library(root)

    def test_rehashed_semantic_corruption_rejected(self):
        self.no_execution()
        mutations = [
            ('candidate_count', lambda report: report.update(candidate_count=1)),
            ('domain', lambda report: report.update(domain_sha256='0' * 64)),
            ('ids', lambda report: report['frontier'][0].update(selected_ids=[99999])),
            ('count', lambda report: report['frontier'][0].update(num_selected=999)),
            ('poses', lambda report: report['frontier'][0]['poses'][0].__setitem__(0, -999.)),
            ('coverage', lambda report: report['frontier'][0].update(covered_count=-1)),
            ('canonical', lambda report: report['frontier'][0]['canonical'][0].update(branch_id='north')),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                root = self.clone()
                report = self.read(root, 'frontier.json')
                mutate(report)
                self.rewrite(root, 'frontier.json', report)
                with self.assertRaises(ValueError):
                    transfer.load_topology_library(root)
        root = self.clone()
        records = self.read(root, 'canonical_candidates.json')
        records[0]['installation_line_id'] = 'wrong'
        self.rewrite(root, 'canonical_candidates.json', records)
        with self.assertRaisesRegex(ValueError, 'Canonical/world'):
            transfer.load_topology_library(root)

    def test_pickle_archive_rejected_even_with_updated_hash(self):
        root = self.clone()
        np.savez(root / 'candidates.npz', poses=np.array([object()], dtype=object),
                 targets=np.zeros((1, 3)), masks=np.zeros((1, 1), dtype=bool))
        manifest = self.read(root, 'manifest.json')
        manifest['artifact_hashes']['candidates.npz'] = digest(root / 'candidates.npz')
        self.rewrite(root, 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'allow_pickle=False'):
            transfer.load_topology_library(root)

    def test_empty_layouts_complete_without_projection(self):
        self.no_execution()
        root = self.clone()
        self.layouts(root, [[], []])
        with patch.object(transfer, 'project_poses', side_effect=AssertionError('empty projection')):
            problem = transfer.prepare_problem(root, target_spec(), [1, 2])
        self.assertEqual(problem['pool_ids'], [])
        for record in problem['transfers']:
            self.assertEqual(record['status'], 'complete')
            self.assertEqual(record['selected_ids'], [])
            self.assertEqual(record['num_selected'], 0)
            self.assertEqual(record['projection_displacement_m'], [])

    def test_real_projection_collision_unavailable_not_repaired(self):
        self.no_execution()
        root = self.clone()
        source = transfer.load_topology_library(root)
        spec = target_spec(geometry=dict(junction_x_m=15., junction_y_m=55.))
        geometry, _, _ = builder._domain(self.config, transfer.explicit_scenario(spec))
        source_segments, target_segments = installation_segments(source['geometry']), installation_segments(geometry)
        by_projection = {}
        pair = None
        for i, pose in enumerate(source['poses']):
            coords = segment_coordinates([pose], source['geometry'], source_segments)
            raw = segment_to_target(coords, geometry, target_segments)
            key = tuple(project_poses(raw, geometry, self.config, coords)[0])
            if key in by_projection:
                pair = [by_projection[key], i]
                break
            by_projection[key] = i
        self.assertIsNotNone(pair, 'Fixture must exercise a real same-side grid collision')
        self.layouts(root, [[], pair])
        problem = transfer.prepare_problem(root, spec, [1, 2])
        self.assertEqual(problem['pool_ids'], [])
        record = problem['transfers'][1]
        self.assertEqual(record['status'], 'unavailable')
        self.assertIn('collapsed', record['reason'])
        self.assertNotIn('selected_ids', record)
        self.assertNotIn('covered_count', record)
        self.assertEqual(len(problem['proposals']), 1)

    def test_target_domain_uses_existing_size_guard(self):
        source = transfer.load_topology_library(self.root)
        source['config'] = deepcopy(source['config'])
        source['config'].candidate['pitch_step_deg'] = 1e-20
        with patch.object(transfer, 'load_topology_library', return_value=source):
            with self.assertRaisesRegex(ValueError, 'installation axis'):
                transfer.prepare_problem(self.root, target_spec(), [1])


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Geometry-only Track B adapter checks using explicitly synthetic source masks."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_components as source
import library_component_transfer as adapter
from src.core.deployment_library import build_frontier
from src.environment import ConnectedCorridorGeometry, LidarProfile


class ComponentTransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = source.default_spec()
        self.spec['geometry'].update(length_m=40, road_width_m=4, bend_length_m=20,
                                     mesh_step_m=2)
        self.spec['candidate'].update(station_step_m=10, heights_m=[3, 4],
                                     relative_yaws_deg=[-20, 0], pitches_deg=[-10])
        self.spec['target'].update(step_m=2, heights_m=[1.5])
        self.profile = LidarProfile('SYNTHETIC_TEST_ONLY', 360, 26.9, 30)
        geometry, archive = source._domain(self.spec)
        # Each row has a distinct synthetic target; these are not physical results.
        archive['masks'] = np.eye(len(archive['poses']), len(archive['targets']), dtype=bool)
        report = build_frontier(archive['poses'], archive['masks'], archive['targets'],
                                [1, 2, 3], alternatives_per_k=3, enumeration_limit=0,
                                context=dict(geometry=asdict(geometry), sensor=asdict(self.profile)))
        for row in report['frontier']:
            for solution in [row, *row['alternatives']]:
                solution['canonical'] = archive['canonical'][solution['selected_ids']].tolist()
        hashes = {}
        for name in source.SOURCE_FILES:
            destination = self.root / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((source.BASE / name).read_bytes())
            hashes[name] = source.digest(destination)
        metadata = dict(config=self.spec, geometry=asdict(geometry), sensor=asdict(self.profile),
                        source_hashes=hashes, domain_sha256=report['domain_sha256'])
        np.savez(self.root / 'library.npz', **archive)
        np.savez(self.root / 'geometry.npz', vertices=np.empty((0, 3)), triangles=np.empty((0, 3)))
        self.write_json('metadata.json', metadata)
        self.write_json('frontier.json', report)
        (self.root / 'inputs').mkdir()
        self.write_json('inputs/config.json', self.spec)
        self.write_json('inputs/sensor.json', asdict(self.profile))
        self.write_json('manifest.json', dict(schema=source.MANIFEST_SCHEMA, status='complete',
            source_hashes=hashes, domain_sha256=report['domain_sha256'],
            input_hashes={name: source.digest(self.root / name)
                          for name in ('inputs/config.json', 'inputs/sensor.json')},
            artifact_hashes={name: source.digest(self.root / name) for name in source.ARTIFACTS}))
        self.target = dict(geometry=dict(length_m=80, road_width_m=6, turn_deg=30,
                                         bend_length_m=32, mesh_step_m=2), modules=2)

    def write_json(self, name, data):
        (self.root / name).write_text(json.dumps(data, allow_nan=False))

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob('*') if path.is_file()}

    def prepare(self, target=None, ks=(1, 3)):
        return adapter.prepare_problem(self.root, self.target if target is None else target, ks)

    def test_prepare_is_read_only_and_deterministic(self):
        before, target = self.snapshot(), deepcopy(self.target)
        with patch.object(ConnectedCorridorGeometry, 'create_environment_mesh',
                          side_effect=AssertionError('mesh')), \
                patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('rays')), \
                patch('src.core.deployment_library.build_frontier', side_effect=AssertionError('solve')), \
                patch.object(Path, 'write_text', side_effect=AssertionError('write')), \
                patch.object(Path, 'write_bytes', side_effect=AssertionError('write')), \
                patch('numpy.savez', side_effect=AssertionError('write')):
            first, second = self.prepare(), self.prepare()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(target, self.target)
        self.assertEqual(first['profile'], self.profile)
        for name in ('poses', 'targets'):
            np.testing.assert_array_equal(first[name], second[name])
        for name in ('context', 'transfers', 'pool_ids', 'proposals', 'k_values'):
            self.assertEqual(first[name], second[name])
        json.dumps(first['context'], allow_nan=False)
        json.dumps(first['transfers'], allow_nan=False)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        self.assertEqual(first['context']['source_hashes'], manifest['source_hashes'])

    def test_full_domain_membership_and_exact_aligned_projection(self):
        result = self.prepare()
        spec = deepcopy(self.spec)
        spec['geometry'].update(self.target['geometry'])
        geometry, archive = source._domain(spec)
        self.assertIsInstance(result['geometry'], ConnectedCorridorGeometry)
        np.testing.assert_array_equal(result['poses'], archive['poses'])
        np.testing.assert_array_equal(result['targets'], archive['targets'])
        self.assertEqual(result['poses'].shape, (64, 5))
        self.assertEqual(result['targets'].shape, (80, 3))
        union = set()
        for record in result['transfers']:
            self.assertEqual(record['status'], 'complete')
            self.assertLessEqual(record['num_selected'], record['budget_k'])
            union.update(record['selected_ids'])
            for projection in record['projection']:
                cid = projection['target_id']
                mapped = projection['mapped_canonical']
                self.assertEqual(projection['position_displacement_m'], 0.)
                np.testing.assert_allclose(archive['canonical'][cid], mapped)
                self.assertEqual(int(mapped[0] // 40), record['module_index'])
                np.testing.assert_allclose(result['poses'][cid], geometry.poses([mapped])[0])
        self.assertEqual(result['pool_ids'], sorted(union))
        self.assertEqual({r['budget_k'] for r in result['transfers']}, {1, 2, 3})
        self.assertTrue(any(r['alternative_index'] is None for r in result['transfers']))
        self.assertTrue(any(r['alternative_index'] == 0 for r in result['transfers']))
        self.assertEqual(result['context']['budget_mode'], 'at_most_k')
        self.assertEqual(result['context']['operational_constraints'], [])
        for proposal in result['proposals']:
            self.assertEqual(set(proposal), {'selected_ids', 'budget_k', 'origin'})
            self.assertTrue(set(proposal['selected_ids']) <= union)

    def test_module_ports_and_boundary_ownership(self):
        result = self.prepare(dict(geometry=dict(length_m=90, road_width_m=6), modules=2))
        modules = result['context']['modules']
        self.assertEqual(len(modules), 2)
        left, right = modules[0]['ports'][1], modules[1]['ports'][0]
        self.assertEqual(left['position_m'], right['position_m'])
        np.testing.assert_allclose(left['outward_tangent'], -np.array(right['outward_tangent']))
        self.assertEqual(left['station_m'], 45.)
        self.assertTrue(left['internal'])
        self.assertFalse(modules[0]['ports'][0]['internal'])
        ids = modules[1]['candidate_ids']
        self.assertTrue(np.any(result['poses'][ids, 0] == 45.))
        all_ids = [cid for module in modules for cid in module['candidate_ids']]
        self.assertEqual(sorted(all_ids), list(range(len(result['poses']))))
        self.assertEqual(len(all_ids), len(set(all_ids)))
        boundary = result['context']['boundary_context']
        self.assertFalse(boundary['periodic'])
        self.assertFalse(boundary['wrap'])
        self.assertTrue(boundary['neighbor_coverage_must_be_reevaluated'])

    def test_nearest_projection_retains_side_and_orientation(self):
        result = self.prepare(dict(geometry=dict(length_m=100, road_width_m=8,
                                  turn_deg=-30, bend_length_m=32), modules=3))
        spec = deepcopy(self.spec)
        spec['geometry'] = asdict(result['geometry'])
        _, archive = source._domain(spec)
        displaced = False
        for record in result['transfers']:
            module_ids = result['context']['modules'][record['module_index']]['candidate_ids']
            for projection in record['projection']:
                mapped = np.array(projection['mapped_canonical'])
                cid = projection['target_id']
                candidate = archive['canonical'][cid]
                self.assertIn(cid, module_ids)
                np.testing.assert_array_equal(candidate[1:], mapped[1:])
                eligible = [i for i in module_ids if np.array_equal(archive['canonical'][i, 1:], mapped[1:])]
                point = result['geometry'].poses([mapped])[0, :3]
                distances = np.linalg.norm(result['poses'][eligible, :3] - point, axis=1)
                self.assertEqual(cid, eligible[int(np.argmin(distances))])
                displaced |= projection['position_displacement_m'] > 0
        self.assertTrue(displaced)

    def test_collision_unavailable_but_empty_layout_valid(self):
        loaded = source.load_component_library(self.root)
        row = loaded['frontier']['frontier'][0]
        # Same side/orientation at two source stations collapses into one target station.
        row.update(budget_k=2, selected_ids=[0, 8], canonical=loaded['archive']['canonical'][[0, 8]].tolist(),
                   alternatives=[dict(selected_ids=[], canonical=[])])
        loaded['frontier']['frontier'] = [row]
        with patch.object(adapter, 'load_component_library', return_value=loaded):
            result = self.prepare(dict(geometry=dict(length_m=20, bend_length_m=20), modules=2))
        unavailable = [r for r in result['transfers'] if r['status'] == 'unavailable']
        self.assertEqual(len(unavailable), 2)
        for record in unavailable:
            self.assertEqual(record['reason'], 'projection_collision')
            self.assertIsNone(record['selected_ids'])
            self.assertIsNone(record['num_selected'])
            self.assertNotIn('covered_count', record)
        complete = [r for r in result['transfers'] if r['status'] == 'complete']
        self.assertEqual(len(complete), 2)
        self.assertTrue(all(r['selected_ids'] == [] and r['num_selected'] == 0 for r in complete))
        self.assertEqual(result['pool_ids'], [])

    def test_target_domain_independent_of_source_layouts_and_no_mutation(self):
        loaded = source.load_component_library(self.root)
        before = deepcopy(loaded)
        with patch.object(adapter, 'load_component_library', return_value=loaded):
            first = self.prepare()
            self.assertEqual(loaded['frontier'], before['frontier'])
            self.assertEqual(loaded['metadata'], before['metadata'])
            for name in loaded['archive']:
                np.testing.assert_array_equal(loaded['archive'][name], before['archive'][name])
            loaded['frontier']['frontier'] = []
            second = self.prepare()
        np.testing.assert_array_equal(first['poses'], second['poses'])
        np.testing.assert_array_equal(first['targets'], second['targets'])
        self.assertEqual(second['pool_ids'], [])
        self.assertEqual(second['transfers'], [])

    def test_invalid_inputs_and_nonempty_module_guards(self):
        targets = [None, {}, dict(geometry={}, modules=True), dict(geometry={}, modules=1),
                   dict(geometry={}, modules=9), dict(geometry={}, modules=2.5),
                   dict(geometry={}, modules=2, graph={}), dict(geometry={'bad': 1}, modules=2),
                   dict(geometry={'length_m': float('nan')}, modules=2),
                   dict(geometry={'length_m': float('inf')}, modules=2),
                   dict(geometry={'length_m': '80'}, modules=2),
                   dict(geometry={'road_width_m': True}, modules=2),
                   dict(geometry={'length_m': 41}, modules=2),
                   dict(geometry={'length_m': 20}, modules=8)]
        for target in targets:
            with self.subTest(target=target), self.assertRaises(ValueError):
                adapter.prepare_problem(self.root, target, [1])
        for ks in ([], [0], [True], [1.5], [2, 1], [1, 1], None, '12'):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                self.prepare(ks=ks)

    def test_size_guards_before_target_allocation(self):
        for geometry, error in ((dict(length_m=1000), 'max_candidates'),
                                (dict(length_m=500, road_width_m=80), 'max_mask_entries'),
                                (dict(mesh_step_m=1e-20), 'max_mesh_sections')):
            with self.subTest(geometry=geometry), \
                    patch.object(ConnectedCorridorGeometry, 'target_samples',
                                 side_effect=AssertionError('allocation')), \
                    self.assertRaisesRegex(ValueError, error):
                self.prepare(dict(geometry=geometry, modules=2))

    def test_tampered_source_rejected(self):
        with (self.root / 'library.npz').open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()

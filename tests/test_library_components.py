"""Bounded Track B source-library checks; no transfer/generalization claim."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_components as library
from src.core.visibility import MeshVisibility
from src.environment import ConnectedCorridorGeometry, LidarProfile


class ComponentLibraryTests(unittest.TestCase):
    def spec(self, angle=0):
        spec = library.default_spec()
        spec['geometry'].update(length_m=12, road_width_m=4, bend_length_m=12,
                                turn_deg=angle, mesh_step_m=2)
        spec['candidate'].update(station_step_m=6, heights_m=[3],
                                 relative_yaws_deg=[0], pitches_deg=[-10])
        spec['target'].update(step_m=2, heights_m=[1.5])
        spec['max_k'] = 3
        return spec

    def profile(self):
        return LidarProfile('SMOKE', 360, 180, 10)

    def assert_timings(self, metadata, *, worker=False):
        phases = {'geometry_domain_mesh', 'candidate_raycasts', 'frontier', 'independent_validation'}
        timings = metadata['phase_wall_sec']
        self.assertEqual(set(timings), phases | ({'worker_total'} if worker else set()))
        for name, value in timings.items():
            with self.subTest(timing=name):
                self.assertTrue(np.isfinite(value))
                self.assertGreaterEqual(value, 0)
        if worker:
            self.assertGreaterEqual(timings['worker_total'], sum(timings[name] for name in phases))

    def test_ports_follow_heading_not_xy_tuple(self):
        for angle in (0, -30, 30):
            geometry = ConnectedCorridorGeometry(**self.spec(angle)['geometry'])
            entry, exit_port = library.component_ports(geometry)
            self.assertEqual([entry['name'], exit_port['name']], ['entry', 'exit'])
            xy, headings = geometry.centerline(np.array([0., geometry.length_m]))
            for index, port in enumerate((entry, exit_port)):
                np.testing.assert_allclose(port['position_m'], [*xy[index], 0])
                expected = [np.cos(headings[index]), np.sin(headings[index]), 0]
                np.testing.assert_allclose(port['tangent'], expected)
                np.testing.assert_allclose(port['outward_tangent'],
                                           np.array(expected) * (-1 if index == 0 else 1))
                self.assertAlmostEqual(np.linalg.norm(port['tangent']), 1)
            np.testing.assert_allclose(exit_port['tangent'],
                [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0], atol=1e-12)

    def test_actual_raycast_frontiers_and_all_candidate_masks(self):
        for angle in (0, 30):
            with self.subTest(angle=angle):
                built = library.build_component_library(self.spec(angle), self.profile())
                self.assert_timings(built['metadata'])
                archive, report = built['archive'], built['frontier']
                self.assertEqual(archive['poses'].shape, (4, 5))
                self.assertEqual(archive['masks'].shape, (4, 12))
                self.assertEqual(archive['masks'].dtype, bool)
                self.assertEqual(report['k_values'], [1, 2, 3])
                self.assertTrue(report['enumeration']['completed'])
                self.assertEqual(report['enumeration']['subset_count'], 15)
                self.assertTrue(np.all(np.diff([r['covered_count'] for r in report['frontier']]) >= 0))
                geometry = ConnectedCorridorGeometry(**built['metadata']['geometry'])
                np.testing.assert_allclose(geometry.poses(archive['canonical']), archive['poses'])
                np.testing.assert_allclose(geometry.points(archive['target_coordinates']),
                                           archive['targets'], atol=1e-6)
                provider = MeshVisibility(geometry.create_environment_mesh(), archive['targets'], self.profile())
                direct = np.asarray([provider.mask(pose) for pose in archive['poses']])
                np.testing.assert_array_equal(direct, archive['masks'])
                self.assertGreater(int(direct.sum()), 0)
                for row in report['frontier']:
                    self.assertEqual(row['certification'], 'discrete_optimal')
                    self.assertEqual(row['selected_ids'], row['alternatives'][0]['selected_ids'])
                    for solution in [row, *row['alternatives']]:
                        ids = solution['selected_ids']
                        count = int(np.any(direct[ids], axis=0).sum())
                        self.assertEqual(solution['covered_count'], count)
                        self.assertLessEqual(solution['num_selected'], row['budget_k'])
                        self.assertTrue(solution['validated_direct_raycast'])
                        self.assertEqual(solution['canonical'], archive['canonical'][ids].tolist())
                        np.testing.assert_allclose(
                            geometry.poses(np.asarray(solution['canonical']).reshape(-1, 5)),
                            np.asarray(solution['poses']).reshape(-1, 5))
                boundary = built['metadata']['boundary_context']
                self.assertTrue(boundary['neighbor_coverage_must_be_reevaluated'])
                self.assertFalse(boundary['periodic'])
                self.assertFalse(boundary['wrap'])
                json.dumps(report, allow_nan=False)

    def test_independent_scene_detects_mask_corruption(self):
        actual_init = MeshVisibility.__init__
        actual_mask = MeshVisibility.mask
        providers = []

        def track(provider, *args, **kwargs):
            actual_init(provider, *args, **kwargs)
            providers.append(provider)

        def corrupt(provider, pose):
            result = actual_mask(provider, pose)
            return ~result if len(providers) > 1 else result

        with patch.object(MeshVisibility, '__init__', track), patch.object(MeshVisibility, 'mask', corrupt):
            with self.assertRaisesRegex(ValueError, 'Independent raycast'):
                library.build_component_library(self.spec(), self.profile())

    def test_identical_masks_are_not_pruned_and_empty_layout_is_valid(self):
        profile = LidarProfile('SHORT_RANGE', 360, 180, .01)
        built = library.build_component_library(self.spec(), profile)
        self.assertEqual(built['archive']['masks'].shape, (4, 12))
        self.assertFalse(built['archive']['masks'].any())
        geometry = ConnectedCorridorGeometry(**built['metadata']['geometry'])
        for row in built['frontier']['frontier']:
            self.assertEqual(row['selected_ids'], [])
            self.assertEqual(row['canonical'], [])
            self.assertEqual(row['covered_count'], 0)
            self.assertTrue(row['validated_direct_raycast'])
            self.assertEqual(geometry.poses(np.asarray(row['canonical']).reshape(-1, 5)).shape, (0, 5))

    def test_frozen_roundtrip_and_nooverwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new_component'
            library.execute_library(self.spec(30), self.profile(), output)
            loaded = library.load_component_library(output)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assert_timings(loaded['metadata'], worker=True)
            self.assertEqual(loaded['metadata']['phase_wall_sec']['worker_total'], manifest['worker_wall_sec'])
            self.assertEqual(loaded['metadata']['source_hashes'], manifest['source_hashes'])
            self.assertEqual(manifest['source_hashes']['library_components.py'], library.digest(library.__file__))
            self.assertIn('src/core/deployment_library.py', manifest['source_hashes'])
            self.assertEqual(manifest['source_hashes']['src/core/deployment_library.py'],
                library.digest(output / 'source/src/core/deployment_library.py'))
            self.assertEqual(loaded['metadata']['sensor'], asdict(self.profile()))
            self.assertFalse(loaded['metadata']['legacy_k_directory_compatible'])
            self.assertEqual(loaded['metadata']['geometry']['turn_deg'], 30)
            self.assertEqual(loaded['frontier']['candidate_count'], 4)
            geometry = ConnectedCorridorGeometry(**loaded['metadata']['geometry'])
            np.testing.assert_allclose(geometry.poses(loaded['archive']['canonical']),
                                       loaded['archive']['poses'])
            self.assertGreater(len(loaded['geometry_archive']['triangles']), 0)
            before = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError):
                library.execute_library(self.spec(), self.profile(), output)
            after = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            with (output / 'library.npz').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                library.load_component_library(output)

    def test_limited_fov_model_version_required_on_load(self):
        from src.environment import OPTICAL_MODEL

        profile = LidarProfile('LIMITED', 270., 90., 10.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            library.execute_library(self.spec(), profile, root)
            loaded = library.load_component_library(root)
            self.assertEqual(loaded['metadata']['optical_model'], OPTICAL_MODEL)
            self.assertEqual(loaded['frontier']['context']['optical_model'], OPTICAL_MODEL)
            manifest = json.loads((root / 'manifest.json').read_text())
            metadata = json.loads((root / 'metadata.json').read_text())
            metadata.pop('optical_model')
            (root / 'metadata.json').write_text(json.dumps(metadata))
            manifest['artifact_hashes']['metadata.json'] = library.digest(root / 'metadata.json')
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'optical model mismatch'):
                library.load_component_library(root)

    def test_default_dry_run_no_mesh_solve_or_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'must_not_exist'
            with patch.object(ConnectedCorridorGeometry, 'create_environment_mesh',
                              side_effect=AssertionError('Unexpected mesh')), \
                    patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Unexpected raycast')), \
                    patch('src.core.deployment_library.build_frontier', side_effect=AssertionError('Unexpected solve')), \
                    redirect_stdout(io.StringIO()) as text:
                library.main(['--output', str(output)])
            self.assertEqual(list(Path(tmp).iterdir()), [])
            plan = json.loads(text.getvalue())
            self.assertFalse(plan['execute'])
            self.assertEqual(plan['k_values'], [1, 2, 3])
            self.assertEqual(plan['scope'], 'single_finite_component_smoke_only')

    def test_execute_existing_output_and_dangling_symlink_rejected_before_raycast(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'existing'
            output.mkdir()
            marker = output / 'old.txt'
            marker.write_text('preserved')
            dangling = Path(tmp) / 'link'
            dangling.symlink_to(Path(tmp) / 'missing')
            for target in (output, dangling):
                with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Unexpected raycast')), \
                        redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                    library.main(['--output', str(target), '--execute'])
            self.assertEqual(marker.read_text(), 'preserved')
            self.assertFalse((Path(tmp) / 'missing').exists())

    def test_size_guards_reject_before_domain_allocation(self):
        for key, value in [('max_candidates', 3), ('max_targets', 11),
                           ('max_mask_entries', 47), ('max_mesh_sections', 5)]:
            spec = self.spec()
            spec['limits'][key] = value
            with self.subTest(key=key), \
                    patch.object(ConnectedCorridorGeometry, 'target_samples',
                                 side_effect=AssertionError('Allocated targets')), \
                    self.assertRaisesRegex(ValueError, key):
                library.plan_component(spec, self.profile())

    def test_invalid_sampling_and_unsupported_scope(self):
        changes = [
            ('candidate', 'station_step_m', 5), ('candidate', 'station_step_m', 1e-20),
            ('candidate', 'heights_m', [3, 3]), ('candidate', 'heights_m', [True]),
            ('candidate', 'relative_yaws_deg', [56]), ('candidate', 'pitches_deg', [-30]),
            ('target', 'step_m', 3), ('target', 'heights_m', [1.5, 1.5]),
            ('target', 'heights_m', [.2]), ('geometry', 'mesh_step_m', 1e-20),
        ]
        for section, key, value in changes:
            spec = self.spec()
            spec[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                library.plan_component(spec, self.profile())
        for extra in ({'components': []}, {'requirements': {}}, {'periodic': True}):
            with self.assertRaises(ValueError):
                library.plan_component({**self.spec(), **extra}, self.profile())
        for k in (0, True, 1.5, 1000000000):
            spec = deepcopy(self.spec())
            spec['max_k'] = k
            with self.assertRaises(ValueError):
                library.plan_component(spec, self.profile())

    def test_worker_cannot_reenter_live_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flags in ([], ['--execute']):
                with self.assertRaises(ValueError):
                    library.main(['--worker-root', tmp, *flags])


if __name__ == '__main__':
    unittest.main()

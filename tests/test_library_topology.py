"""Small actual source raycasts and no-execution guards for Track A."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_topology as library
from prototype_offline import absolute, digest
from src.environment import MapGeometry, get_scenario, load_config


class TopologyLibraryTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('configs/library_topology_smoke.yaml')
        self.scenario = get_scenario(1)

    def test_plan_never_raycast_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not_created'
            with patch.object(library, 'MeshVisibility', side_effect=AssertionError('no rays')), \
                    patch.object(library, 'build_frontier', side_effect=AssertionError('no solve')), \
                    patch.object(library.subprocess, 'run', side_effect=AssertionError('no worker')), \
                    redirect_stdout(io.StringIO()):
                plan = library.main(['--output', str(output)])
            self.assertFalse(output.exists())
            self.assertTrue(plan['enumeration_will_complete'])

    def test_actual_raycast_roundtrip_and_frozen_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'source'
            root = library.execute_library(self.config, self.scenario, output, [1, 2, 3])
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            for name, expected in manifest['artifact_hashes'].items():
                self.assertEqual(digest(root / name), expected)
            for name, expected in manifest['source_hashes'].items():
                self.assertEqual(digest(root / 'source' / name), expected)
            report = json.loads((root / 'frontier.json').read_text())
            with np.load(root / 'candidates.npz', allow_pickle=False) as archive:
                self.assertEqual(archive['masks'].dtype, bool)
                previous = 0
                for row in report['frontier']:
                    self.assertGreaterEqual(row['covered_count'], previous)
                    previous = row['covered_count']
                    for solution in [row, *row['alternatives']]:
                        ids = solution['selected_ids']
                        count = int(np.any(archive['masks'][ids], axis=0).sum())
                        self.assertEqual(count, solution['covered_count'])
                        rebuilt = absolute(solution['canonical'], MapGeometry(self.scenario))
                        np.testing.assert_allclose(rebuilt, archive['poses'][ids], atol=1e-7)
                        self.assertTrue(solution['validated_direct_raycast'])
                    self.assertEqual(row['certification'], 'discrete_optimal')
            validation = report['physical_validation']
            self.assertGreater(validation['build_work']['counts']['rays_submitted'], 0)
            self.assertGreater(validation['validation_work']['counts']['rays_submitted'], 0)
            self.assertTrue(all(np.isfinite(v) and v >= 0 for v in validation['phase_wall_sec'].values()))
            with self.assertRaises(FileExistsError):
                library.execute_library(self.config, self.scenario, output, [1])

    def test_invalid_and_pruned_configuration_rejected(self):
        for field, value in (('max_orientations_per_position', 10), ('z_step', 0),
                             ('pitch_step_deg', 1e-20), ('yaw_step_deg', float('nan'))):
            cfg = load_config('configs/library_topology_smoke.yaml')
            cfg.candidate[field] = value
            with self.assertRaises(ValueError):
                library.plan_library(cfg, self.scenario, [1])
        for ks in ([], [2, 1], [1, 1], [True]):
            with self.assertRaises(ValueError):
                library.plan_library(self.config, self.scenario, ks)

    def test_source_unsupported_fields_rejected_without_geometry_or_output(self):
        for key in ('constraints', 'obstacles', 'forbidden_xy_boxes', 'cost_profile'):
            scenario = replace(self.scenario, raw={**self.scenario.raw, key: {}})
            with patch.object(library, 'MapGeometry', side_effect=AssertionError('no geometry')):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Unsupported Track A'):
                    library.plan_library(self.config, scenario, [1])

    def test_limited_fov_new_source_roundtrip_and_version(self):
        from library_topology_transfer import load_topology_library
        from src.environment import OPTICAL_MODEL

        config = deepcopy(self.config)
        config.raw['lidar_profiles'][config.lidar_profile]['h_fov_deg'] = 270.
        with tempfile.TemporaryDirectory() as directory:
            root = library.execute_library(config, self.scenario, Path(directory) / 'source', [1])
            loaded = load_topology_library(root)
            self.assertEqual(loaded['manifest']['plan']['optical_model'], OPTICAL_MODEL)
            self.assertEqual(loaded['frontier']['context']['optical_model'], OPTICAL_MODEL)

    def test_zero_coverage_empty_layouts_and_alternatives(self):
        self.config.raw['lidar_profiles']['VELODYNE64_R30']['max_range_m'] = .001
        result = library.build_library(self.config, self.scenario, [1, 2], alternatives_per_k=3)
        for row in result['frontier']['frontier']:
            self.assertEqual(row['num_selected'], 0)
            self.assertEqual(row['canonical'], [])
            self.assertTrue(row['validated_direct_raycast'])
        self.assertEqual(result['frontier']['minimum_to_all_targets']['status'],
                         'unattainable_in_declared_domain')

    def test_internal_worker_requires_explicit_execution(self):
        with self.assertRaises(ValueError):
            library.main(['--worker-root', '/tmp/not-a-library'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Library planning and cold-target transfer checks; no full MILP run."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_offline import explicit_scenario
from prototype_library import load_suite, offline_command, parser, run, evaluate_target
from src.environment import MapGeometry, load_config


class LibraryExpansionTest(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads(Path('configs/library_expansion.json').read_text())

    def test_plan_sources_and_targets_disjoint(self):
        suite = load_suite(self.suite)
        self.assertEqual(sum(len(f['sources']) for f in suite['families']), 4)
        self.assertEqual(sum(len(f['targets']) for f in suite['families']), 4)
        for f in suite['families']:
            self.assertEqual([s['road_width'] for s in f['sources']], [7., 14.])

    def test_reject_leakage_duplicates_and_extrapolation(self):
        for mutation in ('duplicate', 'geometry_leak', 'outside', 'bad_name'):
            suite = copy.deepcopy(self.suite)
            family = suite['families'][0]
            if mutation == 'duplicate':
                family['targets'][0]['id'] = family['sources'][0]['id']
            elif mutation == 'geometry_leak':
                family['targets'][0] = dict(family['sources'][0], id=490001)
            elif mutation == 'outside':
                family['targets'][0]['road_width'] = 16
            else:
                family['name'] = '../unsafe'
            with self.assertRaises(ValueError):
                load_suite(suite)

    def test_explicit_geometry_and_validation(self):
        row = self.suite['families'][0]['sources'][0]
        g = MapGeometry(explicit_scenario(row))
        self.assertEqual((g.junction_x_m, g.junction_y_m), (35., 55.))
        for changes in ({'road_width': float('nan')}, {'topology': 'CROSS'}, {'num_lanes': 0}):
            with self.assertRaises(ValueError):
                explicit_scenario(dict(row, **changes))

    def test_dry_run_no_writes_no_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'not-created'
            args = parser().parse_args(['--output', str(output)])
            with patch('prototype_library.subprocess.run') as process:
                run(args)
            process.assert_not_called()
            self.assertFalse(output.exists())
            cmd = offline_command(args, Path(directory), self.suite['families'][0]['sources'][0])
            self.assertIn('--scenario-json', cmd)
            self.assertIn('--max-edges', cmd)
            self.assertNotIn('--road-width', cmd)

    def test_transfer_scores_all_initials_and_records_unavailable(self):
        row = self.suite['families'][0]['targets'][0]
        geometry = MapGeometry(explicit_scenario(row))
        cfg = load_config()
        low = (Path('low'), {'certification': 'best_known'}, {}, geometry, np.zeros((1, 5)), cfg)
        high = (Path('high'), {'certification': 'best_known'}, {}, geometry, np.zeros((2, 5)), cfg)
        poses = np.array([[10.25, 50., 3., 90., -20.]])
        with tempfile.TemporaryDirectory() as directory:
            with patch('prototype_library.prepare_segment_placements', return_value=(
                {'transfer_low': poses, 'transfer_high': poses},
                {'status': 'unavailable', 'reason': 'different actual sensor counts'})), \
                patch('prototype_library.validate_poses'), \
                patch('prototype_library.digest', return_value='test-hash'), \
                patch('prototype_library.export_view'), \
                patch('prototype_library.MeshVisibility') as provider:
                provider.return_value.targets = np.array([[0., 0., 1.], [1., 0., 1.]])
                provider.return_value.mask.return_value = np.array([True, False])
                record = evaluate_target(Path(directory), row, 3, low, high)
            self.assertEqual(record['placements']['interpolation']['status'], 'unavailable')
            self.assertEqual(record['placements']['transfer_low']['coverage_pct'], 50.)
            self.assertFalse((Path(directory)/'cache').exists())

    def test_small_end_to_end_library_and_cold_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = copy.deepcopy(self.suite)
            suite['families'] = suite['families'][:1]
            suite['families'][0]['targets'] = suite['families'][0]['targets'][:1]
            suite_path = root/'suite.json'
            suite_path.write_text(json.dumps(suite))
            cfg = load_config()
            cfg.candidate.update(xy_step_m=10., z_min=3., z_max=3., z_step=1.,
                                 yaw_step_deg=55., pitch_step_deg=20.)
            cfg.target['coverage_z_slices'] = [1.5]
            config_path = root/'config.yaml'
            config_path.write_text(yaml.safe_dump(cfg.raw))
            output = root/'run'
            args = parser().parse_args(['--suite', str(suite_path), '--config', str(config_path),
                '--output', str(output), '--k-values', '1', '--execute'])
            with patch('prototype_library.export_view'):
                run(args)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['library']), 2)
            self.assertEqual(len(manifest['transfers']), 1)
            for source in manifest['library']:
                self.assertTrue((Path(source['root'])/'candidates.npz').exists())
                self.assertEqual(source['results'][0]['certification'], 'discrete_optimal')
            report = json.loads((output/'transfers/s420001/k1/report.json').read_text())
            self.assertFalse(report['full_target_cache'])
            self.assertGreater(report['runtime']['counts']['rays_submitted'], 0)
            self.assertTrue((output/'index.html').exists())
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)

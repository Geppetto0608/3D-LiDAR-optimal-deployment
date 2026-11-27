"""Fixed-source transfer and independent audits on different target geometries."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import connected_constraint_study as runner
import connected_corridor_study as corridor
from src.environment import ConnectedCorridorGeometry
from validate_connected_constraints import audit


class ConnectedHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.base = corridor.load_suite(ROOT / 'configs/connected_corridor_pilot.json')
        self.study = json.loads((ROOT / 'configs/connected_constraint_holdout.json').read_text())

    def test_requirements_and_selection_budget_remain_fixed(self):
        development = json.loads((ROOT / 'configs/connected_constraint_study.json').read_text())
        self.assertEqual(self.study['profiles'], development['profiles'])
        self.assertEqual(self.study['alternatives_per_site'], development['alternatives_per_site'])
        suite = runner.validate_study(self.study, self.base)
        self.assertEqual(suite['geometry'], self.base['geometry'])
        self.assertEqual(len(runner.study_jobs(self.study)), 24)
        for target in self.study['targets']:
            self.assertNotEqual(target['road_width_m'], self.base['geometry']['road_width_m'])
            self.assertNotIn(target['turn_deg'], self.base['angles_deg'])

    def test_invalid_or_repeated_target_and_allocation_rejected(self):
        for update in [dict(road_width_m=8.5), dict(bend_length_m=1), dict(length_m=90), dict(name='../bad')]:
            study = deepcopy(self.study)
            study['targets'][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                runner.validate_study(study, self.base)
        study = deepcopy(self.study)
        study['targets'][1].update(study['targets'][0], name='duplicate_shape')
        with self.assertRaisesRegex(ValueError, 'Duplicate held-out geometry'):
            runner.validate_study(study, self.base)
        study['targets'].append({**study['targets'][2], 'name': 'extra'})
        with self.assertRaises(ValueError):
            runner.validate_study(study, self.base)

    def test_transfer_matches_whole_target_grid_without_source_mutation(self):
        source_geometry = ConnectedCorridorGeometry(length_m=40, bend_length_m=0)
        source = corridor.candidates(source_geometry, self.base['candidate'])
        before = source.copy()
        for target in self.study['targets']:
            job = dict(angle_deg=target['turn_deg'], road_width_m=target['road_width_m'],
                       bend_length_m=target['bend_length_m'])
            geometry = runner.target_geometry(self.base, job)
            transferred = runner.transfer_source(source, geometry)
            expected = corridor.candidates(geometry, self.base['candidate'])
            self.assertTrue(np.array_equal(transferred, expected))
            poses = geometry.poses(transferred)
            _, heading = geometry.centerline(transferred[:, 0])
            relative = (poses[:, 3]-np.rad2deg(heading)-np.where(transferred[:, 1] > 0, -90, 90)+180)%360-180
            np.testing.assert_allclose(relative, transferred[:, 3], atol=1e-10)
            np.testing.assert_array_equal(transferred[:, 2:], np.tile(source[:, 2:], (2, 1)))
        np.testing.assert_array_equal(source, before)

    def test_plan_does_not_query_visibility_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'uncreated'
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Target queried during plan')):
                with redirect_stdout(io.StringIO()) as stream:
                    runner.main(['--study', str(ROOT / 'configs/connected_constraint_holdout.json'), '--output', str(output)])
            self.assertFalse(output.exists())
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan['method_runs'], 168)
            self.assertEqual(plan['source_geometry']['road_width_m'], 7)
            self.assertEqual(plan['source_geometry']['length_m'], 40)

    def test_frozen_width_transfer_audit_and_tampering(self):
        base = deepcopy(self.base)
        base['geometry'].update(length_m=40, bend_length_m=12, mesh_step_m=1)
        base['candidate'].update(station_step_m=10, pitches_deg=[-10])
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        study = dict(schema='connected_constraint_holdout_v1', alternatives_per_site=2,
            targets=[dict(name='width8', road_width_m=8, turn_deg=30, bend_length_m=12),
                     dict(name='width10', road_width_m=10, turn_deg=30, bend_length_m=12)],
            profiles=[dict(name='base', interface={}, forbidden_installation=[]),
                      dict(name='all_blocked', interface={}, forbidden_installation=[
                          dict(station_range_m=[0, 40], side='both')])])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'base.json').write_text(json.dumps(base))
            (tmp / 'study.json').write_text(json.dumps(study))
            output = tmp / 'result'
            run = subprocess.run([sys.executable, str(ROOT / 'connected_constraint_study.py'),
                '--base-suite', str(tmp / 'base.json'), '--study', str(tmp / 'study.json'),
                '--output', str(output), '--execute'], cwd=ROOT, capture_output=True, text=True, timeout=120)
            logs = ''.join(p.read_text() for p in (output / 'worker.log', output / 'audit.log') if p.exists())
            self.assertEqual(run.returncode, 0, run.stderr+logs)
            checked = json.loads((output / 'independent_validation.json').read_text())
            self.assertEqual(checked['conditions'], 4)
            self.assertEqual(checked['reference_optima_checked'], 8)
            self.assertTrue(checked['fixed_source_geometry_verified'])
            self.assertTrue(checked['road_relative_transfer_verified'])
            metadata = json.loads((output / 'library/metadata.json').read_text())
            self.assertEqual(metadata['geometry']['road_width_m'], 7)
            path = output / 'case_00_00/geometry.npz'
            with np.load(path, allow_pickle=False) as archive:
                changed = {key: archive[key] for key in archive.files}
            changed['canonical_candidates'][0, 1] = -6
            np.savez_compressed(path, **changed)
            case_manifest = output / 'case_00_00/manifest.json'
            values = json.loads(case_manifest.read_text())
            values['geometry_sha256'] = corridor.digest(path)
            case_manifest.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, 'road-relative source transfer'):
                audit(output)


if __name__ == '__main__':
    unittest.main(verbosity=2)

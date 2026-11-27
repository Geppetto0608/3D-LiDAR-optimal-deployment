"""Constraint frames, exact screening, diverse source modes and frozen study integration."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from itertools import combinations, product
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
import connected_constraint_study as study_runner
import connected_corridor_study as corridor
from src.environment import ConnectedCorridorGeometry
from src.methods.connected_pairs import assess_ids, exact_pairs
from src.methods.source_modes import diverse_source_alternatives
from src.roadside.corridor_requirements import CorridorRequirements, validate_profile
from validate_connected_constraints import audit


class ConnectedConstraintTests(unittest.TestCase):
    def fixture(self, profile=None):
        geometry = ConnectedCorridorGeometry(length_m=20, bend_length_m=8, turn_deg=30)
        canonical = np.array([[2, -6, 3, 0, -10], [6, 6, 3, 0, -10],
                              [14, -6, 5, 0, -10], [18, 6, 3, 0, -10]], dtype=float)
        poses = geometry.poses(canonical)
        targets, coordinates = geometry.target_samples(heights=(1.5,))
        if profile is None:
            profile = dict(name='example', interface=dict(min_coverage_pct=95, min_overlap_ratio=.5),
                forbidden_installation=[dict(station_range_m=[0, 10], side='right')])
        spec = dict(schema='constrained_coverage_v2', budget_k=2,
            weights=dict(overlap_penalty=.05, installation_cost_weight=.2),
            cost=dict(base_per_sensor=10, per_height_m=1),
            constraints=dict(max_installation_cost=28, min_xy_separation_m=3))
        req = CorridorRequirements(spec, targets, coordinates, canonical, poses, profile,
            length_m=20, interface_half_length_m=4)
        return req, poses, canonical, coordinates

    def test_road_band_is_not_global_overlap_or_world_axis_rectangle(self):
        req, poses, canonical, coordinates = self.fixture()
        masks = np.vstack([np.ones(len(coordinates), dtype=bool), ~req.interface_mask])
        assessment = req.evaluate(poses[[1, 2]], masks.sum(axis=0), sensor_masks=masks)
        self.assertGreater(assessment.overlap_ratio, .5)
        self.assertEqual(assessment.regions['interface']['overlap_ratio'], 0)
        self.assertIn('region:interface:overlap_min', assessment.violations)
        masks[1] = True
        good = req.evaluate(poses[[1, 2]], masks.sum(axis=0), sensor_masks=masks)
        self.assertTrue(good.feasible)
        blocked = req.evaluate(poses[[0, 2]], masks.sum(axis=0), sensor_masks=masks)
        self.assertIn('forbidden_corridor_installation', blocked.violations)
        self.assertEqual(req.eligible_ids(range(4)), [1, 2, 3])
        unknown = poses[[1, 2]].copy()
        unknown[0, 0] += .1
        with self.assertRaisesRegex(ValueError, 'declared discrete grid'):
            req.evaluate(unknown, masks.sum(axis=0), sensor_masks=masks)

    def test_pair_screen_matches_full_shared_constraint_enumeration(self):
        req, poses, canonical, coordinates = self.fixture()
        masks = np.random.default_rng(51).random((4, len(coordinates))) < .7
        masks[1:3, req.interface_mask] = True
        ids = req.eligible_ids(range(4))
        modules = np.array([0, 0, 1, 1])
        for quota in (None, modules):
            feasible = []
            for pair in combinations(ids, 2):
                if quota is not None and quota[pair[0]] == quota[pair[1]]:
                    continue
                value = assess_ids(pair, poses, masks.__getitem__, req)
                if value.feasible:
                    feasible.append((value.objective_score, list(pair)))
            wanted = sorted(feasible, key=lambda x: (-x[0], x[1]))[0]
            actual = exact_pairs(ids, poses, masks.__getitem__, req, module_ids=quota,
                                 pair_mask_filter=req.filter_visibility_pairs)
            self.assertAlmostEqual(actual['assessment']['objective_score'], wanted[0])
            self.assertEqual(actual['selected_ids'], wanted[1])
        empty_masks = np.zeros_like(masks)
        missing = exact_pairs(ids, poses, empty_masks.__getitem__, req,
                              pair_mask_filter=req.filter_visibility_pairs)
        self.assertIsNone(missing['selected_ids'])
        self.assertEqual(missing['visibility_screened_pairs'], 3)
        with self.assertRaisesRegex(ValueError, 'one boolean'):
            exact_pairs(ids, poses, masks.__getitem__, req, pair_mask_filter=lambda a, b: np.ones(len(b)))

    def test_diversity_preserves_high_and_low_mount_direction_modes(self):
        canonical = np.array(list(product([2, 6], [-6, 6], [3, 5], [-45, 0, 45], [-20, -10])), dtype=float)
        scores = 100-20*(canonical[:, 2]-3)-3*np.abs(canonical[:, 3])/45 + (canonical[:, 4] == -20)
        top = corridor.select_source_alternatives(canonical, scores, 4)
        diverse = diverse_source_alternatives(canonical, scores, 4)
        self.assertEqual(len(top), len(diverse))
        self.assertTrue(np.all(canonical[top, 2] == 3))
        for site in np.unique(canonical[:, :2], axis=0):
            selected = [i for i in diverse if np.array_equal(canonical[i, :2], site)]
            modes = set(map(tuple, canonical[selected][:, [2, 3]]))
            self.assertEqual(len(modes), 4)
            self.assertIn((3., 0.), modes)
            self.assertIn((5., -45.), modes)
            self.assertIn((5., 45.), modes)
        self.assertEqual(diverse, diverse_source_alternatives(canonical, scores, 4))
        with self.assertRaises(ValueError):
            diverse_source_alternatives(canonical, scores, 7)

    def test_strict_profile_and_default_plan_guards(self):
        for profile in [dict(name='x', interface=dict(min_overlap_ratio=2), forbidden_installation=[]),
            dict(name='x', interface={}, forbidden_installation=[dict(station_range_m=[0, 5], side='north')]),
            dict(name='x', interface={}, forbidden_installation=[dict(station_range_m=[0, 21], side='both')])]:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_profile(profile, 20)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'uncreated'
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Unexpected raycast')):
                with redirect_stdout(io.StringIO()) as stream:
                    study_runner.main(['--output', str(root)])
            self.assertFalse(root.exists())
            self.assertEqual(json.loads(stream.getvalue())['conditions'], 16)
            self.assertEqual(json.loads(stream.getvalue())['method_runs'], 112)
            root.mkdir()
            with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                study_runner.main(['--output', str(root), '--execute'])
            with self.assertRaises(ValueError):
                study_runner.main(['--worker-root', str(root)])

    def test_frozen_study_distinguishes_blocked_transfer_and_grid_infeasibility(self):
        base = corridor.load_suite(ROOT / 'configs/connected_corridor_pilot.json')
        base['geometry'].update(length_m=40, bend_length_m=12, mesh_step_m=1)
        base['candidate'].update(station_step_m=10, relative_yaws_deg=[-45, 0, 45], pitches_deg=[-10])
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        study = dict(schema='connected_constraint_study_v1', angles_deg=[30, 60], bend_length_m=12,
            alternatives_per_site=2, profiles=[dict(name='base', interface={}, forbidden_installation=[]),
                dict(name='constrained', interface=dict(min_overlap_ratio=.3),
                     forbidden_installation=[dict(station_range_m=[0, 10], side='right')]),
                dict(name='all_blocked', interface={},
                     forbidden_installation=[dict(station_range_m=[0, 40], side='both')])])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'base.json').write_text(json.dumps(base))
            (tmp / 'study.json').write_text(json.dumps(study))
            output = tmp / 'run'
            run = subprocess.run([sys.executable, str(ROOT / 'connected_constraint_study.py'),
                '--base-suite', str(tmp / 'base.json'), '--study', str(tmp / 'study.json'),
                '--output', str(output), '--execute'], cwd=ROOT, capture_output=True, text=True, timeout=120)
            logs = ''.join(p.read_text() for p in (output / 'worker.log', output / 'audit.log') if p.exists())
            self.assertEqual(run.returncode, 0, run.stderr+logs)
            checked = json.loads((output / 'independent_validation.json').read_text())
            self.assertEqual(checked['conditions'], 6)
            self.assertEqual(checked['method_rows'], 42)
            self.assertEqual(checked['reference_optima_checked'], 12)
            self.assertEqual(checked['source_pool_optima_checked'], 12)
            manifest = json.loads((output / 'manifest.json').read_text())
            for job in manifest['jobs']:
                if job['profile'] != 'all_blocked':
                    continue
                directory = output / job['directory']
                reference = json.loads((directory / 'reference_global/result.json').read_text())
                initial = json.loads((directory / 'independent_transfer/result.json').read_text())
                self.assertIsNone(reference['assessment'])
                self.assertEqual(reference['status'], 'infeasible_in_declared_grid')
                self.assertFalse(initial['assessment']['feasible'])
                self.assertFalse((directory / 'top_joint/deployment.npz').exists())
            path = output / manifest['jobs'][0]['directory'] / 'top_joint/result.json'
            changed = json.loads(path.read_text())
            changed['assessment']['objective_score'] += 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'pool optimum'):
                audit(output)


if __name__ == '__main__':
    unittest.main(verbosity=2)

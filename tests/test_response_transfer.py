"""Source residual/mapping contracts, identical optical controls and real smoke."""
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
import connected_response_transfer_study as runner
import test_connected_hierarchy as fixtures
from src.methods.connected_hierarchy import BudgetedObservations, select_pool
from src.methods.response_transfer import correspondence, nearest_source, transfer_residual
from validate_response_transfer import audit


class ResponseTransferTests(unittest.TestCase):
    def test_only_source_optically_valid_occlusion_transfers_unknowns_use_target_optics(self):
        optical = np.ones((2, 4), bool)
        so = np.array([[1, 1, 0], [1, 0, 1]], bool)
        sm = np.array([[1, 0, 0], [0, 0, 1]], bool)
        actual = transfer_residual(optical, so, sm, np.array([1, 0]), np.array([0, 1, 2, -1]))
        np.testing.assert_array_equal(actual, [[0, 1, 1, 1], [1, 0, 1, 1]])
        np.testing.assert_array_equal(transfer_residual(optical, so, so, np.array([1, 0]), np.array([0, 1, 2, -1])), optical)
        self.assertTrue(optical.all())
        for ids in (np.array([-2, 0, 1, 2]), np.array([0., 1., 2., -1.])):
            with self.assertRaises(ValueError):
                transfer_residual(optical, so, sm, np.array([1, 0]), ids)

    def test_mapping_preserves_station_side_height_attitude_and_normalized_lateral(self):
        sc = np.array([[2, -6.5, 3, 0, -10], [2, 6.5, 3, 0, -10]], float)
        tc = sc.copy()
        tc[:, 1] = [-7.5, 7.5]
        st = np.array([[1, -3, 1], [1, -1, 1], [1, 1, 1], [1, 3, 1]], float)
        tt = np.array([[1, -4.5, 1], [1, -1.5, 1], [1, 1.5, 1], [1, 4.5, 1], [2, 1, 1]], float)
        cids, tids = correspondence(sc, st, tc, tt, 8, 12)
        np.testing.assert_array_equal(cids, [0, 1])
        np.testing.assert_array_equal(tids, [0, 1, 2, 3, -1])
        tc[0, 3] = 5
        with self.assertRaisesRegex(ValueError, 'pose grid'):
            correspondence(sc, st, tc, tt, 8, 12)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            correspondence(np.repeat(sc, 2, axis=0), st, sc, tt, 8, 12)

    def test_source_choice_uses_predeclared_geometry_distance_and_stable_name_tie(self):
        sources = [dict(name='b', road_width_m=10, turn_deg=60, bend_length_m=22),
            dict(name='a', road_width_m=10, turn_deg=60, bend_length_m=22)]
        target = dict(road_width_m=11, turn_deg=65, bend_length_m=23)
        name, distance = nearest_source(sources, target)
        self.assertEqual(name, 'a')
        self.assertAlmostEqual(distance, (1/7)**2+(5/90)**2+(1/20)**2)

    def test_opt_in_transforms_preserve_default_seed_policy_and_only_remove_optical_bits(self):
        context, masks = fixtures.HierarchyTests().fixture()
        a = BudgetedObservations(context, lambda i: masks[i], per_site=2)
        b = BudgetedObservations(context, lambda i: masks[i], per_site=2)
        old = select_pool('probe_context', context, a, 2, probe_prior='geometry', fast_screen=True)
        new = select_pool('probe_context', context, b, 2, probe_prior='geometry', fast_screen=True,
            prediction_transform=lambda optical: optical)
        for key in old:
            if key != 'selection_sec':
                self.assertEqual(old[key], new[key])
        c = BudgetedObservations(context, lambda i: masks[i], per_site=2)
        changed = select_pool('probe_context', context, c, 2, probe_prior='geometry', fast_screen=True,
            prediction_transform=lambda optical: np.zeros_like(optical))
        self.assertEqual([e for e in new['events'] if e['phase'] == 'seed'],
            [e for e in changed['events'] if e['phase'] == 'seed'])
        self.assertEqual(new['prior_values'], changed['prior_values'])
        with self.assertRaisesRegex(ValueError, 'boolean subset'):
            select_pool('probe_context', context, BudgetedObservations(context, lambda i:masks[i], per_site=1), 1,
                probe_prior='geometry', prediction_transform=lambda optical: np.ones_like(optical))
        with self.assertRaises(ValueError):
            select_pool('geometry_single', context, BudgetedObservations(context, lambda i:masks[i], per_site=1), 1,
                prediction_transform=lambda optical:optical)

    def test_plan_no_files_rays_or_training_and_design_disjointness(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'not_created'
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Plan queried visibility')):
                with redirect_stdout(io.StringIO()) as text:
                    runner.main(['--output', str(output)])
            plan = json.loads(text.getvalue())
            self.assertEqual((plan['sources'], plan['conditions'], plan['online_outputs'], plan['references']), (5, 24, 288, 24))
            self.assertFalse(plan['lower_refinement'])
            self.assertFalse(output.exists())
            design = runner.read(ROOT/'configs/connected_response_transfer.json')
            design['targets'][0] = design['sources'][0]
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                runner.design_inputs(design, runner.read(ROOT/'configs/connected_constraint_holdout.json'),
                    runner.corridor.load_suite(ROOT/'configs/connected_corridor_pilot.json'))

    def test_frozen_real_smoke_empty_eligible_case_audit_and_tamper_rejection(self):
        base = deepcopy(runner.corridor.load_suite(ROOT/'configs/connected_corridor_pilot.json'))
        base['geometry'].update(length_m=40, bend_length_m=12, mesh_step_m=1)
        base['candidate']['station_step_m'] = 10
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        study = dict(schema='connected_constraint_holdout_v1', alternatives_per_site=2, targets=[],
            profiles=[dict(name='base', interface={}, forbidden_installation=[]),
                dict(name='blocked', interface={}, forbidden_installation=[dict(station_range_m=[0,40], side='both')])])
        design = dict(schema='connected_response_transfer_v1', query_budgets=[1,2],
            sources=[dict(name='straight', road_width_m=8, turn_deg=0, bend_length_m=0),
                dict(name='curve', road_width_m=8, turn_deg=50, bend_length_m=12)],
            targets=[dict(name='target', road_width_m=10, turn_deg=30, bend_length_m=12)])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, data in [('base', base), ('study', study), ('design', design)]:
                runner.corridor.save(tmp/(name+'.json'), data)
            root = tmp/'run'
            cmd = [sys.executable, str(ROOT/'connected_response_transfer_study.py'), '--design', str(tmp/'design.json'),
                '--study', str(tmp/'study.json'), '--base-suite', str(tmp/'base.json'), '--output', str(root), '--execute']
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=ROOT)
            logs = ''.join(p.read_text() for p in (root/'worker.log', root/'audit.log') if p.exists())
            self.assertEqual(run.returncode, 0, run.stderr+logs)
            result = runner.read(root/'independent_validation.json')
            self.assertEqual(result['online_records'], 16)
            self.assertEqual(result['full_optima'], 2)
            self.assertEqual(result['checked_deployments'], 9)
            summary = runner.read(root/'summary.json')
            self.assertEqual(summary['groups']['b1_geometry']['feasible'], 1)
            summary['groups']['b1_geometry']['trials'] += 1
            runner.corridor.save(root/'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'Summary'):
                audit(root)
            refused = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=ROOT)
            self.assertNotEqual(refused.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

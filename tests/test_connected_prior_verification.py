"""Fresh physical evidence is separate from frozen experiment results."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import connected_prior_ablation as runner
import connected_prior_verification as verification


class VerificationTests(unittest.TestCase):
    def test_plan_is_read_only_and_output_cannot_be_inside_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'missing', Path(tmp)/'verification'
            with patch.object(verification, 'run', side_effect=AssertionError('Plan executed')):
                with redirect_stdout(io.StringIO()) as text:
                    verification.main([str(root), '--output', str(output)])
            self.assertFalse(json.loads(text.getvalue())['execute'])
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, 'outside'):
                verification.run(root, root/'audit')

    def test_real_fresh_recheck_preserves_originals_and_handles_infeasible_case(self):
        base = deepcopy(runner.corridor.load_suite(ROOT/'configs/connected_corridor_pilot.json'))
        base['geometry'].update(length_m=40, bend_length_m=12, mesh_step_m=1)
        base['candidate']['station_step_m'] = 10
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        study = dict(schema='connected_constraint_holdout_v1', alternatives_per_site=2,
            targets=[dict(name='development', road_width_m=8, turn_deg=30, bend_length_m=12)],
            profiles=[dict(name='base', interface={}, forbidden_installation=[]),
                dict(name='blocked', interface={}, forbidden_installation=[dict(station_range_m=[0, 40], side='both')])])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner.corridor.save(tmp/'base.json', base)
            runner.corridor.save(tmp/'study.json', study)
            root, output = tmp/'run', tmp/'verification'
            with redirect_stdout(io.StringIO()):
                runner.main(['--base-suite', str(tmp/'base.json'), '--study', str(tmp/'study.json'),
                    '--query-budgets', '1', '--random-seeds', '0', '--output', str(root), '--execute'])
                checked = verification.run(root, output)
            self.assertTrue(checked['originals_unchanged'])
            self.assertEqual(checked['status'], 'complete')
            physical = json.loads((output/'physical_validation.json').read_text())
            self.assertEqual(physical['deployment_records_checked'], 8)
            self.assertEqual(len(physical['geometries']), 2)
            self.assertEqual(physical['source_vs_geometry']['1']['common_feasible_conditions'], 1)
            with self.assertRaises(FileExistsError):
                verification.run(root, output)


if __name__ == '__main__':
    unittest.main(verbosity=2)

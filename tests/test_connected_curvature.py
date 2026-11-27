"""Factorial controls, source-pool nesting and a frozen-source execution check."""
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
import connected_corridor_study as corridor
import connected_curvature_sweep as sweep_runner
from validate_connected_curvature import audit


class ConnectedCurvatureTests(unittest.TestCase):
    def inputs(self):
        base = corridor.load_suite(ROOT / 'configs/connected_corridor_pilot.json')
        sweep = json.loads((ROOT / 'configs/connected_curvature_sweep.json').read_text())
        return base, sweep

    def test_plan_is_factorial_and_does_not_write_or_raycast(self):
        base, sweep = self.inputs()
        sweep_runner.validate_sweep(sweep, base)
        jobs = sweep_runner.plan_jobs(sweep)
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len({(j['bend_length_m'], j['alternatives_per_site'], j['angle_deg']) for j in jobs}), 36)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'uncreated'
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Unexpected raycast')):
                with redirect_stdout(io.StringIO()) as stream:
                    sweep_runner.main(['--output', str(output)])
            self.assertFalse(output.exists())
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan['source_builds'], 1)
            self.assertEqual(plan['geometries'], 12)
            self.assertEqual(plan['method_runs'], 216)
            self.assertEqual(plan['target_pool_sizes'], {'1': 40, '2': 80, '4': 160})
            output.mkdir()
            with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                sweep_runner.main(['--output', str(output), '--execute'])
            with self.assertRaises(ValueError):
                sweep_runner.main(['--worker-root', str(output)])

    def test_invalid_sweeps_and_folded_geometry_are_rejected(self):
        base, sweep = self.inputs()
        variants = [dict(bend_lengths_m=[0]), dict(bend_lengths_m=[20, 20]),
                    dict(bend_lengths_m=[5]), dict(angles_deg=[90], bend_lengths_m=[12]),
                    dict(angles_deg=[True]), dict(alternatives_per_site=[0]),
                    dict(alternatives_per_site=[1.0]), dict(alternatives_per_site=[13]),
                    dict(angles_deg=list(range(9))), dict(unexpected=True)]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                sweep_runner.validate_sweep({**sweep, **variant}, base)

    def test_source_pool_prefixes_keep_ties_and_source_winner(self):
        canonical = np.array([[s, d, z, 0, -10] for s in (2, 6) for d in (-6, 6) for z in (3, 5)], dtype=float)
        scores = np.array([1, 1, 3, 2, 4, 5, 6, 7.])
        one = corridor.select_source_alternatives(canonical, scores, 1)
        two = corridor.select_source_alternatives(canonical, scores, 2)
        self.assertEqual(one, [0, 2, 5, 7])
        self.assertEqual(two, [0, 1, 2, 3, 5, 4, 7, 6])
        self.assertTrue(set(one).issubset(two))
        self.assertIn(int(np.argmax(scores)), one)
        with self.assertRaises(ValueError):
            corridor.select_source_alternatives(canonical, scores[:-1], 1)

    def test_only_requested_factors_change(self):
        base, sweep = self.inputs()
        original = deepcopy(base)
        for bend in sweep['bend_lengths_m']:
            for count in sweep['alternatives_per_site']:
                variant = sweep_runner.variant_suite(base, sweep, bend, count)
                self.assertEqual(variant['requirements'], base['requirements'])
                self.assertEqual(variant['candidate'], base['candidate'])
                self.assertEqual(variant['geometry']['length_m'], 80)
                self.assertEqual(variant['geometry']['road_width_m'], 7)
                self.assertEqual(variant['geometry']['bend_length_m'], bend)
        self.assertEqual(base, original)

    def test_frozen_execution_and_independent_tamper_rejection(self):
        base, sweep = self.inputs()
        base['geometry'].update(length_m=40, bend_length_m=20, mesh_step_m=1)
        base['candidate'].update(station_step_m=10, heights_m=[3, 5],
                                 relative_yaws_deg=[0], pitches_deg=[-10])
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        sweep.update(angles_deg=[30, 60], bend_lengths_m=[20, 12], alternatives_per_site=[1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'base.json').write_text(json.dumps(base))
            (tmp / 'sweep.json').write_text(json.dumps(sweep))
            output = tmp / 'result'
            completed = subprocess.run([sys.executable, str(ROOT / 'connected_curvature_sweep.py'),
                '--base-suite', str(tmp / 'base.json'), '--sweep', str(tmp / 'sweep.json'),
                '--output', str(output), '--execute'], cwd=ROOT, capture_output=True, text=True, timeout=120)
            logs = ''.join(p.read_text() for p in (output / 'worker.log', output / 'audit.log') if p.exists())
            self.assertEqual(completed.returncode, 0, completed.stderr + logs)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['jobs']), 8)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['method_rows'], 48)
            validation = json.loads((output / 'independent_validation.json').read_text())
            self.assertEqual(validation['checked_deployments'], 48)
            self.assertEqual(validation['reference_optima_checked'], 16)
            self.assertEqual(validation['source_builds'], 1)
            self.assertTrue(validation['source_pools_nested'])
            self.assertTrue((output / 'coverage.png').exists())
            self.assertTrue((output / 'objective_gap.png').exists())
            metadata = json.loads((output / 'library/metadata.json').read_text())
            self.assertEqual(metadata['work']['unique_candidate_queries'], 8)
            path = output / manifest['jobs'][0]['directory'] / 'reference_global/result.json'
            result = json.loads(path.read_text())
            result['assessment']['objective_score'] += 1
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'optimum mismatch'):
                audit(output)


if __name__ == '__main__':
    unittest.main(verbosity=2)

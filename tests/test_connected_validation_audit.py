"""Adversarial saved-result checks and independent diagnostic regressions."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
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
import connected_validation_audit as diagnostics
from src.environment import ConnectedCorridorGeometry
from src.methods.connected_diagnostics import pair_table
from src.methods.source_modes import response_source_alternatives
from validate_connected_constraints import audit, check_method_contract


class ValidationMathTests(unittest.TestCase):
    def test_exact_seeded_compression_exposes_greedy_gap(self):
        c = np.array([[s, -6, 3, 0, p] for s in (2, 6) for p in (-25, -23, -21, -19, -17, -15)], float)
        masks = np.random.default_rng(18).random((12, 40)) < .5
        scores = masks.sum(axis=1).astype(float)
        selected, _ = response_source_alternatives(c, masks, scores, 3, intersection_weight=1.05)
        exact, rows = diagnostics.compression_audit(c, masks, scores, {'response': selected}, 3, 1.05)
        self.assertEqual(len(exact), 6)
        self.assertTrue(any(r['response_worst_regret'] > r['exact_worst_regret']+1e-8 for r in rows))
        self.assertTrue(all(r['exact_worst_regret'] <= r['response_worst_regret']+1e-8 for r in rows))
        self.assertTrue(all(r['seed_id'] in r['exact_ids'] for r in rows))
        _, unseeded = diagnostics.compression_audit(c, masks, scores, {'response': selected}, 3, 1.05, preserve_seed=False)
        self.assertTrue(all(a['exact_worst_regret'] <= b['exact_worst_regret']+1e-8 for a, b in zip(unseeded, rows)))
        with self.assertRaisesRegex(ValueError, 'allocation'):
            diagnostics.compression_audit(c, masks, scores, {'response': selected}, 7, 1.05)

    def test_constraint_activity_distinguishes_binding_and_redundant(self):
        poses = np.array([[0, 0, 3, 0, -10], [5, 0, 3, 0, -10], [10, 0, 5, 0, -10]], float)
        masks = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], bool)
        spec = dict(budget_k=2, constraints=dict(max_installation_cost=26, mounting_z_range_m=[3, 5]))
        spec['cost'] = dict(base_per_sensor=10, per_height_m=1)
        allowed = np.ones(3, bool)
        interface = np.array([0, 1, 1, 0], bool)
        table = pair_table(poses, masks, spec, allowed, interface, {})
        rows = {r['constraint']: r for r in diagnostics.constraint_audit(poses, masks, allowed, interface, spec, {}, table)}
        self.assertEqual(rows['mounting_height']['additional_feasible_pairs_if_removed'], 0)
        self.assertEqual(rows['installation_cost']['additional_feasible_pairs_if_removed'], 2)
        self.assertEqual(rows['installation_cost']['j_increase_if_removed'], 1)

    def test_shared_source_site_requires_one_compatible_optimal_subset(self):
        canonical = np.array([[2, -6, 3, 0, p] for p in (-20, -15, -10)]
                             + [[6, -6, 3, 0, -10]], float)
        rows = [dict(site=[2, -6], optimal_subsets=[[0, 1], [1, 2]]),
                dict(site=[6, -6], optimal_subsets=[[3]])]
        ids, compatible = diagnostics.compression_family_domain(canonical, rows)
        self.assertEqual(set(ids), set(range(8)))
        self.assertFalse(compatible[0, 6])
        self.assertTrue(compatible[1, 6])
        self.assertTrue(compatible[0, 7])

    def test_independent_segment_triangle_is_two_sided_and_segment_bounded(self):
        vertices = np.array([[1, -2, -2], [1, 2, -2], [1, 0, 2]], float)
        faces = np.array([[0, 1, 2]])
        origin = np.zeros(3)
        self.assertTrue(diagnostics.segment_blocked(origin, np.array([2, 0, 0]), vertices, faces))
        self.assertTrue(diagnostics.segment_blocked(origin, np.array([2, 0, 0]), vertices, faces[:, ::-1]))
        self.assertFalse(diagnostics.segment_blocked(origin, np.array([.5, 0, 0]), vertices, faces))
        self.assertFalse(diagnostics.segment_blocked(origin, np.array([2, 10, 0]), vertices, faces))

    def test_independent_frames_and_full_circle_optics(self):
        for angle in (-75, 0, 75):
            geometry = ConnectedCorridorGeometry(turn_deg=angle, bend_length_m=24)
            coords = np.array([[s, d, 1.5] for s in (0, 28, 40, 52, 80) for d in (-3, 3)], float)
            actual, _ = diagnostics.independent_road_frame(coords, vars(geometry))
            np.testing.assert_allclose(actual, geometry.points(coords), atol=1e-10)
        profile = dict(h_fov_deg=360, v_fov_deg=26.9, max_range_m=3)
        pose = np.array([0, 0, 3, 0, 0], float)
        targets = np.array([[1, 0, 3], [-1, 0, 3], [0, 0, 4], [4, 0, 3]], float)
        np.testing.assert_array_equal(diagnostics.optical_mask(pose, targets, profile), [1, 1, 0, 0])

    def test_selection_domain_checked_even_without_objective_difference(self):
        modules = np.array([0, 0, 1, 1])
        allowed = np.ones(4, bool)
        with self.assertRaisesRegex(ValueError, 'eligible source pool'):
            check_method_contract('response_joint', dict(selected_ids=[1, 2]), {},
                {'response': [0, 2]}, modules, allowed, 0, 2, 8)
        with self.assertRaisesRegex(ValueError, 'quota'):
            check_method_contract('response_joint', dict(selected_ids=[0, 1]), {},
                {'response': [0, 1, 2]}, modules, allowed, 0, 2, 8)


class SavedResultAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls.temporary.name)
        base = deepcopy(corridor.load_suite(ROOT/'configs/connected_corridor_pilot.json'))
        base['geometry'].update(length_m=40, bend_length_m=12, mesh_step_m=1)
        base['candidate'].update(station_step_m=10)
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3,
            mounting_z_range_m=[3, 5], min_single_failure_coverage_pct=0)
        study = dict(schema='connected_constraint_holdout_v1', alternatives_per_site=2,
            targets=[dict(name='temporary', road_width_m=8, turn_deg=30, bend_length_m=12)],
            profiles=[dict(name='base', interface={}, forbidden_installation=[])])
        (cls.tmp/'base.json').write_text(json.dumps(base))
        (cls.tmp/'study.json').write_text(json.dumps(study))
        cls.root = cls.tmp/'result'
        command = [sys.executable, str(ROOT/'connected_constraint_study.py'), '--response-pool',
            '--base-suite', str(cls.tmp/'base.json'), '--study', str(cls.tmp/'study.json'),
            '--output', str(cls.root), '--execute']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        if result.returncode:
            logs = ''.join(p.read_text() for p in (cls.root/'worker.log', cls.root/'audit.log') if p.exists())
            raise RuntimeError(result.stderr+logs)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def mutate_json(self, path, mutate, message):
        original = path.read_bytes()
        try:
            value = json.loads(original)
            mutate(value)
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, message):
                audit(self.root)
        finally:
            path.write_bytes(original)

    def test_ls_initialization_provenance_tampering_rejected(self):
        path = self.root/'case_00_00/response_joint_ls/result.json'
        def mutate(value):
            first = value['initialization']['selected_ids'][0]
            value['initialization']['selected_ids'][0] = (first//12)*12+(first+1)%12
        self.mutate_json(path, mutate, 'initialization provenance')

    def test_component_and_failure_metric_tampering_rejected(self):
        path = self.root/'case_00_00/response_joint/result.json'
        self.mutate_json(path, lambda v: v['assessment']['components'].__setitem__('installation_cost', 0), 'component mismatch')
        self.mutate_json(path, lambda v: v['assessment']['robustness'].__setitem__('single_failure_coverage_pct', 100), 'failure metric')

    def test_csv_values_previously_not_audited_are_rejected(self):
        path = self.root/'comparison.csv'
        original = path.read_bytes()
        rows = list(csv.DictReader(io.StringIO(original.decode())))
        for field in ('coverage_pct', 'objective_gap', 'candidate_queries', 'rays_submitted', 'online_sec'):
            try:
                changed = deepcopy(rows)
                changed[0][field] = float(changed[0][field])+1
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(changed[0]))
                    writer.writeheader()
                    writer.writerows(changed)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, f'Reported {field}'):
                    audit(self.root)
            finally:
                path.write_bytes(original)

    def test_paired_and_summary_tampering_rejected(self):
        self.mutate_json(self.root/'summary.json', lambda v: v.__setitem__('method_rows', 999), 'Summary method_rows')
        self.mutate_json(self.root/'summary.json', lambda v: v['pairs'][0].__setitem__('response_j', -999), 'Summary paired response_j')
        path = self.root/'paired.csv'
        original = path.read_bytes()
        try:
            path.write_text(original.decode().replace('response_j', 'wrong_response_j'))
            with self.assertRaises((ValueError, KeyError)):
                audit(self.root)
        finally:
            path.write_bytes(original)

    def test_complete_small_audit_no_open3d_calls_and_originals_unchanged(self):
        output = self.root/'audit'
        with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('New Open3D query')):
            with redirect_stdout(io.StringIO()):
                diagnostics.main(['--root', str(self.root), '--output', str(output)])
                self.assertFalse(output.exists())
                diagnostics.main(['--root', str(self.root), '--output', str(output), '--execute'])
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(summary['conditions'], 1)
        self.assertEqual(summary['ls_replays'], 3)
        self.assertGreater(summary['independent_triangle_checks'], 0)
        self.assertEqual(json.loads((output/'manifest.json').read_text())['status'], 'complete')
        with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
            diagnostics.main(['--root', str(self.root), '--output', str(output), '--execute'])
        with self.assertRaisesRegex(ValueError, 'replace input'), redirect_stdout(io.StringIO()):
            diagnostics.main(['--root', str(self.root), '--output', str(self.root), '--execute'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Paired mesh-only stress inputs, analytical rays, and frozen worker integration."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_occlusion_study as study
import library_transfer_study as target_run
from src.core.library_control_evaluation import evaluate_matched_controls
from src.core.scene_occlusion import SCHEMA, validate_spec, with_occlusion
from src.core.visibility import MeshVisibility
from src.environment import LidarProfile


class OcclusionTests(unittest.TestCase):
    def test_sweep_cartesian_design_is_frozen_distinct_and_read_only(self):
        spec = json.loads((study.BASE / 'configs/library_occlusion_large_20261122.json').read_text())
        original = deepcopy(spec)
        with patch.object(Path, 'write_text', side_effect=AssertionError), \
                patch.object(Path, 'write_bytes', side_effect=AssertionError):
            targets = study.expand_design(spec['design'])
        self.assertEqual(spec, original)
        self.assertEqual(targets, study.expand_design(spec['design']))
        self.assertEqual(len(targets), 56)
        self.assertEqual(sum(t['track'] == 'A' for t in targets), 24)
        self.assertEqual(sum(t['track'] == 'B' for t in targets), 32)
        self.assertEqual(len({t['name'] for t in targets}), 56)
        fingerprints = []
        for row in targets:
            target = deepcopy(row['target'])
            target.pop('id', None)
            fingerprints.append(json.dumps(target, sort_keys=True))
        self.assertEqual(len(set(fingerprints)), 56)
        from prototype_offline import explicit_scenario
        for row in targets:
            if row['track'] == 'A':
                self.assertEqual(explicit_scenario(row['target']).topology, 'T_JUNCTION')
        self.assertEqual([t['target']['id'] for t in targets if t['track'] == 'A'],
                         list(range(440001, 440025)))

    def test_sweep_guards_and_legacy_limit(self):
        spec = json.loads((study.BASE / 'configs/library_occlusion_large_20261122.json').read_text())
        changes = [('track_a', 'road_widths_m', [8, 8]), ('track_a', 'road_widths_m', [True]),
                   ('track_a', 'junction_centers_m', [[0, float('nan')]]),
                   ('track_a', 'scenario_id_start', True), ('track_a', 'scenario_id_start', 1),
                   ('track_b', 'modules', 0),
                   ('track_b', 'turn_degrees', ['30']), ('track_b', 'lengths_m', list(range(1, 65)))]
        for section, key, value in changes:
            design = deepcopy(spec['design'])
            design[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.expand_design(design)
        legacy = json.loads((study.BASE / 'configs/library_occlusion_20261122.json').read_text())
        legacy['targets'] = legacy['targets'] * 5
        with patch.object(target_run, 'prepare', side_effect=AssertionError), self.assertRaises(ValueError):
            study.prepare_suite(legacy)

    def problem(self):
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh.create_box().translate([100., 100., 100.])
        return dict(geometry=SimpleNamespace(create_environment_mesh=lambda: mesh),
            poses=np.array([[0, 0, 3, 0, -10], [-1, 0, 3, 0, -10],
                            [0, 4, 3, 0, -10], [4, 4, 3, 0, -10]], float),
            targets=np.array([[4, 0, 1.5], [4, 3, 1.5], [6, 0, 1.5], [6, 3, 1.5]], float),
            profile=LidarProfile('TEST', 360, 180, 30), pool_ids=[0, 1],
            k_values=[1, 2], transfers=[], context=dict(track='B'))

    def spec(self):
        return dict(schema=SCHEMA, boxes=[dict(center_m=[2., 0., 2.25],
                    size_m=[.2, 2., 4.5], yaw_deg=0.)])

    def test_strict_schema(self):
        spec = self.spec()
        bad = [None, {}, dict(spec, cost=1), dict(spec, schema='old'), dict(spec, boxes={})]
        for key, values in [('center_m', [[0, 0], [0, True, 1], [0, 0, float('nan')]]),
                            ('size_m', [[0, 2, 3], [-1, 2, 3], ['1', 2, 3]]),
                            ('yaw_deg', [True, '0', float('inf')])]:
            for value in values:
                changed = deepcopy(spec)
                changed['boxes'][0][key] = value
                bad.append(changed)
        changed = deepcopy(spec)
        changed['boxes'][0]['opacity'] = .5
        bad.append(changed)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_spec(value)
        self.assertEqual(validate_spec(spec), spec)

    def test_reject_enclosed_points_and_candidates_without_filtering(self):
        problem = self.problem()
        for key in ('poses', 'targets'):
            spec = self.spec()
            spec['boxes'][0]['center_m'] = problem[key][0, :3].tolist()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                with_occlusion(problem, spec)
        self.assertNotIn('scene_occlusion', problem['context'])

    def test_empty_mesh_identity_and_no_mutation(self):
        problem, spec = self.problem(), self.spec()
        old_vertices = np.asarray(problem['geometry'].create_environment_mesh().vertices).copy()
        wrapped = with_occlusion(problem, spec)
        self.assertIs(wrapped['poses'], problem['poses'])
        self.assertIs(wrapped['targets'], problem['targets'])
        self.assertIs(wrapped['pool_ids'], problem['pool_ids'])
        spec['boxes'][0]['center_m'][0] = 999
        self.assertEqual(wrapped['context']['scene_occlusion']['boxes'][0]['center_m'][0], 2.)
        wrapped['geometry'].create_environment_mesh()
        np.testing.assert_array_equal(old_vertices, problem['geometry'].create_environment_mesh().vertices)
        clear = with_occlusion(problem, dict(schema=SCHEMA, boxes=[]))
        np.testing.assert_array_equal(old_vertices, clear['geometry'].create_environment_mesh().vertices)
        with self.assertRaisesRegex(ValueError, 'already attached'):
            with_occlusion(wrapped, dict(schema=SCHEMA, boxes=[]))

    def test_actual_blocked_ray_and_rotated_box(self):
        problem = self.problem()
        clear = MeshVisibility(problem['geometry'].create_environment_mesh(), problem['targets'], problem['profile'])
        blocked = with_occlusion(problem, self.spec())
        observed = MeshVisibility(blocked['geometry'].create_environment_mesh(), problem['targets'], problem['profile'])
        self.assertTrue(clear.mask(problem['poses'][0])[0])
        self.assertFalse(observed.mask(problem['poses'][0])[0])
        rotated = self.spec()
        rotated['boxes'][0].update(center_m=[0, 2, 2.25], yaw_deg=90.)
        problem['targets'] = np.array([[0, 4, 1.5]])
        geometry = with_occlusion(problem, rotated)['geometry']
        provider = MeshVisibility(geometry.create_environment_mesh(), problem['targets'], problem['profile'])
        self.assertFalse(provider.mask(problem['poses'][0])[0])

    def test_paired_exact_evaluations_and_tamper_guard(self):
        problem = self.problem()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            for condition, spec in [('clear', dict(schema=SCHEMA, boxes=[])), ('occluded', self.spec())]:
                result = evaluate_matched_controls(with_occlusion(problem, spec), [0, 1])
                path = root / 'runs' / condition
                path.mkdir(parents=True)
                np.savez(path / 'target.npz', **result['archive'])
                (path / 'report.json').write_text(json.dumps(result['report']))
                cases.append(dict(name=condition, pair='toy', condition=condition))
            validation = study.verify_pairs(root, cases)
            self.assertTrue(validation['passed'])
            self.assertTrue(validation['pairs'][0]['effective_occlusion'])
            path = root / 'runs/occluded/report.json'
            changed = json.loads(path.read_text())
            changed['arms'][0]['physical_query_ids'] = [3, 2]
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'query plans differ'):
                study.verify_pairs(root, cases)

    def test_suite_dry_plan_no_mesh_no_ranking_no_writes(self):
        spec = json.loads((study.BASE / 'configs/library_occlusion_20261122.json').read_text())
        problem = self.problem()
        problem['geometry'].junction_x_m = 50
        problem['geometry'].junction_y_m = 50
        problem['geometry'].scenario = SimpleNamespace(road_width=10)
        problem['geometry'].length_m = 160
        problem['geometry'].road_width_m = 10
        problem['geometry'].centerline = lambda s: (np.column_stack([s, np.zeros_like(s)]), np.zeros_like(s))
        problem['geometry'].create_environment_mesh = lambda: self.fail('dry run created mesh')

        def prepare(track, *args):
            return dict(problem, context=dict(track=track))

        with patch.object(target_run, 'prepare', side_effect=prepare), \
                patch('src.core.library_control_evaluation.plan_control', side_effect=AssertionError), \
                patch.object(Path, 'write_text', side_effect=AssertionError), \
                patch.object(Path, 'write_bytes', side_effect=AssertionError):
            _, cases = study.prepare_suite(spec)
        self.assertEqual(len(cases), 8)
        self.assertTrue(all(case['plan']['expected_arm_count'] == 34 for case in cases))
        spec['screens']['cost'] = 1
        with patch.object(target_run, 'prepare', side_effect=prepare), self.assertRaises(ValueError):
            study.prepare_suite(spec)

    def test_frozen_sequential_worker_integration(self):
        from test_library_component_transfer import ComponentTransferTests

        fixture = ComponentTransferTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target.json'
            target.write_text(json.dumps(fixture.target))
            spec = json.loads((study.BASE / 'configs/library_occlusion_20261122.json').read_text())
            spec.update(sources=dict(A=str(fixture.root), B=str(fixture.root)), k_values=[1],
                        control_seeds=[0, 1], targets=[dict(name='tiny', track='B', target_json=str(target))])
            before = fixture.snapshot()
            with redirect_stdout(io.StringIO()):
                output = study.execute_suite(spec, root / 'run', ['test-only-synthetic-source'])
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['completed'], ['tiny_clear', 'tiny_occluded'])
            self.assertTrue(manifest['original_sources_unchanged'])
            self.assertEqual(before, fixture.snapshot())
            self.assertTrue(json.loads((output / 'pair_validation.json').read_text())['passed'])
            for condition in ('clear', 'occluded'):
                run = output / 'runs' / f'tiny_{condition}'
                child = json.loads((run / 'manifest.json').read_text())
                self.assertIn('inputs/occlusion.json', child['input_hashes'])
                report = json.loads((run / 'report.json').read_text())
                self.assertEqual(len(report['comparison']), 6)
            with self.assertRaises(FileExistsError):
                study.execute_suite(spec, output, [])

    def test_frozen_sweep_worker_integration(self):
        from test_library_component_transfer import ComponentTransferTests

        fixture = ComponentTransferTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = json.loads((study.BASE / 'configs/library_occlusion_large_20261122.json').read_text())
            spec.update(sources=dict(A=str(fixture.root), B=str(fixture.root)), k_values=[1],
                        control_seeds=[0, 1])
            # Exercise the sweep worker path with one real B target; the complete
            # A/B Cartesian expansion is checked separately without target physics.
            rows = [dict(name='tiny', track='B', target=fixture.target)]
            before = fixture.snapshot()
            with patch.object(study, 'expand_design', return_value=rows), redirect_stdout(io.StringIO()):
                output = study.execute_suite(spec, root / 'sweep', ['synthetic-sweep-integration'])
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['schema'], study.SWEEP_SCHEMA)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['expected_comparison_rows'], 12)
            self.assertLessEqual(manifest['batch_timeout_sec'], 7200)
            self.assertEqual(before, fixture.snapshot())
            cases = json.loads((output / 'inputs/cases.json').read_text())
            self.assertTrue(all(case['target_path'] is None for case in cases))


if __name__ == '__main__':
    unittest.main()

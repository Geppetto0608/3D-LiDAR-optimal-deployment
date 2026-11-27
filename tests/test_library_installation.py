"""Unary constraints, paired archive invariants and frozen four-cell execution."""
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
from src.core.library_installation import SCHEMA, allowed_candidates, validate_result, validate_spec, with_installation
from src.core.scene_occlusion import SCHEMA as OCCLUSION_SCHEMA, with_occlusion


class InstallationTests(unittest.TestCase):
    def problem(self):
        from test_library_occlusion_study import OcclusionTests
        problem = OcclusionTests().problem()
        problem.update(pool_ids=[0, 1, 2], transfers=[
            dict(status='complete', selected_ids=[0, 1], num_selected=2, budget_k=2,
                 projection=[dict(target_id=1)]),
            dict(status='complete', selected_ids=[0, 2], num_selected=2, budget_k=2),
            dict(status='unavailable', budget_k=1, reason='old_unavailable')],
            proposals=[dict(selected_ids=[0, 1], budget_k=2, origin='invalid'),
                       dict(selected_ids=[0, 2], budget_k=2, origin='survives')])
        return problem

    def spec(self, bounds=None):
        return dict(schema=SCHEMA, forbidden_regions=[[-1.1, -.1, -.9, .1]] if bounds is None else bounds)

    def test_strict_schema_and_inclusive_all_orientations(self):
        bad = [None, {}, dict(self.spec(), cost=1), dict(self.spec(), schema='old'),
               dict(schema=SCHEMA, forbidden_regions={}), self.spec([[0, 0, 0, 1]]),
               self.spec([[0, False, 1, 1]]), self.spec([[0, 0, float('nan'), 1]]),
               self.spec([[0, 0, '1', 1]]), self.spec([[0, 0, 1]]), self.spec([[0, 0, 1, 1]] * 101)]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_spec(spec)
        poses = np.array([[0, 0, 3, 0, -10], [1, 1, 6, 180, -25], [1.01, 1, 3, 0, -10]])
        np.testing.assert_array_equal(allowed_candidates(poses, self.spec([[0, 0, 1, 1]])), [False, False, True])

    def test_filter_remap_unavailable_layout_and_surviving_pool(self):
        problem, spec = self.problem(), self.spec()
        before = deepcopy({k: problem[k] for k in ('context', 'transfers', 'proposals', 'pool_ids')})
        poses = problem['poses'].copy()
        changed = with_installation(problem, spec)
        self.assertIs(changed['geometry'], problem['geometry'])
        self.assertIs(changed['targets'], problem['targets'])
        np.testing.assert_array_equal(changed['poses'], poses[[0, 2, 3]])
        self.assertEqual(changed['pool_ids'], [0, 1])
        self.assertEqual(changed['transfers'][0]['status'], 'unavailable')
        self.assertNotIn('selected_ids', changed['transfers'][0])
        self.assertEqual(changed['transfers'][0]['excluded_original_ids'], [1])
        self.assertEqual(changed['transfers'][0]['original_transfer']['projection'], [dict(target_id=1)])
        self.assertEqual(changed['transfers'][1]['selected_ids'], [0, 1])
        self.assertEqual(changed['proposals'], [dict(selected_ids=[0, 1], budget_k=2, origin='survives')])
        self.assertEqual(before, {k: problem[k] for k in before})
        np.testing.assert_array_equal(problem['poses'], poses)
        spec['forbidden_regions'].clear()
        self.assertEqual(len(changed['context']['installation']['spec']['forbidden_regions']), 1)
        with self.assertRaisesRegex(ValueError, 'already attached'):
            with_installation(changed, self.spec())

    def test_empty_exclusions_preserve_domain_and_pool(self):
        problem = self.problem()
        changed = with_installation(problem, self.spec([]))
        np.testing.assert_array_equal(changed['poses'], problem['poses'])
        self.assertEqual(changed['pool_ids'], problem['pool_ids'])
        self.assertEqual(changed['context']['installation']['eligible_to_original_ids'], [0, 1, 2, 3])

    def test_empty_library_pool_is_retained_and_evaluated(self):
        problem = self.problem()
        spec = self.spec([[-2, -1, 1, 5]])
        changed = with_installation(problem, spec)
        self.assertEqual(changed['pool_ids'], [])
        self.assertTrue(changed['context']['installation']['empty_library_pool'])
        result = evaluate_matched_controls(changed, [0, 1])
        validate_result(result, spec)
        self.assertEqual(result['report']['plan']['query_budget'], 0)
        self.assertEqual(len(result['report']['comparison']), 12)
        self.assertTrue(all(row['covered_count'] == 0 for row in result['report']['comparison']))
        self.assertGreater(result['report']['full_frontier']['frontier'][0]['covered_count'], 0)
        np.testing.assert_array_equal(result['archive']['original_candidate_ids'], [3])

    def test_all_full_candidates_excluded_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Empty full candidate domain'):
            with_installation(self.problem(), self.spec([[-100, -100, 100, 100]]))

    def test_geometry_recipe_rejects_bends_and_unknown_fields(self):
        from src.environment import ConnectedCorridorGeometry
        problem = self.problem()
        problem['geometry'] = ConnectedCorridorGeometry(length_m=120, road_width_m=8, turn_deg=30, bend_length_m=32)
        recipe = dict(band_length_m=20, a_east_station_m=22, b_station_fraction=.25)
        self.assertEqual(study.installation_spec(problem, recipe)['forbidden_regions'], [[20., -9., 40., 9.]])
        for bad in (dict(recipe, b_station_fraction=.5), dict(recipe, band_length_m=True),
                    dict(recipe, cost=1), dict(recipe, b_station_fraction=1)):
            with self.subTest(recipe=bad), self.assertRaises(ValueError):
                study.installation_spec(problem, bad)

    def test_large_plan_no_physics_selection_or_writes(self):
        spec = json.loads((study.BASE / 'configs/library_siting_large_20261123.json').read_text())
        previous = json.loads((study.BASE / 'configs/library_occlusion_large_20261122.json').read_text())
        for key in ('sources', 'design', 'k_values', 'control_seeds', 'screens', 'enumeration_limit'):
            self.assertEqual(spec[key], previous[key])
        problem = self.problem()
        problem['k_values'] = [1, 2, 3]
        geometry = SimpleNamespace(junction_x_m=50, junction_y_m=50, map_len=100,
            scenario=SimpleNamespace(road_width=10), sidewalk_width=5, sidewalk_width_m=5,
            length_m=120, road_width_m=10, bend_length_m=32,
            centerline=lambda s: (np.column_stack([s, np.zeros_like(s)]), np.zeros_like(s)),
            create_environment_mesh=lambda: self.fail('dry plan created mesh'))
        problem['geometry'] = geometry
        def prepare(track, *args):
            return dict(problem, context=dict(track=track))
        with patch.object(target_run, 'prepare', side_effect=prepare), \
                patch('src.core.library_control_evaluation.plan_control', side_effect=AssertionError), \
                patch.object(Path, 'write_text', side_effect=AssertionError), \
                patch.object(Path, 'write_bytes', side_effect=AssertionError):
            _, cases = study.prepare_suite(spec)
        self.assertEqual(len(cases), 224)
        self.assertEqual(len({c['name'] for c in cases}), 224)
        self.assertEqual(len({c['pair'] for c in cases}), 112)
        self.assertEqual(sum(c['plan']['expected_arm_count'] * len(c['plan']['k_values']) for c in cases), 22848)
        for invalid in (dict(spec, cost=1), dict(spec, schema=study.SWEEP_SCHEMA)):
            with self.assertRaises(ValueError):
                study.prepare_suite(invalid)

    def test_paired_physics_and_tamper_checks(self):
        from test_library_occlusion_study import OcclusionTests
        cases = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for level, installation in [('open', self.spec([])), ('restricted', self.spec())]:
                for condition, occlusion in [('clear', dict(schema=OCCLUSION_SCHEMA, boxes=[])),
                                              ('occluded', OcclusionTests().spec())]:
                    name = f'toy_{level}_{condition}'
                    result = evaluate_matched_controls(with_occlusion(
                        with_installation(self.problem(), installation), occlusion), [0, 1])
                    validate_result(result, installation)
                    path = root / 'runs' / name
                    path.mkdir(parents=True)
                    np.savez(path / 'target.npz', **result['archive'])
                    np.savez(path / 'mesh.npz', **result['mesh'])
                    (path / 'report.json').write_text(json.dumps(result['report']))
                    cases.append(dict(name=name, pair=f'toy_{level}', base_map='toy',
                        condition=condition, installation_level=level, installation=installation))
            self.assertTrue(study.verify_pairs(root, cases)['passed'])
            check = study.verify_installation_pairs(root, cases)
            self.assertTrue(check['passed'])
            self.assertEqual(check['pairs'][0]['excluded_candidates'], 1)
            self.assertEqual(check['pairs'][0]['query_budget_before'], 3)
            self.assertEqual(check['pairs'][0]['query_budget_after'], 2)
            path = root / 'runs/toy_restricted_clear/target.npz'
            with np.load(path) as stored:
                original = {key: stored[key].copy() for key in stored.files}
            for field in ('original_candidate_ids', 'masks', 'pool_ids', 'targets'):
                changed = deepcopy(original)
                changed[field].flat[0] = not changed[field].flat[0] if field == 'masks' else 99
                np.savez(path, **changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    study.verify_installation_pairs(root, cases)
                np.savez(path, **original)

    def test_frozen_four_case_worker_and_input_hashes(self):
        from test_library_component_transfer import ComponentTransferTests
        fixture = ComponentTransferTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = json.loads((study.BASE / 'configs/library_siting_large_20261123.json').read_text())
            spec.update(sources=dict(A=str(fixture.root), B=str(fixture.root)), k_values=[1], control_seeds=[0, 1])
            spec['installation_exclusion'].update(band_length_m=10, b_station_fraction=.1875)
            rows = [dict(name='tiny', track='B', target=fixture.target)]
            before = fixture.snapshot()
            with patch.object(study, 'expand_design', return_value=rows), redirect_stdout(io.StringIO()):
                output = study.execute_suite(spec, root / 'new_parent/run', ['test-only-synthetic-source'])
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['expected_comparison_rows'], 24)
            self.assertEqual(len(manifest['completed']), 4)
            self.assertEqual(before, fixture.snapshot())
            self.assertTrue(manifest['original_sources_unchanged'])
            for name in ('pair_validation.json', 'installation_validation.json'):
                self.assertTrue(json.loads((output / name).read_text())['passed'])
            validation = json.loads((output / 'installation_validation.json').read_text())
            self.assertGreater(validation['pairs'][0]['excluded_candidates'], 0)
            self.assertIn('installation_validation_sha256', manifest)
            for name in manifest['completed']:
                child = json.loads((output / 'runs' / name / 'manifest.json').read_text())
                self.assertIn('inputs/installation.json', child['input_hashes'])
            with self.assertRaises(FileExistsError):
                study.execute_suite(spec, output, [])


if __name__ == '__main__':
    unittest.main()

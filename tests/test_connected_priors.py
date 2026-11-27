"""Prior-only ablations, screened-rule parity and frozen real-raycast integration."""
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
import connected_prior_ablation as runner
import test_connected_hierarchy as existing
from src.methods.connected_hierarchy import (BudgetedObservations, optical_predictions,
    predicted_partner, probe_prior_values, select_pool)
from validate_connected_priors import audit, check_upper_metadata, optical_prior


class PriorTests(unittest.TestCase):
    def fixture(self, constraints=None):
        return existing.HierarchyTests().fixture(constraints)

    def test_legacy_source_and_explicit_source_choices_are_identical(self):
        context, masks = self.fixture()
        old = BudgetedObservations(context, lambda i: masks[i], per_site=4)
        new = BudgetedObservations(context, lambda i: masks[i], per_site=4)
        a = select_pool('probe_context', context, old, 4)
        b = select_pool('probe_context', context, new, 4, probe_prior='source', fast_screen=True)
        for field in ('candidate_ids', 'events', 'contexts'):
            self.assertEqual(a[field], b[field])
        for x, y in zip(a['decisions'], b['decisions']):
            x = {('prior_fallback' if k == 'source_fallback' else k): v for k, v in x.items()}
            self.assertEqual(x, y)

    def test_geometry_and_random_never_read_source_in_seed_tie_or_fallback(self):
        class Poison:
            def __array__(self, *args, **kwargs):
                raise AssertionError('Source prior read by a source-free arm')
        context, masks = self.fixture(dict(min_coverage_pct=100))
        context.source_prior = Poison()
        masks[:] = False
        predicted = np.zeros_like(masks)
        for mode, seed in [('geometry', None), ('random', 7)]:
            obs = BudgetedObservations(context, lambda i: masks[i], per_site=3)
            with patch('src.methods.connected_hierarchy.optical_predictions', return_value=predicted):
                result = select_pool('probe_context', context, obs, 3, probe_prior=mode, random_seed=seed, fast_screen=True)
            prior = np.asarray(result['prior_values'])
            self.assertTrue(all(d['prior_fallback'] for d in result['decisions']))
            for group in context.groups:
                expected = sorted(group, key=lambda i: (-prior[i], i))[:3]
                actual = [e['candidate_id'] for e in result['events'] if e['candidate_id'] in group]
                self.assertEqual(actual, expected)

    def test_geometry_prior_matches_independent_tilted_plane_objective(self):
        context, _ = self.fixture()
        predicted = optical_predictions(context)
        actual = probe_prior_values('geometry', context, predicted)
        independent = optical_prior(context.poses, context.targets, context.sensor,
            context.allowed, context.requirements.spec)
        np.testing.assert_allclose(actual, independent, atol=1e-8, rtol=0)

    def test_screened_predictions_match_unscreened_full_constraints(self):
        constraints = dict(max_installation_cost=28, min_xy_separation_m=3,
            min_coverage_pct=60, max_overlap_ratio=.8, min_single_failure_coverage_pct=25)
        context, masks = self.fixture(constraints)
        rng = np.random.default_rng(29)
        obs = BudgetedObservations(context, lambda i: masks[i], initial={i: masks[i] for i in range(len(masks))})
        for thresholds in ({}, dict(min_coverage_pct=95), dict(min_overlap_ratio=.5), dict(max_overlap_ratio=.2)):
            context.requirements.profile['interface'] = thresholds
            predicted = rng.random(masks.shape) < .6
            for cid in range(len(masks)):
                a = predicted_partner(cid, list(range(len(masks))), predicted, obs, context)
                b = predicted_partner(cid, list(range(len(masks))), predicted, obs, context, fast_screen=True)
                self.assertEqual(a, b)

    def test_random_prior_determinism_diversity_and_budget_prefixes(self):
        context, masks = self.fixture()
        first_choices = []
        for mode, seed in [('source', None), ('geometry', None)] + [('random', s) for s in range(5)]:
            previous = []
            for b in (1, 2, 4):
                obs = BudgetedObservations(context, lambda i: masks[i], per_site=b)
                result = select_pool('probe_context', context, obs, b, probe_prior=mode, random_seed=seed, fast_screen=True)
                events = [e['candidate_id'] for e in result['events']]
                self.assertEqual(events[:len(previous)], previous)
                self.assertEqual(len(events), b*len(context.groups))
                previous = events
                if mode == 'random' and b == 1:
                    first_choices.append(tuple(events))
            if mode == 'random':
                a = probe_prior_values(mode, context, optical_predictions(context), random_seed=seed)
                b = probe_prior_values(mode, context, optical_predictions(context), random_seed=seed)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(len(np.unique(a)), len(a))
        self.assertEqual(len(set(first_choices)), 5)

    def test_invalid_prior_options_fail_closed(self):
        context, masks = self.fixture()
        p = optical_predictions(context)
        for mode, seed in [('missing', None), ('random', None), ('random', True), ('random', -1),
                           ('random', 2**32), ('source', 0), ('geometry', 0)]:
            with self.subTest(mode=mode, seed=seed), self.assertRaises(ValueError):
                probe_prior_values(mode, context, p, random_seed=seed)
        for budgets, seeds in [([2, 1], [0]), ([1, 1], [0]), ([3], [0]), ([True], [0]),
                               ([1], []), ([1], [0, 0]), ([1], [True]), ([1], list(range(9)))]:
            with self.assertRaises(ValueError):
                runner.validate_options(dict(schema='connected_prior_ablation_v1', query_budgets=budgets, random_seeds=seeds))
        with self.assertRaises(ValueError):
            select_pool('geometry_single', context, BudgetedObservations(context, lambda i: masks[i], per_site=1),
                1, probe_prior='geometry')

    def test_plan_is_read_only_and_has_no_lower_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'not_created'
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Plan cast rays')):
                with redirect_stdout(io.StringIO()) as text:
                    runner.main(['--output', str(root)])
            plan = json.loads(text.getvalue())
            self.assertEqual(plan['conditions'], 24)
            self.assertEqual(plan['online_outputs'], 504)
            self.assertEqual(plan['references'], 24)
            self.assertEqual(len(plan['arms']), 21)
            self.assertFalse(plan['lower_refinement'])
            self.assertFalse(root.exists())
            with self.assertRaises(ValueError):
                runner.main(['--worker-root', str(root)])

    def test_pair_certificate_checks_status_domain_and_counts(self):
        result = dict(selected_ids=[0, 2], exhaustive=True, fixed_sensor_count=2, candidate_count=4,
            combinations_considered=6, quota='none', allocation_mode='free', status='optimal_in_declared_grid',
            full_constraint_assessments=2, visibility_screened_pairs=0, elapsed_sec=.01)
        check_upper_metadata(result, 4)
        for key, bad in [('status', 'continuous_optimum'), ('candidate_count', 3), ('combinations_considered', 5),
                         ('quota', 'one_per_module'), ('fixed_sensor_count', 3), ('full_constraint_assessments', -1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_upper_metadata({**result, key: bad}, 4)

    def test_frozen_real_smoke_audits_failures_priors_statistics_and_no_overwrite(self):
        base = deepcopy(corridor.load_suite(ROOT/'configs/connected_corridor_pilot.json'))
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
            (tmp/'base.json').write_text(json.dumps(base))
            (tmp/'study.json').write_text(json.dumps(study))
            root = tmp/'run'
            cmd = [sys.executable, str(ROOT/'connected_prior_ablation.py'), '--base-suite', str(tmp/'base.json'),
                '--study', str(tmp/'study.json'), '--query-budgets', '1', '2', '--random-seeds', '0', '1',
                '--output', str(root), '--execute']
            run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
            logs = ''.join(p.read_text() for p in (root/'worker.log', root/'audit.log') if p.exists())
            self.assertEqual(run.returncode, 0, run.stderr+logs)
            v = json.loads((root/'independent_validation.json').read_text())
            self.assertEqual(v['online_rows'], 16)
            self.assertEqual(v['reference_optima_checked'], 2)
            self.assertEqual(v['selection_policy_replays'], 16)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['groups']['b1_random']['trials'], 4)
            blocked = json.loads((root/'case_01_00/b1_geometry/result.json').read_text())
            self.assertIsNone(blocked['assessment'])
            self.assertEqual(blocked['work']['unique_candidate_queries'], 0)
            manifest_path = root/'manifest.json'
            original_manifest = manifest_path.read_text()
            for field in ('source_hashes', 'input_hashes'):
                manifest = json.loads(original_manifest)
                manifest[field].pop(next(iter(manifest[field])))
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    audit(root)
                manifest_path.write_text(original_manifest)
            manifest = json.loads(original_manifest)
            manifest['plan']['online_outputs'] += 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Plan counts'):
                audit(root)
            manifest_path.write_text(original_manifest)
            cm_path = root/'case_00_00/manifest.json'
            original_cm = cm_path.read_text()
            mesh_path = root/'case_00_00/geometry.npz'
            original_mesh = mesh_path.read_bytes()
            with np.load(mesh_path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            arrays['vertices'][0, 2] += .5
            np.savez_compressed(mesh_path, **arrays)
            cm = json.loads(original_cm)
            cm['geometry_sha256'] = corridor.digest(mesh_path)
            cm_path.write_text(json.dumps(cm))
            with self.assertRaisesRegex(ValueError, 'Geometry array: vertices'):
                audit(root)
            mesh_path.write_bytes(original_mesh)
            cm_path.write_text(original_cm)
            result_path = root/'case_00_00/b1_source/result.json'
            original_result = result_path.read_text()
            result = json.loads(original_result)
            result['status'] = 'continuous_optimum'
            result_path.write_text(json.dumps(result))
            cm = json.loads(original_cm)
            cm['online_hashes']['b1_source/result.json'] = corridor.digest(result_path)
            cm_path.write_text(json.dumps(cm))
            with self.assertRaisesRegex(ValueError, 'Pair-domain status'):
                audit(root)
            result_path.write_text(original_result)
            cm_path.write_text(original_cm)
            csv_path = root/'comparison.csv'
            original_csv = csv_path.read_text()
            for field in ('profile', 'target'):
                with csv_path.open(newline='') as f:
                    rows = list(csv.DictReader(f))
                rows[0][field] = 'wrong_condition'
                with csv_path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, f'CSV {field}'):
                    audit(root)
                csv_path.write_text(original_csv)
            summary_path = root/'summary.json'
            original_summary = summary_path.read_text()
            summary['groups']['b1_random']['trials'] += 1
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'Summary'):
                audit(root)
            summary_path.write_text(original_summary)
            # A rewritten hash is not sufficient to fake a source-free geometry prior.
            path = root/'case_00_00/b1_geometry/selection.json'
            selection = json.loads(path.read_text())
            selection['prior_values'][0] += 1
            path.write_text(json.dumps(selection))
            cm_path = root/'case_00_00/manifest.json'
            cm = json.loads(cm_path.read_text())
            cm['online_hashes']['b1_geometry/selection.json'] = corridor.digest(path)
            result_path = root/'case_00_00/b1_geometry/result.json'
            result = json.loads(result_path.read_text())
            result['selection_sha256'] = corridor.digest(path)
            result_path.write_text(json.dumps(result))
            cm['online_hashes']['b1_geometry/result.json'] = corridor.digest(result_path)
            cm_path.write_text(json.dumps(cm))
            with self.assertRaisesRegex(ValueError, 'Independent prior values'):
                audit(root)
            rejected = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

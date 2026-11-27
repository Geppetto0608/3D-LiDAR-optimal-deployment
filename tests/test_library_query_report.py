"""Synthetic stored-data tests: no physics, planners, training, or raw writes."""
import copy
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import library_query_report as report
from src.core.deployment_library import build_frontier


class QueryReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'raw' / 'a_w9'
        self.output = self.base / 'summary'
        self.arrays = dict(poses=np.arange(25, dtype=float).reshape(5, 5),
                           targets=np.arange(24, dtype=float).reshape(8, 3),
                           masks=np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 1, 0, 0, 0],
                                           [1, 1, 0, 1, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 1, 1],
                                           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool),
                           pool_ids=np.array([0, 1], dtype=np.int64))
        self.data = self.make_report([0, 1, 2])
        self.write_case(self.root, self.data)

    def frontier(self, ids):
        result = build_frontier(self.arrays['poses'][ids], self.arrays['masks'][ids],
                                self.arrays['targets'], [1, 2, 3], alternatives_per_k=1)
        for row in result['frontier']:
            for layout in [row, *row['alternatives']]:
                layout['full_candidate_ids'] = [ids[i] for i in layout['selected_ids']]
                layout['validated_direct_raycast'] = True
        return result

    def make_report(self, seeds):
        full = self.frontier(list(range(5)))
        domains = dict(library=[0, 1], geometry_single=[0, 2], geometry_union=[2, 3],
                       geometry_portfolio=[1, 3])
        arms = []
        for name, seed, ids in [(name, None, ids) for name, ids in domains.items()] + [
                (f'random_{seed}', seed, [[0, 4], [2, 3], [1, 4]][index % 3])
                for index, seed in enumerate(seeds)]:
            arms.append(dict(arm_id=name, method='random' if seed is not None else name, seed=seed,
                             query_plan=dict(selected_ids=ids, query_order=list(reversed(ids))),
                             physical_query_ids=list(reversed(ids)), frontier=self.frontier(ids),
                             work=dict(direct_pose_evaluations=2), phase_wall_sec=dict(visibility=0.1)))
        result = dict(schema=report.SCHEMA, plan=dict(track='A', context={}, candidate_count=5,
                      pool_candidate_count=2, target_count=8, k_values=[1, 2, 3], control_seeds=seeds,
                      expected_arm_count=len(arms), query_budget=2), arms=arms, full_frontier=full,
                      validation=dict(passed=True, saved_solution_records=6 * (1 + len(arms))))
        result['comparison'] = self.comparison_rows(result)
        return result

    @staticmethod
    def comparison_rows(data):
        return [dict(arm_id=arm['arm_id'], method=arm['method'], seed=arm['seed'],
                     budget_k=row['budget_k'], query_count=data['plan']['query_budget'],
                     covered_count=row['covered_count'], coverage_pct=row['coverage_pct'],
                     reference_covered_count=full['covered_count'], reference_coverage_pct=full['coverage_pct'],
                     full_shortfall_count=full['covered_count'] - row['covered_count'])
                for arm in data['arms']
                for row, full in zip(arm['frontier']['frontier'], data['full_frontier']['frontier'])]

    def refresh_hashes(self, root):
        manifest = dict(schema='library_target_run_v1', status='complete', artifact_hashes={
            name: sha256((root / name).read_bytes()).hexdigest()
            for name in ('report.json', 'comparison.csv', 'target.npz', 'mesh.npz')})
        for field, prefix, names in [
                ('input_hashes', '', ['inputs/target.json', 'inputs/settings.json', 'inputs/protocol.md']),
                ('source_hashes', 'source', ['frozen.py']),
                ('library_hashes', 'inputs/library', ['library.json'])]:
            manifest[field] = {name: sha256((root / prefix / name).read_bytes()).hexdigest()
                               for name in names if (root / prefix / name).exists()}
        (root / 'manifest.json').write_text(json.dumps(manifest))

    def write_case(self, root, data, arrays=None):
        root.mkdir(parents=True, exist_ok=True)
        for name, content in [('inputs/target.json', '{}'), ('inputs/settings.json', '{}'),
                              ('inputs/protocol.md', 'Frozen protocol'), ('source/frozen.py', '# Frozen source'),
                              ('inputs/library/library.json', '{}')]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (root / 'report.json').write_text(json.dumps(data))
        np.savez_compressed(root / 'target.npz', **(self.arrays if arrays is None else arrays))
        np.savez_compressed(root / 'mesh.npz', vertices=np.zeros((0, 3)))
        with (root / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=report.RAW_FIELDS)
            writer.writeheader()
            writer.writerows(data['comparison'])
        self.refresh_hashes(root)

    def mutate(self, change):
        data = copy.deepcopy(self.data)
        change(data)
        (self.root / 'report.json').write_text(json.dumps(data))
        self.refresh_hashes(self.root)

    def assert_rejected(self):
        with self.assertRaises(ValueError):
            report.aggregate_reports([self.root], self.output)
        self.assertFalse(self.output.exists())

    def test_aggregates_recomputed_values_and_provenance(self):
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch('src.core.deployment_library.build_frontier', side_effect=AssertionError('No solve')):
            self.assertEqual(report.aggregate_reports([self.root], self.output), self.output)
        self.assertEqual(before, {str(p.relative_to(self.root)): p.read_bytes()
                                  for p in self.root.rglob('*') if p.is_file()})
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['reproduction']['status'], 'not_provided')
        self.assertEqual(summary['condition_count'], 3)
        self.assertFalse(summary['validation']['fresh_physics_performed'])
        self.assertEqual(summary['source_sha256'], report._digest(Path(report.__file__)))
        self.assertEqual(summary['material_passport']['OriginSkill'], 'academic-research-suite')
        self.assertEqual(summary['validation']['frozen_hash_checks'], 'passed')
        for field in ('input_hashes', 'source_hashes', 'library_hashes'):
            self.assertEqual(summary['sources'][0][field],
                             json.loads((self.root / 'manifest.json').read_text())[field])
        matches = summary['full_reference_matches']
        for method in (*report.DETERMINISTIC, 'random'):
            expected = [row for row in self.data['comparison'] if row['method'] == method]
            self.assertEqual(matches[method]['row_count'], len(expected))
            self.assertEqual(matches[method]['match_count'],
                             sum(row['covered_count'] == row['reference_covered_count'] for row in expected))
            self.assertEqual(sum(row['match_count'] for row in matches[method]['by_k'].values()),
                             matches[method]['match_count'])
        self.assertEqual(sum(row['match_count'] for row in matches['random']['by_seed'].values()),
                         matches['random']['match_count'])
        with (self.output / 'condition_summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            k = int(row['budget_k'])
            values = [arm['frontier']['frontier'][k - 1]['coverage_pct']
                      for arm in self.data['arms'] if arm['method'] == 'random']
            self.assertAlmostEqual(float(row['random_mean_pct']), np.mean(values))
            self.assertAlmostEqual(float(row['random_sample_sd_pct']), np.std(values, ddof=1))
            for name, expected in zip(('p10', 'median', 'p90'), np.quantile(values, [.1, .5, .9], method='linear')):
                self.assertAlmostEqual(float(row[f'random_{name}_pct']), expected)
            self.assertEqual(sum(int(row[f'library_vs_random_{name}']) for name in ('wins', 'ties', 'losses')), 3)
        with (self.output / 'raw_comparison.csv').open() as stream:
            raw = list(csv.DictReader(stream))
        self.assertEqual(len(raw), 21)
        self.assertEqual(sum(row['arm_id'] == 'library' for row in raw), 3)
        self.assertEqual(sum(row['method'] == 'random' for row in raw), 9)
        self.assertIn('not a generalization evaluation', (self.output / 'REPORT.md').read_text())

    def test_equal_map_weight_not_seed_weight(self):
        other = self.base / 'raw' / 'b_turn45'
        second = self.make_report([0, 1, 2])
        for arm in second['arms']:
            if arm['method'] == 'random':
                arm.update(query_plan=dict(selected_ids=[0, 4], query_order=[0, 4]),
                           physical_query_ids=[0, 4], frontier=self.frontier([0, 4]))
        second['comparison'] = self.comparison_rows(second)
        self.write_case(other, second)
        report.aggregate_reports([other, self.root], self.output)
        summary = json.loads((self.output / 'summary.json').read_text())
        for k in report.KS:
            first_values = [a['frontier']['frontier'][k - 1]['coverage_pct'] for a in self.data['arms']
                            if a['method'] == 'random']
            second_value = second['arms'][-1]['frontier']['frontier'][k - 1]['coverage_pct']
            self.assertAlmostEqual(summary['by_k'][str(k)]['random_mean_pct'],
                                   (np.mean(first_values) + second_value) / 2)
        self.assertEqual(summary['case_count'], 2)

    def test_single_seed_sd_is_undefined(self):
        self.write_case(self.root, self.make_report([42]))
        report.aggregate_reports([self.root], self.output)
        with (self.output / 'condition_summary.csv').open() as stream:
            single = list(csv.DictReader(stream))
        self.assertTrue(all(row['random_sample_sd_pct'] == '' for row in single))

    def test_mismatched_case_seeds_fail_closed(self):
        other = self.base / 'raw' / 'b_turn45'
        self.write_case(other, self.make_report([0, 1, 3]))
        with self.assertRaisesRegex(ValueError, 'same control seeds'):
            report.aggregate_reports([self.root, other], self.output)
        self.assertFalse(self.output.exists())

    def test_four_cases_have_twelve_utility_conditions(self):
        data = self.make_report(list(range(30)))
        self.write_case(self.root, data)
        roots = [self.root]
        for name in ('a_w12', 'b_turn45', 'b_turn60'):
            root = self.base / 'raw' / name
            self.write_case(root, data)
            roots.append(root)
        report.aggregate_reports(roots, self.output)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertTrue(summary['complete_four_case_design'])
        self.assertEqual(summary['condition_count'], 12)
        self.assertTrue(all(sum(counts.values()) == 12 for counts in summary['library_utility'].values()))

    def test_four_case_label_requires_fixed_names_and_seeds(self):
        for seeds, names in [([0, 1, 2], sorted(report.DEVELOPMENT_CASES)),
                             (list(range(30)), ['a_w9', 'a_w12', 'b_turn45', 'other'])]:
            with self.subTest(seeds=seeds, names=names):
                roots = [self.base / 'raw' / name for name in names]
                for root in roots:
                    self.write_case(root, self.make_report(seeds))
                output = self.base / f'summary_{len(seeds)}'
                report.aggregate_reports(roots, output)
                self.assertFalse(json.loads((output / 'summary.json').read_text())['complete_four_case_design'])

    def test_completion_and_hashes_are_mandatory(self):
        for field, value in [('status', 'running'), ('schema', 'wrong'),
                             ('schema', 'oldlibrary_target_run_v1'), ('artifact_hashes', {})]:
            with self.subTest(field=field):
                self.refresh_hashes(self.root)
                manifest = json.loads((self.root / 'manifest.json').read_text())
                manifest[field] = value
                (self.root / 'manifest.json').write_text(json.dumps(manifest))
                self.assert_rejected()
        self.refresh_hashes(self.root)
        with (self.root / 'mesh.npz').open('ab') as stream:
            stream.write(b'tampered')
        self.assert_rejected()

    def test_frozen_files_are_verified_even_when_artifacts_match(self):
        for name in ('inputs/target.json', 'inputs/settings.json', 'inputs/protocol.md',
                     'source/frozen.py', 'inputs/library/library.json'):
            for action in ('change', 'remove'):
                with self.subTest(name=name, action=action):
                    self.write_case(self.root, self.data)
                    path = self.root / name
                    if action == 'change':
                        path.write_text('tampered')
                    else:
                        path.unlink()
                    self.assert_rejected()

    def test_frozen_maps_required_and_paths_confined(self):
        for field in ('input_hashes', 'source_hashes', 'library_hashes'):
            for value in (None, {}, {'../escape': '0' * 64}, {'/absolute': '0' * 64}):
                with self.subTest(field=field, value=value):
                    self.refresh_hashes(self.root)
                    manifest = json.loads((self.root / 'manifest.json').read_text())
                    if value is None:
                        manifest.pop(field)
                    else:
                        manifest[field] = value
                    (self.root / 'manifest.json').write_text(json.dumps(manifest))
                    self.assert_rejected()

    def test_frozen_snapshot_symlink_directory_rejected(self):
        source = self.root / 'source'
        moved = self.base / 'moved_source'
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
        self.assert_rejected()

    def test_protocol_is_optional_when_not_declared(self):
        (self.root / 'inputs/protocol.md').unlink()
        self.refresh_hashes(self.root)
        report.aggregate_reports([self.root], self.output)

    def test_non_boolean_masks_and_invalid_arrays(self):
        for name, value in [('masks', self.arrays['masks'].astype(np.uint8)),
                            ('pool_ids', np.array([0., 1.])),
                            ('pool_ids', np.array([0, 0])),
                            ('poses', np.zeros((5, 5))),
                            ('targets', np.zeros((8, 3)))]:
            with self.subTest(name=name):
                self.write_case(self.root, self.data, dict(self.arrays, **{name: value}))
                self.assert_rejected()

    def test_missing_duplicate_arms_and_seed_metadata(self):
        mutations = [lambda d: d['arms'].pop(),
                     lambda d: d['arms'].__setitem__(-1, copy.deepcopy(d['arms'][0])),
                     lambda d: d['plan'].__setitem__('control_seeds', [0, 0, 2]),
                     lambda d: d['arms'][-1].__setitem__('seed', 100),
                     lambda d: d['plan'].__setitem__('expected_arm_count', 99),
                     lambda d: d['validation'].__setitem__('passed', False),
                     lambda d: d['validation'].__setitem__('saved_solution_records', 1)]
        for change in mutations:
            with self.subTest(change=change):
                self.mutate(change)
                self.assert_rejected()

    def test_query_ids_and_exact_physical_order(self):
        mutations = [lambda d: d['arms'][0]['query_plan'].__setitem__('selected_ids', [0, 0]),
                     lambda d: d['arms'][0]['query_plan'].__setitem__('selected_ids', [False, 1]),
                     lambda d: d['arms'][0]['query_plan'].__setitem__('selected_ids', [0, 5]),
                     lambda d: d['arms'][0]['work'].__setitem__('direct_pose_evaluations', 3),
                     lambda d: d['plan'].__setitem__('query_budget', 3),
                     lambda d: d['arms'][0].__setitem__('physical_query_ids', [0, 1])]
        for change in mutations:
            with self.subTest(change=change):
                self.mutate(change)
                self.assert_rejected()

    def test_frontier_counts_ids_poses_and_certificates(self):
        changes = [('full_candidate_ids', [4]), ('full_candidate_ids', [0, 0]),
                   ('full_candidate_ids', [0.0]), ('covered_count', 0), ('coverage_pct', 0),
                   ('poses', [[0, 0, 0, 0, 0]]), ('num_selected', 0), ('upper_bound_count', 8),
                   ('certification', 'best_known'), ('certified_gap', .1),
                   ('bound_source', 'union_and_single_counts'), ('budget_k', 2)]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                self.mutate(lambda d: d['arms'][0]['frontier']['frontier'][0].__setitem__(field, value))
                self.assert_rejected()
        self.mutate(lambda d: d['arms'][0]['frontier']['enumeration'].__setitem__('completed', False))
        self.assert_rejected()
        self.mutate(lambda d: d['arms'][0]['frontier']['frontier'][0]['alternatives'][0].__setitem__('covered_count', 0))
        self.assert_rejected()

    def test_csv_consistency_missing_and_duplicate_rows(self):
        original = (self.root / 'comparison.csv').read_text()
        lines = original.splitlines()
        for content in ['\n'.join(lines[:-1]), original + lines[1] + '\n', original.replace('37.5', '99.9')]:
            with self.subTest(content=content[:30]):
                (self.root / 'comparison.csv').write_text(content)
                self.refresh_hashes(self.root)
                self.assert_rejected()

    def test_report_comparison_consistency_and_completeness(self):
        mutations = [lambda d: d.pop('comparison'), lambda d: d['comparison'].pop(),
                     lambda d: d['comparison'].append(d['comparison'][0]),
                     lambda d: d['comparison'][0].__setitem__('covered_count', 0),
                     lambda d: d['comparison'][0].__setitem__('coverage_pct', 99.),
                     lambda d: d['comparison'][0].__setitem__('budget_k', True),
                     lambda d: d['comparison'][0].__setitem__('seed', 0),
                     lambda d: d['comparison'][0].__setitem__('reference_covered_count', 0)]
        for change in mutations:
            with self.subTest(change=change):
                self.mutate(change)
                self.assert_rejected()

    def test_report_comparison_row_order_is_not_significant(self):
        self.mutate(lambda d: d['comparison'].reverse())
        report.aggregate_reports([self.root], self.output)

    def test_rejects_output_reuse_and_nested_raw_output(self):
        self.output.mkdir()
        sentinel = self.output / 'sentinel'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            report.aggregate_reports([self.root], self.output)
        self.assertEqual(sentinel.read_text(), 'keep')
        with self.assertRaises(ValueError):
            report.aggregate_reports([self.root], self.root / 'new')
        with self.assertRaises(ValueError):
            report.aggregate_reports([self.root, self.root], self.base / 'another')

    def test_manifest_path_escape_and_symlink(self):
        manifest = json.loads((self.root / 'manifest.json').read_text())
        manifest['artifact_hashes']['../outside'] = '0' * 64
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        self.assert_rejected()
        self.refresh_hashes(self.root)
        (self.root / 'mesh.npz').unlink()
        (self.root / 'mesh.npz').symlink_to(self.root / 'target.npz')
        self.assert_rejected()

    def make_previous(self):
        previous = self.base / 'previous'
        old_root = previous / self.root.name
        self.write_case(old_root, self.data)
        old = dict(schema='library_target_evaluation_v1', plan=self.data['plan'],
                   pool_frontier=self.data['arms'][0]['frontier'], full_frontier=self.data['full_frontier'],
                   validation=dict(passed=True))
        (old_root / 'report.json').write_text(json.dumps(old))
        self.refresh_hashes(old_root)
        return previous

    def test_previous_exact_match_is_separate_structured_result(self):
        previous = self.make_previous()
        report.aggregate_reports([self.root], self.output, previous)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['reproduction']['status'], 'passed')
        self.assertEqual(len(summary['reproduction']['cases']), 1)

    def test_previous_frozen_hashes_are_also_verified(self):
        previous = self.make_previous()
        (previous / self.root.name / 'source/frozen.py').write_text('changed old source')
        with self.assertRaisesRegex(ValueError, 'source_hashes hash mismatch'):
            report.aggregate_reports([self.root], self.output, previous)
        self.assertFalse(self.output.exists())

    def test_previous_array_and_incumbent_mismatch_fail_closed(self):
        for field in ('poses', 'targets', 'masks', 'incumbent'):
            with self.subTest(field=field):
                previous = self.make_previous()
                old_root = previous / self.root.name
                if field == 'incumbent':
                    old = json.loads((old_root / 'report.json').read_text())
                    old['pool_frontier']['frontier'][0]['full_candidate_ids'] = [4]
                    (old_root / 'report.json').write_text(json.dumps(old))
                else:
                    arrays = {name: value.copy() for name, value in self.arrays.items()}
                    if field == 'masks':
                        arrays[field][0, 0] = ~arrays[field][0, 0]
                    else:
                        arrays[field][0, 0] += .5
                    np.savez_compressed(old_root / 'target.npz', **arrays)
                self.refresh_hashes(old_root)
                with self.assertRaises(ValueError):
                    report.aggregate_reports([self.root], self.output, previous)
                self.assertFalse(self.output.exists())

    def test_cli_and_no_heavy_imports(self):
        completed = subprocess.run([sys.executable, '-B', report.__file__, '--roots', str(self.root),
                                    '--output', str(self.output)], capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), str(self.output))
        code = ('import sys; import library_query_report; '
                'assert not any(x in sys.modules for x in '
                '["open3d", "torch", "src.core.visibility", "src.core.library_control_evaluation"])')
        imported = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=20)
        self.assertEqual(imported.returncode, 0, imported.stderr)


if __name__ == '__main__':
    unittest.main()

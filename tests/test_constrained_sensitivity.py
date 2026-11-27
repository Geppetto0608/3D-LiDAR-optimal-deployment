"""Read-only planning and honest paired/conditional sensitivity aggregation."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constrained_sensitivity import collect, materialize_cases, parser, make_plan, run, summarize, trace_at, write_report


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/constrained_sensitivity.json').read_text())
        self.base = json.loads(Path('configs/coverage_constraints_example.json').read_text())

    def test_one_factor_changes_preserve_base_and_objective(self):
        original = deepcopy(self.base)
        cases = materialize_cases(self.config, self.base)
        self.assertEqual(len(cases), 9)
        self.assertEqual(self.base, original)
        self.assertEqual(cases[0]['requirements'], original)
        for case in cases:
            with self.subTest(case=case['id']):
                changed = case['requirements']
                self.assertEqual(changed['weights'], original['weights'])
                self.assertEqual(changed['cost'], original['cost'])
                self.assertEqual(changed['budget_k'], original['budget_k'])
                if case['axis'] == 'region_overlap':
                    self.assertEqual(changed['constraints'], original['constraints'])
                else:
                    self.assertEqual(changed['regions'], original['regions'])

    def test_invalid_cases_and_unsafe_ids_are_rejected(self):
        bad = [dict(id='../escape', axis='baseline', value=None),
               dict(id='x', axis='unknown', value=1),
               dict(id='x', axis='region_overlap', region='absent', value=.5),
               dict(id='x', axis='cost_budget', value=float('nan')),
               dict(id='x', axis='forbidden_half_extent', center=[1, 2], value=0),
               dict(id='x', axis='region_overlap', region='junction', value=1.1),
               dict(id='x', axis='cost_budget', value=True)]
        for case in bad:
            config = deepcopy(self.config)
            config['cases'].append(case)
            with self.subTest(case=case), self.assertRaises(ValueError):
                materialize_cases(config, self.base)
        config = deepcopy(self.config)
        config['cases'].append(deepcopy(config['cases'][0]))
        with self.assertRaises(ValueError):
            materialize_cases(config, self.base)

    def test_plan_has_all_pairs_without_source_results_or_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'absent'
            args = parser().parse_args(['--output', str(root), '--library', str(root / 'missing')])
            with patch('constrained_sensitivity.subprocess.run', side_effect=AssertionError('No process')), \
                 redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(run(args), 0)
            plan = json.loads(stream.getvalue())
            self.assertEqual(len(plan['jobs']), 27)
            self.assertFalse(root.exists())
            self.assertEqual(len({(job['case_id'], job['seed']) for job in plan['jobs']}), 27)
            self.assertEqual(plan['split'], 'development')
        for flags in (['--seeds', '0', '0'], ['--seeds', '-1'], ['--seconds', 'nan'],
                      ['--cases', 'absent'], ['--steps', '0']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                make_plan(parser().parse_args(flags))

    def test_actual_timestamps_do_not_use_late_final_improvements(self):
        trace = [dict(elapsed_sec=2., objective_score=10., coverage_pct=90.),
                 dict(elapsed_sec=4., objective_score=12., coverage_pct=89.),
                 dict(elapsed_sec=6.5, objective_score=20., coverage_pct=95.)]
        self.assertIsNone(trace_at(trace, 1.))
        self.assertEqual(trace_at(trace, 6.)['objective_score'], 12.)
        self.assertEqual(trace_at(trace, 7.)['objective_score'], 20.)
        for invalid in ([trace[1], trace[0]],
                        [trace[0], dict(elapsed_sec=3., objective_score=9., coverage_pct=95.)]):
            with self.assertRaises(ValueError):
                trace_at(invalid, 6.)

    def test_failures_are_not_zero_scores_or_extra_independent_maps(self):
        repairs = [dict(case_id='x', seed=0, repair_status='repaired', worker_wall_sec=3.,
                        feasible_pose_sha256='same'),
                   dict(case_id='x', seed=1, repair_status='no_feasible_found', worker_wall_sec=4.,
                        feasible_pose_sha256=None)]
        rows = []
        for method, score, cov in [('local_search', 10., 90.), ('mappo', 9., 89.)]:
            rows.append(dict(case_id='x', seed=0, method=method, status='complete',
                initial_pose_sha256='same', objective_score=score, coverage_pct=cov,
                optimization_sec=2., worker_wall_sec=5., repair_worker_wall_sec=3.))
            rows.append(dict(case_id='x', seed=1, method=method, status='not_run',
                initial_pose_sha256=None, objective_score=None, coverage_pct=None))
        report = summarize(repairs, rows, ['x'], [0, 1, 2])
        r = report['repair_summary'][0]
        self.assertEqual(r['requested_repeats'], 3)
        self.assertEqual(r['processed_repeats'], 2)
        self.assertEqual(r['repair_success_rate'], .5)
        for result in report['method_summary']:
            self.assertEqual(result['completed'], 1)
            self.assertIsNone(result['objective_score_sd'])
            self.assertGreater(result['objective_score_mean'], 0.)
        self.assertEqual(len(report['paired']), 1)
        self.assertEqual(report['paired'][0]['mappo_minus_ls_objective'], -1.)
        self.assertEqual(report['paired'][0]['mappo_minus_ls_coverage_pp'], -1.)
        self.assertEqual(rows[1]['objective_score'], None)
        bad = deepcopy(rows)
        bad[2]['initial_pose_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'initial'):
            summarize(repairs, bad, ['x'], [0, 1, 2])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'keep'
            marker.write_text('unchanged')
            with self.assertRaises(FileExistsError):
                run(parser().parse_args(['--execute', '--output', temp]))
            self.assertEqual(marker.read_text(), 'unchanged')

    def test_report_keeps_blocked_scores_blank(self):
        repairs = [dict(case_id='x', seed=0, repair_status='already_feasible', worker_wall_sec=3.,
                        feasible_pose_sha256='same'),
                   dict(case_id='x', seed=1, repair_status='infeasible_fixed_count', worker_wall_sec=3.,
                        feasible_pose_sha256=None)]
        rows = [dict(case_id='x', seed=seed, method=method,
                     status='complete' if seed == 0 else 'not_run',
                     initial_pose_sha256='same' if seed == 0 else None,
                     objective_score=10. if seed == 0 else None,
                     coverage_pct=90. if seed == 0 else None,
                     optimization_sec=2. if seed == 0 else None,
                     worker_wall_sec=5. if seed == 0 else None)
                for seed in (0, 1) for method in ('local_search', 'mappo')]
        manifest = dict(status='complete', cases=[dict(id='x')], seeds=[0, 1], jobs=[{}, {}], quality_times=[2., 6.])
        quality = [dict(case_id='x', seed=0, method=method, refinement_deadline_sec=t,
                        objective_score=10., coverage_pct=90.)
                   for method in ('local_search', 'mappo') for t in manifest['quality_times']]
        with tempfile.TemporaryDirectory() as temp, \
             patch('constrained_sensitivity.collect', return_value=(manifest, repairs, rows, quality, {})):
            root = Path(temp)
            write_report(root, plot=False)
            with (root / 'raw_methods.csv').open() as stream:
                stored = list(csv.DictReader(stream))
            self.assertEqual([r['objective_score'] for r in stored[2:]], ['', ''])
            validation = json.loads((root / 'validation_summary.json').read_text())
            self.assertEqual(validation['completed_method_runs'], 2)
            self.assertEqual(validation['blocked_method_runs'], 2)
            self.assertTrue((root / 'REPORT.md').exists())

    def test_tampered_frozen_plan_is_rejected_before_aggregation(self):
        from prototype_offline import digest, save
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'frozen').mkdir()
            path = root / 'frozen/study_plan.json'
            save(path, dict(original=True))
            save(root / 'manifest.json', dict(frozen_hashes={'frozen/study_plan.json': digest(path)}))
            save(path, dict(original=False))
            with self.assertRaisesRegex(ValueError, 'Frozen study input changed'):
                collect(root)

    def test_quality_summary_preserves_observation_and_success_denominators(self):
        from constrained_sensitivity import summarize_quality
        rows = [dict(case_id='x', seed=seed, method='local_search', refinement_deadline_sec=t,
                     objective_score=score, coverage_pct=90. if score is not None else None)
                for seed, t, score in [(0, 2., None), (1, 2., 9.), (0, 6., 10.), (1, 6., 11.)]]
        result = summarize_quality(rows, ['x', 'blocked'], [0, 1, 2], [2., 6.])
        early = next(r for r in result if r['case_id'] == 'x' and r['method'] == 'local_search'
                     and r['refinement_deadline_sec'] == 2.)
        self.assertEqual(early['requested_repeats'], 3)
        self.assertEqual(early['completed_repeats'], 2)
        self.assertEqual(early['observed_by_deadline'], 1)
        self.assertEqual(early['objective_score_mean'], 9.)
        self.assertIsNone(early['objective_score_sd'])
        late = next(r for r in result if r['case_id'] == 'x' and r['method'] == 'local_search'
                    and r['refinement_deadline_sec'] == 6.)
        self.assertEqual(late['objective_score_mean'], 10.5)
        self.assertEqual(late['observed_by_deadline'], 2)
        for row in result:
            if row['case_id'] == 'blocked':
                self.assertEqual(row['observed_by_deadline'], 0)
                self.assertIsNone(row['objective_score_mean'])
        with self.assertRaises(ValueError):
            summarize_quality(rows + [rows[0]], ['x'], [0, 1, 2], [2., 6.])


if __name__ == '__main__':
    unittest.main(verbosity=2)

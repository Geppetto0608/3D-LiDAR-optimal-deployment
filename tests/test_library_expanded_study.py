"""Expansion design/plan contracts; actual tiny execution uses --smoke separately."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_expanded_study as study


class ExpandedStudyTests(unittest.TestCase):
    def test_default_matches_frozen_config(self):
        spec = json.loads((study.BASE / 'configs/library_expanded_20261123.json').read_text())
        self.assertEqual(spec, study.default_spec())
        with patch.object(study, 'MeshVisibility', side_effect=AssertionError('dry-run raycast')), \
             patch.object(study, 'optimize_frontier', side_effect=AssertionError('dry-run solve')), \
             patch.object(study, 'save', side_effect=AssertionError('dry-run write')):
            planned = study.plan(spec)
        self.assertEqual(planned['sources'], 12)
        self.assertEqual(planned['target_cases'], 16)
        self.assertEqual(planned['expected_comparison_rows'], 5376)
        self.assertEqual(planned['k_values'], list(range(1, 9)))

    def test_source_selection_no_coverage(self):
        spec = study.default_spec()
        sources = study.source_specs(spec)
        for target in spec['targets']:
            nearest = study.select_sources(sources, target, 1)
            three = study.select_sources(sources, target, 3)
            self.assertEqual(three[:1], nearest)
            self.assertTrue(all(s['track'] == target['track'] for s in three))
            if target['track'] == 'A':
                self.assertTrue(all(s['topology'] == target['topology'] for s in three))

    def test_smoke_plan_and_invalid_certificate(self):
        spec = study.smoke_spec()
        planned = study.plan(spec)
        self.assertEqual(planned['sources'], 2)
        self.assertEqual(planned['target_cases'], 8)
        self.assertEqual(planned['expected_comparison_rows'], 1152)
        for change in [dict(minimum_seconds=2.), dict(relative_gap=.5), dict(interval_seconds=0.)]:
            bad = copy.deepcopy(spec)
            bad['optimization']['stop_policy'].update(change)
            with self.assertRaises(ValueError):
                study.plan(bad)

    def test_reject_source_target_identity_and_unsafe_names(self):
        spec = study.smoke_spec()
        spec['targets'][0].update(width_m=8., center_m=[50., 50.])
        with self.assertRaisesRegex(ValueError, 'duplicates a source'):
            study.plan(spec)
        spec = study.smoke_spec()
        spec['targets'][0]['name'] = '../bad'
        with self.assertRaises(ValueError):
            study.plan(spec)

    def test_summary_averages_seeds_before_conditions(self):
        base = dict(track='A', source_count=3, requested_q=32, budget_k=8, method='random',
            case='a', coverage_pct=10., overlap_pct=5., num_selected=8, actual_q=30,
            loss_lower_pp=0., loss_upper_pp=2., online_through_k_sec=1., online_all_k_sec=1.,
            reference_coverage_pct=12., reference_online_all_k_sec=2.)
        rows = [base, dict(base, coverage_pct=30.), dict(base, case='b', coverage_pct=60.)]
        summary = study.summarize(rows)
        self.assertEqual(summary[0]['condition_count'], 2)
        self.assertEqual(summary[0]['coverage_pct_mean'], 40.)


if __name__ == '__main__':
    unittest.main()

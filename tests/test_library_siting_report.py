"""Aggregation units and paired-factor arithmetic; no simulator imports."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_siting_report as report


class SitingReportTests(unittest.TestCase):
    def test_numpy_counts_serialize_without_silent_nonfinite_or_object_conversion(self):
        value = dict(count=np.int64(2), passed=np.bool_(True))
        self.assertEqual(json.loads(json.dumps(value, default=report.json_scalar)), dict(count=2, passed=True))
        with self.assertRaises(ValueError):
            json.dumps(dict(value=np.float64('nan')), allow_nan=False, default=report.json_scalar)
        with self.assertRaises(TypeError):
            json.dumps(dict(value=object()), default=report.json_scalar)

    def row(self, level='open', occlusion='clear'):
        values = dict(base_map='toy', track='A', installation=level, occlusion=occlusion,
                      budget_k=1, query_count=3, reference_coverage_pct=90., reference_covered_count=90,
                      random_mean_pct=75., random_full_reference_match_count=2, random_seed_count=30)
        for method, count in [('library', 80), ('geometry_single', 70), ('geometry_union', 90),
                              ('geometry_portfolio', 85)]:
            values[f'{method}_coverage_pct'] = float(count)
            values[f'{method}_covered_count'] = count
        return values

    def test_cell_means_and_deterministic_vs_seed_denominators(self):
        rows = [self.row(), self.row()]
        rows[1].update(library_coverage_pct=90., library_covered_count=90, random_mean_pct=85.)
        result = report.summarize(rows)
        self.assertEqual(result['conditions'], 2)
        self.assertEqual(result['methods']['library']['mean_coverage_pct'], 85.)
        self.assertEqual(result['methods']['library']['mean_shortfall_pp'], 5.)
        self.assertEqual(result['methods']['library']['exact_match_count'], 1)
        self.assertEqual(result['methods']['random_mean']['mean_coverage_pct'], 80.)
        self.assertEqual(result['methods']['random_mean']['seed_rows'], 60)
        self.assertEqual(result['methods']['geometry_portfolio']['library_vs_method'],
                         dict(wins=1, ties=0, losses=1))

    def test_factor_pairing_and_shortfall_sign(self):
        rows = [self.row(level, occ) for level in ('open', 'restricted') for occ in ('clear', 'occluded')]
        rows[2].update(query_count=2, reference_coverage_pct=85., library_coverage_pct=70.)
        effects = report.paired_effects(rows)
        self.assertEqual(len(effects), 4)
        effect = next(r for r in effects if r['factor'] == 'installation' and r['fixed'] == 'clear')
        self.assertEqual(effect['query_before'], 3)
        self.assertEqual(effect['query_after'], 2)
        self.assertEqual(effect['library_delta_pp'], -10.)
        self.assertEqual(effect['reference_delta_pp'], -5.)
        self.assertEqual(effect['library_shortfall_delta_pp'], 5.)
        with self.assertRaisesRegex(ValueError, 'Missing paired'):
            report.paired_effects(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            report.paired_effects([*rows, deepcopy(rows[0])])

    def test_group_denominators_keep_tracks_and_k_separate(self):
        rows = [self.row(), dict(self.row(), track='B', base_map='other', budget_k=2)]
        groups = report.grouped_summary(rows)
        self.assertEqual(groups['all']['conditions'], 2)
        self.assertEqual(groups['track/A']['conditions'], 1)
        self.assertEqual(groups['track/B']['conditions'], 1)
        self.assertEqual(groups['K/1']['conditions'], 1)
        self.assertEqual(groups['factor_K/B/open/clear/2']['conditions'], 1)


if __name__ == '__main__':
    unittest.main()

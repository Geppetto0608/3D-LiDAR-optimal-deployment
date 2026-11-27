"""Fixed-Q, source-selection, held-out isolation and aggregation regression tests."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_mechanism_study as study


class MechanismTest(unittest.TestCase):
    def test_frozen_config_and_counts(self):
        spec = study.default_spec()
        self.assertEqual(json.loads((study.BASE / 'configs/library_mechanism_20261127.json').read_text()), spec)
        self.assertEqual(len(study.variants(spec)), 10)
        refs = sum(map(len, spec['target_k'].values())) * len(spec['scene_levels']) * len(spec['profiles'])
        self.assertEqual(refs, 184)
        self.assertEqual(refs * (len(study.variants(spec)) + 2 + len(spec['random_seeds'])), 2760)
        self.assertEqual(len(spec['split']['reserved']), 8)

    def test_split_rejects_renamed_reseeded_geometry_and_reserved_execution(self):
        spec = study.default_spec()
        split = spec['split']
        study.validate_split(split, split['sources'], spec['targets'])
        for original in (split['sources'][0], split['development'][0]):
            bad = deepcopy(split)
            bad['reserved'][0] = dict(original, name='renamed', seed=9001)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                study.validate_split(bad, bad['sources'], spec['targets'])
        with self.assertRaisesRegex(ValueError, 'sealed'):
            study.validate_split(split, split['sources'], [split['reserved'][0]])
        bad = deepcopy(split)
        bad['reserved'][0]['seed'] = split['sources'][0]['seed']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            study.validate_split(bad, bad['sources'], spec['targets'])

    def test_geometry_identity_normalizes_numbers_defaults_and_modules(self):
        d = study.default_split()['sources'][12]
        changed = dict(d, length_m=int(d['length_m']), bend_length_m=48, modules=6, seed=9999)
        self.assertEqual(study.geometry_key(d), study.geometry_key(changed))

    def test_random_sources_reproducible_compatible_and_order_independent(self):
        split = study.default_split()
        for target in split['development']:
            a = study.select_sources(split['sources'], target, 3, 'random', 7)
            b = study.select_sources(list(reversed(split['sources'])), target, 3, 'random', 7)
            self.assertEqual(a, b)
            self.assertEqual(len({d['name'] for d in a}), 3)
            self.assertTrue(all(d['track'] == target['track'] for d in a))
            if target['track'] == 'A':
                self.assertTrue(all(d['topology'] == target['topology'] for d in a))
            nearest = study.select_sources(split['sources'], target, 3, 'nearest', None)
            self.assertEqual(nearest, study.retrieve(split['sources'], target, 3))

    def test_incumbent_is_explicit_and_alternatives_deduplicated(self):
        row = dict(selected_ids=[2, 1], alternatives=[dict(selected_ids=[4]),
                   dict(selected_ids=[1, 2]), dict(selected_ids=[5])])
        data = dict(profiles=dict(union=dict(frontier=[row])))
        self.assertEqual(study.source_solutions(data, 1), [[2, 1]])
        self.assertEqual(study.source_solutions(data, 2), [[2, 1], [4]])
        self.assertEqual(study.source_solutions(data, 12), [[2, 1], [4], [5]])
        row['selected_ids'] = None
        row['alternatives'] = []
        self.assertEqual(study.source_solutions(data, 12), [])

    def test_fixed_q_fill_is_explicit_even_for_empty_pool(self):
        for pool in ([], [2], [2, 4, 1], list(range(10))):
            chosen, fill = study.fill_shortlist(pool, list(range(8)), 8, 10)
            self.assertEqual(len(chosen) + len(fill), 8)
            self.assertFalse(set(chosen) & set(fill))
            self.assertEqual(chosen, pool[:8])
        for pool, order, q, n in [([1, 1], [0, 1], 2, 3), ([0], [0], 2, 3),
                                  ([], [0], 4, 3), ([9], [0, 1], 2, 3)]:
            with self.assertRaises(ValueError):
                study.fill_shortlist(pool, order, q, n)

    def test_planning_fixed_q_does_not_query_physics(self):
        spec = study.smoke_spec()
        spec['query_budget'] = 8
        problem = dict(descriptor=spec['targets'][0], poses=np.array([[i, 0, 3, 0, -10] for i in range(20)]),
                       targets=np.zeros((5, 3)), profile=object())
        masks = np.random.default_rng(0).random((20, 5)) < .5
        def pool(root, sources, target, v):
            count = v['source_count'] + v['alternative_count']
            return dict(order=list(range(min(count, 20))), sources=[], planning_sec=.01)
        frontier = dict(frontier=[dict(alternatives=[dict(selected_ids=[18, 19])])])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study.core.save(root / 'library_spec.json', study.core.default_spec())
            with patch.object(study, 'transfer_pool', side_effect=pool), \
                    patch.object(study.core, '_optical_masks', return_value=masks), \
                    patch.object(study.core, '_profile_values', return_value={}), \
                    patch.object(study.budget, 'optimize', return_value=frontier), \
                    patch.object(study.core, 'query', side_effect=AssertionError('physical oracle')):
                plans = study.make_plans(root, root, spec, problem, np.ones(20))
            self.assertEqual(len(plans), 22)
            self.assertTrue(all(p['actual_q'] == p['requested_q'] == 8 for p in plans))
            self.assertTrue(all(len(p['selected_ids']) == len(set(p['selected_ids'])) == 8 for p in plans))
            self.assertTrue(all(len(p['comparison_metadata']['query_ids_sha256']) == 64 for p in plans))
            self.assertTrue(any(p['geometry_fill_ids'] for p in plans))
            for p in plans:
                if p['method'].startswith('library_'):
                    self.assertEqual(set(p['selected_ids']), set(p['library_ids'] + p['geometry_fill_ids']))
                    self.assertEqual(p['comparison_metadata']['geometry_fill_count'], len(p['geometry_fill_ids']))
            self.assertEqual(json.loads((root / 'query_plans.json').read_text()), plans)

    def test_transfer_round_robin_and_no_source_mask_read(self):
        split = study.default_split()
        target = split['development'][0]
        sources = study.select_sources(split['sources'], target, 3, 'nearest', None)
        v = dict(source_count=3, alternative_count=1, retrieval='nearest', seed=None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, d in enumerate(sources):
                folder = root / 'library' / d['name']
                folder.mkdir(parents=True)
                row = dict(selected_ids=[i], alternatives=[])
                study.core.save(folder / 'layouts.json', dict(profiles=dict(union=dict(frontier=[row]))))
                study.core.save(folder / 'result.json', dict(artifact_hashes={
                    'layouts.json': study.core.digest(folder / 'layouts.json')}))
            # No source candidates.npz exists: transfer uses poses/metadata only.
            with patch.object(study, 'transfer_ids', side_effect=[([0, 3], []), ([1, 3], []), ([2], [])]):
                result = study.transfer_pool(root, sources, dict(descriptor=target), v)
            self.assertEqual(result['order'], [0, 1, 2, 3])

    def test_dry_run_no_output_solve_query_or_reserved_domain(self):
        original = study.core.domain
        reserved = {d['name'] for d in study.default_split()['reserved']}
        def guarded(d, sampling):
            self.assertNotIn(d['name'], reserved)
            return original(d, sampling)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'absent'
            with patch.object(study.budget, 'read_library', return_value=(study.core.default_spec(), {})), \
                    patch.object(study.core, 'domain', side_effect=guarded), \
                    patch.object(study.core, 'query', side_effect=AssertionError('raycast')), \
                    patch.object(study.budget, 'solve_frontier', side_effect=AssertionError('solve')), \
                    redirect_stdout(StringIO()):
                study.main(['--smoke', '--output', str(output)])
            self.assertFalse(output.exists())
        for args in (['--worker', '/tmp/not-run'], ['--detach']):
            with redirect_stdout(StringIO()), self.assertRaises(SystemExit):
                study.main(args)

    def test_report_requires_complete_seeds_and_preserves_infeasibility(self):
        spec = study.smoke_spec()
        spec['targets'] = spec['targets'][:1]
        spec['target_k'] = {spec['targets'][0]['name']: [2]}
        spec['profiles'] = {'base': spec['profiles']['base']}
        spec['random_seeds'] = [0, 1, 2]
        methods = [(v['method'], v['seed']) for v in study.variants(spec)]
        methods += [('geometry_union', None), ('geometry_portfolio', None)] + [('random', s) for s in spec['random_seeds']]
        rows = [dict(case=spec['targets'][0]['name'] + '_dense', profile='base', budget_k=2,
                     method=m, seed=s, actual_q=32, requested_q=32, status='feasible',
                     query_ids_sha256='test_query_set',
                     coverage_pct=(60 + 10*s if m == 'random' else 80)) for m, s in methods]
        rows[0].update(status='infeasible', coverage_pct=None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study.core.save(root / 'spec.json', spec)
            study.core.write_csv(root / 'comparison.csv', rows)
            study.aggregate_pairs(root)
            with (root / 'paired_effects.csv').open() as stream:
                pairs = list(csv.DictReader(stream))
            difference = next(r for r in pairs if r['contrast'] == 'anchor_vs_random')
            self.assertEqual(float(difference['coverage_difference_pp']), 10.)
            self.assertEqual(difference['identical_query_set_all_trials'], 'True')
            self.assertTrue(any(r['coverage_difference_pp'] == '' for r in pairs))
            for bad in (rows[:-1], rows + [rows[0]]):
                study.core.write_csv(root / 'comparison.csv', bad)
                with self.assertRaisesRegex(RuntimeError, 'conditions/seeds'):
                    study.aggregate_pairs(root)
            rows[1]['actual_q'] = 31
            study.core.write_csv(root / 'comparison.csv', rows)
            with self.assertRaisesRegex(RuntimeError, 'query budgets'):
                study.aggregate_pairs(root)


if __name__ == '__main__':
    unittest.main()

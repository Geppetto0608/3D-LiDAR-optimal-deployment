"""Exact conditional selection parity, budget isolation and fresh-worker audit."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geometry_placement as runner
import test_connected_hierarchy as fixtures
import test_geometry_placement as worker_fixtures
from src.methods.connected_hierarchy import (BudgetedObservations, batch_predicted_partners,
    predicted_partner, select_pool)


class ConditionalBatchTests(unittest.TestCase):
    def fixture(self, constraints=None):
        return fixtures.HierarchyTests().fixture(constraints)

    def compare_partners(self, context, truth, predicted, candidates, partners, fast=True):
        observed = BudgetedObservations(context, lambda i: (_ for _ in ()).throw(AssertionError('Extra query')),
            initial={i: truth[i] for i in partners})
        expected = {cid: predicted_partner(cid, partners, predicted, observed, context, fast_screen=fast)
            for cid in candidates}
        self.assertEqual(batch_predicted_partners(candidates, partners, predicted, observed, context,
            fast_screen=fast), expected)
        self.assertEqual(observed.events, [])
        return expected

    def test_random_masks_nonuniform_importance_costs_and_constraint_screens(self):
        cases = [{}, dict(min_coverage_pct=75, min_overlap_ratio=.2, max_overlap_ratio=.7,
            min_single_failure_coverage_pct=40, min_xy_separation_m=5, max_installation_cost=28),
            dict(min_coverage_pct=100, min_overlap_ratio=.5, max_overlap_ratio=.5,
                min_single_failure_coverage_pct=50, max_sensors_per_pole=1)]
        for seed in range(20):
            rng = np.random.default_rng(seed)
            for constraints in cases:
                context, truth = self.fixture(constraints)
                context.requirements.importance = rng.uniform(1, 10, len(context.targets))
                context.requirements.spec['cost']['regions'] = [
                    dict(bounds=[0, -10, 5, 10], surcharge=.125)]
                truth = rng.random(truth.shape) < .55
                predicted = rng.random(truth.shape) < .65
                partners = sorted(g[0] for g in context.groups)
                candidates = [i for g in context.groups for i in g[1:]]
                for interface in ({}, dict(min_coverage_pct=75, min_overlap_ratio=.25, max_overlap_ratio=.75)):
                    context.requirements.profile['interface'] = interface
                    self.compare_partners(context, truth, predicted, candidates, partners)

    def test_quantized_ties_extreme_weights_and_additional_full_constraints(self):
        context, truth = self.fixture()
        truth[:] = False
        partners = [g[0] for g in context.groups]
        candidates = [g[1] for g in context.groups]
        predicted = truth.copy()
        predicted[:, :4] = True
        for i in partners:
            truth[i, 4:8] = True
        for scale in (1e-12, 1., 1e12):
            context.requirements.importance[:] = scale
            context.requirements.importance[4] += 1e-9
            for fast in (False, True):
                self.compare_partners(context, truth, predicted, candidates, partners, fast)
        # This constraint is deliberately not in the packed necessary screen.
        context.requirements.spec['constraints']['forbidden_regions'] = [[0, 0, 5, 10]]
        self.compare_partners(context, truth, predicted, candidates, partners)

    def test_threshold_boundaries_packed_padding_and_single_survivor(self):
        context, truth = self.fixture()
        truth[:] = False
        partners = [g[0] for g in context.groups[1:]]
        cid = context.groups[0][1]
        predicted = truth.copy()
        predicted[cid, :5] = True
        truth[partners[0], 3:9] = True
        truth[partners[1:], :2] = True
        for delta in (-2e-9, -5e-10, 0., 5e-10, 2e-9):
            context.requirements.spec['constraints'] = dict(min_coverage_pct=75+delta,
                min_overlap_ratio=2/12+delta, max_overlap_ratio=2/12+delta,
                min_single_failure_coverage_pct=100*5/12+delta,
                max_installation_cost=26+delta)
            self.compare_partners(context, truth, predicted, [cid], partners)
        context.requirements.spec['constraints'] = dict(min_coverage_pct=75)
        result = self.compare_partners(context, truth, predicted, [cid], partners)
        self.assertEqual(result[cid][1], partners[0])

    def test_empty_and_same_site_partners_and_unmeasured_truth_are_not_used(self):
        context, truth = self.fixture()
        cid = context.groups[0][1]
        self.assertEqual(self.compare_partners(context, truth, truth, [cid], []), {cid: (None, None)})
        result = self.compare_partners(context, truth, truth, [cid], context.groups[0][:1])
        self.assertEqual(result[cid], (None, None))
        self.assertEqual(self.compare_partners(context, truth, truth, [], []), {})
        obs = BudgetedObservations(context, lambda i: truth[i])
        with self.assertRaises(KeyError):
            batch_predicted_partners([cid], [context.groups[1][0]], truth, obs, context, fast_screen=True)

    def test_whole_policy_identical_across_priors_transforms_rounds_and_budgets(self):
        for seed in range(4):
            context, truth = self.fixture(dict(min_coverage_pct=60, max_overlap_ratio=.75))
            rng = np.random.default_rng(seed)
            predicted = rng.random(truth.shape) < .75
            keep = rng.random(truth.shape) < .9
            for mode in ('geometry', 'source', 'random'):
                results = []
                for budget in (1, 2, 4):
                    options = dict(probe_prior=mode, random_seed=7 if mode == 'random' else None,
                        fast_screen=True, prediction_transform=lambda a: a & keep)
                    pair = []
                    for engine in ('scalar', 'batch'):
                        obs = BudgetedObservations(context, lambda i: truth[i], per_site=budget)
                        with patch('src.methods.connected_hierarchy.optical_predictions', return_value=predicted.copy()):
                            selected = select_pool('probe_context', context, obs, budget, partner_engine=engine, **options)
                        pair.append({k:v for k,v in selected.items() if k not in ('selection_sec', 'partner_engine')})
                        self.assertEqual(len(obs.events), budget*len(context.groups))
                    self.assertEqual(*pair)
                    results.append(pair[0])
                for small, large in zip(results, results[1:]):
                    self.assertEqual(small['events'], large['events'][:len(small['events'])])

    def test_generic_subclass_and_unscreened_modes_fall_back_to_scalar(self):
        context, truth = self.fixture()
        base = type(context.requirements)
        class CustomRequirements(base):
            pass
        context.requirements.__class__ = CustomRequirements
        ids = [context.groups[0][1]]
        partners = [g[0] for g in context.groups[1:]]
        self.compare_partners(context, truth, truth, ids, partners)
        context.requirements.__class__ = base
        self.compare_partners(context, truth, truth, ids, partners, fast=False)

    def test_batch_engine_is_opt_in_and_rejects_mismatched_selectors(self):
        context, truth = self.fixture()
        for kind, engine in [('probe_context', 'unknown'), ('geometry_single', 'batch'), ('probe_context', True)]:
            with self.assertRaises(ValueError):
                select_pool(kind, context, BudgetedObservations(context, lambda i: truth[i], per_site=2),
                    2, partner_engine=engine)
        design, base = worker_fixtures.GeometryPlacementTests().inputs()
        _, _, arms = runner.validate_inputs(design, base, ['adaptive', 'adaptive_batch', 'full'], [2], 3)
        self.assertEqual([a['name'] for a in arms], ['adaptive_q2', 'adaptive_batch_q2', 'full'])

    def test_real_workers_replay_both_engines_and_reject_rehashed_trace_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = worker_fixtures.GeometryPlacementTests().run_smoke(Path(tmp),
                ['adaptive', 'adaptive_batch', 'full'], [2], 1, 'parity')
            validation = runner.read(root/'independent_validation.json')
            self.assertEqual((validation['workers'], validation['scalar_batch_policy_pairs']), (6, 2))
            manifest = runner.read(root/'manifest.json')
            job = next(j for j in manifest['jobs'] if j['arm']['method'] == 'adaptive_batch')
            path = root/'runs'/job['id']/'selection.json'
            selection = runner.read(path)
            selection['decisions'][0]['predicted_j'] = -123
            runner.corridor.save(path, selection)
            job['output_hashes']['selection.json'] = runner.corridor.digest(path)
            runner.corridor.save(root/'manifest.json', manifest)
            from validate_geometry_placement import audit
            with self.assertRaisesRegex(ValueError, 'Policy replay'):
                audit(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)

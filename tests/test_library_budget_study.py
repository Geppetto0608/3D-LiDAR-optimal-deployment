"""Independent enumeration and workflow guards for low-K constrained transfer."""
from contextlib import redirect_stdout
from io import StringIO
from itertools import combinations
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_budget_study as study
from src.methods.rsu_coverage_milp import assess, solve_frontier


class CostCriticalSolverTest(unittest.TestCase):
    def setUp(self):
        self.poses = np.array([[x, 0, z, 0, -10] for x, z in
                              [(0, 3), (0, 5), (2, 3), (4, 3), (8, 3), (12, 3)]], float)
        self.critical = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
        self.req = dict(overlap_ratio=0., critical_overlap_ratio=0., min_separation_m=3.)
        self.costs = np.array([1.3, 1.8, 1., 1.2, 1.1, 1.4])

    def test_brute_force_independent_cost_critical_and_overlap(self):
        for seed in range(3):
            masks = np.random.default_rng(seed).random((6, 9)) < .5
            for floor, overlap in [(0., 0.), (.6, 0.), (.8, .2)]:
                req = dict(self.req, overlap_ratio=overlap, critical_overlap_ratio=overlap)
                result = solve_frontier(self.poses, masks, self.critical, [1, 2, 3], req,
                    candidate_costs=self.costs, cost_limits={1: 1.25, 2: 2.5, 3: 3.75},
                    critical_coverage_ratio=floor, seconds=5., stop_gap=0., alternatives=3)
                for row in result['frontier']:
                    k = row['budget_k']
                    feasible = []
                    for size in range(k + 1):
                        for selected in combinations(range(6), size):
                            ids = list(selected)
                            xy = self.poses[ids, :2]
                            if any(np.linalg.norm(xy[a] - xy[b]) < 3. - 1e-7
                                   for a, b in combinations(range(size), 2)):
                                continue
                            if self.costs[ids].sum() > 1.25 * k + 1e-7:
                                continue
                            density = masks[ids].sum(axis=0)
                            if (np.count_nonzero(density[:4]) < np.ceil(floor * 4 - 1e-9)
                                    or np.count_nonzero(density >= 2) < np.ceil(overlap * 9 - 1e-9)
                                    or np.count_nonzero(density[:4] >= 2) < np.ceil(overlap * 4 - 1e-9)):
                                continue
                            feasible.append(int(np.count_nonzero(density)))
                    if not feasible:
                        self.assertEqual(row['status'], 'infeasible')
                        self.assertIsNone(row['metrics'])
                        self.assertIsNone(row['upper_bound_count'])
                    else:
                        self.assertEqual(row['metrics']['covered_count'], max(feasible))
                        self.assertTrue(row['discrete_optimal'])
                        for entry in row['alternatives']:
                            ids = entry['selected_ids']
                            self.assertLessEqual(self.costs[ids].sum(), k * 1.25 + 1e-7)
                            self.assertGreaterEqual(np.count_nonzero(masks[ids].any(axis=0)[:4]), np.ceil(floor * 4 - 1e-9))

    def test_zero_budget_and_critical_infeasibility(self):
        masks = np.ones((6, 9), bool)
        for floor in (0., .1):
            row = solve_frontier(self.poses, masks, self.critical, [2], self.req,
                candidate_costs=self.costs, cost_limits={2: 0.}, critical_coverage_ratio=floor,
                seconds=5.)['frontier'][0]
            self.assertEqual(row['status'], 'infeasible' if floor else 'feasible')
            if not floor:
                self.assertEqual(row['selected_ids'], [])
                self.assertEqual(row['metrics']['installation_cost'], 0.)

    def test_cost_and_critical_violations_not_reward_penalties(self):
        masks = np.zeros((6, 9), bool)
        value = assess(self.poses, masks, [1], self.critical, self.req, 2,
            candidate_costs=self.costs, cost_budget=1., critical_coverage_ratio=.6)
        self.assertFalse(value['feasible'])
        self.assertIn('critical_coverage', value['violations'])
        self.assertIn('installation_cost', value['violations'])

    def test_invalid_new_inputs(self):
        masks = np.ones((6, 9), bool)
        for kw in [dict(candidate_costs=[1]), dict(candidate_costs=[-1] * 6),
                   dict(candidate_costs=[np.nan] * 6), dict(cost_limits={1: 1}),
                   dict(candidate_costs=self.costs, cost_limits={1: 2, 2: 1}),
                   dict(candidate_costs=self.costs, cost_limits={1: 2}),
                   dict(critical_coverage_ratio=1.1), dict(critical_coverage_ratio=np.nan)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                solve_frontier(self.poses, masks, self.critical, [1, 2], self.req, **kw)

    def test_legacy_defaults_equal_explicit_unconstrained_costs(self):
        masks = np.random.default_rng(7).random((6, 9)) < .5
        old = solve_frontier(self.poses, masks, self.critical, [1, 2], self.req, seconds=5.)
        new = solve_frontier(self.poses, masks, self.critical, [1, 2], self.req,
                             seconds=5., candidate_costs=self.costs)
        self.assertEqual(old['schema'], 'rsu_constrained_frontier_v1')
        for a, b in zip(old['frontier'], new['frontier']):
            self.assertEqual(a['metrics']['covered_count'], b['metrics']['covered_count'])


class BudgetStudyTest(unittest.TestCase):
    def test_config_and_predeclared_count(self):
        spec = study.default_spec()
        path = study.BASE / 'configs/library_lowk_cost_20261124.json'
        self.assertEqual(json.loads(path.read_text()), spec)
        refs = sum(map(len, spec['target_k'].values())) * len(spec['profiles']) * len(spec['scene_levels'])
        self.assertEqual(refs, 184)
        self.assertEqual(refs * (4 + len(spec['random_seeds'])), 1288)

    def test_costs_geometry_only_and_orientation_invariant(self):
        spec = study.smoke_spec()
        for d in spec['targets']:
            problem = study.domain(d, spec['sampling'])
            costs, info = study.installation_costs(problem, spec['cost_model'])
            alternate = dict(problem, poses=problem['poses'].copy(), targets=np.zeros((1, 3)), critical=None)
            alternate['poses'][:, 3:] += 10.
            changed, other = study.installation_costs(alternate, spec['cost_model'])
            np.testing.assert_array_equal(costs, changed)
            self.assertEqual(info, other)
            self.assertGreater(np.ptp(costs), 0.)
            self.assertTrue(np.all(costs >= 1.))

    def test_smoke_keeps_source_installation_modes(self):
        full = study.default_spec()['sampling']
        smoke = study.smoke_spec()['sampling']
        for key in ('installation_step_m', 'heights_m', 'relative_yaws_deg', 'pitches_deg'):
            self.assertEqual(full[key], smoke[key])
        for d in study.smoke_spec()['targets']:
            problem = study.domain(d, smoke)
            if d['track'] == 'A':
                groups = {(c['branch_id'], c['installation_line_id'], c['z_m']) for c in problem['canonical']}
                for branch in ('east', 'west', 'south'):
                    for side in ('left', 'right'):
                        for z in (3., 5.):
                            self.assertIn((branch, side, z), groups)

    def test_frozen_library_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'source').mkdir()
            (root / 'library').mkdir()
            artifact = root / 'library' / 'fixture.txt'
            artifact.write_text('source')
            provenance = dict(hashes={'library/fixture.txt': study.core.digest(artifact)})
            for folder in (root, root / 'source'):
                study.core.save(folder / 'library_provenance.json', provenance)
                study.core.save(folder / 'library_spec.json', {})
            study.verify_library(root)
            artifact.write_text('changed')
            with self.assertRaises(RuntimeError):
                study.verify_library(root)

    def test_dry_run_does_not_solve_query_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'not_created'
            fake = study.core.default_spec()
            with patch.object(study, 'read_library', return_value=(fake, {})), \
                    patch.object(study, 'solve_frontier', side_effect=AssertionError('solve')), \
                    patch.object(study.core, 'query', side_effect=AssertionError('query')), \
                    redirect_stdout(StringIO()):
                study.main(['--smoke', '--output', str(output)])
            self.assertFalse(output.exists())
        with self.assertRaises(SystemExit):
            study.main(['--worker', '/tmp/no-run'])

    def test_plans_precede_queries_and_use_same_budget(self):
        spec = study.smoke_spec()
        spec['profiles'] = dict(cost=spec['profiles']['cost'])
        problem = study.domain(spec['targets'][0], spec['sampling'])
        costs, _ = study.installation_costs(problem, spec['cost_model'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study.core.save(root / 'library_spec.json', study.core.default_spec())
            pool = dict(order=list(range(min(6, len(costs)))), planning_sec=.1)
            with patch.object(study.core, 'library_pool', return_value=pool), \
                    patch.object(study.core, 'query', side_effect=AssertionError('physical query during planning')):
                plans = study.make_plans(root, root, spec, problem, costs)
            self.assertEqual(len(plans), 5)
            self.assertEqual({p['actual_q'] for p in plans}, {6})
            self.assertEqual({p['requested_q'] for p in plans}, {8})
            self.assertTrue((root / 'query_plans.json').is_file())

    def test_missing_rows_are_not_zero_filled(self):
        spec = study.smoke_spec()
        problem = study.domain(spec['targets'][0], spec['sampling'])
        costs, _ = study.installation_costs(problem, spec['cost_model'])
        for status in ('infeasible', 'no_feasible_found'):
            self.assertIsNone(study.check_row(problem, None, costs, spec, 'cost',
                dict(status=status, metrics=None), None))
        with self.assertRaises(RuntimeError):
            study.check_row(problem, None, costs, spec, 'cost', dict(status='feasible', metrics=None), None)

    def test_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, 'plan', return_value={}):
            with self.assertRaises(FileExistsError):
                study.execute(study.smoke_spec(), Path(tmp), Path(tmp), False)


if __name__ == '__main__':
    unittest.main()

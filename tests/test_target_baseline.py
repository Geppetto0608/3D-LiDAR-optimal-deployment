"""Plan/guard and result-join checks; no real optimization required."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_target_baseline import command, plan, report, run
from prototype_offline import save


class TargetBaselineTests(unittest.TestCase):
    def test_plan_groups_budgets_not_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [dict(status='complete', scenario=dict(id=420001), budget_k=k,
                         source_roots=['source_low', 'source_high'])
                    for k in [3, 4] for _ in range(3)]
            save(root/'manifest.json', dict(status='complete', jobs=jobs))
            result = plan(root)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['k_values'], [3, 4])
            jobs[0]['status'] = 'failed'
            save(root/'manifest.json', dict(status='complete', jobs=jobs))
            with self.assertRaisesRegex(ValueError, 'must be complete'):
                plan(root)

    def test_commands_do_not_use_online_warm_start_or_cache(self):
        item = dict(scenario=dict(id=420001), k_values=[3, 4])
        planned = command(Path('/tmp/test'), item, 300)
        self.assertNotIn('--execute', planned)
        self.assertNotIn('--reuse-candidates', planned)
        self.assertEqual(planned[planned.index('--k-values')+1:planned.index('--time-limit')], ['3', '4'])
        self.assertEqual(command(Path('/tmp/test'), item, 300, True)[-1], '--execute')

    def test_dry_run_does_not_create_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'untouched'
            args = SimpleNamespace(study='unused', output=str(root), seconds=300., execute=False)
            with patch('prototype_target_baseline.plan', return_value=[dict(k_values=[3, 4])]), \
                    contextlib.redirect_stdout(io.StringIO()):
                run(args)
            self.assertFalse(root.exists())

    def make_report_fixture(self, root):
        study = root/'online'
        source = root/'baselines/s420001'
        (source/'k3').mkdir(parents=True)
        online = dict(scenario_id=420001, K=3, initialization='transfer_low',
            initial_coverage_pct=50., local_search_coverage_pct=80., mappo_mean_pct=75.,
            mappo_sample_sd_pct=None, local_search_online_sec=3., mappo_mean_online_sec=61.)
        save(root/'online_summary.json', dict(pairs=[online]))
        save(source/'manifest.json', dict(status='complete', k_values=[3],
             candidate_stats=dict(num_candidates=2), candidate_build_sec=10.))
        save(source/'k3/result.json', dict(coverage_pct=90., num_targets=2, certification='best_known',
            wall_time_sec=305., solver=dict(solver_status='9', objective=1.8, best_bound=None,
                                         mip_gap=None, solve_time_sec=300.)))
        save(source/'k3/validation.json', dict(passed=True))
        targets = np.array([[0., 0., 1.], [1., 1., 1.]])
        np.savez(source/'candidates.npz', target_points=targets)
        for method in ('local_search', 'mappo_seed0'):
            path = study/'jobs'/f's420001_k3_transfer_low_{method}'
            path.mkdir(parents=True)
            np.savez(path/'coverage.npz', targets=targets)
        save(root/'online_domain.json', {f's420001_k3_transfer_low_{m}': dict(inside_milp_xyz_domain=m=='local_search')
                                        for m in ('local_search', 'mappo_seed0')})
        return dict(status='complete', online_study=str(study), seeds=[0],
                    targets=[dict(status='complete', scenario=dict(id=420001), k_values=[3], worker_wall_sec=320.)])

    def test_report_keeps_noncertified_status_and_time_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_report_fixture(root)
            report(root, manifest)
            result = json.loads((root/'comparison.json').read_text())
            row = result['pairs'][0]
            self.assertEqual(row['local_search_minus_milp_pp'], -10.)
            self.assertEqual(row['candidate_plus_k_work_sec'], 315.)
            self.assertEqual(row['certification'], 'best_known')
            self.assertIsNone(row['best_bound'])
            self.assertTrue(row['local_search_inside_milp_domain'])
            self.assertFalse(row['all_mappo_trials_inside_milp_domain'])

    def test_report_rejects_equal_count_but_different_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_report_fixture(root)
            path = root/'online/jobs/s420001_k3_transfer_low_local_search/coverage.npz'
            np.savez(path, targets=np.array([[5., 5., 1.], [6., 6., 1.]]))
            with self.assertRaisesRegex(ValueError, 'coordinates differ'):
                report(root, manifest)


if __name__ == '__main__':
    unittest.main()

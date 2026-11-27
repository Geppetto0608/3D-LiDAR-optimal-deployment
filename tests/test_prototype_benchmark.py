"""Width-sweep protocol and sample statistics; no large solve/training."""
from argparse import Namespace
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_benchmark import aggregate, map_key, validate_arguments


def settings(**overrides):
    values = dict(widths=[8,9,11,12,13],scenarios=[0,1,2],seeds=[0,1,2],gpus=[1,2,3],
                  k=3,steps=2000,radius=10.,baseline_topm=10,time_limit=300.,gap=.05)
    values.update(overrides)
    return Namespace(**values)


class WidthBenchmarkTest(unittest.TestCase):
    def test_internal_worker_cannot_bypass_execute_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'not_created'
            result=subprocess.run([sys.executable,'prototype_benchmark.py','--output',str(output),
                '--worker-scenario','0','--worker-width','8'],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('requires --execute',result.stderr)
            self.assertFalse(output.exists())

    def test_protocol_and_stable_map_keys(self):
        args = settings()
        validate_arguments(args)
        self.assertEqual(len(args.scenarios)*len(args.widths),15)
        self.assertEqual(len(args.scenarios)*len(args.widths)*3*len(args.seeds),135)
        self.assertEqual(map_key(2,11),map_key(2,11.0))

    def test_rejects_leakage_duplicates_and_unsafe_full_candidate_workers(self):
        for values in [dict(widths=[7]),dict(widths=[14]),dict(widths=[10]),dict(widths=[np.nan]),
                       dict(widths=[8,8]),dict(seeds=[0,0]),dict(seeds=[-1]),dict(gpus=[]),
                       dict(baseline_topm=0),dict(radius=0),dict(k=4),dict(gap=1),
                       dict(time_limit=np.inf)]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                validate_arguments(settings(**values))
        validate_arguments(settings(baseline_topm=0,gpus=[1]))

    def test_sample_sd_and_policy_are_separate(self):
        rows = [dict(scenario=1,width=8,method='interpolation_mappo',seed=seed,
                     coverage_pct=value,gain_pp=value-70,final_policy_coverage_pct=value-5,
                     compute_component_sum_sec=10.) for seed,value in enumerate([80,82,84])]
        summary = aggregate(rows)[0]
        self.assertEqual(summary['n'],3)
        self.assertEqual(summary['coverage_mean'],82)
        self.assertEqual(summary['coverage_sd'],2)
        self.assertEqual(summary['gain_mean_pp'],12)
        self.assertEqual(summary['final_policy_mean'],77)
        self.assertEqual(summary['compute_component_mean_sec'],10)

    def test_deterministic_baseline_is_not_a_repeated_trial(self):
        row=dict(scenario=0,width=9,method='greedy_top10',seed=None,coverage_pct=90,
                 gain_pp=None,final_policy_coverage_pct=None,compute_component_sum_sec=12)
        summary=aggregate([row])[0]
        self.assertEqual(summary['n'],1)
        self.assertIsNone(summary['coverage_sd'])
        with self.assertRaises(ValueError):
            aggregate([row,row])
        with self.assertRaises(ValueError):
            aggregate([dict(row,coverage_pct=np.nan)])


if __name__ == '__main__':
    unittest.main(verbosity=2)

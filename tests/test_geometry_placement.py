"""Source-free defaults, cold-ready boundaries and isolated real-worker smoke."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import geometry_placement as runner
from validate_geometry_placement import audit


class GeometryPlacementTests(unittest.TestCase):
    def inputs(self):
        base = deepcopy(runner.corridor.load_suite(ROOT/'configs/connected_corridor_pilot.json'))
        base['geometry'].update(length_m=40,bend_length_m=12,mesh_step_m=1)
        base['candidate']['station_step_m'] = 10
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28,min_xy_separation_m=3)
        design = dict(schema='geometry_placement_v1',
            targets=[dict(name='development',road_width_m=10,turn_deg=30,bend_length_m=12)],
            profiles=[dict(name='base',interface={},forbidden_installation=[]),
                dict(name='blocked',interface={},forbidden_installation=[dict(station_range_m=[0,40],side='both')])])
        return design,base

    def test_default_plan_uses_no_source_assets_and_no_full_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'not_created'
            with patch.object(runner.corridor,'source_library',side_effect=AssertionError('Source library used')):
                with patch('src.core.visibility.MeshVisibility',side_effect=AssertionError('Plan cast rays')):
                    with redirect_stdout(io.StringIO()) as text:
                        runner.main(['--output',str(output)])
            plan = json.loads(text.getvalue())
            self.assertTrue(plan['source_free'])
            self.assertFalse(plan['source_assets_required'])
            self.assertFalse(plan['reference_requested'])
            self.assertEqual(plan['arms'],[dict(name='adaptive_q4',method='adaptive',query_budget=4)])
            self.assertEqual((plan['conditions'],plan['fresh_workers'],plan['repeats']),(12,12,1))
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                runner.main(['--worker-input','missing'])

    def test_invalid_configuration_is_rejected_and_order_is_cyclic(self):
        design,base = self.inputs()
        for methods,budgets,repeats in [([], [1],1),(['adaptive','adaptive'],[1],1),(['unknown'],[1],1),
            (['static'],[2,1],1),(['adaptive'],[True],1),(['adaptive'],[1],0),(['adaptive'],[1],True)]:
            with self.subTest(methods=methods,budgets=budgets,repeats=repeats),self.assertRaises(ValueError):
                runner.validate_inputs(design,base,methods,budgets,repeats)
        _,cases,arms = runner.validate_inputs(design,base,['adaptive','static','full'],[1,2],2)
        jobs = runner.worker_jobs(cases,arms,2)
        self.assertEqual(len(jobs),20)
        self.assertEqual([j['arm'] for j in jobs[:5]],arms)
        self.assertEqual([j['arm'] for j in jobs[5:10]],arms[1:]+arms[:1])
        self.assertEqual(len({j['id'] for j in jobs}),20)

    def test_cold_ready_clock_excludes_postdecision_work_and_rejects_invalid_markers(self):
        marker = dict(event='decision_complete',completed_monotonic_ns=3_000_000_000,source_free=True)
        timing = runner.cold_timing(1_000_000_000,6_000_000_000,marker)
        self.assertEqual((timing['cold_ready_sec'],timing['worker_wall_sec']),(2.,5.))
        for key,value in [('completed_monotonic_ns',0),('completed_monotonic_ns',7_000_000_000),
                          ('completed_monotonic_ns',3.),('source_free',False),('event','export_complete')]:
            with self.assertRaises(ValueError):
                runner.cold_timing(1_000_000_000,6_000_000_000,{**marker,key:value})

    def test_worker_does_not_construct_or_transfer_source_library(self):
        design,base = self.inputs()
        suite,cases,arms = runner.validate_inputs(design,base,['adaptive'],[1],1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_path = tmp/'input.json'
            runner.corridor.save(input_path,dict(suite=suite,case=cases[0],profile=design['profiles'][0],arm=arms[0]))
            with patch.object(runner.corridor,'source_library',side_effect=AssertionError('Source library built')):
                with patch.object(runner.constraints,'transfer_source',side_effect=AssertionError('Source transferred')):
                    with redirect_stdout(io.StringIO()) as text:
                        runner.run_worker(input_path,ROOT/'configs/experiment.yaml',tmp/'run')
            self.assertTrue(json.loads(text.getvalue())['source_free'])
            result = runner.read(tmp/'run/solution/result.json')
            self.assertTrue(result['source_free'])
            self.assertEqual(result['work']['unique_candidate_queries'],8)
            with self.assertRaises(FileExistsError):
                runner.run_worker(input_path,ROOT/'configs/experiment.yaml',tmp/'run')

    def test_statistics_do_not_turn_repeats_into_maps_or_claim_equal_quality_speedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(cases=[dict(directory='case',profile='base',target_name='toy')],
                plan=dict(repeats=3,arms=[dict(name='adaptive_q2',method='adaptive',query_budget=2),
                    dict(name='full',method='full',query_budget=None)]),jobs=[])
            for arm in manifest['plan']['arms']:
                for rep in range(3):
                    name = arm['name']+str(rep)
                    manifest['jobs'].append(dict(id=name,case='case',arm=arm,repeat=rep,order_position=rep,status='complete'))
                    result = dict(selected_ids=[0,1],assessment=dict(objective_score=10 if arm['method']=='full' else 9,coverage_pct=90),
                        work=dict(unique_candidate_queries=12 if arm['method']=='full' else 2,rays_submitted=100),
                        optical_point_tests=0,decision_sec=.1)
                    (root/'runs'/name/'solution').mkdir(parents=True)
                    runner.corridor.save(root/'runs'/name/'solution/result.json',result)
                    runner.corridor.save(root/'runs'/name/'timing.json',dict(cold_ready_sec=2*rep+(2 if arm['method']=='full' else 1),worker_wall_sec=10))
            raw,comparison,summary = runner.tables(root,manifest)
            self.assertEqual((len(raw),len(comparison),summary['conditions']),(6,2,1))
            row = next(r for r in comparison if r['method']=='adaptive')
            self.assertEqual(row['cold_ready_median_sec'],3)
            self.assertEqual((row['cold_ready_q25_sec'],row['cold_ready_q75_sec']),(2,4))
            self.assertEqual(summary['groups']['adaptive_q2']['mean_feasible_j_gap'],1)
            self.assertIsNone(summary['groups']['adaptive_q2']['mean_equal_j_latency_ratio'])
            path = root/'runs/adaptive_q20/solution/result.json'
            value = runner.read(path)
            value['assessment']['objective_score'] = 8
            runner.corridor.save(path,value)
            with self.assertRaisesRegex(ValueError,'Non-deterministic'):
                runner.tables(root,manifest)

    def run_smoke(self,tmp,methods,budgets,repeats,name):
        design,base = self.inputs()
        runner.corridor.save(tmp/'design.json',design)
        runner.corridor.save(tmp/'base.json',base)
        root = tmp/name
        cmd = [sys.executable,str(ROOT/'geometry_placement.py'),'--study',str(tmp/'design.json'),
            '--base-suite',str(tmp/'base.json'),'--methods',*methods,'--query-budgets',*map(str,budgets),
            '--repeats',str(repeats),'--output',str(root),'--execute']
        result = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=240)
        errors = ''.join(p.read_text() for p in sorted(root.glob('logs/*.stderr')) if result.returncode)
        if (root/'audit.log').exists() and result.returncode:
            errors += (root/'audit.log').read_text()
        self.assertEqual(result.returncode,0,result.stderr+errors)
        return root,cmd

    def test_real_fresh_workers_full_reference_repeat_audit_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,cmd = self.run_smoke(Path(tmp),['adaptive','static','full'],[1,2],2,'comparison')
            validation = runner.read(root/'independent_validation.json')
            self.assertEqual((validation['workers'],validation['full_case_optima'],validation['timing_rows']),(20,2,20))
            self.assertEqual(validation['checked_deployments'],10)
            summary = runner.read(root/'summary.json')
            self.assertEqual(summary['groups']['adaptive_q1']['cases'],2)
            self.assertEqual(summary['groups']['adaptive_q1']['feasible_cases'],1)
            summary['groups']['adaptive_q1']['cases'] += 1
            runner.corridor.save(root/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'Summary'):
                audit(root)
            self.assertFalse((root/'library').exists())
            refused = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(refused.returncode,0)

    def test_real_default_path_does_not_create_a_full_reference_or_query_unvisited_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,_ = self.run_smoke(Path(tmp),['adaptive'],[1],1,'standalone')
            validation = runner.read(root/'independent_validation.json')
            self.assertEqual(validation['full_case_optima'],0)
            self.assertEqual(validation['fresh_candidate_queries'],8)
            self.assertEqual(validation['checked_deployments'],1)
            summary = runner.read(root/'summary.json')['groups']['adaptive_q1']
            self.assertEqual(summary['reference_cases'],0)
            self.assertIsNone(summary['mean_feasible_j_gap'])


if __name__=='__main__':
    unittest.main(verbosity=2)

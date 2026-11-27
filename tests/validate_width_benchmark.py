"""Independently re-evaluate a completed width sweep; no training or cache bake."""
from pathlib import Path
import argparse
import contextlib
import csv
import io
import json
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_benchmark import aggregate, collect
from prototype_offline import digest, save, width_scenario
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, get_lidar_profile, load_config
from src.methods.local_mappo import validate_poses
from validate_prototype_refinement import validate as validate_refinement


def validate(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['failed_maps']:
        raise ValueError('Expected a fully completed benchmark')
    for path,expected in manifest['frozen_prototype_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen prototype changed: {path}')
    refinement_runs=0
    baseline_runs=0
    for plan in manifest['plans']:
        directory=root/plan['key']
        for seed in manifest['arguments']['seeds']:
            with contextlib.redirect_stdout(io.StringIO()):
                validate_refinement(directory/f'seed{seed}')
            refinement_runs+=3
        baseline=directory/'baselines'
        cfg=load_config(baseline/'config.yaml')
        geometry=MapGeometry(width_scenario(plan['width'],plan['scenario']))
        provider=MeshVisibility(geometry.create_environment_mesh(),
            geometry.target_points(cfg.target['coverage_z_slices']),get_lidar_profile(cfg.lidar_profile,cfg))
        records=json.loads((baseline/'report.json').read_text())['results']
        assert set(records)=={'greedy','milp'}
        for rec in records.values():
            poses=np.load(baseline/rec['poses_npy'],allow_pickle=False)
            validate_poses(poses,geometry,cfg)
            assert len(poses)==rec['num_selected']<=manifest['arguments']['k']
            density=np.stack([provider.mask(p) for p in poses]).sum(axis=0)
            assert rec['target_count']==len(density)
            assert rec['covered_count']==int((density>0).sum())
            assert rec['overlap_target_count']==int((density>1).sum())
            assert rec['uncovered_count']==int((density==0).sum())
            assert np.isclose(rec['coverage_pct'],100*(density>0).mean(),atol=1e-7,rtol=0)
            with np.load(baseline/f'{rec["method"]}_coverage.npz') as data:
                np.testing.assert_array_equal(data['density'],density)
                np.testing.assert_array_equal(data['target_points'],provider.targets)
            baseline_runs+=1
        assert records['milp']['coverage_pct']>=records['greedy']['coverage_pct']-1e-7
        print(f'Validated {plan["key"]}',flush=True)
    rows,_=collect(root,manifest)
    with (root/'trials.csv').open() as stream:
        saved=list(csv.DictReader(stream))
    assert len(rows)==len(saved)==len(manifest['plans'])*(5+3*len(manifest['arguments']['seeds']))
    for actual,stored in zip(rows,saved):
        assert actual['method']==stored['method']
        assert actual['result_file']==stored['result_file']
        assert np.isclose(actual['coverage_pct'],float(stored['coverage_pct']),atol=1e-9,rtol=0)
    expected=aggregate(rows)
    with (root/'summary.csv').open() as stream:
        saved=list(csv.DictReader(stream))
    assert len(expected)==len(saved)==len(manifest['plans'])*8
    for actual,stored in zip(expected,saved):
        for key,value in actual.items():
            if value is None:
                assert stored[key]==''
            elif isinstance(value,(int,float)):
                assert np.isclose(value,float(stored[key]),atol=1e-9,rtol=0)
            else:
                assert value==stored[key]
    result=dict(status='passed',maps=len(manifest['plans']),mappo_runs=refinement_runs,
        baseline_runs=baseline_runs,raw_rows=len(rows),summary_rows=len(expected),
        command=[sys.executable,*sys.argv],validator_sha256=digest(__file__),
        scope='frozen prototypes, all refinement poses/logs/policies, direct baseline density, budgets, CSV/SD parity')
    save(root/'final_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    validate(parser.parse_args().root)

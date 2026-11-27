"""Re-evaluate saved refinement results without training or full visibility cache.

Usage: .venv/bin/python tests/validate_prototype_refinement.py RESULTS_ROOT
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_offline import digest, save, width_scenario
from prototype_refine import METHODS
from prototype_compatible import allocation
from prototype_transfer import prepare_relative_placements
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, get_lidar_profile, load_config
from src.methods.local_mappo import validate_poses


def validate(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] not in ('complete','partial'):
        raise ValueError('Experiment has not finished successfully')
    checked, unavailable = [], []
    for item in manifest['scenarios']:
        sid = item['scenario']
        directory = root/f's{sid}'
        report = json.loads((directory/'report.json').read_text())
        config = load_config(directory/'config.yaml')
        geometry = MapGeometry(width_scenario(report['held_out_width_m'],sid))
        sources = []
        for label in ('low','high'):
            library = directory/'library'/label
            descriptor = json.loads((library/'geometry.json').read_text())
            width = descriptor['geometry']['scenario']['road_width']
            source_geometry = MapGeometry(width_scenario(width,sid))
            source_poses = np.load(library/'poses.npy',allow_pickle=False)
            source_result = json.loads((library/'result.json').read_text())
            validate_poses(source_poses,source_geometry,config)
            sources.append((source_poses,source_geometry,source_result))
        if sources[1][2].get('installation_allocation') is not None:
            quotas = {(q['branch_id'],q['installation_line_id']):q['count']
                      for q in sources[1][2]['installation_allocation']}
            assert allocation(sources[0][0],sources[0][1]) == quotas
            assert allocation(sources[1][0],sources[1][1]) == quotas
        reconstructed, metadata = prepare_relative_placements(
            sources[0][0],sources[1][0],sources[0][1],sources[1][1],geometry,config)
        assert metadata['status'] == report['interpolation']['status']
        for name,poses in reconstructed.items():
            np.testing.assert_array_equal(poses,np.load(directory/f'{name}.npy',allow_pickle=False))
        provider = MeshVisibility(geometry.create_environment_mesh(),
            geometry.target_points(config.target['coverage_z_slices']),get_lidar_profile(config.lidar_profile,config))
        if set(report['results']) != set(METHODS):
            raise ValueError('Missing or extra result slots')
        for name,record in report['results'].items():
            if record['status'] == 'unavailable':
                if 'interpolation' not in name or not record.get('reason'):
                    raise ValueError('Unexpected missing result')
                unavailable.append(f's{sid}/{name}')
                continue
            if record['status'] != 'complete':
                raise ValueError(f'Incomplete {sid}/{name}')
            poses = np.load(directory/record['poses_npy'],allow_pickle=False)
            validate_poses(poses,geometry,config)
            if len(poses) != report['budget_k']:
                raise ValueError('Unexpected sensor count')
            density = np.stack([provider.mask(p) for p in poses]).sum(axis=0)
            assert np.isclose(record['coverage_pct'],100*(density>0).mean(),atol=1e-7,rtol=0)
            assert record['covered_count'] == int((density>0).sum())
            assert record['overlap_target_count'] == int((density>1).sum())
            assert record['uncovered_count'] == int((density==0).sum())
            assert record['target_count'] == len(density)
            with np.load(directory/f'{name}_coverage.npz') as data:
                np.testing.assert_array_equal(data['density'],density)
                np.testing.assert_array_equal(data['target_points'],provider.targets)
            if name.endswith('_mappo'):
                meta = record['refinement']
                initial = np.load(directory/name/'initial.npy')
                np.testing.assert_array_equal(initial,np.load(directory/f'{name.removesuffix("_mappo")}.npy'))
                assert np.all(np.linalg.norm(poses[:,:2]-initial[:,:2],axis=1) <= meta['local_radius_m']+1e-5)
                assert record['coverage_pct'] >= meta['initial_coverage_pct']-1e-7
                assert not meta['full_visibility_cache'] and not meta['reused_checkpoint']
                assert meta['ppo_updates'] == len(meta['ppo_diagnostics']) > 0
                for update in meta['ppo_diagnostics']:
                    assert all(np.isfinite(v) for v in update.values())
                with np.load(directory/name/'train_log.npz') as data:
                    np.testing.assert_array_equal(data['step'],np.arange(1,meta['training_steps']+1))
                    assert data['eval_step'][-1] == meta['training_steps']
                    assert all(np.isfinite(v).all() for v in data.values())
                policy = np.load(directory/name/'final_policy.npy')
                policy_density = np.stack([provider.mask(p) for p in policy]).sum(axis=0)
                assert np.isclose(meta['final_policy_coverage_pct'],100*(policy_density>0).mean(),atol=1e-7,rtol=0)
            checked.append(f's{sid}/{name}')
    assert len(checked) == manifest['completed_results']
    assert len(unavailable) == manifest['unavailable_results']
    assert len(checked)+len(unavailable) == manifest['expected_result_slots']
    result = dict(status='passed',checked=len(checked),unavailable=len(unavailable),
                  checked_results=checked,unavailable_results=unavailable,
                  command=[sys.executable,*sys.argv],validator_sha256=digest(__file__),
                  scope='prototype allocation, reconstructed transfer/interpolation, saved poses, '
                        'direct coverage/density, local radius, PPO logs, final policy coverage')
    save(root/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    validate(parser.parse_args().root)

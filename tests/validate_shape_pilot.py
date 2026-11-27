"""Re-evaluate shape-pilot deployments without training or solving."""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_offline import digest, save
from src.core.visibility import MeshVisibility
from src.environment import MapGeometry, Scenario, get_lidar_profile, load_config
from src.methods.local_mappo import validate_poses


def validate(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Expected a completed experiment')
    report = dict(status='running', command=[sys.executable, *sys.argv],
                  validator_sha256=digest(__file__), checked=[])
    save(root/'validation.json', report)
    try:
        for path, expected in {**manifest['source_hashes'], **manifest['prototype_hashes']}.items():
            if digest(path) != expected:
                raise ValueError(f'Source/input changed: {path}; use the recorded snapshot')
        for job in manifest['jobs']:
            directory = root/job['key']
            rec = json.loads((directory/'result.json').read_text())
            if rec['status'] != 'complete':
                raise ValueError(f'Incomplete job: {job["key"]}')
            desc = json.loads((directory/'geometry.json').read_text())
            geometry = MapGeometry(Scenario(**desc['geometry']['scenario']))
            cfg = load_config(directory/'config.yaml')
            poses = validate_poses(np.load(directory/'deployment.npy', allow_pickle=False), geometry, cfg)
            targets = geometry.target_points(cfg.target['coverage_z_slices'])
            provider = MeshVisibility(geometry.create_environment_mesh(), targets,
                                      get_lidar_profile(cfg.lidar_profile, cfg))
            density = np.stack([provider.mask(p) for p in poses]).sum(axis=0)
            with np.load(directory/'coverage.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['targets'], targets)
                np.testing.assert_array_equal(saved['density'], density)
            assert rec['covered_count'] == int((density > 0).sum())
            assert rec['target_count'] == len(targets)
            assert np.isclose(rec['coverage_pct'], 100*(density > 0).mean(), atol=1e-8, rtol=0)
            assert len(poses) == rec['num_selected'] <= manifest['arguments']['k']
            if rec['selection']:
                assert rec['coverage_pct'] + 1e-6 >= rec['selection']['coverage_pct']
                initial = np.load(directory/'initial.npy', allow_pickle=False)
                assert np.all(np.linalg.norm(poses[:, :2] - initial[:, :2], axis=1)
                              <= manifest['arguments']['radius'] + 1e-5)
            if job['method'] == 'mappo':
                with np.load(directory/'refinement/train_log.npz', allow_pickle=False) as log:
                    assert len(log['step']) == rec['method_metadata']['training_steps']
                    assert all(np.isfinite(value).all() for value in log.values())
            if manifest['arguments']['save_3d']:
                assert (directory/'deployment.html').is_file()
            report['checked'].append(dict(key=job['key'], coverage_pct=rec['coverage_pct']))
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        save(root/'validation.json', report)
    print(f'{len(report["checked"])}/{len(manifest["jobs"])} direct raycast validations passed')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    validate(parser.parse_args().root)

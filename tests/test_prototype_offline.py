import sys
import unittest
import itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prototype_offline import canonical, absolute, describe, certified
from src.environment import MapGeometry, get_scenario, load_config
from src.core.candidate import Candidate, CandidateSet
from src.methods.milp import _solve_with_gurobi


class PrototypeTest(unittest.TestCase):
    def test_batched_milp_matches_full_and_brute_force(self):
        rng = np.random.default_rng(12)
        masks = rng.random((7, 11)) > 0.5
        masks[-1] = False
        candidates = CandidateSet([Candidate(i, 0, 0, 3, 0, 0, mask)
            for i, mask in enumerate(masks)], np.zeros((11, 3)), 'toy', 'toy', {})
        config = load_config()
        config.solver.update(time_limit_sec=10, mip_gap=0, threads=1)
        rewards = np.arange(1, 12, dtype=float)
        costs = np.arange(7, dtype=float) * 3
        for weighted in (False, True):
            coeff = (rewards, costs) if weighted else None
            for k in (0, 1, 2, 3):
                def score(ids):
                    covered = np.any(masks[list(ids)], axis=0)
                    return float(rewards @ covered - costs[list(ids)].sum()) if weighted else int(covered.sum())
                optimum = max(score(ids) for size in range(k+1)
                    for ids in itertools.combinations(range(7), size))
                for batch_size in (0, 4):
                    config.solver['sparse_target_batch_size'] = batch_size
                    result = _solve_with_gurobi(candidates, k, config,
                        start_ids=[0] if k else [], coefficients=coeff)
                    self.assertAlmostEqual(score(result['selected_ids']), optimum)
                    self.assertAlmostEqual(result['objective'], optimum)
                    self.assertAlmostEqual(result['best_bound'], optimum)

    def test_canonical_roundtrip(self):
        geometry=MapGeometry(get_scenario(0))
        poses=np.array([[10.25,44.25,3,90,-10],[80.25,55.75,6,300,-25]])
        mapped=canonical(poses,geometry)
        np.testing.assert_allclose(absolute(mapped,geometry),poses)
        self.assertEqual(mapped[0]['installation_line_id'],'south')
        self.assertEqual(mapped[1]['installation_line_id'],'north')

    def test_descriptor(self):
        descriptor=describe(MapGeometry(get_scenario(0)),load_config())
        self.assertEqual(descriptor['branches'][0]['width_m'],7)
        self.assertEqual(descriptor['provenance']['type'],'osm_metadata_parametric')
        self.assertEqual(len(descriptor['installation_lines']),2)
        for sid in (1,2):
            geometry = MapGeometry(get_scenario(sid))
            poses = np.array([[20.25,44.25,3,90,-20],[55.75,20.25,4,180,-15]])
            np.testing.assert_allclose(absolute(canonical(poses,geometry),geometry),poses)
            self.assertEqual(len(describe(geometry,load_config())['branches']), sid+2)

    def test_optimality_requires_bound(self):
        self.assertTrue(certified(dict(solver_status_code=2,objective=100,best_bound=100),100))
        self.assertFalse(certified(dict(solver_status_code=2,objective=100,best_bound=104),100))
        self.assertFalse(certified(dict(solver_status_code=9,objective=100,best_bound=100),100))
        self.assertFalse(certified(dict(solver_status_code=2,objective=None,best_bound=None),100))


if __name__=='__main__':
    unittest.main(verbosity=2)

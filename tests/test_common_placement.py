import hashlib
import itertools
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.core.candidate import Candidate, CandidateSet
from src.core.models import PlacementProblem, SensorPose
from src.core.objective import ObjectiveWeights
from src.core.evaluation import evaluate_selection
from src.environment import load_config, get_scenario, evaluate_coverage
from src.methods.greedy import solve_problem as greedy
from src.methods.milp import solve_problem as milp, build_oriented_candidates


class CommonPlacementTests(unittest.TestCase):
    def problem(self, weights, importance=None, costs=None):
        masks = [[True, True, False], [False, True, True], [False, False, True]]
        candidates = [Candidate(i, i, 0, 1, 0, 0, np.array(m)) for i, m in enumerate(masks)]
        return PlacementProblem(CandidateSet(candidates, np.zeros((3, 3)), '', '', {}), 2,
                                weights, importance, costs)

    def test_weighted_milp_matches_brute_force(self):
        config = load_config()
        config.solver['solver'] = 'pulp'
        for weights in (ObjectiveWeights(), ObjectiveWeights(importance_weight=2, overlap_penalty=3,
                         sensor_count_penalty=0.2, installation_cost_weight=1)):
            problem = self.problem(weights, [1, 1, 5], [2, 0, 0])
            expected = max(evaluate_selection(problem, s).objective_score
                           for k in range(3) for s in itertools.combinations(range(3), k))
            result = milp(problem, config)
            self.assertAlmostEqual(result.objective_score, expected)

    def test_weights_change_selection(self):
        first = greedy(self.problem(ObjectiveWeights()))
        weighted = self.problem(ObjectiveWeights(importance_weight=10), [1, 1, 5])
        self.assertEqual(first.selected_ids[0], 0)
        self.assertEqual(greedy(weighted).selected_ids[0], 1)
        self.assertLessEqual(greedy(self.problem(ObjectiveWeights(sensor_count_penalty=100))).objective_score, 0)

    def test_pose_and_unsupported_objectives(self):
        self.assertEqual(SensorPose(1, 2, 3, 90, -10).to_legacy_pose5(), [1, 2, 3, 90, -10])
        with self.assertRaises(ValueError):
            SensorPose(1, 2, 3, 90, -10, 5).to_legacy_pose5()
        with self.assertRaises(NotImplementedError):
            ObjectiveWeights(robustness_weight=1)

    def test_robot_self_occlusion(self):
        import open3d as o3d
        from src.core.visibility import MeshVisibility
        from src.environment import LidarProfile
        mesh = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
        provider = MeshVisibility(mesh, [[2, .5, .5], [-2, .5, .5]], LidarProfile('test', 360, 180, 10))
        np.testing.assert_array_equal(provider.mask([-0.1, .5, .5, 0, 0]), [False, True])

    @unittest.skipUnless('--roadside-parity' in sys.argv, 'Explicit full candidate parity check')
    def test_roadside_golden(self):
        config = load_config()
        scenario = get_scenario(0)
        candidates = build_oriented_candidates(scenario, config.lidar_profile, config)
        self.assertEqual(len(candidates.target_points), 7200)
        self.assertEqual(len(candidates.candidates), 7985)
        digest = hashlib.sha256(b''.join(np.packbits(c.covered).tobytes() for c in candidates.candidates)).hexdigest()
        self.assertEqual(digest, 'a4e017d77f79700e477071ce1b41f4e05e70a3a1713156669565d47875ed333d')
        result = greedy(PlacementProblem(candidates, 1))
        self.assertEqual(result.selected_ids, [2427])
        np.testing.assert_array_equal(result.poses, [[30.25, 44.25, 3., 85., -20.]])
        evaluation = evaluate_coverage(result.poses, scenario, config.lidar_profile, config)
        self.assertEqual(evaluation.covered_count, 4205)
        self.assertAlmostEqual(result.coverage_pct, evaluation.coverage_pct)


if __name__ == '__main__':
    unittest.main(argv=[arg for arg in sys.argv if arg != '--roadside-parity'], verbosity=2)

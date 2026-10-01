import sys
import unittest
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from placement_ui import app, state


class WorkbenchTest(unittest.TestCase):
    def setUp(self):
        state.clear()
        self.client = app.test_client()

    def test_weights_and_roi_reuse_visibility(self):
        loaded = self.client.post('/api/load', json={'mode': 'robot'})
        self.assertEqual(loaded.status_code, 200)
        cached = state['problem'].candidates
        baseline = self.client.post('/api/solve', json={'budget_k': 1, 'weights': {'importance_weight': 1}})
        self.assertEqual(baseline.json['selected_ids'], [0])
        rear = self.client.post('/api/solve', json={'budget_k': 1, 'weights': {'importance_weight': 1},
                 'region': {'bounds': [-3, -1, -1, 2], 'factor': 10}})
        self.assertEqual(rear.json['selected_ids'], [1])
        self.assertIs(state['problem'].candidates, cached)
        denied = self.client.post('/api/solve', json={'budget_k': 1, 'weights': {}, 'method': 'mappo', 'mappo_steps': 0})
        self.assertEqual(denied.status_code, 400)

    def test_fresh_mappo_training(self):
        self.client.post('/api/load', json={'mode': 'robot'})
        payload = dict(budget_k=2, weights={'importance_weight': 1}, method='mappo', mappo_steps=64, seed=5)
        first = self.client.post('/api/solve', json=payload)
        self.assertEqual(first.status_code, 200, first.json)
        meta = first.json['metadata']
        self.assertTrue(meta['parameters_changed'])
        self.assertEqual(meta['steps'], 64)
        self.assertEqual(meta['ppo_updates'], 1)
        self.assertFalse(meta['pretrained'])
        from src.core.evaluation import evaluate_selection
        from src.core.models import PlacementProblem
        from src.core.objective import ObjectiveWeights
        base = state['problem']
        problem = PlacementProblem(base.candidates, 2, ObjectiveWeights(importance_weight=1), base.importance, base.costs)
        evaluated = evaluate_selection(problem, first.json['selected_ids'])
        self.assertAlmostEqual(evaluated.objective_score, first.json['objective_score'])
        second = self.client.post('/api/solve', json=payload)
        self.assertEqual(first.json['metadata']['history'], second.json['metadata']['history'])
        self.assertEqual(first.json['metadata']['update_history'], second.json['metadata']['update_history'])
        payload['weights'] = {'coverage_weight': 0, 'sensor_count_penalty': 100}
        penalized = self.client.post('/api/solve', json=payload)
        self.assertEqual(penalized.json['selected_ids'], [])

    def test_compute_controls(self):
        self.client.post('/api/load', json={'mode': 'robot'})
        response = self.client.post('/api/solve', json=dict(method='greedy', budget_k=2, weights={}, greedy_steps=0))
        self.assertEqual(response.json['selected_ids'], [])
        for key, value in [('budget_k', 1.5), ('greedy_steps', -1), ('budget_k', True)]:
            payload = dict(method='greedy', budget_k=2, weights={})
            payload[key] = value
            self.assertEqual(self.client.post('/api/solve', json=payload).status_code, 400)
        from unittest.mock import patch
        from src.core.evaluation import evaluate_selection
        def fake_solver(problem, config):
            self.assertEqual(config.solver['time_limit_sec'], 17)
            self.assertEqual(config.solver['mip_gap'], 0.12)
            return evaluate_selection(problem, [])
        with patch('src.methods.milp.solve_problem', side_effect=fake_solver):
            response = self.client.post('/api/solve', json=dict(method='milp', budget_k=2, weights={}, milp_seconds=17, milp_gap=0.12))
            self.assertEqual(response.status_code, 200)

    def test_custom_mesh_validation(self):
        payload = json.loads(Path('configs/mesh_example.json').read_text())
        response = self.client.post('/api/load', json={'mode': 'custom', 'data': payload})
        self.assertEqual(response.status_code, 200)
        payload['triangles'][0][0] = 999
        response = self.client.post('/api/load', json={'mode': 'custom', 'data': payload})
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Independent small-state checks and a frozen-input read-only integration test."""
from contextlib import redirect_stdout
import io
from itertools import combinations, product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import connected_failure_diagnostics as diagnostic
from src.methods.connected_diagnostics import (PairTable, best_assignment, coordinate_neighbors,
    labelled_reachability, omission_witness, pair_table, search_diagnostic)
from src.roadside.corridor_requirements import CorridorRequirements


def artificial_table(n, entries):
    scores = np.full((n, n), -np.inf)
    valid = np.zeros((n, n), dtype=bool)
    for (a, b), value in entries.items():
        scores[a, b] = scores[b, a] = value
        valid[a, b] = valid[b, a] = True
    return PairTable(scores, valid, np.zeros((n, n)), np.zeros((n, n)), np.zeros(n))


class ConnectedDiagnosticsTests(unittest.TestCase):
    def test_repeated_input_reads_cannot_replace_the_initial_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'example.json'
            path.write_text('{}')
            inputs = diagnostic.Inputs(Path(tmp))
            inputs.json('example.json')
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                inputs.json('example.json')

    def test_pair_table_checks_every_pair_against_shared_evaluator(self):
        canonical = np.array(list(product([2, 6], [-6, 6], [3, 5], [0, 45], [-10])), dtype=float)
        poses = canonical.copy()
        coordinates = np.column_stack([np.linspace(.5, 9.5, 12), np.ones(12), np.full(12, 1.5)])
        masks = np.random.default_rng(54).random((len(poses), 12)) < .7
        spec = dict(schema='constrained_coverage_v2', budget_k=2,
            weights=dict(coverage_weight=1.3, importance_weight=.2, overlap_penalty=.15,
                         sensor_count_penalty=.1, installation_cost_weight=.2),
            cost=dict(base_per_sensor=10, per_height_m=1),
            constraints=dict(max_installation_cost=28, min_xy_separation_m=3,
                max_sensors_per_pole=1, mounting_z_range_m=[3, 5], min_coverage_pct=30,
                max_overlap_ratio=.8, min_single_failure_coverage_pct=20))
        profile = dict(name='example', interface=dict(min_coverage_pct=50, min_overlap_ratio=.15),
            forbidden_installation=[dict(station_range_m=[0, 3], side='left')])
        req = CorridorRequirements(spec, coordinates, coordinates, canonical, poses, profile,
            length_m=10, interface_half_length_m=2)
        table = pair_table(poses, masks, spec, req.allowed, req.interface_mask, profile['interface'])
        for pair in combinations(range(len(poses)), 2):
            ids = list(pair)
            expected = req.evaluate(poses[ids], masks[ids].sum(axis=0), sensor_masks=masks[ids])
            self.assertEqual(bool(table.feasible[pair]), expected.feasible)
            self.assertAlmostEqual(table.scores[pair], expected.objective_score)
            self.assertEqual(table.scores[pair], table.scores[pair[::-1]])
        self.assertFalse(np.diag(table.feasible).any())
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            pair_table(poses, masks, {**spec, 'constraints': dict(service_hubs=[])},
                       req.allowed, req.interface_mask, {})

    def test_omission_checks_all_tied_optima_not_one_reference_pose(self):
        table = artificial_table(4, {(0, 3): 10, (1, 2): 10, (0, 2): 8, (1, 3): 8})
        modules = np.array([0, 0, 1, 1])
        present = omission_witness(table, [1, 2], modules)
        self.assertEqual(present['optimal_pair_count'], 2)
        self.assertEqual(present['minimum_target_candidate_additions'], 0)
        self.assertEqual(present['witness_ids'], [1, 2])
        self.assertEqual(present['essential_candidate_ids'], [])
        missing = omission_witness(table, [0, 2], modules)
        self.assertEqual(missing['minimum_target_candidate_additions'], 1)
        self.assertEqual(missing['missing_ids'], [3])
        absent = best_assignment(table, [0], [0])
        self.assertIsNone(absent['score'])

    def graph_fixture(self, middle=8):
        canonical = np.array([[s, d, 3, 0, -10] for d in (-6, 6) for s in (2, 6, 10)], dtype=float)
        table = artificial_table(6, {(0, 3): 10, (1, 3): middle, (2, 3): 12})
        return canonical, table

    def test_widest_path_matches_independent_threshold_enumeration(self):
        for middle in (8, 10):
            canonical, table = self.graph_fixture(middle)
            neighbors = coordinate_neighbors(canonical)
            actual = labelled_reachability(table, [0, 1, 2], [3], neighbors, [0, 3])
            # Independently test every possible objective threshold on the chain.
            achievable = []
            for level in sorted(set(table.scores[table.feasible])):
                seen, queue = {0}, [0]
                if table.scores[0, 3] < level:
                    continue
                while queue:
                    cid = queue.pop()
                    for nxt in (cid-1, cid+1):
                        if (0 <= nxt < 3 and nxt not in seen and table.feasible[nxt, 3]
                                and table.scores[nxt, 3] >= level):
                            seen.add(nxt)
                            queue.append(nxt)
                if 2 in seen:
                    achievable.append(float(level))
            self.assertEqual(actual['bottleneck_score'], max(achievable))
            self.assertEqual(actual['minimum_objective_dip'], 10-middle)
            self.assertEqual(actual['witness_path'], [[0, 3], [1, 3], [2, 3]])

    def test_side_anchor_disconnect_and_search_are_different_restrictions(self):
        canonical, table = self.graph_fixture()
        table.scores[1, 2] = table.scores[2, 1] = 15
        table.feasible[1, 2] = table.feasible[2, 1] = True
        full = best_assignment(table, range(6), range(6))
        limited = search_diagnostic(table, canonical, [0, 3], [0, 3], 4)
        self.assertEqual(full['score'], 15)
        self.assertEqual(limited['fixed_side_optimum']['score'], 12)
        self.assertEqual(limited['anchor_optimum']['score'], 10)
        allowed = search_diagnostic(table, canonical, [0, 3], [0, 3], 8)
        self.assertEqual(allowed['feasible_component_optimum']['score'], 12)
        self.assertEqual(allowed['one_coordinate_best_j'], 10)
        self.assertEqual(allowed['one_agent_relocation_optimum']['score'], 12)
        blocked = artificial_table(6, {(0, 3): 10, (2, 3): 12})
        broken = search_diagnostic(blocked, canonical, [0, 3], [0, 3], 8)
        self.assertEqual(broken['anchor_optimum']['score'], 12)
        self.assertEqual(broken['feasible_component_optimum']['score'], 10)
        with self.assertRaisesRegex(ValueError, 'different feasible action components'):
            search_diagnostic(blocked, canonical, [0, 3], [2, 3], 8)

    def test_joint_coordination_barrier_is_not_one_agent_relocation(self):
        canonical, _ = self.graph_fixture()
        table = artificial_table(6, {(0, 3): 10, (1, 3): 8, (0, 4): 8, (1, 4): 15})
        result = search_diagnostic(table, canonical, [0, 3], [0, 3], 4)
        self.assertEqual(result['one_agent_relocation_optimum']['score'], 10)
        self.assertEqual(result['feasible_component_optimum']['score'], 15)
        self.assertEqual(result['feasible_component_optimum']['minimum_objective_dip'], 2)

    def test_marginal_response_includes_overlap_cost_and_is_conditional(self):
        masks = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool)
        costs = np.array([13., 13., 15.])
        weights = dict(overlap_penalty=.05, installation_cost_weight=.2)
        value = diagnostic.candidate_response(0, 1, masks, costs, weights)
        self.assertEqual(value['unique_count'], 1)
        self.assertAlmostEqual(value['marginal_j'], 1-.05-2.6)
        other = diagnostic.candidate_response(0, 2, masks, costs, weights)
        self.assertEqual(other['unique_count'], 0)
        self.assertAlmostEqual(other['marginal_j'], -.1-2.6)

    def test_frozen_input_integration_never_queries_visibility_and_rejects_tampering(self):
        base = json.loads((ROOT/'configs/connected_corridor_pilot.json').read_text())
        base['geometry'].update(length_m=40, bend_length_m=12, mesh_step_m=1)
        base['candidate'].update(station_step_m=10, pitches_deg=[-10])
        base['interface_half_length_m'] = 5
        base['requirements']['constraints'] = dict(max_installation_cost=28, min_xy_separation_m=3)
        study = dict(schema='connected_constraint_holdout_v1', alternatives_per_site=2,
            targets=[dict(name='w8', road_width_m=8, turn_deg=30, bend_length_m=12),
                     dict(name='w10', road_width_m=10, turn_deg=30, bend_length_m=12)],
            profiles=[dict(name='base', interface={}, forbidden_installation=[])])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp/'base.json').write_text(json.dumps(base))
            (tmp/'study.json').write_text(json.dumps(study))
            root, output = tmp/'study', tmp/'analysis'
            run = subprocess.run([sys.executable, str(ROOT/'connected_constraint_study.py'),
                '--base-suite', str(tmp/'base.json'), '--study', str(tmp/'study.json'),
                '--output', str(root), '--execute'], cwd=ROOT, capture_output=True, text=True, timeout=120)
            logs = ''.join(p.read_text() for p in (root/'worker.log', root/'audit.log') if p.exists())
            self.assertEqual(run.returncode, 0, run.stderr+logs)
            before = {str(p.relative_to(root)): diagnostic.digest(p) for p in root.rglob('*') if p.is_file()}
            with patch('src.core.visibility.MeshVisibility', side_effect=AssertionError('Unexpected new raycast')) as visibility:
                with redirect_stdout(io.StringIO()) as stream:
                    diagnostic.main(['--root', str(root), '--output', str(output)])
                self.assertFalse(output.exists())
                self.assertFalse(json.loads(stream.getvalue())['writes_files'])
                with redirect_stdout(io.StringIO()):
                    diagnostic.generate(root, output)
                visibility.assert_not_called()
            result = json.loads((output/'manifest.json').read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['diagnostic_rows'], 4)
            self.assertEqual(before, {str(p.relative_to(root)): diagnostic.digest(p) for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):
                diagnostic.generate(root, output)
            with self.assertRaises(ValueError):
                diagnostic.generate(root, root)
            path = root/'source_pools.json'
            changed = json.loads(path.read_text())
            changed['best_source_id'] = -1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'source pool hash'):
                diagnostic.generate(root, tmp/'failed')
            self.assertFalse((tmp/'failed').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

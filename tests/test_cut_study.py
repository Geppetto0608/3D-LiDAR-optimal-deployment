"""Aggregation keeps solver bound, primal-heuristic result, and missing jobs separate."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_cut_study import summarize
from prototype_offline import save


class CutStudyTests(unittest.TestCase):
    def test_aggregation_does_not_count_pending_as_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root/'done'
            out.mkdir()
            save(out/'result.json', dict(prior_coverage_pct=80., coverage_pct=99., objective=99,
                certified_upper_bound=100., certified_gap=1/99, discrete_optimal=False,
                solver_best_bound=100., bound_source='min(solver_bound, target_count)', solver_status=9,
                swap=dict(coverage_count=98, elapsed_sec=1.), solver_sec=2., archive_load_sec=.1,
                total_sec=3., cuts=4))
            jobs = [dict(status='complete',output=str(out),scenario_id=1,k=3),dict(status='pending')]
            summarize(root,jobs)
            value = json.loads((root/'summary.json').read_text())
            self.assertEqual(value['completed'],1)
            self.assertEqual(value['expected'],2)
            self.assertEqual(value['results'][0]['swap_objective'],98)
            self.assertFalse(value['results'][0]['discrete_optimal'])


if __name__ == '__main__':
    unittest.main()

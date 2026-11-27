"""Artifact-only summary tests; no optimization or training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype_junctions import report


class JunctionSummaryTest(unittest.TestCase):
    def test_missing_results_stay_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report(root)
            data = json.loads((root/'summary.json').read_text())
            self.assertEqual(data['offline'], [])
            self.assertEqual(data['transfers'], [])
            self.assertIn('TBD', (root/'SUMMARY.md').read_text())

    def test_partial_report_keeps_unavailable_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer = root/'review'
            transfer.mkdir()
            payload = dict(status='partial', interpolation=dict(status='unavailable'),
                results={'transfer_low': {'coverage_pct': 12.5}})
            (transfer/'report.json').write_text(json.dumps(payload))
            report(root, transfer_s1='review')
            data = json.loads((root/'summary.json').read_text())
            self.assertEqual(data['transfers'][0]['data'], payload)
            self.assertIn('review/report.json', data['sources'])
            text = (root/'SUMMARY.md').read_text()
            self.assertIn('N/A', text)
            self.assertIn('12.500000', text)
            self.assertNotIn('interpolation.html', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)

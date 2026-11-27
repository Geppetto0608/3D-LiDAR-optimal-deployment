"""Publication is an excerpt of audited data, not a new experiment or validation."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_research_summary as export


class ExportTest(unittest.TestCase):
    def fixture(self, root):
        report = root / 'private_report'
        report.mkdir()
        for name in export.TABLES:
            (report / name).write_bytes(b'method,value\r\nfixture,1\r\n')
        audit = {key: 1 for key in export.VALIDATION_FIELDS}
        audit.update(status='passed', direct_raycast=True, raw_inputs_unchanged=True,
                     rerun_optimization=False, root='/private/host/path',
                     input_hashes={'private': 'omitted'},
                     summary={key: 1 for key in export.SUMMARY_FIELDS})
        (report / 'validation_summary.json').write_text(json.dumps(audit))
        return report

    def test_plan_export_check_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, output = self.fixture(root), root / 'public'
            self.assertEqual(export.export(report, output)['status'], 'plan_only')
            self.assertFalse(output.exists())
            export.export(report, output, execute=True)
            self.assertEqual(export.export(report, output, check=True)['status'], 'checked')
            for name in export.TABLES:
                self.assertEqual((report / name).read_bytes(), (output / name).read_bytes())
            excerpt = json.loads((output / 'validation_excerpt.json').read_text())
            self.assertNotIn('root', excerpt)
            self.assertNotIn('input_hashes', excerpt)
            with self.assertRaises(FileExistsError):
                export.export(report, output, execute=True)
            (output / export.TABLES[0]).write_text('changed')
            with self.assertRaises(ValueError):
                export.export(report, output, check=True)

    def test_rejects_unverified_or_path_bearing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.fixture(root)
            (report / export.TABLES[0]).write_text('method,path\nfixture,/home/private/user\n')
            with self.assertRaises(ValueError):
                export.export(report, root / 'public')
            audit = json.loads((report / 'validation_summary.json').read_text())
            audit['status'] = 'failed'
            (report / 'validation_summary.json').write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                export.export(report, root / 'public')
            self.assertFalse((root / 'public').exists())


if __name__ == '__main__':
    unittest.main()

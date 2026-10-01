"""Browser integration: requires Playwright Chromium and a Gurobi license."""
import json
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server
from placement_ui import app


class ComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server('127.0.0.1', 0, app, threaded=True)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True, args=['--no-sandbox'])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width': 1440, 'height': 1000})
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.page.goto(f'http://127.0.0.1:{self.server.server_port}')
        self.page.wait_for_function("document.getElementById('status').textContent.includes('환경 준비 완료')", timeout=60000)

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors, [])

    def test_nine_real_combinations_and_export(self):
        p = self.page
        p.check('input[name=method][value=milp]')
        p.check('input[name=method][value=mappo]')
        p.select_option('#budgetMode', 'range')
        p.fill('#budget', '3')
        p.fill('#budgetEnd', '5')
        p.fill('#mappo_steps', '32')
        p.fill('#milp_seconds', '2')
        p.click('#solve')
        p.wait_for_function("document.getElementById('status').textContent.includes('비교 완료')", timeout=120000)
        self.assertTrue(p.evaluate("rows.length===9 && rows.every(r=>r.status==='완료')"))
        for method in ['greedy', 'milp', 'mappo']:
            self.assertIn(method.upper(), p.locator('#'+method+'Options legend').inner_text())
        self.assertEqual(p.evaluate("document.getElementById('comparison').data[0].type"), 'bar')
        p.click('#clearResults')
        p.get_by_role('checkbox', name='GREEDY K=3 그래프', exact=True).check()
        p.get_by_role('checkbox', name='MAPPO K=5 그래프', exact=True).check()
        self.assertEqual(p.evaluate("document.getElementById('comparison').data.flatMap(t=>t.y).filter(v=>v!==null).length"), 2)
        self.assertEqual(p.evaluate('rows.length'), 9)
        p.select_option('#chartType', 'scatter')
        self.assertEqual(p.evaluate("document.getElementById('comparison').data[0].type"), 'scatter')
        p.click('#selectAllResults')
        p.locator('input[name=resultK][value="4"]').uncheck()
        self.assertEqual(p.evaluate("document.getElementById('comparison').data.flatMap(t=>t.y).filter(v=>v!==null).length"), 6)
        p.locator('input[name=resultMethod][value=milp]').uncheck()
        self.assertEqual(p.evaluate("document.getElementById('comparison').data.flatMap(t=>t.y).filter(v=>v!==null).length"), 4)
        p.click('#selectAllResults')
        self.assertTrue(p.evaluate('rows.every(({result:r})=>Math.abs(r.coverage_pct+r.uncovered_pct-100)<1e-8)'))
        p.locator('input[name=viewResult]').first.check()
        self.assertIn('GREEDY', p.locator('#selectedTitle').inner_text())
        for metric in ['overlap_pct', 'uncovered_pct', 'coverage_pct']:
            p.select_option('#chartMetric', metric)
            self.assertEqual(p.evaluate("document.getElementById('comparison').data.length"), 3)
        with p.expect_download() as download:
            p.click('#download')
        bundle = json.loads(Path(download.value.path()).read_text())
        self.assertEqual(len(bundle['runs']), 9)
        for run in bundle['runs']:
            r = run['result']
            self.assertEqual(r['weights']['coverage_weight'], bundle['settings']['weights']['coverage_weight'])
            self.assertEqual(r['uncovered_count'], r['density'].count(0))
            self.assertEqual(r['overlap_count'], sum(d >= 2 for d in r['density']))
        p.set_viewport_size({'width': 390, 'height': 844})
        p.wait_for_timeout(500)
        self.assertTrue(p.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        p.locator('#coverage_weight').fill('2')
        self.assertTrue(p.locator('#download').is_disabled())
        self.assertEqual(p.locator('#comparisonRows tr').count(), 0)

    def test_failure_is_not_a_zero_score_and_single_budget(self):
        p = self.page
        def fail_milp(route):
            if route.request.post_data_json['method'] == 'milp':
                route.fulfill(status=400, content_type='application/json', body='{"error":"solver unavailable"}')
            else:
                route.continue_()
        p.route('**/api/solve', fail_milp)
        p.check('input[name=method][value=milp]')
        p.fill('#budget', '2')
        p.click('#solve')
        p.wait_for_function("document.getElementById('status').textContent.includes('비교 완료')")
        self.assertEqual(p.locator('#comparisonRows tr').count(), 2)
        self.assertTrue(p.evaluate("rows[1].status==='실패' && !rows[1].result"))
        self.assertIsNone(p.evaluate("document.getElementById('comparison').data[1].y[0]"))


if __name__ == '__main__':
    unittest.main(verbosity=2)

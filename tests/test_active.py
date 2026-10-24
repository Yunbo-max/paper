"""Query-by-committee tests with authentically recorded sandbox-run metadata.

The candidate replay subprocess is mocked; oracle observations use a live localhost
HTTP server. These tests do NOT claim a real Docker or GPU run.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paperrev.active import _distance, select_probes, evaluate_selection
from paperrev.blackbox import capture_api, replay_candidate
from paperrev.workbench import Workbench, hash_file, read_json, sha256, write_once


class LocalOracle(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        answer = json.dumps({'answer': {'value': 2 * request['x']}}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)


class ActiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), LocalOracle)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.wb = Workbench.create(self.root / 'lab', 'Query-by-committee')
        self.pool = self.root / 'pool.jsonl'
        self.data = [{'id': 'q' + str(i), 'input': {'x': i}, 'parent_id': 'P' + str(i // 2)}
                     for i in range(6)]
        self.save_pool()
        self.evids = []

    def save_pool(self):
        self.pool.write_text(''.join(json.dumps(z, sort_keys=True) + '\n' for z in self.data))

    def add_model(self, name: str, function, *, alter_run=None):
        folder = self.wb.root / 'candidates' / name
        (folder / 'src').mkdir(parents=True)
        code = 'def predict(record):\n    return {"value": 2*record["x"]}\n'
        (folder / 'src/reproduction.py').write_text(code)
        write_once(folder / 'candidate.json', {
            'schema': 'paperrev/candidate-v2', 'candidate_id': name,
            'files': {'src/reproduction.py': sha256(code.encode('utf-8'))},
        })
        rows = [{'id': r['id'], 'output': {'value': function(r['input']['x'])}}
                for r in self.data]
        stdout = ''.join(json.dumps(row) + '\n' for row in rows)
        fake = SimpleNamespace(returncode=0, stdout=stdout, stderr='')
        with patch('paperrev.blackbox.shutil.which', return_value='/usr/bin/docker'), \
             patch('paperrev.blackbox.subprocess.run', return_value=fake):
            result = replay_candidate(self.wb, name, self.pool, image='python:3.12-slim', execute=True)
        self.assertEqual(result['status'], 'ran')
        self.evids.append(result['evidence_id'])
        if alter_run:
            path = self.wb.root / 'runs' / (result['run_id'] + '.json')
            altered = read_json(path)
            altered.update(alter_run)
            path.write_text(json.dumps(altered))
        return result

    def two_models(self):
        a = self.add_model('A', lambda x: 2 * x)
        b = self.add_model('B', lambda x: 3 * x)
        return a, b

    def selection(self, **kw):
        return select_probes(self.wb, self.pool, self.evids, **kw)

    def capture(self, selected, limit=25):
        return capture_api(self.wb, selected['cases_path'], origin=f'http://127.0.0.1:{self.server.server_address[1]}',
                           endpoint_path='/decision', output_pointer='/answer', execute=True, limit=limit)

    def assessment(self, selected, observed):
        return evaluate_selection(self.wb, pool_file=self.pool,
                                  selection_evidence_id=selected['evidence_id'],
                                  observation_evidence_id=observed['evidence_id'])

    def test_numeric_distance_and_nested_changes(self):
        self.assertEqual(_distance({'p': [1, 2]}, {'p': [1, 2]}, 1), 0)
        self.assertEqual(_distance({'x': 1}, {'y': 1}, 1), 1)
        self.assertEqual(_distance(True, 1, 1), 1)
        self.assertEqual(_distance({'ok': False}, {'ok': True}, 1), 1)
        self.assertGreater(_distance({'a': [1, 5]}, {'a': [1, 8]}, 1), 0)
        with self.assertRaises(ValueError):
            _distance(float('nan'), 2, 1)

    def test_greedy_selects_diverse_informative_queries(self):
        self.two_models()
        result = self.selection(budget=3, max_per_parent=1)
        self.assertEqual(len(result['selected_ids']), 3)
        self.assertNotIn('q0', result['selected_ids'])
        lookup = {z['id']: z['parent_id'] for z in self.data}
        self.assertEqual(len({lookup[x] for x in result['selected_ids']}), 3)
        self.assertEqual(result['candidate_pairs'], 1)
        self.assertTrue(result['no_external_api_called'])
        self.assertEqual(self.wb.get(result['evidence_id'])['confidence'], 'derived')
        self.assertEqual(result['selection_id'], self.selection(budget=3, max_per_parent=1)['selection_id'])

    def test_pair_coverage_with_three_models(self):
        self.add_model('A', lambda x: 2*x)
        self.add_model('B', lambda x: 3*x)
        self.add_model('C', lambda x: 2*x if x < 3 else 2*x + 4)
        result = self.selection(budget=2)
        self.assertEqual(result['candidate_pairs'], 3)
        self.assertGreaterEqual(result['pairs_covered'], 2)
        self.assertEqual(len(result['selected_ids']), 2)

    def test_budget_is_a_hard_bound(self):
        self.two_models()
        selected = self.selection(budget=1)
        self.assertEqual(len(selected['selected_ids']), 1)
        self.assertEqual(self.wb.get(selected['evidence_id'])['result']['selected'], 1)

    def test_query_cost_discounts_expensive_cases(self):
        self.data[5]['query_cost'] = 1000
        self.save_pool()
        self.two_models()
        chosen = self.selection(budget=1)
        self.assertNotEqual(chosen['selected_ids'][0], 'q5')

    def test_rejects_invalid_budget_and_costs(self):
        self.two_models()
        for values in ({'budget': 0}, {'numeric_scale': 0}, {'min_distance': float('nan')}, {'max_per_parent': 0}):
            with self.assertRaises(ValueError):
                self.selection(**values)
        self.data[1]['query_cost'] = -1
        self.save_pool()
        with self.assertRaises(ValueError):
            self.selection()

    def test_equal_models_produce_no_queries(self):
        self.add_model('A', lambda x: 2*x)
        self.add_model('B', lambda x: 2*x)
        outcome = self.selection()
        self.assertEqual(outcome['status'], 'no-discriminatory-inputs')
        self.assertEqual(outcome['selected_ids'], [])
        with self.assertRaisesRegex(ValueError, 'nothing to evaluate'):
            self.assessment(outcome, {'evidence_id': self.evids[0]})

    def test_candidate_code_tampering_rejected(self):
        self.two_models()
        (self.wb.root / 'candidates' / 'A' / 'src/reproduction.py').write_text('print("tampered")')
        with self.assertRaisesRegex(ValueError, 'code changed'):
            self.selection()

    def test_run_ledger_tampering_rejected(self):
        self.add_model('A', lambda x: 2*x, alter_run={'image': 'malicious:latest'})
        self.add_model('B', lambda x: 3*x)
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.selection()

    def test_candidate_output_tampering_rejected(self):
        r, _ = self.two_models()
        Path(r['output_path']).write_text('garbage\n')
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.selection()

    def test_wrong_input_pool_rejected(self):
        self.two_models()
        self.data[0]['input']['x'] = 100
        self.save_pool()
        with self.assertRaisesRegex(ValueError, 'different input pool'):
            self.selection()

    def test_duplicate_committee_candidate_rejected(self):
        self.two_models()
        with self.assertRaises(ValueError):
            select_probes(self.wb, self.pool, [self.evids[0], self.evids[0]])

    def test_unauthorized_inferred_observation_rejected(self):
        self.two_models()
        forged = self.wb.record(kind='candidate-behavior', authority='analyst-inference', confidence='inferred',
                                subject={'candidate_id': 'Z'}, result={'status': 'ran'})
        with self.assertRaises(ValueError):
            select_probes(self.wb, self.pool, [self.evids[0], forged['evidence_id']])

    def test_oracle_resolves_finite_committee(self):
        self.two_models()
        selected = self.selection(budget=3)
        obs = self.capture(selected)
        result = self.assessment(selected, obs)
        self.assertEqual(result['coverage'], 'complete')
        by_id = {x['candidate_id']: x for x in result['candidate_assessments']}
        self.assertEqual(by_id['A']['status'], 'compatible-on-tested-inputs')
        self.assertEqual(by_id['B']['status'], 'disagreed-on-observed-inputs')
        self.assertGreater(len(by_id['B']['mismatch_ids']), 0)
        self.assertEqual(result['compatible'], 1)
        self.assertEqual(result['contradicted_on_observed'], 1)
        bundle = self.root / 'export.json'
        self.wb.export_bundle(bundle)
        self.assertTrue(Workbench.verify_bundle(bundle)['verified'])

    def test_partial_oracle_cannot_certify_survivor(self):
        self.two_models()
        selected = self.selection(budget=3)
        obs = self.capture(selected, limit=1)
        result = self.assessment(selected, obs)
        self.assertEqual(result['coverage'], 'partial')
        by_id = {x['candidate_id']: x for x in result['candidate_assessments']}
        self.assertEqual(by_id['A']['status'], 'inconclusive')
        self.assertEqual(result['compatible'], 0)

    def test_selected_file_tampering_rejected(self):
        self.two_models()
        selected = self.selection()
        Path(selected['cases_path']).write_text('something else\n')
        with self.assertRaisesRegex(ValueError, 'modified'):
            evaluate_selection(self.wb, pool_file=self.pool,
                               selection_evidence_id=selected['evidence_id'],
                               observation_evidence_id=self.evids[0])

    def test_manifest_tampering_rejected(self):
        self.two_models()
        chosen = self.selection()
        path = self.wb.root / 'selections' / (chosen['selection_id'] + '.json')
        manifest = read_json(path)
        manifest['params']['budget'] = 1000
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'modified'):
            evaluate_selection(self.wb, pool_file=self.pool,
                               selection_evidence_id=chosen['evidence_id'],
                               observation_evidence_id=self.evids[0])

    def test_oracle_source_mismatch_rejected(self):
        self.two_models()
        chosen = self.selection()
        other_capture = capture_api(self.wb, self.pool,
                                    origin=f'http://127.0.0.1:{self.server.server_address[1]}',
                                    endpoint_path='/decision', output_pointer='/answer', execute=True)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.assessment(chosen, other_capture)

    def test_observation_metadata_tampering_rejected(self):
        self.two_models()
        chosen = self.selection()
        obs = self.capture(chosen)
        meta = self.wb.root / 'observations' / (obs['capture_id'] + '.json')
        original = read_json(meta)
        original['endpoint']['path'] = '/evil'
        meta.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            self.assessment(chosen, obs)

    def test_cli_select_and_evaluate_end_to_end(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from paperrev.engine_cli import main as cli
        self.two_models()
        out = StringIO()
        with redirect_stdout(out):
            status = cli(['active-select', '--workspace', str(self.wb.root),
                          '--pool', str(self.pool),
                          '--candidate-run', self.evids[0], '--candidate-run', self.evids[1],
                          '--budget', '2'])
        self.assertEqual(status, 0)
        selected = json.loads(out.getvalue())
        self.assertEqual(len(selected['selected_ids']), 2)
        observed = self.capture(selected)
        out = StringIO()
        with redirect_stdout(out):
            status = cli(['active-evaluate', '--workspace', str(self.wb.root),
                          '--pool', str(self.pool),
                          '--selection', selected['evidence_id'],
                          '--observation', observed['evidence_id']])
        self.assertEqual(status, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report['compatible'], 1)
        self.assertEqual(report['contradicted_on_observed'], 1)

    def test_mcp_adapter_advertises_selection_and_evaluation(self):
        import sys
        import types
        from unittest.mock import patch as mock_patch
        from paperrev.mcp_server import build_server

        class FakeFastMCP:
            def __init__(self, **kwargs):
                self.registered = {}

            def tool(self):
                def decorate(fn):
                    self.registered[fn.__name__] = fn
                    return fn
                return decorate

        mcp = types.ModuleType('mcp')
        server = types.ModuleType('mcp.server')
        fastmcp = types.ModuleType('mcp.server.fastmcp')
        fastmcp.FastMCP = FakeFastMCP
        with mock_patch.dict(sys.modules, {'mcp': mcp, 'mcp.server': server,
                                            'mcp.server.fastmcp': fastmcp}):
            adapter = build_server()
        self.assertIn('select_active_probes', adapter.registered)
        self.assertIn('evaluate_active_probes', adapter.registered)
        self.two_models()
        selected = adapter.registered['select_active_probes'](
            str(self.wb.root), str(self.pool), self.evids, budget=2)
        self.assertEqual(len(selected['selected_ids']), 2)
        obs = self.capture(selected)
        outcome = adapter.registered['evaluate_active_probes'](
            str(self.wb.root), str(self.pool), selected['evidence_id'], obs['evidence_id'])
        self.assertEqual(outcome['compatible'], 1)

    def test_rejects_too_small_cost_and_numeric_overflow(self):
        self.data[2]['query_cost'] = 1e-300
        self.save_pool()
        self.two_models()
        with self.assertRaises(ValueError):
            self.selection()
        self.assertEqual(_distance(10**500, 10**500, 1), 0.0)
        self.assertEqual(_distance(10**500, 10**501, 1), 1.0)



if __name__ == '__main__':
    unittest.main()

"""End-to-end observable HTTP comparisons and negative provenance controls."""
from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from paperrev.blackbox import capture_api, make_interventions
from paperrev.contrast import analyze_contrasts
from paperrev.workbench import Workbench


class EchoMath(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        obj = json.loads(raw)
        result = {"value": 2 * obj["n"], "kind": "stable", "nested": {"even": obj["n"] % 2 == 0}}
        body = json.dumps({"output": result}, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Contrasts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), EchoMath)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=3)
        cls.server.server_close()

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.wb = Workbench.create(self.root / "lab", "Finite black-box hypothesis tests")
        self.base = self.root / "base.jsonl"
        self.base.write_text(''.join(json.dumps({"id": str(i), "input": {"n": i}}) + "\n" for i in [2, 3, 4]))
        self.spec = self.root / "changes.json"
        self.spec.write_text(json.dumps({
            "schema": "paperrev/interventions-v1",
            "changes": [{"id": "plus-one", "path": "/n", "op": "add", "value": 1}],
            "checks": [
                {"id": "C-1", "intervention_id": "plus-one", "output_pointer": "/value", "relation": "shift", "expected_delta": 2},
                {"id": "C-2", "intervention_id": "plus-one", "output_pointer": "/kind", "relation": "same"},
                {"id": "C-3", "intervention_id": "plus-one", "output_pointer": "/value", "relation": "same"},
                {"id": "C-4", "intervention_id": "plus-one", "output_pointer": "/nested/even", "relation": "different"},
            ],
        }))
        self.plan = make_interventions(self.wb, self.base, self.spec)
        self.port = self.server.server_address[1]
        self.origin = f"http://127.0.0.1:{self.port}"

    def obs(self, cases_path, limit=25, origin=None):
        return capture_api(self.wb, cases_path, origin=origin or self.origin,
                           endpoint_path="/predict", execute=True, output_pointer="/output", limit=limit)

    def run_analysis(self, base_capture=None, probe_capture=None, **kw):
        b = base_capture or self.obs(self.base)
        p = probe_capture or self.obs(self.plan["path"])
        return analyze_contrasts(self.wb,
            baseline_evidence_id=b["evidence_id"],
            intervention_evidence_id=p["evidence_id"],
            intervention_plan_evidence_id=kw.get("plan_id", self.plan["evidence_id"]),
            base_cases_path=kw.get("base_path", self.base),
            probe_cases_path=kw.get("probe_path", self.plan["path"]),
            spec_path=kw.get("spec_path", self.spec))

    def test_observed_shift_and_counterexample(self):
        report = self.run_analysis()
        by_id = {x["id"]: x for x in report["checks"]}
        self.assertEqual(report["coverage"], "complete")
        self.assertEqual(by_id["C-1"]["status"], "supported_on_pairs")
        self.assertEqual(by_id["C-1"]["mean_observed_delta"], 2.0)
        self.assertEqual(by_id["C-2"]["status"], "supported_on_pairs")
        self.assertEqual(by_id["C-3"]["status"], "counterexample_observed")
        self.assertEqual(len(by_id["C-3"]["counterexample_case_ids"]), 3)
        self.assertEqual(by_id["C-4"]["status"], "supported_on_pairs")
        self.assertEqual(self.wb.get(report["evidence_id"])["confidence"], "derived")
        out = self.root / "evidence.json"
        self.wb.export_bundle(out)
        self.assertTrue(Workbench.verify_bundle(out)["verified"])

    def test_partial_probe_cannot_support_full_claim(self):
        b = self.obs(self.base)
        p = self.obs(self.plan["path"], limit=1)
        report = self.run_analysis(b, p)
        self.assertEqual(report["coverage"], "partial")
        self.assertEqual(report["checks"][0]["status"], "unknown")
        self.assertEqual(report["checks"][0]["paired"], 1)
        # Contradictions remain valid finite counterexamples even in partial samples.
        self.assertEqual(report["checks"][2]["status"], "counterexample_observed")

    def test_damaged_observation_rejected(self):
        b = self.obs(self.base)
        p = self.obs(self.plan["path"])
        file = self.wb.root / p["reference_file"]
        file.write_text(file.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, "changed after observation"):
            self.run_analysis(b, p)

    def test_metadata_tampering_rejected(self):
        b = self.obs(self.base)
        p = self.obs(self.plan["path"])
        manifest = self.wb.root / "observations" / (p["capture_id"] + ".json")
        data = json.loads(manifest.read_text())
        data["endpoint"]["origin"] = "http://attacker.example"
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "metadata changed after observation"):
            self.run_analysis(b, p)

    def test_cli_evaluation_matches_library(self):
        from paperrev.engine_cli import main as cli
        b = self.obs(self.base)
        p = self.obs(self.plan["path"])
        args = ["analyze-interventions", "--workspace", str(self.wb.root),
                "--baseline", b["evidence_id"], "--probes", p["evidence_id"],
                "--plan-evidence", self.plan["evidence_id"],
                "--cases", str(self.base), "--probe-cases", str(self.plan["path"]),
                "--spec", str(self.spec)]
        from contextlib import redirect_stdout
        from io import StringIO
        sink = StringIO()
        with redirect_stdout(sink):
            self.assertEqual(cli(args), 0)
        result = json.loads(sink.getvalue())
        self.assertEqual(result["summary"]["supported_on_pairs"], 3)
        self.assertEqual(result["summary"]["counterexample_observed"], 1)

    def test_modified_probe_file_rejected(self):
        b = self.obs(self.base)
        p = self.obs(self.plan["path"])
        path = Path(self.plan["path"])
        value = json.loads(path.read_text().splitlines()[0])
        value["input"]["other"] = "unauthorized-change"
        lines = path.read_text().splitlines()
        lines[0] = json.dumps(value)
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "Evidence does not bind"):
            self.run_analysis(b, p)

    def test_wrong_spec_rejected(self):
        b = self.obs(self.base)
        p = self.obs(self.plan["path"])
        wrong = self.root / "wrong.json"
        spec = json.loads(self.spec.read_text())
        spec["changes"][0]["value"] = 3
        wrong.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "Evidence does not bind"):
            self.run_analysis(b, p, spec_path=wrong)

    def test_wrong_origin_rejected(self):
        b = self.obs(self.base)
        p = self.obs(self.plan["path"], origin=f"http://localhost:{self.port}")
        with self.assertRaisesRegex(ValueError, "incompatible target"):
            self.run_analysis(b, p)

    def test_unknown_output_field_is_not_failure(self):
        spec = json.loads(self.spec.read_text())
        spec["checks"] = [{"id": "C-missing", "intervention_id": "plus-one", "output_pointer": "/never-present", "relation": "same"}]
        # A changed diagnostic spec requires new generated probe-plan evidence.
        self.spec.write_text(json.dumps(spec))
        self.plan = make_interventions(self.wb, self.base, self.spec)
        result = self.run_analysis()
        self.assertEqual(result["checks"][0]["status"], "unknown")
        self.assertEqual(result["checks"][0]["missing_or_incomparable"], 3)

    def test_wrong_plan_authority_rejected(self):
        spoof = self.wb.record(kind="intervention-plan", authority="analyst-inference",
                               confidence="inferred", subject={}, result={})
        with self.assertRaisesRegex(ValueError, "Evidence does not bind"):
            self.run_analysis(plan_id=spoof["evidence_id"])


if __name__ == "__main__":
    unittest.main()

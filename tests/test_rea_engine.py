"""End-to-end offline tests of the REA-inspired paper reconstruction engine."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paperrev.workbench import Workbench, canonical, read_json, sha256
from paperrev.replicate import (ModelClient, ingest_pdf, register_plan, propose_plan,
                                 create_candidate, generate_candidate)
from paperrev.reproduce import run_tests, compare_outputs, verify_claims, candidate_manifest
from paperrev.engine_cli import main as cli


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.ctx = tempfile.TemporaryDirectory()
        self.addCleanup(self.ctx.cleanup)
        self.root = Path(self.ctx.name)
        self.wb = Workbench.create(self.root / "lab", "Tiny scientific reproduction")

    def source_and_plan(self, *, unknowns=None):
        try:
            import pymupdf
        except ImportError:
            self.skipTest("optional pymupdf not installed")
        path = self.root / "paper.pdf"
        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_text((40, 50), "Every element is multiplied by two.")
        doc.save(path)
        doc.close()
        source = ingest_pdf(self.wb, path)
        paper_sha = source["source_sha256"]
        plan = {
            "schema": "paperrev/plan-v2",
            "claims": [{"id": "C1", "statement": "Output equals twice each element.",
                        "quote": "Every element is multiplied by two.", "page": 1}],
            "modules": [{"name": "transform", "purpose": "double elements", "claim_ids": ["C1"],
                         "inputs": "array", "outputs": "array", "algorithm": "2*x"}],
            "unknowns": unknowns if unknowns is not None else [],
        }
        saved = register_plan(self.wb, paper_sha, plan)
        return source, plan, saved

    def proposal(self):
        return {"files": [
            {"path": "src/reproduction.py", "content": "def predict(xs):\n    return [2*x for x in xs]\n"},
            {"path": "tests/test_reproduction.py", "content": "import unittest\nfrom reproduction import predict\n\nclass T(unittest.TestCase):\n    def test_double(self):\n        self.assertEqual(predict([1,2]),[2,4])\n"},
        ], "assumptions": ["Integer arithmetic"]}

    def test_evidence_integrity_and_idempotence(self):
        kw = dict(kind="observation", authority="controlled-execution", confidence="observed",
                  subject={"sample": 1}, result={"value": 2})
        a = self.wb.record(**kw)
        self.assertEqual(a, self.wb.record(**kw))
        self.assertEqual(self.wb.get(a["evidence_id"]), a)
        path = self.wb.root / "evidence" / f'{a["evidence_id"]}.json'
        rec = read_json(path)
        rec["result"]["value"] = 3
        path.write_text(json.dumps(rec))
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            self.wb.get(a["evidence_id"])

    def test_unknown_requires_observation_for_resolution(self):
        e = self.wb.record(kind="guess", authority="analyst-inference", confidence="inferred",
                           subject={}, result={"text": "might work"})
        unk = self.wb.record_unknown("Unknown optimizer?", related_claim_ids=["C1"],
                                      evidence_ids=[e["evidence_id"]])
        with self.assertRaises(ValueError):
            self.wb.update_unknown(unk["unknown_id"], expected_revision=1,
                                   status="resolved", resolution="AdamW",
                                   evidence_ids=[e["evidence_id"]])
        run = self.wb.record(kind="test-result", authority="controlled-execution", confidence="observed",
                             subject={}, result={"value": "verified"})
        new = self.wb.update_unknown(unk["unknown_id"], expected_revision=1,
                                      status="resolved", resolution="an observed specific fact",
                                      evidence_ids=[run["evidence_id"]])
        self.assertEqual(new["revision"], 2)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.wb.update_unknown(unk["unknown_id"], expected_revision=1,
                                   status="investigating")
        self.assertEqual(self.wb.unresolved_for("C1"), [])

    def test_pdf_plan_grounding_rejects_hallucinated_quote(self):
        source, plan, saved = self.source_and_plan()
        self.assertEqual(saved["status"], "planned-not-implemented")
        self.assertEqual(saved["claims"], 1)
        corrupted = json.loads(json.dumps(plan))
        corrupted["claims"][0]["quote"] = "This theorem has a 99 percent accuracy."
        with self.assertRaisesRegex(ValueError, "unverified quote"):
            register_plan(self.wb, source["source_sha256"], corrupted)

    def test_candidate_materialization_and_hash_guard(self):
        _, _, saved = self.source_and_plan()
        code = create_candidate(self.wb, plan_sha=saved["plan_sha256"], proposal=self.proposal(), label="v1")
        self.assertEqual(code["status"], "syntax-checked-not-executed")
        self.assertEqual(len(candidate_manifest(self.wb, code["candidate_id"])["files"]), 2)
        path = Path(code["candidate_path"]) / "src" / "reproduction.py"
        path.write_text("print('different')")
        with self.assertRaisesRegex(ValueError, "code changed"):
            candidate_manifest(self.wb, code["candidate_id"])

    def test_candidate_rejects_path_traversal_and_syntax_error(self):
        _, _, saved = self.source_and_plan()
        bad = self.proposal()
        bad["files"][0]["path"] = "src/../../outside.py"
        with self.assertRaises(ValueError):
            create_candidate(self.wb, plan_sha=saved["plan_sha256"], proposal=bad, label="bad")
        bad = self.proposal()
        bad["files"][0]["content"] = "def broken(:\n"
        with self.assertRaises(SyntaxError):
            create_candidate(self.wb, plan_sha=saved["plan_sha256"], proposal=bad, label="bad")

    def test_docker_is_required_and_mocked_command_is_restricted(self):
        _, _, saved = self.source_and_plan()
        code = create_candidate(self.wb, plan_sha=saved["plan_sha256"], proposal=self.proposal(), label="run")
        with patch("paperrev.reproduce.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Docker not found"):
                run_tests(self.wb, code["candidate_id"], image="python:3.12-slim")
        proc = SimpleNamespace(returncode=0, stdout="ok", stderr="Ran 1 test in 0.00s\nOK")
        with patch("paperrev.reproduce.shutil.which", return_value="/usr/bin/docker"), \
             patch("paperrev.reproduce.subprocess.run", return_value=proc) as execute:
            result = run_tests(self.wb, code["candidate_id"], image="python:3.12-slim")
        self.assertEqual(result["status"], "pass")
        args = execute.call_args.args[0]
        for restriction in ("--network=none", "--read-only", "--cap-drop=ALL",
                            "--security-opt=no-new-privileges", "--pull=never", "--pids-limit=64"):
            self.assertIn(restriction, args)
        self.assertIn("type=bind,source=", " ".join(args))
        self.assertEqual(self.wb.get(result["evidence_id"])["confidence"], "observed")

    def test_behavior_comparison_requires_complete_id_match(self):
        ref = self.root / "reference.jsonl"
        candidate = self.root / "generated.jsonl"
        ref.write_text('\n'.join(json.dumps(x) for x in [
            {"id": "one", "output": {"p": [0.1, 0.9]}},
            {"id": "two", "output": {"p": [0.3, 0.7]}}]) + "\n")
        candidate.write_text(json.dumps({"id": "one", "output": {"p": [0.100001, 0.899999]}}) + "\n")
        compare = compare_outputs(self.wb, ref, candidate, atol=0.00001)
        self.assertEqual(compare["coverage"], "partial")
        self.assertEqual(compare["paired_cases"], 1)
        self.assertEqual(compare["matched_fraction"], 1.0)
        _, _, plan = self.source_and_plan()
        assessment = verify_claims(self.wb, plan["plan_sha256"], {"C1": {
            "kind": "output-match", "evidence_id": compare["evidence_id"], "min_fraction": 1.0}})
        self.assertEqual(assessment["unknown"], 1)
        self.assertFalse(assessment["paper_reproduced"])

    def test_complete_comparison_can_pass_scoped_obligation(self):
        ref = self.root / "r.jsonl"
        got = self.root / "g.jsonl"
        ref.write_text('{"id":1,"output":42}\n')
        got.write_text('{"id":1,"output":42}\n')
        comparison = compare_outputs(self.wb, ref, got)
        _, _, plan = self.source_and_plan()
        report = verify_claims(self.wb, plan["plan_sha256"], {"C1": {
            "kind": "output-match", "evidence_id": comparison["evidence_id"], "min_fraction": 1.0}})
        self.assertEqual(report["claims"][0]["status"], "pass")
        self.assertFalse(report["paper_reproduced"])

    def test_open_unknown_blocks_success(self):
        _, _, plan = self.source_and_plan(unknowns=[{"question": "Which normalization?", "claim_ids": ["C1"]}])
        ref, got = self.root / "ref.jsonl", self.root / "got.jsonl"
        ref.write_text('{"id":1,"output":1}\n')
        got.write_text('{"id":1,"output":1}\n')
        cmp = compare_outputs(self.wb, ref, got)
        report = verify_claims(self.wb, plan["plan_sha256"], {"C1": {
            "kind": "output-match", "evidence_id": cmp["evidence_id"], "min_fraction": 1.0}})
        self.assertEqual(report["claims"][0]["status"], "unknown")
        self.assertTrue(report["claims"][0]["unresolved_unknowns"])

    def test_bundle_verification_and_tamper(self):
        e = self.wb.record(kind="thing", authority="paper", confidence="reported", subject={}, result={"number": 9})
        output = self.root / "bundle.json"
        self.wb.export_bundle(output)
        self.assertEqual(Workbench.verify_bundle(output)["records"], 1)
        bundle = read_json(output)
        bundle["records"][0]["result"]["number"] = 11
        output.write_text(json.dumps(bundle))
        with self.assertRaisesRegex(ValueError, "tampered"):
            Workbench.verify_bundle(output)

    def test_model_url_needs_local_or_explicit_secure_remote(self):
        with self.assertRaises(ValueError):
            ModelClient("https://example.org", "mock")
        with self.assertRaises(ValueError):
            ModelClient("http://example.org", "mock", allow_remote=True)
        self.assertEqual(ModelClient("http://localhost:8000", "mock").model, "mock")

    def test_fake_llm_end_to_end_prepares_code_and_tests_without_running_it(self):
        src, plan, _ = self.source_and_plan()
        # Separate workbench demonstrates model-driven stages after ingesting the paper.
        second = Workbench.create(self.root / "agent", "Agent reconstruction")
        ingest_pdf(second, self.root / "paper.pdf")
        class FakeClient:
            def __init__(self): self.calls = []
            def ask(self, system, prompt):
                self.calls.append(prompt)
                return plan if len(self.calls) == 1 else self_proposal
        self_proposal = self.proposal()
        client = FakeClient()
        authored = propose_plan(second, src["source_sha256"], client)
        candidate = generate_candidate(second, authored["plan_sha256"], "agent-v1", client)
        self.assertTrue(Path(candidate["candidate_path"]).joinpath("src/reproduction.py").exists())
        self.assertEqual(candidate["status"], "syntax-checked-not-executed")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("UNTRUSTED", __import__("paperrev.replicate", fromlist=["PLAN_SYSTEM"]).PLAN_SYSTEM)

    def test_empty_outputs_compare_exactly(self):
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        a.write_text('{"id":"empty","output":[]}\n')
        b.write_text('{"id":"empty","output":[]}\n')
        cmp = compare_outputs(self.wb, a, b)
        self.assertEqual(cmp["matched_fraction"], 1.0)
        self.assertEqual(cmp["matching_values"], 1)

    def test_untracked_candidate_code_is_rejected(self):
        _, _, plan = self.source_and_plan()
        new = create_candidate(self.wb, plan_sha=plan["plan_sha256"], label="v2", proposal=self.proposal())
        extra = Path(new["candidate_path"]) / "tests" / "test_extra.py"
        extra.write_text("print('not in manifest')")
        with self.assertRaisesRegex(ValueError, "untracked file"):
            candidate_manifest(self.wb, new["candidate_id"])

    def test_clean_exit_without_tests_is_not_a_pass(self):
        _, _, plan = self.source_and_plan()
        new = create_candidate(self.wb, plan_sha=plan["plan_sha256"], label="v2", proposal=self.proposal())
        done = SimpleNamespace(returncode=0, stdout="nothing run", stderr="")
        with patch("paperrev.reproduce.shutil.which", return_value="/usr/bin/docker"), \
             patch("paperrev.reproduce.subprocess.run", return_value=done):
            result = run_tests(self.wb, new["candidate_id"], image="python:3.12-slim")
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(self.wb.get(result["evidence_id"])["coverage"], "partial")

    def test_unknown_bundle_tamper_detected(self):
        self.wb.record_unknown("Which optimizer?", related_claim_ids=["C1"])
        out = self.root / "evidence-export.json"
        self.wb.export_bundle(out)
        self.assertTrue(Workbench.verify_bundle(out)["verified"])
        content = read_json(out)
        content["unknowns"][0]["status"] = "resolved"
        out.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "unknown history integrity"):
            Workbench.verify_bundle(out)

    def test_docker_timeout_records_cleanup_status(self):
        _, _, plan = self.source_and_plan()
        new = create_candidate(self.wb, plan_sha=plan["plan_sha256"], label="timed", proposal=self.proposal())
        import subprocess
        def fake_run(args, **kw):
            if args[1] == "run":
                raise subprocess.TimeoutExpired(cmd=args, timeout=kw.get("timeout", 1), output=b"partial")
            return SimpleNamespace(returncode=0, stdout="removed", stderr="")
        with patch("paperrev.reproduce.shutil.which", return_value="/usr/bin/docker"), \
             patch("paperrev.reproduce.subprocess.run", side_effect=fake_run) as caller:
            result = run_tests(self.wb, new["candidate_id"], image="python:3.12-slim")
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(caller.call_count, 2)
        record = read_json(self.wb.root / "runs" / (result["run_id"] + ".json"))
        self.assertEqual(record["container_cleanup"], "forced-remove-accepted")

    def test_cli_core_paths(self):
        # We can invoke the non-network CLI against the existing session.
        self.assertEqual(cli(["inspect", "--workspace", str(self.wb.root)]), 0)
        self.assertEqual(cli(["verify-bundle", "--file", "/does-not-exist"]), 2)


if __name__ == "__main__":
    unittest.main()

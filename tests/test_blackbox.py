"""Offline behavioral-reconstruction tests; no real API billing or Docker execution."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paperrev.blackbox import _endpoint, _pointer, capture_api, cases, make_interventions, replay_candidate, check_reconstruction, generate_from_observation
from paperrev.workbench import Workbench, sha256, write_once


class Reply:
    status = 200
    def __init__(self, obj): self.obj = obj
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, n): return json.dumps(self.obj).encode()


class Opener:
    def __init__(self, fn): self.fn = fn; self.requests = []
    def open(self, req, timeout):
        x = json.loads(req.data)
        self.requests.append(x)
        return Reply(self.fn(x))


class BlackBoxTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.wb = Workbench.create(self.root / "work", "Black-box test")
        self.inputs = self.root / "cases.jsonl"
        self.inputs.write_text('{"id":1,"input":{"x":1,"options":["a","b"]}}\n'
                               '{"id":2,"input":{"x":2,"options":["a","b"]}}\n')

    def candidate(self):
        folder = self.wb.root / "candidates" / "candidate-1"
        (folder / "src").mkdir(parents=True)
        code = 'def predict(x):\n    return {"score":2*x["x"]}\n'
        (folder / "src" / "reproduction.py").write_text(code)
        write_once(folder / "candidate.json", {
            "schema": "paperrev/candidate-v2", "candidate_id": "candidate-1",
            "files": {"src/reproduction.py": sha256(code.encode())},
        })
        return "candidate-1"

    def observed(self):
        with patch("paperrev.blackbox.urllib.request.build_opener",
                   return_value=Opener(lambda x: {"result": {"score": 2 * x["x"]}})):
            return capture_api(self.wb, self.inputs, origin="http://127.0.0.1:8000",
                               endpoint_path="/predict", output_pointer="/result", execute=True)

    def test_auth_and_origin_scope(self):
        with self.assertRaises(ValueError):
            capture_api(self.wb, self.inputs, origin="https://remote.example",
                        endpoint_path="/predict", execute=False)
        with self.assertRaises(ValueError):
            _endpoint("http://remote.example", "/predict", True)
        with self.assertRaises(ValueError):
            _endpoint("https://remote.example", "//other", True)
        self.assertEqual(_pointer({"a/b": [1, 2]}, "/a~1b/1"), 2)

    def test_capture_real_output_and_scope(self):
        obs = self.observed()
        self.assertEqual(obs["coverage"], "complete")
        data = [json.loads(x) for x in (self.wb.root / obs["reference_file"]).read_text().splitlines()]
        self.assertEqual(data[0]["output"], {"score": 2})
        self.assertEqual(data[1]["output"], {"score": 4})
        self.assertEqual(self.wb.get(obs["evidence_id"])["confidence"], "observed")
        with patch("paperrev.blackbox.urllib.request.build_opener",
                   return_value=Opener(lambda x: {"score": x["x"]})):
            partial = capture_api(self.wb, self.inputs, origin="http://localhost:8000",
                                  endpoint_path="/predict", limit=1, execute=True)
        self.assertEqual(partial["coverage"], "partial")

    def test_input_interventions_are_not_observations(self):
        spec = self.root / "changes.json"
        spec.write_text(json.dumps({"schema": "paperrev/interventions-v1", "changes": [
            {"id": "increment", "path": "/x", "op": "add", "value": 1},
            {"id": "reorder", "path": "/options", "op": "reverse"}]}))
        before = self.inputs.read_bytes()
        transformed = make_interventions(self.wb, self.inputs, spec)
        self.assertEqual(transformed["cases"], 4)
        self.assertEqual(cases(transformed["path"])[0]["input"]["x"], 2)
        self.assertEqual(cases(transformed["path"])[2]["input"]["options"], ["b", "a"])
        self.assertEqual(self.inputs.read_bytes(), before)
        self.assertEqual(self.wb.get(transformed["evidence_id"])["confidence"], "inferred")

    def test_docker_only_and_scoped_comparison(self):
        ident = self.candidate()
        obs = self.observed()
        with self.assertRaises(ValueError):
            replay_candidate(self.wb, ident, self.inputs, image="python:3.12-slim", execute=False)
        with patch("paperrev.blackbox.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                replay_candidate(self.wb, ident, self.inputs, image="python:3.12-slim", execute=True)
        output = '{"id":1,"output":{"score":2}}\n{"id":2,"output":{"score":4}}\n'
        done = SimpleNamespace(returncode=0, stdout=output, stderr="")
        with patch("paperrev.blackbox.shutil.which", return_value="/usr/bin/docker"), \
             patch("paperrev.blackbox.subprocess.run", return_value=done) as run:
            result = check_reconstruction(self.wb, obs["evidence_id"], ident, self.inputs,
                                          image="python:3.12-slim", execute=True)
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["comparison"]["matched_fraction"], 1.0)
        self.assertEqual(result["comparison"]["coverage"], "complete")
        cmd = run.call_args.args[0]
        for condition in ("--network=none", "--read-only", "--cap-drop=ALL", "--pull=never"):
            self.assertIn(condition, cmd)
        self.assertEqual(len(self.wb.get(result["evidence_id"])["sources"]), 3)

    def test_invalid_output_and_capture_mutation_fail_closed(self):
        ident = self.candidate()
        incomplete = SimpleNamespace(returncode=0, stdout='{"id":1,"output":1}\n', stderr="")
        with patch("paperrev.blackbox.shutil.which", return_value="/usr/bin/docker"), \
             patch("paperrev.blackbox.subprocess.run", return_value=incomplete):
            result = replay_candidate(self.wb, ident, self.inputs, image="python:3.12-slim", execute=True)
        self.assertEqual(result["status"], "invalid-output")
        self.assertIsNone(result["output_path"])
        observation = self.observed()
        (self.wb.root / observation["reference_file"]).write_text('{"id":1,"output":{"score":1234}}\n')
        with self.assertRaises(ValueError):
            check_reconstruction(self.wb, observation["evidence_id"], ident, self.inputs,
                                 image="python:3.12-slim", execute=True)

    def test_observations_condition_independent_code_generation(self):
        observation = self.observed()
        plan_sha = "b" * 64
        write_once(self.wb.root / "plans" / (plan_sha + ".json"), {
            "schema": "paperrev/plan-v2", "claims": [{"id":"C1"}], "modules": [{"name":"prediction"}]})
        class FakeLLM:
            def __init__(self): self.calls = []
            def ask(self, system, prompt):
                self.calls.append((system, json.loads(prompt)))
                return {"files": [
                    {"path": "src/reproduction.py", "content": 'def predict(x):\n    return {"score": 2*x["x"]}\n'},
                    {"path": "tests/test_reproduction.py", "content": "import unittest\n"}
                ], "assumptions": ["finite examples do not identify historical source"]}
        model = FakeLLM()
        record = generate_from_observation(self.wb, observation["evidence_id"],
                                           plan_sha, self.inputs, "behavioral", model)
        self.assertEqual(record["status"], "behavior-grounded-generated-not-verified")
        self.assertEqual(len(model.calls), 1)
        self.assertIn("predict(input)", model.calls[0][0])
        self.assertEqual(len(model.calls[0][1]["captured_examples"]), 2)
        linkage = self.wb.get(record["reconstruction_evidence_id"])
        self.assertIn(observation["evidence_id"], linkage["sources"])


if __name__ == "__main__":
    unittest.main()

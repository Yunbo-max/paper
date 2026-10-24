"""Mocked API checks and mathematical gate-analysis tests. No GPU or network."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paperrev.startlux import analyze, endpoint_for, probe


def fake_response():
    return {
        "model": "fake-for-tests",
        "answers": {
            "answer": {
                "type": "choice", "choice": "A",
                "probabilities": {"A": 0.775, "B": 0.225},
                "details": {
                    "first_pass": {"probabilities": {"A": 0.4, "B": 0.6}, "top": 0.6},
                    "after_thought": {"probabilities": {"A": 0.9, "B": 0.1}, "ms": 4},
                    "thought": {"used": True, "tokens": 100, "ended": "closed"},
                },
            }
        },
    }


class StartLuxTest(unittest.TestCase):
    def test_endpoint_restrictions(self):
        self.assertEqual(endpoint_for("http://localhost:8090", False),
                         "http://localhost:8090/v1/systemone")
        with self.assertRaises(ValueError):
            endpoint_for("https://example.com", False)
        with self.assertRaises(ValueError):
            endpoint_for("http://example.com", True)
        with self.assertRaises(ValueError):
            endpoint_for("https://user:pass@example.com", True)
        self.assertEqual(endpoint_for("https://example.com", True),
                         "https://example.com/v1/systemone")

    def test_probe_requires_explicit_execution(self):
        with self.assertRaises(ValueError):
            probe("unused", "unused", "http://localhost:8090", execute=False)

    def test_probe_records_real_responses_without_copying_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset = root / "samples.jsonl"
            dataset.write_text(json.dumps({
                "id": "s1", "state": "private state", "questions": {
                    "answer": {"type": "choice", "instructions": "choose", "criteria": {"A": "yes", "B": "no"}}
                }, "gold": {"answer": "A"},
            }) + "\n")
            output = root / "records.jsonl"
            with patch("paperrev.startlux._post_json", return_value=fake_response()) as mocked:
                summary = probe(dataset, output, "http://127.0.0.1:8090", execute=True)
            self.assertEqual(summary["status"], "completed")
            body = mocked.call_args.args[1]
            self.assertEqual(body["think_mode"], "always")
            self.assertFalse(body["none_check"])
            self.assertFalse(body["allow_none"])
            self.assertTrue(body["return_details"])
            self.assertFalse(body["return_thought"])
            capture = output.read_text()
            self.assertNotIn("private state", capture)
            self.assertEqual(json.loads(capture)["status"], "ok")

    def test_gate_sweep_can_help_or_harm(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            capture = root / "capture.jsonl"
            rec = {
                "id": "s1", "status": "ok",
                "controls": {"think_mode": "always", "none_check": False, "allow_none": False},
                "questions": {"answer": {"type": "choice"}},
                "gold": {"answer": "A"}, "response": fake_response(),
            }
            capture.write_text(json.dumps(rec) + "\n")
            result = analyze(capture, root / "summary.json", [0.5, 0.7])
            self.assertEqual(result["comparison_scope"]["paired_questions"], 1)
            self.assertEqual(result["strategies"]["gate_0.5"]["accuracy"], 0.0)
            self.assertEqual(result["strategies"]["gate_0.5"]["mean_thought_tokens_estimate"], 0.0)
            self.assertEqual(result["strategies"]["gate_0.7"]["accuracy"], 1.0)
            self.assertEqual(result["strategies"]["gate_0.7"]["mean_thought_tokens_estimate"], 100.0)
            self.assertGreater(result["after_vs_before"]["mean_log_score_gain"], 0.0)

    def test_missing_thought_is_unknown_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rec = {
                "id": "s1", "status": "ok",
                "controls": {"think_mode": "always", "none_check": False, "allow_none": False},
                "questions": {"answer": {"type": "choice"}},
                "gold": {"answer": "A"},
                "response": {"answers": {"answer": {"details": {
                    "first_pass": {"probabilities": {"A": 0.8, "B": 0.2}},
                    "thought": {"used": False},
                }}}},
            }
            path = root / "bad.jsonl"
            path.write_text(json.dumps(rec))
            with self.assertRaisesRegex(ValueError, "no comparable"):
                analyze(path, root / "summary.json", [0.6])


if __name__ == "__main__":
    unittest.main()

"""Dependency-free checks of provenance and strict evidence boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paperrev.core import init_case, inspect_hub_model, inspect_local_model, load_json


class CoreTest(unittest.TestCase):
    def test_case_init_snapshots_and_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "case.json"
            source.write_text(json.dumps({
                "case_id": "toy",
                "claims": [{"id": "C1", "statement": "a claimed mechanism"}],
                "unknowns": [{"id": "U1", "question": "which optimizer?"}],
            }))
            output = root / "run"
            manifest = init_case(source, output)
            self.assertEqual(manifest["status"], "initialized-not-reproduced")
            claims = load_json(output / "claim_ledger.json")["claims"]
            self.assertEqual(claims[0]["assessment"], "not_tested")
            unknown = load_json(output / "unknown_ledger.json")["unknowns"][0]
            self.assertEqual(unknown["status"], "open")
            with self.assertRaises(FileExistsError):
                init_case(source, output)

    def test_init_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "invalid.json"
            path.write_text(json.dumps({"claims": [{"id": "A"}, {"id": "A"}]}))
            with self.assertRaises(ValueError):
                init_case(path, root / "out")

    def test_inspect_local_does_not_read_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"x": "model-00001-of-00002.safetensors", "y": "model-00002-of-00002.safetensors"}
            }))
            result = inspect_local_model(root, root / "inspection.json")
            self.assertEqual(result["tensor_entries_declared"], 2)
            self.assertEqual(len(result["weight_shards_declared"]), 2)
            self.assertFalse(result["weight_data_inspected"])
            self.assertFalse(result["training_method_recovered"])

    def test_hub_pins_sha_and_fetches_only_configs(self):
        sha = "f" * 40
        accessed = []
        def fake_download(url):
            accessed.append(url)
            if "/api/models/" in url:
                return {"sha": sha, "siblings": [{"rfilename": "model-1.safetensors", "size": 100}]}
            if url.endswith("model.safetensors.index.json"):
                return {"weight_map": {"x": "model-1.safetensors"}}
            return {"model_type": "qwen3_5"}
        with tempfile.TemporaryDirectory() as d:
            with patch("paperrev.core._download_json", side_effect=fake_download):
                result = inspect_hub_model("example/model", "main", Path(d) / "hub.json")
            self.assertEqual(result["source"]["revision"], sha)
            self.assertEqual(result["tensor_entries_declared"], 1)
            self.assertEqual(len(accessed), 4)
            self.assertTrue(all(".safetensors" not in x or "index.json" in x for x in accessed))
            self.assertFalse(result["weight_data_inspected"])

    def test_hub_rejects_path_tricks(self):
        with self.assertRaises(ValueError):
            inspect_hub_model("../evil", "main", "/tmp/unwritten.json")


if __name__ == "__main__":
    unittest.main()

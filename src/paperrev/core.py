"""Source snapshots and evidence records. No inferred statement is promoted to proof."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def init_case(case_path: str | Path, output: str | Path) -> dict[str, Any]:
    source = Path(case_path)
    raw = source.read_bytes()
    case = json.loads(raw)
    if not isinstance(case, dict) or not isinstance(case.get("claims"), list):
        raise ValueError("case must contain a 'claims' array")
    identifiers = [x.get("id") for x in case["claims"] if isinstance(x, dict)]
    if len(identifiers) != len(case["claims"]) or any(not isinstance(x, str) or not x for x in identifiers):
        raise ValueError("each claim needs a nonempty string id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate claim ids")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "paperrev/run-v1",
        "case_id": case.get("case_id", "unspecified"),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_sha256": digest_bytes(raw),
        "source": str(source),
        "python": sys.version.split()[0],
        "status": "initialized-not-reproduced",
        "verification_scope": "no GPU experiment or original training reproduction has run",
    }
    claims = [
        {
            **item,
            "assessment": "not_tested",
            "evidence": [],
        }
        for item in case["claims"]
    ]
    unknowns = [
        {**entry, "status": "open"}
        for entry in case.get("unknowns", [])
    ]
    write_json(output / "case_snapshot.json", case)
    write_json(output / "manifest.json", manifest)
    write_json(output / "claim_ledger.json", {"claims": claims, "schema": "paperrev/claims-v1"})
    write_json(output / "unknown_ledger.json", {"unknowns": unknowns, "schema": "paperrev/unknowns-v1"})
    return manifest


def ingest_pdf(pdf_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Extract text with page provenance; image-only content remains explicitly unknown."""
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PDF support requires: pip install '.[pdf]'") from exc
    pdf_path = Path(pdf_path)
    pages: list[dict[str, Any]] = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            content = page.get_text(sort=True)
            pages.append({
                "page": index + 1,
                "text": content,
                "text_sha256": digest_bytes(content.encode("utf-8")),
                "needs_visual_review": not bool(content.strip()),
            })
    result = {
        "schema": "paperrev/pdf-text-v1",
        "source_sha256": digest_file(pdf_path),
        "pages": pages,
        "limitations": [
            "Text extraction does not validate equations, diagrams, tables or page images.",
            "Pages with missing text require manual visual review; no OCR was performed.",
        ],
    }
    write_json(output, result)
    return {"page_count": len(pages), "needs_visual_review": sum(p["needs_visual_review"] for p in pages)}


def inspect_local_model(folder: str | Path, output: str | Path) -> dict[str, Any]:
    """Inspect only small config and index files; never read or claim to verify weights."""
    folder = Path(folder)
    configs: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for filename in ("config.json", "decision_config.json", "model.safetensors.index.json"):
        path = folder / filename
        if path.is_file():
            configs[filename] = load_json(path)
            hashes[filename] = digest_file(path)
    if not configs:
        raise ValueError(f"no known model metadata files found in {folder}")
    idx = configs.get("model.safetensors.index.json", {})
    weight_map = idx.get("weight_map") if isinstance(idx, dict) else None
    result = {
        "schema": "paperrev/model-metadata-v1",
        "source": {"kind": "local-files", "folder": str(folder)},
        "file_sha256": hashes,
        "configs": configs,
        "tensor_entries_declared": len(weight_map) if isinstance(weight_map, dict) else None,
        "weight_shards_declared": sorted(set(weight_map.values())) if isinstance(weight_map, dict) else None,
        "weight_data_inspected": False,
        "training_method_recovered": False,
        "limitations": ["A config or weight index is not proof that weight bytes are present or authentic."],
    }
    write_json(output, result)
    return result


def _download_json(url: str) -> dict[str, Any]:
    headers = {"User-Agent": "paperrev-lab/0.1"}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=40) as response:
        data = json.loads(response.read(12 * 1024 * 1024))
    if not isinstance(data, dict):
        raise ValueError("Hugging Face returned a non-object JSON response")
    return data


def inspect_hub_model(repo: str, revision: str, output: str | Path) -> dict[str, Any]:
    """Fetch metadata from a commit-pinned HF repo. Does not download tensors."""
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repo):
        raise ValueError("expected model id in owner/model format")
    if not re.fullmatch(r"[a-zA-Z0-9_./-]+", revision) or ".." in revision:
        raise ValueError("invalid revision")
    from urllib.parse import quote
    from urllib.error import HTTPError
    info = _download_json("https://huggingface.co/api/models/" + repo + "?blobs=true")
    pinned_sha = info.get("sha")
    if not isinstance(pinned_sha, str) or not re.fullmatch("[0-9a-f]{40}", pinned_sha):
        raise ValueError("HF response does not include a trusted 40-character commit SHA")
    # For main this is a point-in-time HEAD snapshot. A non-main requested ref
    # must not be silently replaced with HEAD.
    if revision != "main":
        info = _download_json("https://huggingface.co/api/models/" + repo + "/revision/" + quote(revision, safe="") + "?blobs=true")
        pinned_sha = info.get("sha")
        if not isinstance(pinned_sha, str) or not re.fullmatch("[0-9a-f]{40}", pinned_sha):
            raise ValueError("requested revision could not be pinned")
    all_files = info.get("siblings", [])
    files = [{"path": r.get("rfilename"), "size_bytes": r.get("size")} for r in all_files if isinstance(r, dict)]
    configs: dict[str, Any] = {}
    config_status: dict[str, str] = {}
    for filename in ("config.json", "decision_config.json", "model.safetensors.index.json"):
        url = "https://huggingface.co/" + repo + "/resolve/" + pinned_sha + "/" + filename
        try:
            configs[filename] = _download_json(url)
            config_status[filename] = "observed"
        except HTTPError as exc:
            if exc.code != 404:
                raise
            config_status[filename] = "not-present-at-snapshot"
    idx = configs.get("model.safetensors.index.json", {})
    weight_map = idx.get("weight_map") if isinstance(idx, dict) else None
    result = {
        "schema": "paperrev/model-metadata-v1",
        "source": {"kind": "huggingface-repository", "repo": repo, "revision": pinned_sha,
                   "url": "https://huggingface.co/" + repo},
        "files": files,
        "configs": configs,
        "config_status": config_status,
        "tensor_entries_declared": len(weight_map) if isinstance(weight_map, dict) else None,
        "weight_shards_declared": sorted(set(weight_map.values())) if isinstance(weight_map, dict) else None,
        "weight_data_inspected": False,
        "training_method_recovered": False,
        "limitations": ["Hub listing and config inspection do not download or verify weight bytes."],
    }
    write_json(output, result)
    return result

"""Content-addressed Evidence and revisioned unknowns, inspired by REA's investigation model.

A workbench never upgrades a source quotation, LLM guess, or test exit into
scientific truth. Records are immutable and independently verifiable by SHA256.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, record: Any) -> None:
    """Never overwrite an immutable output; an existing identical object is idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(record) + b"\n"
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable artifact already exists with different content: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(content)


def atomic_replace(path: Path, record: Any) -> None:
    """Atomic local revision write (not a multi-process compare-and-swap)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise FileExistsError(f"stale temporary state; inspect before retry: {tmp}")
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
            f.write(canonical(record) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


EVIDENCE_AUTHORITIES = {"paper", "repository", "checkpoint-metadata", "controlled-execution", "external-observation", "analyst-inference"}
EVIDENCE_CONFIDENCES = {"observed", "derived", "inferred", "reported"}


class Workbench:
    """User-selected root. No persistent server-global state or implicit network access."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not (self.root / "session.json").is_file():
            raise FileNotFoundError(f"no paperrev session at {self.root}; run init first")

    @classmethod
    def create(cls, root: str | Path, title: str) -> "Workbench":
        root = Path(root).expanduser().resolve()
        if not title.strip():
            raise ValueError("session title cannot be empty")
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("refusing to overwrite existing research workbench")
        root.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": "paperrev/session-v2", "title": title,
                    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "scope": "evidence-first-independent-reconstruction", "format_version": 2}
        write_once(root / "session.json", manifest)
        for folder in ("evidence", "unknowns", "sources", "plans", "candidates", "runs", "comparisons"):
            (root / folder).mkdir(exist_ok=True)
        return cls(root)

    def record(self, *, kind: str, authority: str, confidence: str, subject: dict[str, Any],
               result: dict[str, Any], sources: list[str] | None = None,
               limitations: list[str] | None = None, coverage: str = "bounded") -> dict[str, Any]:
        if authority not in EVIDENCE_AUTHORITIES or confidence not in EVIDENCE_CONFIDENCES:
            raise ValueError("unsupported evidence authority or confidence")
        if coverage not in ("complete", "bounded", "partial", "unknown"):
            raise ValueError("invalid coverage")
        parents = sources or []
        for parent in parents:
            if not isinstance(parent, str) or not parent.startswith("ev_") or not self._evidence_file(parent).exists():
                raise ValueError(f"unknown cited parent evidence: {parent}")
            self.get(parent)  # verify its digest rather than trusting its filename
        identity = {"schema": "paperrev/evidence-v2", "kind": kind, "authority": authority,
                    "confidence": confidence, "subject": subject, "result": result,
                    "sources": parents, "coverage": coverage, "limitations": limitations or []}
        record = {"evidence_id": "ev_" + sha256(canonical(identity)), **identity}
        write_once(self._evidence_file(record["evidence_id"]), record)
        return record

    def _evidence_file(self, evidence_id: str) -> Path:
        if not re.fullmatch(r"ev_[0-9a-f]{64}", evidence_id):
            raise ValueError("invalid evidence id")
        return self.root / "evidence" / (evidence_id + ".json")

    def get(self, evidence_id: str) -> dict[str, Any]:
        record = read_json(self._evidence_file(evidence_id))
        identity = {k: v for k, v in record.items() if k != "evidence_id"}
        if "ev_" + sha256(canonical(identity)) != evidence_id:
            raise ValueError("evidence integrity mismatch")
        return record

    def list_evidence(self, kind: str | None = None, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset must be >=0 and limit 1..100")
        summaries = []
        for path in sorted((self.root / "evidence").glob("ev_*.json")):
            rec = self.get(path.stem)
            if kind is None or rec["kind"] == kind:
                summaries.append({"id": rec["evidence_id"], "kind": rec["kind"],
                                  "authority": rec["authority"], "confidence": rec["confidence"],
                                  "coverage": rec["coverage"], "subject": rec["subject"]})
        return {"total": len(summaries), "items": summaries[offset:offset + limit],
                "next_offset": offset + limit if offset + limit < len(summaries) else None}

    def _unknown_file(self, unknown_id: str) -> Path:
        if not re.fullmatch(r"unk_[0-9a-f]{64}", unknown_id):
            raise ValueError("invalid unknown id")
        return self.root / "unknowns" / (unknown_id + ".json")

    def record_unknown(self, question: str, *, related_claim_ids: list[str] | None = None,
                       evidence_ids: list[str] | None = None, next_probe: str = "") -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("unknown question must be nonempty")
        claim_ids = sorted(set(related_claim_ids or []))
        unknown_id = "unk_" + sha256(canonical({"question": question.strip(), "claims": claim_ids}))
        for evid in evidence_ids or []:
            self.get(evid)
        file = self._unknown_file(unknown_id)
        if file.exists():
            return read_json(file)
        result = {"unknown_id": unknown_id, "question": question.strip(), "claim_ids": claim_ids,
                  "revision": 1, "status": "open", "evidence_ids": sorted(set(evidence_ids or [])),
                  "next_probe": next_probe, "resolution": None}
        result["revision_sha256"] = sha256(canonical(result))
        write_once(file, result)
        return result

    def update_unknown(self, unknown_id: str, *, expected_revision: int, status: str,
                       evidence_ids: list[str] | None = None, resolution: str | None = None) -> dict[str, Any]:
        if status not in ("open", "investigating", "resolved", "unresolved"):
            raise ValueError("invalid unknown status")
        path = self._unknown_file(unknown_id)
        old = read_json(path)
        if old["revision"] != expected_revision:
            raise ValueError("stale unknown revision; re-read before updating")
        for evid in evidence_ids or []:
            self.get(evid)
        if status == "resolved" and (not resolution or not evidence_ids):
            raise ValueError("resolving an unknown requires an explicit resolution and evidence")
        if status == "resolved" and not any(self.get(i)["confidence"] == "observed" for i in evidence_ids or []):
            raise ValueError("unknown cannot be resolved solely by inference")
        if sha256(canonical({k: v for k, v in old.items() if k != "revision_sha256"})) != old.get("revision_sha256"):
            raise ValueError("unknown revision integrity mismatch")
        new = {**old, "revision": expected_revision + 1, "status": status,
               "evidence_ids": sorted(set(old["evidence_ids"] + (evidence_ids or []))),
               "resolution": resolution if status == "resolved" else old["resolution"]}
        new["revision_sha256"] = sha256(canonical({k: v for k, v in new.items() if k != "revision_sha256"}))
        # Retain each prior state for an audit. This has a small multi-process race;
        # a single workbench writer is required in v0.2.
        write_once(self.root / "unknowns" / "history" / f"{unknown_id}_r{expected_revision}.json", old)
        atomic_replace(path, new)
        return new

    def unresolved_for(self, claim_id: str) -> list[str]:
        issues = []
        for path in (self.root / "unknowns").glob("unk_*.json"):
            item = read_json(path)
            if claim_id in item["claim_ids"] and item["status"] != "resolved":
                issues.append(item["unknown_id"])
        return sorted(issues)

    def export_bundle(self, path: str | Path) -> dict[str, Any]:
        """Portable evidence and unknown history; no raw PDF, model weights or secrets."""
        if (out := Path(path).resolve()).is_relative_to(self.root) and out.parts[-2] == "evidence":
            raise ValueError("export destination must not be an Evidence record path")
        manifest = read_json(self.root / "session.json")
        records = [self.get(p.stem) for p in sorted((self.root / "evidence").glob("ev_*.json"))]
        unknowns = [read_json(p) for p in sorted((self.root / "unknowns").glob("unk_*.json"))]
        unknown_history = [read_json(p) for p in sorted((self.root / "unknowns" / "history").glob("unk_*.json"))]
        bundle = {"schema": "paperrev/evidence-bundle-v2", "session": manifest,
                  "records": records, "unknowns": unknowns, "unknown_history": unknown_history,
                  "scope_note": "No source PDF, weights, private data or generated code bytes are in this bundle."}
        write_once(out, bundle)
        return {"path": str(out), "records": len(records), "unknowns": len(unknowns), "sha256": hash_file(out)}

    @staticmethod
    def verify_bundle(path: str | Path) -> dict[str, Any]:
        bundle = read_json(Path(path))
        if bundle.get("schema") != "paperrev/evidence-bundle-v2":
            raise ValueError("unsupported bundle schema")
        seen = set()
        for record in bundle.get("records", []):
            identity = {k: v for k, v in record.items() if k != "evidence_id"}
            evid = "ev_" + sha256(canonical(identity))
            if record.get("evidence_id") != evid or evid in seen:
                raise ValueError("tampered or duplicate evidence in bundle")
            seen.add(evid)
        for record in bundle.get("records", []):
            if any(p not in seen for p in record["sources"]):
                raise ValueError("missing cited parent evidence")
        for unknown in bundle.get("unknowns", []) + bundle.get("unknown_history", []):
            revision = unknown.get("revision_sha256")
            if revision != sha256(canonical({k: v for k, v in unknown.items() if k != "revision_sha256"})):
                raise ValueError("unknown history integrity mismatch")
            if any(ev not in seen for ev in unknown.get("evidence_ids", [])):
                raise ValueError("unknown cites an absent evidence record")
        return {"verified": True, "records": len(seen), "unknowns": len(bundle.get("unknowns", [])),
                "unknown_revisions": len(bundle.get("unknown_history", []))}

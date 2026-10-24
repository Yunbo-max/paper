"""Bounded execution, observable behavior comparison, and claim-level verification.

Generated code never runs implicitly. Standard test execution requires Docker,
no network, a read-only source mount and an explicitly selected local image.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .workbench import Workbench, canonical, hash_file, read_json, sha256, write_once


def candidate_folder(wb: Workbench, candidate_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", candidate_id):
        raise ValueError("invalid candidate id")
    folder = wb.root / "candidates" / candidate_id
    if not folder.is_dir() or folder.is_symlink():
        raise FileNotFoundError("candidate does not exist or is a symlink")
    return folder


def candidate_manifest(wb: Workbench, candidate_id: str) -> dict[str, Any]:
    folder = candidate_folder(wb, candidate_id)
    manifest = read_json(folder / "candidate.json")
    if manifest.get("candidate_id") != candidate_id:
        raise ValueError("candidate identity mismatch")
    for name, expected in manifest.get("files", {}).items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("candidate contains missing or symlinked source")
        if hash_file(path) != expected:
            raise ValueError(f"candidate code changed since generation: {name}")
    allowed = set(manifest.get("files", {})) | {"candidate.json"}
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlinks in candidate are forbidden")
        if path.is_file() and path.relative_to(folder).as_posix() not in allowed:
            raise ValueError("candidate contains an untracked file: " + str(path.relative_to(folder)))
    return manifest


def run_tests(wb: Workbench, candidate_id: str, *, image: str, timeout_sec: int = 120,
              memory_mb: int = 1024, cpus: float = 2.0) -> dict[str, Any]:
    """Run untrusted candidate tests using a preinstalled container image.

    No pip install, network, host fallback or Docker image pull is attempted.
    """
    manifest = candidate_manifest(wb, candidate_id)
    if not image or not re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,180}", image):
        raise ValueError("explicit Docker image name/digest is required")
    if not 5 <= timeout_sec <= 1800 or not 128 <= memory_mb <= 32768 or not 0.25 <= cpus <= 16:
        raise ValueError("out-of-range compute budget")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker not found; refusing to execute generated code on host")
    folder = candidate_folder(wb, candidate_id).resolve()
    container_name = "paperrev-" + uuid.uuid4().hex[:24]
    cmd = [docker, "run", "--rm", "--name", container_name,
           "--pull=never", "--network=none", "--read-only",
           "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
           "--memory", f"{memory_mb}m", "--cpus", str(cpus), "--user", "65534:65534",
           "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-e", "PYTHONPATH=/work/src", "--mount", f"type=bind,source={folder},target=/work,readonly",
           "--workdir", "/work", image, "python", "-m", "unittest", "discover", "-s", "tests", "-v"]
    started = time.monotonic()
    cleanup = "not-needed"
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_sec, check=False)
        status, code, out, err = ("pass" if done.returncode == 0 else "fail"), done.returncode, done.stdout, done.stderr
        found = re.search(r"Ran\s+(\d+)\s+tests?\s+in\s+", out + "\n" + err)
        if done.returncode == 0 and (found is None or int(found.group(1)) < 1):
            status = "inconclusive"  # clean exit without executing any tests is not verification
    except subprocess.TimeoutExpired as exc:
        status, code = "timeout", None
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        # Killing the docker client alone does not prove the daemon released its container.
        try:
            release = subprocess.run([docker, "rm", "-f", container_name], stdin=subprocess.DEVNULL,
                                     capture_output=True, text=True, timeout=20, check=False)
            cleanup = "forced-remove-accepted" if release.returncode == 0 else "unconfirmed"
        except (OSError, subprocess.TimeoutExpired):
            cleanup = "unconfirmed"
    duration = time.monotonic() - started
    record = {"schema": "paperrev/test-run-v2", "candidate_id": candidate_id,
              "candidate_manifest_sha256": hash_file(folder / "candidate.json"),
              "image": image, "timeout_sec": timeout_sec, "memory_mb": memory_mb, "cpus": cpus,
              "argv": cmd[1:], "status": status, "exit_code": code,
              "elapsed_sec": round(duration, 4), "container_cleanup": cleanup,
              "stdout": out[:65536], "stderr": err[:65536],
              "stdout_truncated": len(out) > 65536, "stderr_truncated": len(err) > 65536,
              "scope": "candidate unit tests only; not paper metric reproduction"}
    run_id = "run_" + sha256(canonical(record))[:32]
    write_once(wb.root / "runs" / (run_id + ".json"), record)
    evid = wb.record(kind="candidate-unit-tests", authority="controlled-execution", confidence="observed",
                     subject={"candidate_id": candidate_id, "run_id": run_id,
                              "candidate_manifest_sha256": record["candidate_manifest_sha256"]},
                     result={"status": status, "exit_code": code, "runtime_sec": record["elapsed_sec"],
                             "source_file": str((Path("runs") / (run_id + ".json")))},
                     coverage="complete" if status in ("pass", "fail") else "partial",
                     limitations=["Passing generated unit tests does not reproduce published experimental results."])
    return {"run_id": run_id, "evidence_id": evid["evidence_id"], "status": status,
            "exit_code": code, "duration_sec": record["elapsed_sec"]}


def _rows(path: str | Path) -> dict[str, Any]:
    rows = {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), (str, int)) or "output" not in obj:
            raise ValueError(f"invalid prediction row {line_no}")
        key = str(obj["id"])
        if key in rows:
            raise ValueError(f"duplicate observation id {key}")
        rows[key] = obj["output"]
    if not rows:
        raise ValueError("empty observations")
    return rows


def _compare(a: Any, b: Any, atol: float, rtol: float) -> tuple[int, int]:
    """Returns (matched leaves, compared leaves); shape mismatch is one mismatch."""
    if type(a) is not type(b):
        return (0, 1)
    if isinstance(a, bool):
        return (int(a == b), 1)
    if isinstance(a, (float, int)):
        if not (math.isfinite(a) and math.isfinite(b)):
            return (0, 1)
        return (int(math.isclose(a, b, rel_tol=rtol, abs_tol=atol)), 1)
    if isinstance(a, (str, type(None))):
        return (int(a == b), 1)
    if isinstance(a, list):
        if len(a) != len(b):
            return (0, 1)
        good, total = 0, 0
        for x, y in zip(a, b):
            g, n = _compare(x, y, atol, rtol)
            good, total = good + g, total + n
        return (good, total) if total else (1, 1)
    if isinstance(a, dict):
        if set(a) != set(b):
            return (0, 1)
        good, total = 0, 0
        for k in sorted(a):
            g, n = _compare(a[k], b[k], atol, rtol)
            good, total = good + g, total + n
        return (good, total) if total else (1, 1)
    return (0, 1)


def compare_outputs(wb: Workbench, reference: str | Path, candidate: str | Path, *,
                    atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
    if not 0 <= atol <= 1 or not 0 <= rtol <= 1:
        raise ValueError("atol and rtol must be in 0..1")
    ref, got = _rows(reference), _rows(candidate)
    common = sorted(set(ref) & set(got))
    if not common:
        raise ValueError("no comparable id between reference and candidate")
    matched, total = 0, 0
    discrepancies = []
    for key in common:
        g, n = _compare(ref[key], got[key], atol, rtol)
        matched, total = matched + g, total + n
        if g < n:
            discrepancies.append(key)
    coverage = "complete" if set(ref) == set(got) else "partial"
    result = {"schema": "paperrev/output-comparison-v2", "reference_sha256": hash_file(Path(reference)),
              "candidate_sha256": hash_file(Path(candidate)), "reference_cases": len(ref),
              "candidate_cases": len(got), "paired_cases": len(common),
              "missing_in_candidate": sorted(set(ref) - set(got)),
              "extra_in_candidate": sorted(set(got) - set(ref)),
              "compared_values": total, "matching_values": matched,
              "matched_fraction": matched / total, "discrepancy_ids": discrepancies[:100],
              "tolerance": {"atol": atol, "rtol": rtol}, "coverage": coverage,
              "note": "Output agreement on provided JSONL is not proof of original implementation identity."}
    cmp_id = "cmp_" + sha256(canonical(result))[:32]
    write_once(wb.root / "comparisons" / (cmp_id + ".json"), result)
    evid = wb.record(kind="output-comparison", authority="controlled-execution", confidence="observed",
                     subject={"comparison_id": cmp_id, "reference_sha256": result["reference_sha256"],
                              "candidate_sha256": result["candidate_sha256"]},
                     result={"coverage": coverage, "paired_cases": len(common),
                             "matched_fraction": result["matched_fraction"], "comparison_path": f"comparisons/{cmp_id}.json"},
                     coverage=coverage,
                     limitations=["Correctness of the reference observations is not independently authenticated.",
                                  "Reference and candidate rows can originate from different conditions unless separately controlled."])
    return {"comparison_id": cmp_id, "evidence_id": evid["evidence_id"], **result}


def verify_claims(wb: Workbench, plan_sha: str, bindings: dict[str, Any]) -> dict[str, Any]:
    """Finite obligation verification; intentionally no global paper-success flag.

    bindings: {claim_id: {kind: 'unit-test'|'output-match', evidence_id: str,
                          candidate_id?: str, min_fraction?: float}}
    """
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha):
        raise ValueError("invalid plan SHA")
    plan = read_json(wb.root / "plans" / (plan_sha + ".json"))
    claims = []
    for claim in plan["claims"]:
        cid = claim["id"]
        binding = bindings.get(cid)
        if not isinstance(binding, dict):
            claims.append({"claim_id": cid, "status": "unknown", "reason": "no independently observed verification"})
            continue
        evid = wb.get(binding["evidence_id"])
        reasons = []
        if evid["confidence"] != "observed" or evid["authority"] != "controlled-execution":
            reasons.append("supplied evidence does not observe a controlled execution")
        kind = binding.get("kind")
        if kind == "unit-test":
            if evid["kind"] != "candidate-unit-tests":
                reasons.append("wrong evidence kind")
            if evid["subject"].get("candidate_id") != binding.get("candidate_id"):
                reasons.append("candidate identity mismatch")
            if evid["result"].get("status") == "timeout":
                reasons.append("execution timeout")
            passed = evid["result"].get("status") == "pass"
        elif kind == "output-match":
            if evid["kind"] != "output-comparison":
                reasons.append("wrong evidence kind")
            if evid["result"].get("coverage") != "complete":
                reasons.append("incomplete pair coverage")
            minimum = binding.get("min_fraction")
            if not isinstance(minimum, (int, float)) or not 0 <= minimum <= 1:
                raise ValueError("output-match needs min_fraction in [0,1]")
            passed = evid["result"].get("matched_fraction", -1) >= minimum
        else:
            raise ValueError("unknown binding kind")
        status = "unknown" if reasons else ("pass" if passed else "fail")
        blockers = wb.unresolved_for(cid)
        if blockers and status == "pass":
            status = "unknown"
            reasons.append("active implementation unknowns block claim closure")
        claims.append({"claim_id": cid, "status": status, "verification_kind": kind,
                       "evidence_id": binding["evidence_id"], "unresolved_unknowns": blockers,
                       "reasons": reasons, "interpretation": "scoped check only, not author training reproduction"})
    result = {"schema": "paperrev/claim-verification-v2", "plan_sha256": plan_sha,
              "claims": claims, "pass": sum(c["status"] == "pass" for c in claims),
              "fail": sum(c["status"] == "fail" for c in claims),
              "unknown": sum(c["status"] == "unknown" for c in claims),
              "paper_reproduced": False,
              "limitations": ["A passed unit test does not prove scientific outcome equivalence.",
                              "External dataset provenance and author implementation identity remain separate."]}
    record_id = "verification_" + sha256(canonical(result))[:32]
    write_once(wb.root / "comparisons" / (record_id + ".json"), result)
    return {**result, "report_path": str(wb.root / "comparisons" / (record_id + ".json"))}

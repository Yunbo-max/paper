"""REA-style controlled differential probes of observable model/API behavior.

A hypothesis is a finite relation between the outputs of a base input and a
single declared intervention. The analyzer authenticates both captures, the
intervention manifest, and the *actual* input cases before scoring any pair.
Finite evidence never establishes a globally equivalent model or training run.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

from .blackbox import _pointer, cases
from .workbench import Workbench, canonical, hash_file, read_json, sha256, write_once


_RULES = {"same", "different", "increase", "decrease", "shift"}


def _checks(spec: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if spec.get("schema") != "paperrev/interventions-v1":
        raise ValueError("expected a paperrev/interventions-v1 specification")
    changes = spec.get("changes")
    if not isinstance(changes, list) or not 1 <= len(changes) <= 30:
        raise ValueError("spec requires 1..30 changes")
    by_id = {}
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("intervention change must be an object")
        ident = change.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", ident) or ident in by_id:
            raise ValueError("invalid or duplicate intervention id")
        if change.get("op") not in ("replace", "add", "reverse") or not isinstance(change.get("path"), str):
            raise ValueError("invalid intervention operation or path")
        if not change["path"].startswith("/") or change["path"] == "/":
            raise ValueError("intervention must select a non-root JSON field")
        by_id[ident] = change
    checks = spec.get("checks")
    if not isinstance(checks, list) or not 1 <= len(checks) <= 60:
        raise ValueError("spec requires 1..60 diagnostic checks")
    seen = set()
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("check must be an object")
        cid = check.get("id")
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", cid) or cid in seen:
            raise ValueError("invalid or duplicate check id")
        seen.add(cid)
        if check.get("intervention_id") not in by_id or check.get("relation") not in _RULES:
            raise ValueError(f"check {cid} has unknown intervention or relation")
        if not isinstance(check.get("output_pointer", ""), str):
            raise ValueError("output_pointer must be a string")
        tol = check.get("tolerance", 1e-6)
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) or not 0 <= tol <= 1:
            raise ValueError(f"check {cid} tolerance must be finite in [0,1]")
        if check["relation"] == "shift":
            shift = check.get("expected_delta")
            if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(shift):
                raise ValueError(f"check {cid} shift needs finite expected_delta")
    return by_id, checks


def _put_back(mutated: Any, path: str, value: Any) -> Any:
    """Reverse a mutation solely at the declared pointer; other fields must match."""
    clone = copy.deepcopy(mutated)
    segments = path.split("/")[1:]
    node = clone
    for raw in segments[:-1]:
        if re.search(r"~(?![01])", raw):
            raise ValueError("invalid JSON pointer")
        key = raw.replace("~1", "/").replace("~0", "~")
        node = node[int(key)] if isinstance(node, list) else node[key]
    raw = segments[-1]
    if re.search(r"~(?![01])", raw):
        raise ValueError("invalid JSON pointer")
    leaf = raw.replace("~1", "/").replace("~0", "~")
    node[int(leaf) if isinstance(node, list) else leaf] = copy.deepcopy(value)
    return clone


def _validate_intervention(base: Any, changed: Any, change: dict[str, Any]) -> None:
    path = change["path"]
    old, new = _pointer(base, path), _pointer(changed, path)
    if old == new or _put_back(changed, path, old) != base:
        raise ValueError("probe modifies other input fields, or the declared change is a no-op")
    op = change["op"]
    if op == "replace":
        expected = change.get("value")
    elif op == "reverse":
        if not isinstance(old, list) or len(old) < 2:
            raise ValueError("reverse needs a nontrivial list")
        expected = list(reversed(old))
    else:
        delta = change.get("value")
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (old, delta))):
            raise ValueError("add requires numeric operands")
        expected = old + delta
        if not math.isfinite(expected):
            raise ValueError("non-finite numerical intervention")
    if new != expected:
        raise ValueError("probe does not match the declared intervention semantics")


def _capture(wb: Workbench, evidence_id: str, case_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = wb.get(evidence_id)
    if (evidence.get("kind") != "api-observation" or evidence.get("confidence") != "observed"
            or evidence.get("authority") != "external-observation"):
        raise ValueError("expected an observed, source-bound API capture")
    if case_file.is_symlink() or hash_file(case_file) != evidence["subject"].get("cases_sha256"):
        raise ValueError("capture source cases SHA256 differs from frozen input")
    rel = evidence["result"].get("reference_file")
    if (not isinstance(rel, str) or not re.fullmatch(r"observations/capture_[a-z0-9]+\.jsonl", rel)):
        raise ValueError("capture evidence has an unsafe or missing reference location")
    reference = wb.root / rel
    if reference.is_symlink() or not reference.is_file() or hash_file(reference) != evidence["result"].get("reference_sha256"):
        raise ValueError("API reference capture has changed after observation")
    capture_id = evidence["subject"].get("capture_id")
    if rel != f"observations/{capture_id}.jsonl":
        raise ValueError("capture identity is not consistent with reference path")
    summary_path = wb.root / "observations" / (capture_id + ".json")
    summary = read_json(summary_path)
    metadata_digest = evidence["result"].get("capture_meta_sha256")
    if not isinstance(metadata_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", metadata_digest):
        raise ValueError("capture metadata is not bound to immutable Evidence; recapture with current version")
    if sha256(canonical(summary)) != metadata_digest:
        raise ValueError("capture metadata changed after observation")
    if (summary.get("capture_id") != capture_id or summary.get("reference_sha256") != hash_file(reference)
            or summary.get("source_cases_sha256") != hash_file(case_file)
            or summary.get("observed") != evidence["result"].get("observed")):
        raise ValueError("API capture summary disagrees with immutable evidence")
    responses = {}
    for n, line in enumerate(reference.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or "id" not in item or "output" not in item:
            raise ValueError(f"malformed API capture row {n}")
        key = str(item["id"])
        if key in responses:
            raise ValueError("duplicate API capture row ID")
        responses[key] = item["output"]
    if len(responses) != summary["observed"]:
        raise ValueError("capture output count and summary disagree")
    return summary, responses


def _same(a: Any, b: Any, tolerance: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=tolerance, rel_tol=0)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y, tolerance) for x, y in zip(a, b))
    return a == b


def _finite_scalar(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("order/shift relation requires a finite numerical output")
    return float(value)


def _relation(old: Any, new: Any, check: dict[str, Any]) -> tuple[bool, float | None]:
    tol = float(check.get("tolerance", 1e-6))
    op = check["relation"]
    if op == "same":
        return _same(old, new, tol), None
    if op == "different":
        return not _same(old, new, tol), None
    a, b = _finite_scalar(old), _finite_scalar(new)
    delta = b - a
    if op == "increase":
        return delta > tol, delta
    if op == "decrease":
        return delta < -tol, delta
    return math.isclose(delta, float(check["expected_delta"]), rel_tol=0, abs_tol=tol), delta


def analyze_contrasts(
    wb: Workbench,
    *,
    baseline_evidence_id: str,
    intervention_evidence_id: str,
    intervention_plan_evidence_id: str,
    base_cases_path: str | Path,
    probe_cases_path: str | Path,
    spec_path: str | Path,
) -> dict[str, Any]:
    """Verify one-feature input diffs and assess bounded metamorphic claims.

    Counterexamples are finite observed pairs, not claims about the whole API.
    Missing or incomplete observations remain UNKNOWN; provenance is checked
    before an output may count toward any mathematical statement.
    """
    base_file, probe_file, spec_file = (Path(p) for p in (base_cases_path, probe_cases_path, spec_path))
    for file in (base_file, probe_file, spec_file):
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("source/probe/spec must be regular files <=2 MiB")
    base = cases(base_file, limit=2000)
    probes = cases(probe_file, limit=2000)
    spec = read_json(spec_file)
    changes, checks = _checks(spec)
    plan = wb.get(intervention_plan_evidence_id)
    if (plan.get("kind") != "intervention-plan" or plan.get("confidence") != "inferred"
            or plan.get("subject", {}).get("cases_sha256", plan.get("subject", {}).get("input_sha256")) != hash_file(base_file)
            or plan.get("subject", {}).get("spec_sha256") != hash_file(spec_file)
            or plan.get("result", {}).get("sha256", plan.get("result", {}).get("output_sha256")) != hash_file(probe_file)):
        raise ValueError("intervention-plan Evidence does not bind these exact inputs")
    base_by_id = {str(row["id"]): row for row in base}
    expected = {f"{row['id']}::{ident}" for row in base for ident in changes}
    if {str(row["id"]) for row in probes} != expected or len(probes) != len(expected):
        raise ValueError("intervention inputs must be the complete declared cross product")
    for probe in probes:
        parent, intervention = str(probe.get("parent_id")), probe.get("intervention_id")
        if (not isinstance(intervention, str) or intervention not in changes
                or str(probe["id"]) != parent + "::" + intervention or parent not in base_by_id):
            raise ValueError("probe lacks a valid parent and intervention identity")
        _validate_intervention(base_by_id[parent]["input"], probe["input"], changes[intervention])
    baseline, before = _capture(wb, baseline_evidence_id, base_file)
    intervention, after = _capture(wb, intervention_evidence_id, probe_file)
    if baseline.get("endpoint") != intervention.get("endpoint") or baseline.get("output_pointer") != intervention.get("output_pointer"):
        raise ValueError("incompatible target origins, paths or output projections")
    # The capture summary and Evidence must agree on selected and complete rows.
    for summary, outputs, source in ((baseline, before, base), (intervention, after, probes)):
        ids = {str(row["id"]) for row in source}
        if not set(outputs) <= ids or summary["selected"] > len(source):
            raise ValueError("capture contains outputs not selected from source case set")
        if summary.get("coverage") == "complete" and set(outputs) != ids:
            raise ValueError("capture falsely claims complete input coverage")
    results = []
    for check in checks:
        ident = check["intervention_id"]
        good, bad, missing = [], [], []
        delta_values = []
        for case in base:
            key = str(case["id"])
            probe_key = key + "::" + ident
            if key not in before or probe_key not in after:
                missing.append(key)
                continue
            try:
                old = _pointer(before[key], check.get("output_pointer", ""))
                new = _pointer(after[probe_key], check.get("output_pointer", ""))
                valid, delta = _relation(old, new, check)
            except (ValueError, KeyError, IndexError, TypeError, OverflowError):
                missing.append(key)
                continue
            if valid:
                good.append(key)
            else:
                bad.append(key)
            if delta is not None:
                delta_values.append(delta)
        status = "counterexample_observed" if bad else "supported_on_pairs" if len(good) == len(base) else "unknown"
        results.append({
            "id": check["id"], "intervention_id": ident, "relation": check["relation"],
            "output_pointer": check.get("output_pointer", ""),
            "status": status, "paired": len(good) + len(bad), "supported": len(good),
            "contradicted": len(bad), "missing_or_incomparable": len(missing),
            "counterexample_case_ids": bad[:20], "unknown_case_ids": missing[:20],
            "mean_observed_delta": sum(delta_values) / len(delta_values) if delta_values else None,
        })
    coverage = "complete" if all(r["missing_or_incomparable"] == 0 for r in results) else "partial"
    report = {
        "schema": "paperrev/controlled-contrasts-v1",
        "baseline_evidence_id": baseline_evidence_id,
        "probe_evidence_id": intervention_evidence_id,
        "plan_evidence_id": intervention_plan_evidence_id,
        "baseline_cases_sha256": hash_file(base_file),
        "intervention_cases_sha256": hash_file(probe_file),
        "intervention_spec_sha256": hash_file(spec_file),
        "endpoint": baseline["endpoint"],
        "output_pointer": baseline["output_pointer"],
        "base_cases": len(base), "probe_cases": len(probes),
        "checks": results,
        "coverage": coverage,
        "summary": {"supported_on_pairs": sum(x["status"] == "supported_on_pairs" for x in results),
                    "counterexample_observed": sum(x["status"] == "counterexample_observed" for x in results),
                    "unknown": sum(x["status"] == "unknown" for x in results)},
        "limitations": [
            "Results describe a finite selection of observed input/output pairs, not a global mathematical law.",
            "A changed server/model version or stochastic sampling could explain between-capture differences.",
            "No original implementation, private training state or scientifically correct gold labels are recovered.",
        ],
    }
    digest = sha256(canonical(report))
    relative = Path("comparisons") / ("contrast_" + digest[:32] + ".json")
    write_once(wb.root / relative, report)
    ev = wb.record(kind="intervention-differential", authority="external-observation",
                   confidence="derived", subject={"report_sha256": hash_file(wb.root / relative)},
                   result={"report_path": relative.as_posix(), "coverage": coverage,
                           "checks": report["summary"]},
                   sources=[baseline_evidence_id, intervention_evidence_id, intervention_plan_evidence_id],
                   coverage=coverage, limitations=report["limitations"])
    return {**report, "comparison_path": str(wb.root / relative), "evidence_id": ev["evidence_id"]}

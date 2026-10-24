"""REA-style active black-box probing via *source-authenticated candidate disagreement*.

No code or remote model is executed by this module. It consumes outputs previously
produced by PaperRev's sandboxed candidate replay, chooses useful inputs for the
next *authorized* observation, then compares independently captured target I/O.
This is a version-space heuristic, not Bayesian information gain or author-source
recovery. When all candidates agree, their shared answer might still be wrong.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from .blackbox import cases
from .contrast import _capture
from .reproduce import _compare, _rows, candidate_folder, candidate_manifest
from .workbench import Workbench, canonical, hash_file, read_json, sha256, write_once


def _run_candidate(
    wb: Workbench,
    evidence_id: str,
    pool_sha: str,
    expected_ids: set[str],
) -> tuple[str, dict[str, Any]]:
    """Authenticate one complete Docker replay, its exact code and its output."""
    ev = wb.get(evidence_id)
    if (ev.get("kind") != "candidate-behavior" or
            ev.get("authority") != "controlled-execution" or
            ev.get("confidence") != "observed" or ev.get("coverage") != "complete" or
            ev.get("result", {}).get("status") != "ran"):
        raise ValueError("candidate Evidence must be an observed, complete, successful Docker replay")
    cand = ev.get("subject", {}).get("candidate_id")
    run_id = ev.get("subject", {}).get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"replay_[0-9a-f]{24}", run_id):
        raise ValueError("invalid replay identity")
    if not isinstance(cand, str):
        raise ValueError("replay is missing candidate id")
    run_file = wb.root / "runs" / (run_id + ".json")
    if run_file.is_symlink() or not run_file.is_file():
        raise ValueError("replay run ledger missing or symlinked")
    ledger = read_json(run_file)
    if run_id != "replay_" + sha256(canonical(ledger))[:24]:
        raise ValueError("replay ledger was modified after recording")
    if ledger.get("candidate_id") != cand or ledger.get("status") != "ran":
        raise ValueError("replay identity or status mismatch")
    original_input_hash = ledger.get("case_file_sha256", ledger.get("input_sha256"))
    if original_input_hash != pool_sha:
        raise ValueError("replay predictions belong to a different input pool")
    if ledger.get("cases_expected") != len(expected_ids) or ledger.get("cases_observed") != len(expected_ids):
        raise ValueError("replay case count does not cover the pool")
    folder = candidate_folder(wb, cand)
    candidate_manifest(wb, cand)  # checks every tracked source and forbids extras
    manifest_hash = ledger.get("manifest_sha256", ledger.get("candidate_manifest_sha256"))
    if manifest_hash != hash_file(folder / "candidate.json"):
        raise ValueError("candidate manifest differs from the source used for replay")
    if ev["result"].get("observed") != len(expected_ids) or ev["result"].get("output_sha256") != ledger.get("output_sha256"):
        raise ValueError("replay Evidence does not match its run ledger")
    if not isinstance(ledger.get("output_path"), str):
        raise ValueError("candidate replay lacks an output file")
    output = Path(ledger["output_path"])
    expected_parent = (wb.root / "observations").resolve()
    if (not output.is_absolute() or output.is_symlink() or not output.is_file() or
            output.parent.resolve() != expected_parent or
            not re.fullmatch(r"candidate_[0-9a-f]{24}\.jsonl", output.name)):
        raise ValueError("replay output path is untrusted")
    if hash_file(output) != ledger["output_sha256"]:
        raise ValueError("candidate output has been modified since replay")
    predictions = _rows(output)
    if set(predictions) != expected_ids:
        raise ValueError("replay prediction IDs do not match the frozen input pool")
    return cand, predictions


def _distance(a: Any, b: Any, scale: float) -> float:
    """A bounded max-leaf discrepancy on finite JSON, not an inferred KL."""
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if type(a) is type(b) and a == b else 1.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return 0.0
        try:
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("prediction contains a non-finite scalar")
            difference, largest = abs(a - b), max(abs(a), abs(b))
            if largest > 1e300:
                return 1.0
            return min(1.0, difference / (scale + largest))
        except OverflowError:
            # Python can parse JSON integers beyond the floating-point range.
            # Their equality is exact, but our bounded float distance cannot resolve
            # their relative gap; treat different huge values as a full disagreement.
            return 1.0
    if type(a) is not type(b):
        return 1.0
    if isinstance(a, list):
        if len(a) != len(b):
            return 1.0
        return max((_distance(x, y, scale) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict):
        if set(a) != set(b):
            return 1.0
        return max((_distance(a[k], b[k], scale) for k in a), default=0.0)
    if a is None or isinstance(a, str):
        return 0.0 if a == b else 1.0
    raise ValueError("candidate prediction contains unsupported JSON value")


def _selected_manifest(wb: Workbench, selection_evidence_id: str, pool: Path) -> tuple[dict[str, Any], Path]:
    ev = wb.get(selection_evidence_id)
    if ev.get("kind") != "active-probe-selection" or ev.get("authority") != "analyst-inference" or ev.get("confidence") != "derived":
        raise ValueError("expected an authenticated selection Evidence record")
    selection_id = ev.get("result", {}).get("selection_id")
    if not isinstance(selection_id, str) or not re.fullmatch(r"select_[0-9a-f]{24}", selection_id):
        raise ValueError("invalid selection identity")
    manifest_path = wb.root / "selections" / (selection_id + ".json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("selection manifest is absent or symlinked")
    manifest = read_json(manifest_path)
    if selection_id != "select_" + sha256(canonical(manifest))[:24]:
        raise ValueError("selection manifest has been modified")
    if ev["result"].get("manifest_sha256") != sha256(canonical(manifest)):
        raise ValueError("selection Evidence does not bind the stored manifest")
    if manifest.get("pool_sha256") != hash_file(pool):
        raise ValueError("selected inputs originated from another or modified pool")
    if manifest.get("candidate_evidence_ids") != ev.get("sources"):
        raise ValueError("selection candidate Evidence links are inconsistent")
    path = wb.root / "selections" / (selection_id + ".jsonl")
    if path.is_symlink() or not path.is_file() or hash_file(path) != manifest.get("selected_cases_sha256"):
        raise ValueError("selected input JSONL was modified")
    selected = cases(path) if manifest.get("selected_ids") else []
    if [str(r["id"]) for r in selected] != manifest.get("selected_ids"):
        raise ValueError("selected case IDs and manifest disagree")
    all_cases = {str(r["id"]): r for r in cases(pool)}
    if any(r != all_cases.get(str(r["id"])) for r in selected):
        raise ValueError("selected cases differ from the frozen input pool")
    return manifest, path


def select_probes(
    wb: Workbench, pool_file: str | Path, candidate_evidence_ids: Sequence[str], *,
    budget: int = 5, min_distance: float = 0.05, numeric_scale: float = 1.0,
    max_per_parent: int = 1,
) -> dict[str, Any]:
    """Greedy pair-coverage over sandboxed candidate predictions; **zero oracle calls**.

    The score is cost-adjusted pair discrepancy, discounted after a pair of
    hypotheses has already been separated by a selected case. This is a
    transparent experimental-design heuristic, not a calibrated uncertainty.
    """
    if (not 1 <= budget <= 200 or not 1 <= max_per_parent <= 200 or
            isinstance(min_distance, bool) or not isinstance(min_distance, (int, float)) or
            not math.isfinite(min_distance) or not 0 < min_distance <= 1 or
            isinstance(numeric_scale, bool) or not isinstance(numeric_scale, (int, float)) or
            not math.isfinite(numeric_scale) or numeric_scale <= 0):
        raise ValueError("invalid selection budget or numerical distance parameters")
    identifiers = list(candidate_evidence_ids)
    if not 2 <= len(identifiers) <= 16 or len(set(identifiers)) != len(identifiers):
        raise ValueError("selection requires 2..16 distinct candidate replay Evidence IDs")
    pool = Path(pool_file)
    items = cases(pool)
    pool_sha = hash_file(pool)
    ids = {str(item["id"]) for item in items}
    models = {}
    evidence_by_candidate = {}
    for evid in identifiers:
        name, results = _run_candidate(wb, evid, pool_sha, ids)
        if name in models:
            raise ValueError("same candidate cannot appear in the committee twice")
        models[name] = results
        evidence_by_candidate[name] = evid
    candidates = sorted(models)
    pair_labels = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]]
    indexed = {}
    for item in items:
        ident = str(item["id"])
        group = str(item.get("parent_id", item["id"]))
        raw_cost = item.get("query_cost", 1.0)
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)) or not math.isfinite(raw_cost) or not 1e-6 <= raw_cost <= 1e6:
            raise ValueError(f"invalid positive query_cost on case {ident}")
        distances = {f"{a}|{b}": _distance(models[a][ident], models[b][ident], numeric_scale)
                     for a, b in pair_labels}
        indexed[ident] = {"row": item, "group": group, "cost": float(raw_cost), "distances": distances,
                          "max_disagreement": max(distances.values())}
    chosen, used_groups, covered_pairs = [], {}, set()
    while len(chosen) < budget:
        best = None
        for ident in sorted(indexed):
            e = indexed[ident]
            if ident in chosen or used_groups.get(e["group"], 0) >= max_per_parent:
                continue
            # Novel candidate-pair distinctions receive full weight; redundant pairs
            # retain a small weight for repeat observations / robustness.
            gain = sum((d if pair not in covered_pairs else 0.15 * d)
                       for pair, d in e["distances"].items() if d >= min_distance)
            value = gain / e["cost"]
            if value <= 0:
                continue
            if best is None or value > best[0] + 1e-12 or (abs(value - best[0]) <= 1e-12 and ident < best[1]):
                best = (value, ident)
        if best is None:
            break
        value, ident = best
        chosen.append(ident)
        group = indexed[ident]["group"]
        used_groups[group] = used_groups.get(group, 0) + 1
        covered_pairs.update(pair for pair, d in indexed[ident]["distances"].items() if d >= min_distance)
        indexed[ident]["selection_score"] = round(value, 9)
    selected = [indexed[ident]["row"] for ident in chosen]
    chosen_evidence = [evidence_by_candidate[c] for c in candidates]
    params = {"budget": budget, "min_distance": float(min_distance),
              "numeric_scale": float(numeric_scale), "max_per_parent": max_per_parent,
              "disagreement_metric": "max-leaf-absolute-difference-scaled-by-1+max-magnitude",
              "selection_rule": "greedy uncovered-pair discrepancy per declared query_cost"}
    # The SHA binds the literal JSONL of the selected inputs, not the text of the pool alone.
    selected_text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in selected)
    manifest = {"schema": "paperrev/active-selection-v1", "pool_sha256": pool_sha,
                "pool_cases": len(items), "candidate_ids": candidates,
                "candidate_evidence_ids": chosen_evidence,
                "params": params, "selected_ids": chosen,
                "selected_cases_sha256": sha256(selected_text.encode("utf-8")),
                "pairs_covered": sorted(covered_pairs),
                "selection_scores": [
                    {"id": ident, "score": indexed[ident]["selection_score"],
                     "max_disagreement": round(indexed[ident]["max_disagreement"], 9),
                     "cost": indexed[ident]["cost"]} for ident in chosen],
                "status": "selected" if chosen else "no-discriminatory-inputs",
                "limitations": ["Candidate disagreement does not prove any candidate is correct.",
                                "Only already executed and authenticated Docker replay results were scored.",
                                "Stochastic candidate prediction variation is not modeled here."]}
    selection_id = "select_" + sha256(canonical(manifest))[:24]
    dest = wb.root / "selections" / (selection_id + ".jsonl")
    write_once(wb.root / "selections" / (selection_id + ".json"), manifest)
    # Existing atomic write helper handles JSON only; use exclusive binary write for
    # selected JSONL and reject divergent repeated data.
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.is_symlink() or dest.read_text(encoding="utf-8") != selected_text:
            raise ValueError("existing selection input file has different content")
    else:
        with dest.open("x", encoding="utf-8") as stream:
            stream.write(selected_text)
    ev = wb.record(kind="active-probe-selection", authority="analyst-inference",
                   confidence="derived", subject={"selection_id": selection_id, "pool_sha256": pool_sha},
                   result={"selection_id": selection_id, "manifest_sha256": sha256(canonical(manifest)),
                           "selected_cases_sha256": manifest["selected_cases_sha256"], "selected": len(chosen),
                           "selected_ids": chosen},
                   sources=chosen_evidence, coverage="bounded",
                   limitations=manifest["limitations"])
    return {"selection_id": selection_id, "evidence_id": ev["evidence_id"],
            "cases_path": str(dest), "selected_ids": chosen,
            "score_summary": manifest["selection_scores"], "pairs_covered": len(covered_pairs),
            "candidate_pairs": len(pair_labels), "status": manifest["status"],
            "no_external_api_called": True}


def evaluate_selection(
    wb: Workbench, *, pool_file: str | Path, selection_evidence_id: str,
    observation_evidence_id: str, atol: float = 1e-6, rtol: float = 1e-5,
) -> dict[str, Any]:
    """Finite version-space update from a genuine, authenticated oracle capture."""
    if any(isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or not 0 <= t <= 1
           for t in (atol, rtol)):
        raise ValueError("comparison tolerances must be finite and within [0,1]")
    pool = Path(pool_file)
    manifest, cases_path = _selected_manifest(wb, selection_evidence_id, pool)
    if not manifest["selected_ids"]:
        raise ValueError("no discriminating query was selected; there is nothing to evaluate")
    obs_summary, observed = _capture(wb, observation_evidence_id, cases_path)
    if not set(observed).issubset(set(manifest["selected_ids"])):
        raise ValueError("oracle returned an output for an unselected case")
    if obs_summary.get("coverage") == "complete" and set(observed) != set(manifest["selected_ids"]):
        raise ValueError("complete oracle capture contains missing selected samples")
    pool_sha = hash_file(pool)
    pool_ids = {str(row["id"]) for row in cases(pool)}
    decisions = []
    for expected_id, evid in zip(manifest["candidate_ids"], manifest["candidate_evidence_ids"]):
        candidate_id, rows = _run_candidate(wb, evid, pool_sha, pool_ids)
        if candidate_id != expected_id:
            raise ValueError("committee member identity no longer matches the selection")
        mismatches, matching_leaves, compared_leaves = [], 0, 0
        for ident in manifest["selected_ids"]:
            if ident not in observed:
                continue
            matched, total = _compare(observed[ident], rows[ident], atol, rtol)
            matching_leaves += matched
            compared_leaves += total
            if matched != total:
                mismatches.append(ident)
        status = ("disagreed-on-observed-inputs" if mismatches else
                  "compatible-on-tested-inputs" if len(observed) == len(manifest["selected_ids"]) else "inconclusive")
        decisions.append({"candidate_id": candidate_id, "status": status,
                          "mismatch_ids": mismatches[:200], "cases_checked": len(observed),
                          "fraction_matching_leaves": matching_leaves / compared_leaves if compared_leaves else None})
    coverage = "complete" if len(observed) == len(manifest["selected_ids"]) else "partial"
    report = {"schema": "paperrev/active-evaluation-v1",
              "selection_id": "select_" + sha256(canonical(manifest))[:24],
              "selection_evidence_id": selection_evidence_id,
              "selected_ids": manifest["selected_ids"],
              "candidate_assessments": decisions, "capture_coverage": coverage,
              "selected_cases": len(manifest["selected_ids"]), "observed_cases": len(observed),
              "tolerance": {"atol": float(atol), "rtol": float(rtol)},
              "limitations": ["Finite agreement is not proof of historical author source or mechanism identity.",
                              "A stochastic model may disagree across repeated runs, even with identical code.",
                              "Missing target observations never establish candidate compatibility."]}
    assessment_id = "assessment_" + sha256(canonical(report))[:24]
    write_once(wb.root / "selections" / (assessment_id + ".json"), report)
    ev = wb.record(kind="active-candidate-assessment", authority="analyst-inference", confidence="derived",
                   subject={"assessment_id": assessment_id},
                   result={"observed": len(observed), "selected": len(manifest["selected_ids"]),
                           "candidate_assessments": decisions},
                   sources=[selection_evidence_id, observation_evidence_id] + manifest["candidate_evidence_ids"],
                   coverage=coverage, limitations=report["limitations"])
    return {"assessment_id": assessment_id, "evidence_id": ev["evidence_id"],
            "coverage": coverage, "candidate_assessments": decisions,
            "total": len(decisions), "compatible": sum(d["status"] == "compatible-on-tested-inputs" for d in decisions),
            "contradicted_on_observed": sum(d["status"] == "disagreed-on-observed-inputs" for d in decisions),
            "status": "evaluated" if coverage == "complete" else "partial"}

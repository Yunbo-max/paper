"""StartLux black-/white-box pilot: capture actual decisions and audit gate choices.

Only real responses produce observational evidence. A gate sweep over ALWAYS
captures is a post-hoc counterfactual estimate, not a live AUTO evaluation.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .core import digest_bytes, digest_file, write_json


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {lineno} of {path} must be a JSON object")
        records.append(value)
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def endpoint_for(base_url: str, allow_remote: bool) -> str:
    value = urllib.parse.urlsplit(base_url)
    if value.scheme not in ("http", "https") or not value.hostname or value.username or value.password:
        raise ValueError("base URL must be http(s), with hostname and no embedded credentials")
    if value.query or value.fragment or value.path not in ("", "/"):
        raise ValueError("base URL must be an origin without path, query or fragment")
    is_local = value.hostname in ("localhost", "127.0.0.1", "::1")
    if not is_local and not allow_remote:
        raise ValueError("remote inference requires --allow-remote")
    if not is_local and value.scheme != "https":
        raise ValueError("remote inference requires HTTPS")
    return base_url.rstrip("/") + "/v1/systemone"


def _post_json(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "paperrev-lab/0.1"}
    if os.getenv("STARTLUX_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STARTLUX_API_KEY"]
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise ValueError("model response has no answers object")
    return result


def probe(
    dataset: str | Path,
    output: str | Path,
    base_url: str,
    *,
    execute: bool,
    allow_remote: bool = False,
    max_requests: int = 10,
    think_budget: int = 256,
    mix: float = 0.75,
    timeout: int = 600,
) -> dict[str, Any]:
    """Explicit opt-in HTTP capture: one ALWAYS call per selected dataset entry."""
    if not execute:
        raise ValueError("no inference was run: pass --execute to authorize HTTP requests")
    if max_requests < 1 or think_budget < 1 or timeout < 1 or not 0 <= mix <= 1:
        raise ValueError("limit, budget, timeout must be positive and mix must be in [0,1]")
    url = endpoint_for(base_url, allow_remote)
    items = read_jsonl(dataset)[:max_requests]
    controls = {
        "think_mode": "always",
        "think_budget": think_budget,
        "mix": mix,
        "none_check": False,
        "allow_none": False,
        "return_details": True,
        "return_thought": False,
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite capture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_sha = digest_file(dataset)
    ok = 0
    # Persist each result as it arrives. A partial file never claims completion.
    with output.open("x", encoding="utf-8") as handle:
        for index, item in enumerate(items):
            if not isinstance(item.get("questions"), dict) or "state" not in item:
                raise ValueError(f"dataset entry {index} needs state and questions")
            body = {"state": item["state"], "questions": item["questions"], **controls}
            request_sha = digest_bytes(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            rec: dict[str, Any] = {
                "id": item.get("id", index), "dataset_sha256": dataset_sha,
                "request_sha256": request_sha, "questions": item["questions"],
                "gold": item.get("gold", {}), "controls": controls,
            }
            try:
                rec["response"] = _post_json(url, body, timeout)
                rec["status"] = "ok"
                ok += 1
            except urllib.error.HTTPError as exc:
                rec["status"] = "error"
                rec["error"] = {"kind": "http", "status": exc.code}
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                rec["status"] = "error"
                rec["error"] = {"kind": type(exc).__name__}
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if rec["status"] != "ok":
                break
    return {
        "status": "completed" if ok == len(items) else "partial",
        "successful_requests": ok,
        "requested": len(items),
        "dataset_sha256": dataset_sha,
        "capture_sha256": digest_file(output),
        "capture_path": str(output),
        "note": "uses official /v1/systemone inference API; no training code or weights were recovered",
    }


def _gold_key(spec: dict[str, Any], value: Any) -> str:
    kind = spec.get("type", "choice")
    if kind in ("bool", "noul"):
        if value is True or str(value).lower() in ("true", "yes", "1"):
            return "true"
        if value is False or str(value).lower() in ("false", "no", "0"):
            return "false"
        raise ValueError("unrecognized yes/no gold value")
    if kind == "score":
        if isinstance(value, bool):
            raise ValueError("score gold cannot be boolean")
        if isinstance(value, int):
            return str(value)
        criteria = spec.get("criteria", [])
        if isinstance(criteria, dict) and str(value) in criteria:
            return str(sorted(criteria, key=lambda x: int(x)).index(str(value)))
        return str(value)
    return str(value)


def _probabilities(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError("probabilities are missing")
    p = {str(k): float(v) for k, v in value.items()}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()):
        raise ValueError("probabilities must be finite in [0,1]")
    if abs(sum(p.values()) - 1) > 0.02:
        raise ValueError("probabilities are not normalized")
    return p


def _mix(a: dict[str, float], b: dict[str, float], mix: float) -> dict[str, float]:
    if set(a) != set(b):
        raise ValueError("first and post-thought distributions have different option ids")
    return {k: (1 - mix) * a[k] + mix * b[k] for k in a}


def _metrics(rows: list[dict[str, Any]], variants: list[tuple[dict[str, float], int]]) -> dict[str, float]:
    if len(rows) != len(variants):
        raise ValueError("missing comparison rows")
    n = len(rows)
    right = 0
    brier = 0.0
    nll = 0.0
    thought = 0
    tokens = 0
    for row, (p, cost) in zip(rows, variants):
        gold = row["gold"]
        if gold not in p:
            raise ValueError("gold label missing from probability distribution")
        right += max(p, key=p.get) == gold
        brier += sum((v - int(k == gold)) ** 2 for k, v in p.items())
        nll -= math.log(max(p[gold], 1e-15))
        thought += cost > 0
        tokens += cost
    return {
        "questions": n,
        "accuracy": right / n,
        "mean_brier": brier / n,
        "mean_nll": nll / n,
        "thought_share_estimate": thought / n,
        "mean_thought_tokens_estimate": tokens / n,
    }


def analyze(records_path: str | Path, output: str | Path, gates: list[float], mix: float = 0.75) -> dict[str, Any]:
    """Analyze frozen ALWAYS traces, with comparable complete-pair scope."""
    if not 0 <= mix <= 1 or not gates or any(not 0 <= g <= 1 for g in gates):
        raise ValueError("mix and gates must be in [0,1]")
    records = read_jsonl(records_path)
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("status") != "ok":
            excluded.append({"id": rec.get("id"), "reason": "request-failed"})
            continue
        if rec.get("controls", {}).get("think_mode") != "always":
            raise ValueError("gate counterfactuals require captures with think_mode=always")
        if rec.get("controls", {}).get("none_check") is not False or rec.get("controls", {}).get("allow_none") is not False:
            raise ValueError("gate pilot requires none_check=false and allow_none=false")
        if not isinstance(rec.get("gold"), dict):
            raise ValueError("gold must be a mapping")
        for key, spec in rec.get("questions", {}).items():
            if key not in rec["gold"]:
                excluded.append({"id": rec.get("id"), "question": key, "reason": "no-gold"})
                continue
            response = rec.get("response", {}).get("answers", {}).get(key, {})
            details = response.get("details") or {}
            before = (details.get("first_pass") or {}).get("probabilities")
            after = (details.get("after_thought") or {}).get("probabilities")
            if before is None or after is None:
                excluded.append({"id": rec.get("id"), "question": key, "reason": "missing-comparable-before-or-after"})
                continue
            p0, p1 = _probabilities(before), _probabilities(after)
            if set(p0) != set(p1):
                raise ValueError(f"option set mismatch: {rec.get('id')}/{key}")
            gold = _gold_key(spec, rec["gold"][key])
            if gold not in p0:
                raise ValueError(f"gold option absent: {rec.get('id')}/{key} = {gold}")
            thought = details.get("thought") or {}
            if thought.get("used") is not True:
                excluded.append({"id": rec.get("id"), "question": key, "reason": "no-thought-executed"})
                continue
            tokens = thought.get("tokens")
            if not isinstance(tokens, int) or tokens < 0:
                raise ValueError("missing or invalid thought token count")
            rows.append({"id": rec.get("id"), "question": key, "gold": gold, "p0": p0, "p1": p1,
                         "tokens": tokens, "top": max(p0.values())})
    if not rows:
        raise ValueError("no comparable gold-labeled first/after thought pairs; cannot evaluate gate")
    strategies: dict[str, dict[str, float]] = {
        "no_think": _metrics(rows, [(r["p0"], 0) for r in rows]),
        "after_thought_only": _metrics(rows, [(r["p1"], r["tokens"]) for r in rows]),
        "always_mix": _metrics(rows, [(_mix(r["p0"], r["p1"], mix), r["tokens"]) for r in rows]),
    }
    for gate in sorted(set(gates)):
        selected = [
            (_mix(r["p0"], r["p1"], mix), r["tokens"]) if r["top"] < gate else (r["p0"], 0)
            for r in rows
        ]
        strategies[f"gate_{gate:g}"] = _metrics(rows, selected)
    deltas = [math.log(max(r["p1"][r["gold"]], 1e-15))
              - math.log(max(r["p0"][r["gold"]], 1e-15)) for r in rows]
    result = {
        "schema": "paperrev/startlux-gate-audit-v1",
        "source_capture_sha256": digest_file(records_path),
        "comparison_scope": {"paired_questions": len(rows), "excluded": excluded},
        "strategies": strategies,
        "after_vs_before": {
            "mean_log_score_gain": sum(deltas) / len(deltas),
            "fraction_helped": sum(d > 0 for d in deltas) / len(deltas),
            "fraction_harmed": sum(d < 0 for d in deltas) / len(deltas),
        },
        "verification": "post-hoc gate simulation from actually observed always-think traces",
        "limitations": [
            "This is not a live auto-gate run; generated thoughts can vary with sampling.",
            "Thought tokens are estimated from captured always-think traces; latency is not inferred.",
            "Only complete paired gold-labeled questions contribute to all strategy metrics.",
            "Inference results cannot establish the original SFT or RLCD^2 training recipe.",
        ],
    }
    write_json(output, result)
    return result

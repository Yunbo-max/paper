"""Black-box model reconstruction from actual API observations, REA-style.

Observe a callable target, construct controlled input interventions, and replay
independently generated code against the *same* inputs. Observed behavior is not
historical author source or a recovered private training recipe.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .reproduce import candidate_folder, candidate_manifest, compare_outputs
from .workbench import Workbench, canonical, hash_file, sha256, write_once


def cases(path: str | Path, *, limit: int = 200) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("cases must be a regular JSONL file at most 2 MiB")
    if not 1 <= limit <= 2000:
        raise ValueError("case limit outside 1..2000")
    items, seen = [], set()
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        ident = row.get("id") if isinstance(row, dict) else None
        if not isinstance(ident, (str, int)) or isinstance(ident, bool) or "input" not in row:
            raise ValueError(f"case line {n} needs id and input")
        if str(ident) in seen:
            raise ValueError(f"duplicate case id: {ident}")
        seen.add(str(ident))
        items.append(row)
        if len(items) > limit:
            raise ValueError("case count exceeds declared bound")
    if not items:
        raise ValueError("no cases")
    return items


def _pointer(value: Any, path: str) -> Any:
    """Read a JSON Pointer without silently accepting absent fields."""
    if path == "":
        return value
    if not path.startswith("/"):
        raise ValueError("JSON Pointer must start with /")
    for part in path.split("/")[1:]:
        if re.search(r"~(?![01])", part):
            raise ValueError("invalid JSON Pointer escape")
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and part.isdecimal():
            value = value[int(part)]
        else:
            raise ValueError("JSON Pointer does not select a field")
    return value


def _endpoint(origin: str, endpoint_path: str, allow_remote: bool) -> str:
    parsed = urllib.parse.urlsplit(origin)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("origin must be an HTTP(S) URL without credentials")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("origin must have no path/query/fragment")
    local = parsed.hostname.lower() in ("localhost", "127.0.0.1", "::1")
    if not local and (not allow_remote or parsed.scheme != "https"):
        raise ValueError("remote API calls require explicit --allow-remote and HTTPS")
    if not endpoint_path.startswith("/") or endpoint_path.startswith("//") or any(c in endpoint_path for c in "?#\\"):
        raise ValueError("endpoint path must be a literal same-origin path")
    decoded = urllib.parse.unquote(endpoint_path)
    if decoded.startswith("//") or any(piece in (".", "..") for piece in decoded.split("/")):
        raise ValueError("endpoint path traversal is forbidden")
    return origin.rstrip("/") + endpoint_path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def capture_api(wb: Workbench, case_file: str | Path, *, origin: str, endpoint_path: str,
                execute: bool, allow_remote: bool = False, limit: int = 25,
                timeout: int = 120, output_pointer: str = "") -> dict[str, Any]:
    """POST frozen cases to an explicit local/HTTPS target; persist real outputs."""
    if not execute:
        raise ValueError("--execute required before any API request")
    if not 1 <= timeout <= 600 or not 1 <= limit <= 200:
        raise ValueError("timeout/limit outside bounds")
    url = _endpoint(origin, endpoint_path, allow_remote)
    rows = cases(case_file)
    selected = rows[:limit]
    case_hash = hash_file(Path(case_file))
    stamp = sha256(canonical({"url": url, "cases": case_hash, "nonce": uuid.uuid4().hex}))[:24]
    capture_id = "capture_" + stamp
    dest = wb.root / "observations" / (capture_id + ".jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(_NoRedirect)
    completed, failure = 0, None
    with dest.open("x", encoding="utf-8") as stream:
        for row in selected:
            body = canonical(row["input"])
            if len(body) > 256 * 1024:
                failure = {"id": row["id"], "reason": "input-too-large"}
                break
            headers = {"Content-Type": "application/json", "User-Agent": "paperrev/0.3"}
            if os.getenv("PAPERREV_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["PAPERREV_API_KEY"]
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with opener.open(request, timeout=timeout) as response:
                    if response.status != 200:
                        raise ValueError("non-200 response")
                    blob = response.read(4 * 1024 * 1024 + 1)
                    if len(blob) > 4 * 1024 * 1024:
                        raise ValueError("response too large")
                    output = _pointer(json.loads(blob), output_pointer)
                stream.write(json.dumps({"id": row["id"], "output": output}, ensure_ascii=False,
                                        allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush()
                completed += 1
            except (OSError, ValueError, KeyError, IndexError, TypeError, urllib.error.HTTPError) as exc:
                failure = {"id": row["id"], "reason": type(exc).__name__}
                break
    scope = "complete" if completed == len(rows) else "partial"
    result = {"schema": "paperrev/api-capture-v1", "capture_id": capture_id,
              "source_cases_sha256": case_hash, "reference_file": str(dest.relative_to(wb.root)),
              "reference_sha256": hash_file(dest), "selected": len(selected),
              "source_cases": len(rows), "observed": completed,
              "coverage": scope, "failure": failure,
              "endpoint": {"origin": origin, "path": endpoint_path}, "output_pointer": output_pointer,
              "limitations": ["Only selected JSON POST behavior was observed.",
                              "Model versions, hidden state and sampling can vary between runs.",
                              "An API observation does not reveal private model weights or training source."]}
    write_once(wb.root / "observations" / (capture_id + ".json"), result)
    ev = wb.record(kind="api-observation", authority="external-observation", confidence="observed",
                   subject={"capture_id": capture_id, "cases_sha256": case_hash},
                   result={"reference_file": result["reference_file"],
                           "reference_sha256": result["reference_sha256"],
                           "observed": completed, "expected": len(rows), "failure": failure},
                   coverage=scope, limitations=result["limitations"])
    return {**result, "evidence_id": ev["evidence_id"]}


def make_interventions(wb: Workbench, source_file: str | Path, spec_file: str | Path) -> dict[str, Any]:
    """Create single-input-field mutations as hypotheses, not observed outcomes."""
    originals = cases(source_file)
    spec = json.loads(Path(spec_file).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema") != "paperrev/interventions-v1":
        raise ValueError("invalid intervention schema")
    changes = spec.get("changes")
    if not isinstance(changes, list) or not 1 <= len(changes) <= 30:
        raise ValueError("expected 1..30 intervention definitions")
    output, names = [], set()
    for change in changes:
        name, path, op = change.get("id"), change.get("path"), change.get("op")
        if not isinstance(name, str) or not re.fullmatch("[A-Za-z0-9_.-]{1,40}", name) or name in names:
            raise ValueError("intervention id invalid or duplicated")
        if not isinstance(path, str) or not path.startswith("/") or op not in ("replace", "add", "reverse"):
            raise ValueError("invalid intervention path or operation")
        names.add(name)
        for row in originals:
            updated = copy.deepcopy(row["input"])
            previous = _pointer(updated, path)
            if op == "replace":
                value = copy.deepcopy(change.get("value"))
            elif op == "reverse":
                if not isinstance(previous, list) or len(previous) < 2:
                    raise ValueError("reverse needs list with at least two entries")
                value = list(reversed(previous))
            else:
                delta = change.get("value")
                if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (previous, delta)):
                    raise ValueError("add requires numeric arguments")
                value = previous + delta
                if not math.isfinite(value):
                    raise ValueError("non-finite intervention value")
            keys = path.split("/")[1:]
            parent = updated
            for key in keys[:-1]:
                key = key.replace("~1", "/").replace("~0", "~")
                parent = parent[int(key)] if isinstance(parent, list) else parent[key]
            leaf = keys[-1].replace("~1", "/").replace("~0", "~")
            parent[int(leaf) if isinstance(parent, list) else leaf] = value
            if updated == row["input"]:
                raise ValueError(f"intervention {name} is a no-op for case {row['id']}")
            output.append({"id": f"{row['id']}::{name}", "input": updated,
                           "parent_id": row["id"], "intervention_id": name})
    identity = {"cases_sha256": hash_file(Path(source_file)), "spec_sha256": hash_file(Path(spec_file))}
    run_id = "probes_" + sha256(canonical(identity))[:24]
    target = wb.root / "observations" / (run_id + ".jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in output)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError("conflicting intervention result")
    else:
        with target.open("x", encoding="utf-8") as f:
            f.write(text)
    evid = wb.record(kind="intervention-plan", authority="analyst-inference", confidence="inferred",
                     subject=identity, result={"cases": len(output), "sha256": hash_file(target),
                                               "interventions": sorted(names)},
                     coverage="bounded",
                     limitations=["Interventions are generated inputs; no target has yet been queried."])
    return {"status": "generated-not-observed", "path": str(target),
            "cases": len(output), "sha256": hash_file(target), "evidence_id": evid["evidence_id"]}


RUNNER = (
    "import importlib,json,sys\n"
    "fn=getattr(importlib.import_module(sys.argv[1]),sys.argv[2])\n"
    "with open('/cases/input.jsonl',encoding='utf-8') as f:\n"
    " for line in f:\n"
    "  if not line.strip():continue\n"
    "  row=json.loads(line)\n"
    "  print(json.dumps({'id':row['id'],'output':fn(row['input'])},allow_nan=False),flush=True)\n"
)


def replay_candidate(wb: Workbench, candidate_id: str, case_file: str | Path, *,
                     image: str, execute: bool, module: str = "reproduction",
                     function: str = "predict", timeout: int = 120,
                     memory_mb: int = 1024) -> dict[str, Any]:
    """Run candidate predict(JSON) on matching inputs only inside a Docker sandbox."""
    if not execute:
        raise ValueError("--execute-tests is required for candidate replay")
    if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*", module)
            or not re.fullmatch("[A-Za-z][A-Za-z0-9_]*", function)):
        raise ValueError("invalid module/function")
    if not re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,180}", image or ""):
        raise ValueError("explicit image required")
    if not 5 <= timeout <= 1800 or not 128 <= memory_mb <= 32768:
        raise ValueError("invalid resource bounds")
    # Resolve only after rejecting symlinks in the original user-selected path.
    rows = cases(case_file)
    src = Path(case_file).resolve(strict=True)
    manifest = candidate_manifest(wb, candidate_id)
    folder = candidate_folder(wb, candidate_id).resolve()
    module_path = "src/" + module.replace(".", "/") + ".py"
    if module_path not in manifest["files"]:
        raise ValueError("predictor is not recorded in candidate manifest")
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker unavailable; refusing unsafe host execution")
    if "," in str(src) or "," in str(folder):
        raise ValueError("Docker --mount path cannot contain commas")
    name = "paperrev-replay-" + uuid.uuid4().hex[:18]
    cmd = [docker, "run", "--rm", "--name", name, "--pull=never", "--network=none",
           "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
           "--pids-limit=64", "--memory", f"{memory_mb}m", "--cpus", "2",
           "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
           "--mount", f"type=bind,source={folder},target=/work,readonly",
           "--mount", f"type=bind,source={src},target=/cases/input.jsonl,readonly",
           "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONPATH=/work/src",
           "--workdir", "/work", image, "python", "-c", RUNNER, module, function]
    t0 = time.monotonic()
    exit_code, cleanup = None, "not-needed"
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout, check=False)
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        status = "ran" if exit_code == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        try:
            stopped = subprocess.run([docker, "rm", "-f", name], stdin=subprocess.DEVNULL,
                                     capture_output=True, text=True, timeout=20, check=False)
            cleanup = "removed" if stopped.returncode == 0 else "unconfirmed"
        except (OSError, subprocess.TimeoutExpired):
            cleanup = "unconfirmed"
    answers = []
    if status == "ran":
        try:
            if len(stdout.encode("utf-8")) > 4 * 1024 * 1024:
                raise ValueError("candidate output too large")
            for line in stdout.splitlines():
                if line.strip():
                    answer = json.loads(line)
                    if not isinstance(answer, dict) or "id" not in answer or "output" not in answer:
                        raise ValueError("candidate returned malformed JSONL")
                    answers.append(answer)
            found = [str(row["id"]) for row in answers]
            if len(set(found)) != len(found) or set(found) != {str(row["id"]) for row in rows}:
                raise ValueError("candidate output has missing or duplicate IDs")
        except (ValueError, TypeError):
            status, answers = "invalid-output", []
    output_path = None
    if status == "ran":
        digest = sha256(canonical({"candidate_id": candidate_id, "cases": hash_file(src), "answers": answers}))
        output_path = wb.root / "observations" / ("candidate_" + digest[:24] + ".jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in answers)
        if output_path.exists() and output_path.read_text(encoding="utf-8") != data:
            raise FileExistsError("candidate output digest collision")
        if not output_path.exists():
            output_path.write_text(data, encoding="utf-8")
    record = {"schema": "paperrev/behavioral-replay-v1", "candidate_id": candidate_id,
              "manifest_sha256": hash_file(folder / "candidate.json"),
              "case_file_sha256": hash_file(src), "image": image, "status": status,
              "cases_expected": len(rows), "cases_observed": len(answers),
              "output_path": str(output_path) if output_path else None,
              "output_sha256": hash_file(output_path) if output_path else None,
              "exit_code": exit_code, "container_cleanup": cleanup,
              "timeout_sec": timeout, "memory_mb": memory_mb,
              "duration_sec": round(time.monotonic() - t0, 4),
              "stderr_preview": stderr[:1000],
              "scope_note": "Candidate code behavior on frozen inputs, not paper metric reproduction."}
    run_id = "replay_" + sha256(canonical(record))[:24]
    write_once(wb.root / "runs" / (run_id + ".json"), record)
    evidence = wb.record(kind="candidate-behavior", authority="controlled-execution",
                         confidence="observed", subject={"run_id": run_id, "candidate_id": candidate_id},
                         result={"status": status, "output_sha256": record["output_sha256"],
                                 "observed": len(answers)},
                         coverage="complete" if status == "ran" else "partial",
                         limitations=[record["scope_note"]])
    return {"run_id": run_id, "status": status, "output_path": record["output_path"],
            "observed": len(answers), "evidence_id": evidence["evidence_id"]}


def check_reconstruction(wb: Workbench, observation_id: str, candidate_id: str,
                         case_file: str | Path, *, image: str, execute: bool,
                         atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
    """Authenticate recorded target evidence before sandbox replay and comparison."""
    original = wb.get(observation_id)
    if original["kind"] != "api-observation":
        raise ValueError("only an observed API capture can be the reference")
    if original["subject"]["cases_sha256"] != hash_file(Path(case_file)):
        raise ValueError("case file differs from observed input set")
    reference = wb.root / original["result"]["reference_file"]
    if hash_file(reference) != original["result"]["reference_sha256"]:
        raise ValueError("reference capture file changed after observation")
    run = replay_candidate(wb, candidate_id, case_file, image=image, execute=execute)
    if run["status"] != "ran":
        return {"status": "inconclusive", "replay": run}
    comparison = compare_outputs(wb, reference, run["output_path"], atol=atol, rtol=rtol)
    evidence = wb.record(kind="reconstruction-comparison", authority="controlled-execution",
                         confidence="derived", subject={"candidate_id": candidate_id},
                         result={"paired_cases": comparison["paired_cases"],
                                 "matched_fraction": comparison["matched_fraction"],
                                 "coverage": comparison["coverage"]},
                         sources=[observation_id, run["evidence_id"], comparison["evidence_id"]],
                         coverage=comparison["coverage"],
                         limitations=["Finite I/O agreement is not historical source-code identity."])
    return {"status": "compared", "comparison": comparison, "evidence_id": evidence["evidence_id"]}


def generate_from_observation(wb: Workbench, observation_id: str, plan_sha: str,
                              case_file: str | Path, label: str, client: Any,
                              max_pairs: int = 12) -> dict[str, Any]:
    """Build an independent code candidate using a paper plan AND observed I/O.

    ModelClient.ask is an explicitly invoked model operation. It must be
    authorized at the CLI boundary; observations are passed to the selected
    model endpoint and may include the user's own input data.
    """
    from .replicate import CODE_SYSTEM, create_candidate
    from .workbench import read_json

    if not 1 <= max_pairs <= 24:
        raise ValueError("max_pairs must be in 1..24")
    observation = wb.get(observation_id)
    if observation["kind"] != "api-observation" or observation["coverage"] != "complete":
        raise ValueError("generation requires a complete captured target observation")
    if observation["subject"]["cases_sha256"] != hash_file(Path(case_file)):
        raise ValueError("case data changed since target observation")
    reference = wb.root / observation["result"]["reference_file"]
    if hash_file(reference) != observation["result"]["reference_sha256"]:
        raise ValueError("captured reference bytes were altered")
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha):
        raise ValueError("invalid algorithm plan SHA")
    plan = read_json(wb.root / "plans" / (plan_sha + ".json"))
    source = cases(case_file)
    responses = {}
    for line in reference.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        ident = str(record["id"])
        if ident in responses:
            raise ValueError("duplicated observed response ID")
        responses[ident] = record["output"]
    if set(responses) != {str(row["id"]) for row in source}:
        raise ValueError("reference and source case coverage differ")
    examples = [{"id": row["id"], "input": row["input"],
                 "observed_output": responses[str(row["id"])]}
                for row in source[:max_pairs]]
    prompt = {"paper_plan": plan, "captured_examples": examples,
              "contract": {"file": "src/reproduction.py",
                           "entrypoint": "def predict(input): ...",
                           "output": "JSON serializable value",
                           "unit_tests": "tests/test_reproduction.py"},
              "not_author_source": True,
              "constraints": ["Infer candidate rules from finite observations only.",
                              "Do not claim to know missing original training details.",
                              "Include test cases for observations and at least one edge case.",
                              "Return the same JSON files schema as other PaperRev candidates."]}
    proposal = client.ask(
        CODE_SYSTEM + "\nThe candidate must expose predict(input) in src/reproduction.py "
        "for later Docker black-box replay. Do not claim it is original source.",
        json.dumps(prompt, ensure_ascii=False))
    candidate = create_candidate(wb, plan_sha=plan_sha, proposal=proposal, label=label)
    linked = wb.record(kind="behavior-grounded-candidate", authority="analyst-inference",
                       confidence="inferred", subject={"candidate_id": candidate["candidate_id"],
                                                     "plan_sha256": plan_sha},
                       result={"training_examples": len(examples),
                               "source_observation": observation_id,
                               "candidate_evidence": candidate["evidence_id"]},
                       sources=[observation_id, candidate["evidence_id"]], coverage="bounded",
                       limitations=["The model saw a finite behavior sample only.",
                                    "Passing generated tests cannot establish general equivalence."])
    return {**candidate, "behavior_source_evidence_id": observation_id,
            "reconstruction_evidence_id": linked["evidence_id"],
            "status": "behavior-grounded-generated-not-verified"}

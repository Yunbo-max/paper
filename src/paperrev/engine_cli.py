"""REA-style CLI: every operation is explicit, result-bearing, and locally inspectable."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .workbench import Workbench, read_json
from .replicate import (ModelClient, ingest_pdf, propose_plan, register_plan,
                        create_candidate, generate_candidate)
from .reproduce import compare_outputs, run_tests, verify_claims


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="paperrev", description="Evidence-first paper reverse engineering")
    p = root.add_subparsers(dest="cmd", required=True)
    x = p.add_parser("init", help="Create immutable research workspace")
    x.add_argument("--workspace", required=True)
    x.add_argument("--title", required=True)
    x = p.add_parser("open-paper", help="Inspect a PDF; record page-level evidence")
    x.add_argument("--workspace", required=True)
    x.add_argument("--pdf", required=True)
    x.add_argument("--max-pages", type=int, default=500)
    x = p.add_parser("plan", help="Use local or approved remote LLM to propose a quoted algorithm spec")
    x.add_argument("--workspace", required=True)
    x.add_argument("--source-sha", required=True)
    x.add_argument("--endpoint", default="http://127.0.0.1:8000")
    x.add_argument("--model", required=True)
    x.add_argument("--allow-remote", action="store_true")
    x.add_argument("--execute-llm", action="store_true")
    x.add_argument("--max-chars", type=int, default=48000)
    x = p.add_parser("register-plan", help="Validate a manually curated or agent-written JSON plan")
    x.add_argument("--workspace", required=True)
    x.add_argument("--source-sha", required=True)
    x.add_argument("--plan", required=True)
    x = p.add_parser("generate", help="Generate independent source code from plan (LLM or supplied proposal JSON)")
    x.add_argument("--workspace", required=True)
    x.add_argument("--plan-sha", required=True)
    x.add_argument("--label", required=True)
    x.add_argument("--proposal", help="Local candidate JSON; no LLM/network required")
    x.add_argument("--endpoint", default="http://127.0.0.1:8000")
    x.add_argument("--model")
    x.add_argument("--allow-remote", action="store_true")
    x.add_argument("--execute-llm", action="store_true")
    x = p.add_parser("test", help="Run untrusted generated unit tests inside Docker (never host Python)")
    x.add_argument("--workspace", required=True)
    x.add_argument("--candidate", required=True)
    x.add_argument("--image", required=True, help="already-installed Docker image (never auto-pull)")
    x.add_argument("--timeout", type=int, default=120)
    x.add_argument("--memory-mb", type=int, default=1024)
    x.add_argument("--cpus", type=float, default=2.0)
    x.add_argument("--execute-tests", action="store_true")
    x = p.add_parser("compare", help="Compare ID-matched prediction JSONL with explicit numeric tolerances")
    x.add_argument("--workspace", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--candidate", required=True)
    x.add_argument("--atol", type=float, default=1e-6)
    x.add_argument("--rtol", type=float, default=1e-5)
    x = p.add_parser("verify", help="Verify finite claim obligations against real Evidence IDs")
    x.add_argument("--workspace", required=True)
    x.add_argument("--plan-sha", required=True)
    x.add_argument("--bindings", required=True)
    x = p.add_parser("record-unknown", help="Record an unresolved implementation question")
    x.add_argument("--workspace", required=True)
    x.add_argument("--question", required=True)
    x.add_argument("--claim", action="append", default=[])
    x.add_argument("--evidence", action="append", default=[])
    x.add_argument("--next-probe", default="")
    x = p.add_parser("resolve-unknown", help="Revision-checked unknown update")
    x.add_argument("--workspace", required=True)
    x.add_argument("--id", required=True)
    x.add_argument("--revision", type=int, required=True)
    x.add_argument("--status", choices=["open", "investigating", "unresolved", "resolved"], required=True)
    x.add_argument("--evidence", action="append", default=[])
    x.add_argument("--resolution")
    x = p.add_parser("inspect", help="Read one evidence item by hash or a summary page")
    x.add_argument("--workspace", required=True)
    x.add_argument("--id")
    x.add_argument("--kind")
    x.add_argument("--offset", type=int, default=0)
    x.add_argument("--limit", type=int, default=20)
    x = p.add_parser("export", help="Export an authenticated, portable Evidence bundle")
    x.add_argument("--workspace", required=True)
    x.add_argument("--out", required=True)
    x = p.add_parser("verify-bundle", help="Read and hash-check an exported Evidence bundle")
    x.add_argument("--file", required=True)
    x = p.add_parser("reconstruct", help="Bounded end-to-end: ingest PDF -> plan -> code -> optional Docker tests")
    x.add_argument("--workspace", required=True)
    x.add_argument("--pdf", required=True)
    x.add_argument("--title", required=True)
    x.add_argument("--label", default="paper-reproduction")
    x.add_argument("--endpoint", default="http://127.0.0.1:8000")
    x.add_argument("--model", required=True)
    x.add_argument("--allow-remote", action="store_true")
    x.add_argument("--execute-llm", action="store_true")
    x.add_argument("--execute-tests", action="store_true")
    x.add_argument("--image", help="required when --execute-tests is selected")
    return root


def dispatch(a: argparse.Namespace) -> dict:
    if a.cmd == "init":
        wb = Workbench.create(a.workspace, a.title)
        return {"workspace": str(wb.root), "status": "initialized"}
    if a.cmd == "verify-bundle":
        return Workbench.verify_bundle(a.file)
    if a.cmd == "reconstruct":
        if not a.execute_llm:
            raise ValueError("automatic reconstruction requires --execute-llm; no hidden API call")
        if a.execute_tests and not a.image:
            raise ValueError("--execute-tests requires explicit --image")
        wb = Workbench.create(a.workspace, a.title)
        pdf = ingest_pdf(wb, a.pdf)
        client = ModelClient(a.endpoint, a.model, allow_remote=a.allow_remote)
        plan = propose_plan(wb, pdf["source_sha256"], client)
        candidate = generate_candidate(wb, plan["plan_sha256"], a.label, client)
        report = {"workspace": str(wb.root), "paper": pdf, "plan": plan, "candidate": candidate,
                  "status": "generated-not-reproduced"}
        if a.execute_tests:
            report["tests"] = run_tests(wb, candidate["candidate_id"], image=a.image)
            report["status"] = "unit-tests-observed-not-paper-reproduced"
        return report
    wb = Workbench(a.workspace)
    if a.cmd == "open-paper":
        return ingest_pdf(wb, a.pdf, max_pages=a.max_pages)
    if a.cmd == "plan":
        if not a.execute_llm:
            raise ValueError("--execute-llm required to send paper content to model")
        return propose_plan(wb, a.source_sha, ModelClient(a.endpoint, a.model, allow_remote=a.allow_remote),
                            max_chars=a.max_chars)
    if a.cmd == "register-plan":
        return register_plan(wb, a.source_sha, read_json(Path(a.plan)))
    if a.cmd == "generate":
        if a.proposal:
            return create_candidate(wb, plan_sha=a.plan_sha, label=a.label, proposal=read_json(Path(a.proposal)))
        if not a.execute_llm or not a.model:
            raise ValueError("provide --proposal OR --model plus --execute-llm")
        return generate_candidate(wb, a.plan_sha, a.label,
                                  ModelClient(a.endpoint, a.model, allow_remote=a.allow_remote))
    if a.cmd == "test":
        if not a.execute_tests:
            raise ValueError("--execute-tests is required for generated code execution")
        return run_tests(wb, a.candidate, image=a.image, timeout_sec=a.timeout,
                         memory_mb=a.memory_mb, cpus=a.cpus)
    if a.cmd == "compare":
        return compare_outputs(wb, a.reference, a.candidate, atol=a.atol, rtol=a.rtol)
    if a.cmd == "verify":
        return verify_claims(wb, a.plan_sha, read_json(Path(a.bindings)))
    if a.cmd == "record-unknown":
        return wb.record_unknown(a.question, related_claim_ids=a.claim,
                                 evidence_ids=a.evidence, next_probe=a.next_probe)
    if a.cmd == "resolve-unknown":
        return wb.update_unknown(a.id, expected_revision=a.revision, status=a.status,
                                 evidence_ids=a.evidence, resolution=a.resolution)
    if a.cmd == "inspect":
        return wb.get(a.id) if a.id else wb.list_evidence(a.kind, a.offset, a.limit)
    if a.cmd == "export":
        return wb.export_bundle(a.out)
    raise ValueError(f"unknown subcommand {a.cmd}")


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, SyntaxError, json.JSONDecodeError) as exc:
        print(f"paperrev: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

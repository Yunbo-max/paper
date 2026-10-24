"""Optional Model Context Protocol adapter over the exact CLI/domain operations.

The Python SDK is optional; no server is registered by installing a skill alone.
Use `paperrev-mcp` in a client that connects local MCP stdio tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .replicate import ingest_pdf, register_plan, create_candidate
from .reproduce import candidate_manifest, compare_outputs, run_tests, verify_claims
from .workbench import Workbench, read_json


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("MCP extra required: pip install 'paperrev-lab[mcp]'") from exc

    server = FastMCP(name="paperrev", instructions=(
        "Paper reverse-engineering with exact source evidence. Inspect before claiming. "
        "Plan and generated code are not author source. Test only in explicit bounded Docker sandbox; "
        "a passing test does not mean the entire paper is reproduced."))

    @server.tool()
    def create_session(workspace: str, title: str) -> dict[str, Any]:
        """Create a new workbench without overwriting previous investigations."""
        wb = Workbench.create(workspace, title)
        return {"workspace": str(wb.root)}

    @server.tool()
    def open_paper(workspace: str, pdf_path: str, max_pages: int = 500) -> dict[str, Any]:
        """Inspect a user-supplied PDF and produce page-grounded Evidence."""
        return ingest_pdf(Workbench(workspace), pdf_path, max_pages=max_pages)

    @server.tool()
    def record_plan(workspace: str, paper_sha256: str, plan_path: str) -> dict[str, Any]:
        """Validate claims against exact quotes in the ingested PDF and record an algorithm plan."""
        return register_plan(Workbench(workspace), paper_sha256, read_json(Path(plan_path)))

    @server.tool()
    def generate_code(workspace: str, plan_sha256: str, proposal_path: str, label: str) -> dict[str, Any]:
        """Materialize a candidate package from agent-authored JSON; syntax check without executing."""
        return create_candidate(Workbench(workspace), plan_sha=plan_sha256,
                                proposal=read_json(Path(proposal_path)), label=label)

    @server.tool()
    def inspect_candidate(workspace: str, candidate_id: str) -> dict[str, Any]:
        """Check candidate files still match their recorded hashes."""
        return candidate_manifest(Workbench(workspace), candidate_id)

    @server.tool()
    def run_candidate_tests(workspace: str, candidate_id: str, docker_image: str,
                            timeout_sec: int = 120, memory_mb: int = 1024) -> dict[str, Any]:
        """Explicitly execute untrusted candidate unit tests in a no-network Docker container."""
        return run_tests(Workbench(workspace), candidate_id, image=docker_image,
                         timeout_sec=timeout_sec, memory_mb=memory_mb)

    @server.tool()
    def compare_behavior(workspace: str, reference_jsonl: str, candidate_jsonl: str,
                         atol: float = 1e-6, rtol: float = 1e-5) -> dict[str, Any]:
        """Compare recorded predictions with exact sample IDs and numeric tolerance."""
        return compare_outputs(Workbench(workspace), reference_jsonl, candidate_jsonl, atol=atol, rtol=rtol)

    @server.tool()
    def verify_reconstruction(workspace: str, plan_sha256: str, bindings_path: str) -> dict[str, Any]:
        """Verify scoped claim obligations; unresolved claims remain unknown."""
        return verify_claims(Workbench(workspace), plan_sha256, read_json(Path(bindings_path)))

    @server.tool()
    def record_unknown(workspace: str, question: str, claim_ids: list[str], evidence_ids: list[str],
                       next_probe: str = "") -> dict[str, Any]:
        """Record the reason a mechanism, parameter or training detail is unidentifiable."""
        return Workbench(workspace).record_unknown(question, related_claim_ids=claim_ids,
                                                    evidence_ids=evidence_ids, next_probe=next_probe)

    @server.tool()
    def update_unknown(workspace: str, unknown_id: str, expected_revision: int,
                       status: str, evidence_ids: list[str], resolution: str = "") -> dict[str, Any]:
        """Revision-checked update; resolution needs observed evidence."""
        return Workbench(workspace).update_unknown(unknown_id, expected_revision=expected_revision,
                status=status, evidence_ids=evidence_ids, resolution=resolution or None)

    @server.tool()
    def inspect_evidence(workspace: str, evidence_id: str = "", offset: int = 0,
                         limit: int = 20) -> dict[str, Any]:
        """Retrieve one authenticated Evidence item or a bounded index page."""
        wb = Workbench(workspace)
        return wb.get(evidence_id) if evidence_id else wb.list_evidence(offset=offset, limit=limit)

    @server.tool()
    def export_evidence_bundle(workspace: str, output_path: str) -> dict[str, Any]:
        """Export complete immutable Evidence records and unknown states, not checkpoints or private paper bytes."""
        return Workbench(workspace).export_bundle(output_path)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()

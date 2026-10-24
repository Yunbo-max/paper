---
name: paper-reverse-engineering
description: REA-style evidence-first investigation of incompletely released research papers, from source artifacts through independent code candidate, controlled execution, and scoped claim verification.
metadata:
  version: "2"
---

# Paper Reverse Engineering (PaperRev v0.2)

Use this Skill for **paper to independent executable implementation**, especially when source/training code is not fully released. PaperRev implements an original scientific workflow inspired by REA's investigative code. The Skill is instructions only; installing it does **not** install Python, Docker, an LLM, register MCP tools, or create a runnable agent by itself.

## Route the target first

- PDF only: `open_paper` / `paperrev open-paper`, then inspect page evidence and figures visually as needed.
- Public complete source: use ordinary repository/code tools; do not reverse engineer code you can already read.
- Paper + weights: inspect pinned configs, architecture, and tensors in a separately authorized runtime. Final weights do not determine hidden training data or optimizer history.
- Paper + accessible inference API: collect matched outputs with a pinned model/version, benchmark and controls; distinguish black-box inference from author-internal details.
- Released native binary/app: invoke REA if installed; cite its evidence IDs and scope.

## Never replace evidence with plausibility

Start with `create_session`, `open_paper`, and `record_plan`. A plan must cite an exact quote from an observed PDF page. If a formula, table, figure, or caption needs visual review, do that before promoting it into an executable contract. An LLM-proposed plan is **inferred**, even when literal source quotations pass validation.

Keep a hypothesis branch for every materially ambiguous detail (normalization, dropout, off-policy target, stop-gradient, scheduler, reward reduction, tokenizer, data filtering). Call `record_unknown`; supply a falsification test, required authority and bounded next probe. Do not resolve it merely because generated code passes.

## Build and verify code

1. Register a plan with multiple modules, claim references, input/output shapes, equations, testable invariants and unknowns.
2. Generate a candidate with a coding agent (`generate_code` with an agent-written proposal) or `paperrev generate --execute-llm` against an authorized OpenAI-compatible endpoint.
3. Treat generated files as **untrusted**. Candidate paths/size/syntax are checked without executing code; preserve the exact candidate digest and all rejected/failed hypotheses.
4. Use `run_candidate_tests` or `paperrev test --execute-tests --image ...` with Docker, never unsandboxed host Python. Absence of Docker is a blocker; not a reason to run untrusted code on host.
5. Obtain reference outputs in a controlled setting where possible. Compare by stable test case IDs, nested outputs, numeric tolerance and explicit coverage. Do not treat a benchmark score as a byte-identical original implementation.
6. Use `verify_reconstruction` or `paperrev verify` with real, cited Evidence IDs. Include unknown blockers and incomplete sources. Report pass/fail/unknown for **individual finite obligations**, never announce the whole paper replicated from one test.
7. Export an authenticated Evidence bundle to support independent review and later runs. The export omits source PDF and code bytes; retain those separately with hashes.

## Tool names

MCP: `create_session`, `open_paper`, `record_plan`, `generate_code`, `inspect_candidate`, `run_candidate_tests`, `compare_behavior`, `verify_reconstruction`, `record_unknown`, `update_unknown`, `inspect_evidence`, `export_evidence_bundle`.

CLI: `paperrev init`, `open-paper`, `plan`, `register-plan`, `generate`, `test`, `compare`, `verify`, `record-unknown`, `resolve-unknown`, `inspect`, `export`, `verify-bundle`, `reconstruct`.

MCP functions are scoped adapters; optional direct plan/generation LLM calls live in the CLI. Do not invent unadvertised MCP tools. Check the actual connected server tool list and installed package versions.

## Gate and cost discipline

PDF inspection is local. LLM calls require `--execute-llm`; remote HTTPS requires `--allow-remote`. Tests require `--execute-tests` plus an explicit already-installed Docker image. No automatic GPU scheduling, package installation, GitHub Actions or background execution. Evidence that a model can run does not imply model weights or trained results are available.

## StartLux initial research application

StartLux v1.1 has public model weights, inference code and a technical report. The first branch of study inspects official inference behavior and checkpoint configs; the second can attempt an **independent** RLCD² training implementation only after explicit mathematical specification and evaluations exist. The old `paperrev-startlux` adapter remains available for paired always-think Gate experiments and is not itself a training replication.

Always state what was *actually executed*, with the exact artifact IDs; separately list inferred hypotheses, unknowns and incomplete proof obligations.

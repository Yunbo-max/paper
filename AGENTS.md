# PaperRev Research Agent Contract (v0.2)

**Goal:** build inspectable **independent reproductions** of incompletely released scientific papers, following REA-style evidence identity, incomplete-coverage semantics, unknown registries and verified reconstruction obligations.

## Mandatory target routing

- Only PDF available → extract bounded page/text Evidence, identify equations and images that require separate visual inspection; do not invent missing formulas.
- Open-source complete implementation → inspect that source normally before reimplementing any module.
- Released weights or public API → use the appropriate model runtime and pinned checkpoint/inference configuration; model output does not reveal training history uniquely.
- Packaged native or JavaScript artifact → optionally invoke REA; code here does not claim to decompile binaries.

## Agent lifecycle

1. Create an isolated PaperRev workbench. Keep existing branches/artifacts intact.
2. Ingest the paper and cite exact page/quotes, with hash identities. Source quote match is not mathematical validation.
3. Record a multi-module plan and every unresolved detail. Candidate implementations of ambiguous details must branch; don't silently choose one as history.
4. Generate independent code via agent-authored JSON or a user-authorized model endpoint, then inspect candidate manifest and static syntax.
5. Only run in the declared bounded Docker environment; lack of Docker is a blocker, not permission for host execution.
6. Capture actual output and compare against a versioned reference, then assess explicit per-claim obligations and evidence authority.
7. Leave unresolved questions active unless an actual observed source can close them; do not equate test passing with numerical paper reproduction.

## Exposed interfaces

- CLI: `paperrev --help`, `paperrev-mcp` (optional official MCP SDK).
- Existing separate StartLux proto: `paperrev-startlux`.
- Generic packages: `workbench.py`, `replicate.py`, `reproduce.py`, `engine_cli.py`, `mcp_server.py`.
- Test locally: `python -m unittest discover -s tests -v`.

## Restrictive defaults

Never run API/GPU/model-generation calls without explicit user authorization, documented budget and opt-in flags. Never install missing tools as a side effect. Never use GitHub Actions or create a scheduled task for a one-time investigation. Preserve stdout/stderr, failures and missing evidence; test only with data one may lawfully submit to the selected provider.

## Reporting statuses

- source-audited; plan-produced; candidate-syntax-checked; candidate-unit-test-passed/failed; behavior-compared-partial/complete; finite-claim-check-passed/failed/unknown; paper-reproduction-not-established.
- Always name observed commands and stored artifacts. Never claim source recovered, historical optimizer recovered or paper reproduced solely from a generated repository.

## Observable model target (REA website/app route)

Use `docs/BLACKBOX_WORKFLOW.md` to capture authorized real model output, generate one-feature probes, derive an independent candidate from a source-grounded plan plus observations, and compare via no-network Docker. Preserve every Evidence link, data hash and incomplete-coverage reason. A callable black-box model is not the original private training source, and program-generated contrast inputs are not observed until an actual model returns outputs.

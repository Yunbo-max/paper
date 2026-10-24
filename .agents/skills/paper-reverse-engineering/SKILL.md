---
name: paper-reverse-engineering
description: Evidence-first reconstruction of incompletely released ML papers with source/weight/API inspection, competing implementation hypotheses, mathematical checks and scoped experiments.
---

# Paper Reverse Engineering

Use this skill when a user wants to reproduce an ML paper without complete source and/or training artifacts. This file is **investigation guidance**, not an installed MCP server or a promise that GPU work has run.

## Select evidence route before coding

| Available input | First probes | Authority |
| --- | --- | --- |
| Paper PDF only | Extract paginated text, cross-check figures/equations visually, list unknown details | Source text, not behavior |
| Paper + source repository | Freeze exact repo revision, locate implementations and tests | Observed code, possible incomplete training scope |
| Paper + released weights | Freeze exact HF commit, inspect configs/index, map tensor names; load in model runtime for interventions | Inspectable model, not historical training |
| Paper + executable application | Route shipped binaries and traces to REA when available | Partial static/runtime evidence |
| Paper + remote API | Freeze endpoint and request schema; compare controlled input/output samples | Black-box behavior only |

For an ordinary fully open GitHub codebase, use source tools instead of REA. Jacobian can validate **bounded, typed mathematical subclaims**; it cannot certify neural network generalization from a finite sample.

## Build a scoped claim ledger

For each claim record: ID, source URI + section/page + immutable digest, exact proposition, required verification authority, test fixture, status, implementation dependency and remaining unknowns. Never move source-declared directly to scientifically reproduced without an experiment.

Use case files under cases/ and run paperrev init. Do not overwrite an existing run or overwrite earlier raw results. Keep structured unknowns alongside claims.

## Do not guess missing implementation choices

Represent under-specified details as candidate branches. For each branch:
- a specific choice and compatible explicit paper constraints;
- a distinct, testable prediction;
- the smallest numerical, symbolic or behavioral experiment that separates it from alternatives;
- a falsification criterion and cost estimate.

If no original behavior is observable, a matching candidate is an **independent reconstruction**, not recovered author code. If two candidates match the public evidence, remain agnostic instead of declaring one to be historical truth.

## Reconstruction procedure

1. Inspect source and available artifacts. Build requirements and uncertainty ledger.
2. Separate algorithmic equations, architectural tensors, training objective, dataset/optimizer, inference path and metric harness.
3. Validate mathematical invariants: shapes, gradients, normalizations, bounds and small counterexamples. Use an independent check where feasible.
4. Produce a minimal implementation of only the components not already publicly available. Reuse official inference implementations and reference tests.
5. Test on synthetic fixtures first; then compare on frozen real/held-out inputs. Capture raw outcomes, exact model revision, seeds and environment.
6. Identify which scientific claims pass, fail or are still unknown. Prefer a focused follow-up experiment over broader prompt speculation.
7. Compare at least one competing mechanism and a suitable ablation with equal compute. Preserve all failed branches.
8. Write a reviewable audit with actual observed commands/results, limits, and next discrimination probes.

Do not let code compilation, score proximity or multiple agents' agreement substitute for experimental evidence. If a run fails, preserve partial evidence and stop rather than silently switching the model/dataset.

## StartLux pilot

See docs/STARTLUX_PLAN.md. An available official inference implementation and open 4B/9B/27B AutoThink weights permit observed first and post-thought probability comparisons. The v0.1 analyzer reuses ALWAYS traces to estimate gate strategies; it must **never** be described as a live AUTO run or RLCD² training reproduction.

First milestones: pinned model manifest -> local server sanity -> paired original outputs -> live gate comparisons -> white-box intervention -> independent small-base training reconstruction. Leave unpublished original training data, optimizer states and source unknown.

## Acceptance

Report one of: NOT_STARTED, SOURCE_AUDITED, EXECUTABLE, BEHAVIOR_CHECKED, CLAIM_REPRODUCED_WITH_SCOPE, CONTRADICTED_WITH_WITNESS, INCONCLUSIVE. Every promotion needs linked outputs and a verifier with sufficient authority. Do not claim a whole paper solved or reproduced when only an individual assertion passed.

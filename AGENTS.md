# PaperRev Agent Instructions

## Purpose
Reconstruct scientific claims from **partially released ML papers**. Produce inspectable evidence, runnable candidate code and scoped experiments. Never claim to recover undisclosed historical training source or data from a PDF/final checkpoint alone.

## Route by evidence type
1. Paper PDF: inspect text with page anchors; equations, figures and tables require visual/source cross-check before becoming verified mathematical contracts.
2. Open source repository: normal static/code analysis; use REA only when packaged binaries, runtime behavior or unavailable source make it relevant.
3. Released model weights: pin artifact SHA, inspect config, tensor index and license first. For internals use the model runtime and controlled hooks; a header alone is not a weight inspection.
4. API-only model: explicitly report black-box authority, sample/version drift and limited inference from behavior.
5. Unavailable artifact: record an open unknown and propose a discriminating test; do not invent source.

## Required scientific states
- source-declared: paper/repo says X, not independently checked.
- code-verified: a frozen implementation passed an exact specified contract.
- behavior-observed: actual model responses on a frozen input are stored.
- reproduced-within-scope: measured claim passed with correct baseline, dataset, controls and uncertainty.
- contradicted: a counterexample with source, scope and independent checks is recorded.
- unknown: source/authority absent, partial, inconclusive or materially ambiguous.

A passing unit test is not a behavioral reproduction or a validated causal mechanism.

## Run discipline
- Work on scoped branches; preserve the user's unrelated projects.
- Make the smallest factual observation first, then follow unresolved questions.
- Preserve file hashes, source URLs, revision IDs, environment, seeds, model IDs, controls and exclusions.
- Parallelize independent hypothesis checks only when they do not share mutable files or spend the same compute budget.
- Record both supporting and contradicting observations.
- When a proposed original implementation detail is under-specified, branch into candidate mechanisms. Do not collapse branches on mere plausibility.
- Do not execute remote APIs or GPU jobs without a concrete authorized request, pinned costs/budget and a reproducible test.
- Independent review must verify actual outputs; having a second Agent rephrase a claim does not constitute validation.

## Current implementation
Read README.md and docs/ARCHITECTURE.md. The executable v0.1 supports source case initialization, page-bound PDF text, local/Hub model metadata, and StartLux always-think capture plus **offline** gate analysis. It does not yet implement autonomous code generation, symbolic proof production, weight patching, RL training or independent published benchmark reproduction.

Standard local check: python -m unittest discover -s tests -v.
No GitHub Actions or background execution should be assumed. Never mark a check as run unless its output was actually observed.

## First target
StartLux v1.1 AutoThink (arXiv:2610.17309). Use the author's inference code rather than rewriting it. Start at 4B for smoke behavior; only afterwards schedule 27B analysis on suitable GPU, and finally explore an **independent** RLCD² training candidate.

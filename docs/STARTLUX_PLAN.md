# StartLux v1.1 AutoThink: first reproduction and falsification plan

## Research question

Does selective natural-language deliberation improve the quality of a trained decision model, and is first-pass confidence a sufficient signal for *when* to deliberate? Answer this **before** attempting to reproduce the original RLCD² training process.

Source links:
- Paper: https://arxiv.org/abs/2610.17309
- Public inference code: https://github.com/StartLuxLabs/StartLux-Decision
- Released BF16 weights: https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink
- Smaller 4B pilot: https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink
- First/after probability definition: https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/api.md
- Official eval program: https://github.com/StartLuxLabs/StartLux-Decision/blob/main/startlux_decide/evaluate.py

The released inference code includes readout, mixing, thresholds, HTTP service, and engine adaptations. The v1/finetune LoRA scripts are from an older release, **not** the full v1.1 multi-stage/RLCD² training recipe.

## Phases, required evidence and stop gates

### Phase 0 — Source audit (CPU, no weights)

1. Freeze the paper ID and target exact Hugging Face commit.
2. Inspect config.json, decision_config.json and model.safetensors.index.json. Save hashes and file inventory; do not claim weights are validated.
3. Inventory the open inference code and reported training description. Separate explicit information from unknown: original training data, intermediates, optimizer and full original scripts.
4. Initialize the case claim and unknown ledgers. Review the original paper equations manually, including figure/table content, before filling the ledger.

**Gate:** source hashes, release type (Instant vs AutoThink), first/after probability fields and license are understood. Otherwise do not launch GPU.

### Phase 1 — Behavior pilot (4B before 27B)

Use the official inference engine and checkpoint, not a generated replica. Run no-think, always-think and live-auto with a registered, frozen dataset. The first 3 examples in smoke.jsonl only check plumbing.

StartLux's public response with return_details=true includes first_pass.probabilities, after_thought.probabilities, thought.used and thought.tokens. Set none_check=false and allow_none=false for the first multi-class scoring experiment so the option spaces agree.

**Primary outcome:** paired negative log score, Brier score and accuracy at an equal per-question budget. Also measure thought token count, unanswered/error counts and live latency separately.

**Gate:** real always responses contain complete first and after-thought distributions for the majority of difficult questions, and local readout/mixing checks pass. If not, diagnose API/model config rather than inventing a training fix.

### Phase 2 — Selective deliberation mechanism test

For a question x and gold t:
- p0 = first-pass distribution.
- p1 = post-thought re-tempered distribution.
- p_mix = (1-alpha) p0 + alpha p1.
- Delta(x) = log(p1[t]) - log(p0[t]) (positive is improved gold log score).

For each pre-registered gate g, compute the offline estimator:
- choose p_mix if max(p0) < g, else p0;
- charge thought tokens observed for x only if gate triggers.

Compare g=0.6 and g=0.9, plus no-think and always-think. The estimator is not a real AUTO policy run: stochastic thought generation, engine effects, changed batch scheduling and thought-not-generated scenarios must be checked with **live auto** runs before making policy claims.

**Gate:** benefit/expense trade-off reproduced on a meaningful held-out set and with at least three repeat seeds. If the gate has no consistent benefit over fixed budgets, park the training reproduction and investigate confidence-versus-value-of-thinking instead.

### Phase 3 — White-box analysis (4B, 9B, then 27B)

Load verified public checkpoint weights and map tensor prefixes against the declared Qwen3.5 architecture. Capture selected activations with hooks, compare first vs post thought, and run matched intervention controls on layers or heads. Do not infer SFT/RLCD² stage attribution from a weight difference alone; multiple training histories can lead to similar end weights.

**Gate:** interventions have reproducible effects beyond random/control patches and preserve the exact question/seed/engine conditions.

### Phase 4 — Independent RLCD² reimplementation (training project)

Only after Phase 1–3 establish a valid evaluation loop, reimplement the public objective on a smaller open base using independently sourced labeled decisions:
- First SFT on candidate decision distributions.
- A controlled no-RL or length-penalty baseline.
- Grouped candidate thoughts with improvement in a proper score and a length penalty.
- Compare GRPO-like grouping, KL, entropy/diversity and careful inference-threshold calibration.

This is **independent reconstruction**, not proof of recovering the original author's private training script, data or optimizer. Train-on-synthetic and held-out real benchmarks must remain separate. Save code revision, model revision, data manifests, objective equations, hardware and seeds.

## Controls against misleading conclusions

- Gate by top probability, not TypeSafe's normalized option confidence; these differ when option counts vary.
- Report proper scores (NLL and multiclass Brier) alongside accuracy. A confident wrong prediction matters.
- Never compare 4B and 27B results as if model sizes or precision were controlled.
- Keep model license (CC BY-NC 4.0) distinct from open inference code license (Apache-2.0).
- Distinguish no source code, no *training* code, available open weights, and restricted API access.
- Do not label a smoke test or a post-hoc policy estimate as reproduction of the paper's published benchmark.

## Expected output structure

~~~text
runs/startlux/
  manifest.json
  case_snapshot.json
  claim_ledger.json
  unknown_ledger.json
  model_hub.json
  paper_pages.json
  always.jsonl
  gates.json
  # later: live_auto_seed*.jsonl, reproducibility_report.json
~~~

All generated artifacts are intentionally gitignored; commit concise code, source pointers and reviews, not proprietary checkpoint files or secrets.

# REA Website/App Reverse Engineering Applied to Research Models

## What transfers from REA

REA reconstructs a shipped application's externally observable features by collecting an artifact inventory, live/browser or runtime interactions, and comparable evidence before writing new code. For an unpublished ML training stack, **the public inference endpoint is the observable application**. A paper's stated equations, released checkpoint, and inference service are distinct evidence authorities.

PaperRev provides a narrow but runnable analog:

~~~text
Paper + model/interface metadata
        |
        +-- paperrev open-paper / register-plan -> source-grounded claims
        |
Actual inference behavior -> paperrev observe-api -> retained JSONL + SHA256
        |                                         (real outputs, not fabricated)
        v
One-variable probes  -> paperrev intervene -> new inputs (NOT observations)
        |                  |
        +-- observe-api on modified inputs -> actual counterfactual behavior
        v
Paper plan + captured I/O -> paperrev generate-behavior -> independent Python
        |
        +-- src/reproduction.py: predict(input) -> JSON output
        v
Docker offline replay -> paperrev check-reconstruction
        |
        +-- exact case IDs, tolerances, digest checks, Evidence parent links
        +-- mismatches -> new candidate hypotheses, not evidence of private source
~~~

The current release supports explicitly targeted **stateless JSON POST APIs**. It does not automatically browse arbitrary websites, inspect client JavaScript, learn session cookies, mutate live app data, capture a HAR, or recover server-side source. Those require additional, scoped adapters and separate permissions.

## Prepare a frozen experiment

One JSON object per line in cases.jsonl, with a stable ID and the exact request body for the endpoint:

~~~json
{"id":"delayed","input":{"state":"Shipment arrived after 10 days","questions":{"late":{"type":"noul","instructions":"Was it late?"}}}}
{"id":"prompt","input":{"state":"Shipment arrived on schedule","questions":{"late":{"type":"noul","instructions":"Was it late?"}}}}
~~~

The tool does not need gold labels to imitate an endpoint. **Gold labels and independent benchmarks are needed to evaluate scientific accuracy**. The model response itself is not ground truth.

~~~bash
paperrev init --workspace runs/audit --title 'Black-box reconstruction'
paperrev observe-api --workspace runs/audit --cases cases.jsonl \
  --origin http://127.0.0.1:8090 --path /v1/systemone \
  --output-pointer /answers --execute
~~~

To use a remote endpoint, specify \`--allow-remote\` and an HTTPS origin, and optionally \`PAPERREV_API_KEY\`. Calls are never redirected, do not run without \`--execute\`, and record the number of valid responses versus requested cases.

## Design discriminating interventions

The intervention file is a finite declarative list:

~~~json
{
  "schema":"paperrev/interventions-v1",
  "changes":[
    {"id":"change-state","path":"/state","op":"replace","value":"Shipment arrived after 100 days"}
  ]
}
~~~

~~~bash
paperrev intervene --workspace runs/audit \
  --cases cases.jsonl --spec interventions.json
~~~

Operations are \`replace\` (a specified existing JSON field), \`add\` (numeric) or \`reverse\` (list). They never query the target. To gather independent evidence, call \`observe-api\` again against the generated probe JSONL. Keep that new capture, its source SHA and the original capture separately.

Avoid concluding that a change has a causal explanation beyond the tested input perturbation. Repeated seeds and server-version control are needed for stochastic models.

## Infer a candidate implementation

With a paper grounded algorithm plan already registered:

~~~bash
paperrev generate-behavior --workspace runs/audit --plan-sha PLAN_SHA \
  --observation OBSERVED_EVIDENCE_ID --cases cases.jsonl \
  --label candidate-1 --model my-coding-model \
  --endpoint http://127.0.0.1:8000 --execute-llm
~~~

The selected coding model receives a bounded sample of **source observations** alongside the paper plan, and proposes candidate Python files. The generated implementation is tagged as \`inferred\`. It is **not executed at generation time** and is not represented as the authors' private source. If the model endpoint is remote, \`--allow-remote\` and HTTPS are required; only send information that you are authorized to disclose.

## Replay on exactly the same input set

~~~bash
paperrev replay --workspace runs/audit --candidate CANDIDATE_ID \
  --cases cases.jsonl --image python:3.12-slim --execute-tests

paperrev check-reconstruction --workspace runs/audit \
  --observation OBSERVED_EVIDENCE_ID --candidate CANDIDATE_ID \
  --cases cases.jsonl --image python:3.12-slim --execute-tests
~~~

The generated code must export a callable \`predict(input)\` from \`src/reproduction.py\` (or select another tracked module/function for standalone \`replay\`). The program is invoked inside **read-only, non-root, network-disabled Docker**, with CPU/memory/deadline bounds and an explicit preinstalled image. No pip installs or host fallback occur.

The comparison authenticates the actual reference capture file and case file, replays every input, checks unique output IDs and stores a finite output-equivalence comparison. Missing cases, corrupted observations, execution errors and unknowns cannot silently become successes.

## How to interpret results

| Evidence | Can be concluded | Cannot be concluded |
| --- | --- | --- |
| Real model API output | Selected endpoint returned the recorded value | Its original implementation or training method |
| Single-variable intervention | That controlled input variation changed (or did not change) the observed response | A uniquely identified internal mechanism |
| Candidate replay | Independently generated code returned results for the tested inputs | Original source recovered |
| Comparison score | Finite behavioral compatibility under selected tolerance/scope | Paper's complete scientific findings reproduced |
| Independent held-out benchmark | A specified performance claim under recorded conditions | Training data, optimizer state, and author code reconstructed |

StartLux-Decision makes a suitable first experiment because the released inference service exposes confidence and reasoning controls. Its **official inference source already exists**; PaperRev's generative reconstruction should target *missing mechanism/training hypotheses* rather than needlessly regenerating that public serving engine.

All remote actions and candidate Docker execution require explicit opt-in. No external model, paid API, GPU, browser, or app account is activated by importing this library.

## Metamorphic differentiation: from a probe to a falsifiable mechanism hypothesis (v0.4)

One-variable inputs alone are not evidence of how a model responds. After observing **both** the original JSONL and the generated probe JSONL at the **same API origin/path/output pointer**, run a finite relation audit. This is analogous to REA's static-vs-runtime corroboration: the two observation files and the exact intervention plan are all independently authenticated before the comparison is allowed.

Extend the same `interventions.json` with bounded `checks`:

```json
{
  "schema": "paperrev/interventions-v1",
  "changes": [
    {"id": "plus-day", "path": "/days", "op": "add", "value": 1}
  ],
  "checks": [
    {"id": "monotonic-delay", "intervention_id": "plus-day", "output_pointer": "/late_probability", "relation": "increase"},
    {"id": "unchanged-label", "intervention_id": "plus-day", "output_pointer": "/category", "relation": "same"}
  ]
}
```

Supported relations are `same`, `different` (nested JSON, numeric absolute tolerance), `increase`, `decrease`, and `shift` (scalar numeric difference equals declared `expected_delta`, within `tolerance`). The latter three require finite numbers. All comparisons are finite-sample observations, not universal theorems.

```bash
# The first response returns BASE_EVIDENCE_ID from observe-api.
paperrev observe-api --workspace runs/reverse --cases model_cases.jsonl \
  --origin http://127.0.0.1:8090 --path /v1/systemone \
  --output-pointer /answers --execute

# `intervene` returns PLAN_EVIDENCE_ID and PROBE_CASES_JSONL.
paperrev intervene --workspace runs/reverse \
  --cases model_cases.jsonl --spec interventions.json

# Capture the *same* endpoint's responses to the generated inputs.
paperrev observe-api --workspace runs/reverse --cases PROBE_CASES_JSONL \
  --origin http://127.0.0.1:8090 --path /v1/systemone \
  --output-pointer /answers --limit 200 --execute

paperrev analyze-interventions --workspace runs/reverse \
  --baseline BASE_EVIDENCE_ID --probes PROBE_EVIDENCE_ID \
  --plan-evidence PLAN_EVIDENCE_ID --cases model_cases.jsonl \
  --probe-cases PROBE_CASES_JSONL --spec interventions.json
```

The analyzer checks exact input hashes, the generated probe-plan hash, parent IDs, input-path isolation, original and perturbed API output hashes, **hash-bound endpoint metadata**, and equal output projection. It returns per-check `supported_on_pairs`, `counterexample_observed`, or `unknown`, including paired/missing case counts and finite witness IDs. A partial capture can reveal a real contrary example, but **cannot establish all-case support**. Results are linked to the three parent Evidence IDs and included in the exported Evidence bundle.

Legacy v0.3 API capture metadata lacked a digest in its immutable Evidence record, so those captures must be repeated before v0.4 controlled contrast analysis; trusting an unbound endpoint label would weaken the conclusions. This tool still cannot identify the original model's hidden weights, causal internal computations, or historical training recipe. Stochastic outputs need repeated controlled runs and source-version fingerprints before attributing behavioral shifts to an intervention.

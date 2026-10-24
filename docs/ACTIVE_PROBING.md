# Active Mechanism Probing — REA-style black-box reverse engineering (v0.5)

## What changed

Software reverse engineering does not stop after collecting a trace. The investigator writes competing explanations, then asks the target a **new question on which those explanations disagree**. This module brings that workflow to paper/model reproduction using prior **authenticated Docker candidate replay** results.

The loop is:

```text
Published paper / executable model
       ↓
REA-style Workbench Evidence & implementation unknowns
       ↓
Generate at least 2 distinct candidate implementations
       ↓
Run each candidate on the SAME unqueried input pool in offline Docker
       ↓
active-select: identify inputs that separate candidate predictions
       ↓ [no model API call here]
observe-api: explicitly authorized real model query on SELECTED inputs
       ↓
active-evaluate: update finite candidate compatibility, preserve contrary witnesses
       ↓
Branch/repair candidate(s), generate new pool, repeat
```

The core idea is related to **Query-by-Committee / active learning**. It is a practical mechanism-selection baseline, not an assertion that candidate disagreement or greedy set coverage is a novel theorem, a calibrated information-gain measure, or recovery of authors' private training code.

## Exact source of candidate predictions

Before selection, use `paperrev generate` to register each candidate, then:

```bash
paperrev replay --workspace runs/investigation \
  --candidate candidate-A --cases pool.jsonl \
  --image python:3.12-slim --execute-tests
paperrev replay --workspace runs/investigation \
  --candidate candidate-B --cases pool.jsonl \
  --image python:3.12-slim --execute-tests
```

These commands produce **`candidate-behavior` Evidence IDs**, each linked to a retained Docker replay, code manifest SHA256 and output JSONL SHA256. The selector checks all four independently before it trusts the predictions. A fabricated `output.jsonl` path or a free-form LLM claim is insufficient.

The input `pool.jsonl` contains at most 200 rows, with `id` and `input` as in `observe-api`. Optional `parent_id` groups related perturbations; optional `query_cost` is a positive budget proxy (not measured latency):

```json
{"id":"p1", "parent_id":"root-a", "query_cost":1.0, "input":{"x":1}}
{"id":"p2", "parent_id":"root-b", "query_cost":1.5, "input":{"x":2}}
```

## Choose the next target questions

```bash
paperrev active-select \
  --workspace runs/investigation \
  --pool pool.jsonl \
  --candidate-run ev_FIRST_SANDBOX_REPLAY \
  --candidate-run ev_SECOND_SANDBOX_REPLAY \
  --budget 5 --min-distance 0.05 --numeric-scale 1 \
  --max-per-parent 1
```

For each input x and candidate pair i,j, we calculate a normalized bounded difference across JSON outputs:

\[
  D_{ij}(x) = \max_{\ell\in\text{output leaves}} \min\left(1,\frac{|f_i(x)_\ell-f_j(x)_\ell|}{s+\max(|f_i(x)_\ell|,|f_j(x)_\ell|)}\right)
\]

Categorical differences or shape incompatibility score 1; identical leaves score 0. The `max` is taken over comparable JSON leaves, not over arbitrary latent representation features. `s` is the declared positive `numeric_scale`. This is a heuristic, not an information-theoretic divergence.

A greedy selector favors candidate pairs **not yet separated** by earlier chosen inputs, discounts redundant pair disagreement, divides by an optional declared query-cost proxy and enforces a per-parent cap. It writes an immutable **selection manifest** plus selected JSONL inputs. If all candidates agree within the threshold, it returns `no-discriminatory-inputs`, **not** proof that the committee is right.

## Query the actual target only with explicit authorization

```bash
paperrev observe-api \
  --workspace runs/investigation \
  --cases "SELECTED_CASES_PATH_FROM_ACTIVE_SELECT" \
  --origin http://127.0.0.1:8090 --path /v1/systemone \
  --output-pointer /answers --execute
```

Replace `SELECTED_CASES_PATH_FROM_ACTIVE_SELECT` with the exact `cases_path` emitted by `active-select`. Inference is a separate call and must be authorized. An API output is observed behavior, not an independent scientific gold label.

## Narrow hypotheses with observed target behavior

```bash
paperrev active-evaluate \
  --workspace runs/investigation \
  --pool pool.jsonl \
  --selection ev_SELECTION \
  --observation ev_TARGET_CAPTURE \
  --atol 1e-6 --rtol 1e-5
```

`active-evaluate` verifies the full selection manifest, the unmodified input pool, both classes of evidence, the target capture metadata and every candidate replay/source digest. It then returns for **each candidate**:

- `compatible-on-tested-inputs`: all selected requests were actually observed and their outputs match within tolerance. Not global equivalence.
- `disagreed-on-observed-inputs`: at least one finite discrepancy. Includes concrete mismatch case IDs.
- `inconclusive`: missing observed cases, with no documented disagreement. Partial target captures never mark surviving candidates as compatible.

The assessment is a derived Evidence record citing the selection, target observation and all candidate replay records. No code executes during assessment.

## Boundaries and honest failure modes

1. **Shared committee bugs:** if every candidate agrees but is wrong, disagreement sampling cannot detect the mistake. Include exploratory/random control probes and independent paper claims in a complete research campaign.
2. **Stochastic outputs:** one sampled difference might be sampling noise. Repeat with pinned seeds and identical settings before making a causal mechanism claim.
3. **Restricted candidate family:** choosing an input that separates candidates does not show the original implementation lies inside that family.
4. **Cost:** `query_cost` is an operator-provided proxy; treat the ranking as only as reliable as this proxy.
5. **Execution:** all candidate Python must first run via PaperRev's explicit Docker replay. `active-select` does not run untrusted candidate code on the host.
6. **Authority:** this tool can verify finite behavioral equivalence, not reconstruct undisclosed data, loss functions, original optimizer states or author source.
7. **Privacy:** selected inputs are stored locally and, only after explicit `observe-api --execute`, sent to the selected model endpoint. Do not send sensitive cases to third-party APIs without authorization.

## StartLux first study

On StartLux, use the official inference code for the accessible mechanisms. Generate **alternative explanations of missing aspects** (e.g. confidence gate or modified candidate probabilities), run them on held-out typed questions and select where they differ. The current generic JSON interface is **not** a 27B weight inspection or a training-run reproducer. A later adapter can draw candidate values from controlled activation or weight interventions; only after that would this be a true *white-box* reverse engineering project.

Recommended first benchmark: frozen question variants with different option counts, option order, and distractor text. Separate the inference API's public mechanics from unknown training mechanisms; evaluate the latter only with controlled training and data.

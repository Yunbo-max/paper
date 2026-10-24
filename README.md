# PaperRev — Reverse Engineer Research Papers

**REA-style investigation and reconstruction for ML papers with incomplete releases.** PaperRev is an agent-facing engine that turns source-bound paper facts into a structured algorithm plan, generates independently runnable code, and checks that code against scoped obligations. It deliberately distinguishes a useful independent reproduction from recovery of unpublished original training code.

**Current version v0.5:** This is now a *generic paper-reproduction engine*, not a StartLux-only benchmark script. The existing StartLux prototype remains available as `paperrev-startlux`; StartLux is the first real case for the new engine.

| REA principle | PaperRev implementation |
| --- | --- |
| Target routing | `open-paper` PDF extraction, source-bound plans, candidates, optional model metadata |
| Source identity and graph | SHA256-identified paper/source, linked content-addressed Evidence records |
| Unknown registry | `record-unknown` and revision-checked `resolve-unknown` |
| Investigative tools | Native Python CLI plus optional `paperrev-mcp` server |
| Reconstruction | Optional local/remote OpenAI-compatible LLM, or agent-authored plan/code JSON |
| Controlled execution | Docker-only, offline, read-only candidate mount; no host fallback |
| Comparison | Sample-ID-aligned nested JSONL behavioral checks with numeric tolerance and coverage |
| Verification | Explicit bounded claim obligations; unknowns remain unknown |


## Model/API black-box reconstruction (REA website/app workflow)

PaperRev now observes **actual inference inputs and outputs** rather than relying only on paper prose. Like REA's website/app workflow, it collects bounded evidence, constructs one-feature interventions, proposes independent reconstruction code, replays that code and compares behavior on the **same cases**.

The source is a real model/API endpoint; its undisclosed training pipeline is not automatically reconstructed.

~~~bash
paperrev init --workspace runs/audit --title "Black-box reconstruction"
paperrev observe-api --workspace runs/audit --cases examples/blackbox_cases.jsonl \
  --origin http://127.0.0.1:8090 --path /v1/systemone \
  --output-pointer /answers --execute

paperrev intervene --workspace runs/audit \
  --cases examples/blackbox_cases.jsonl --spec examples/interventions.json

# After paperrev open-paper and register-plan have produced PLAN_SHA:
paperrev generate-behavior --workspace runs/audit --plan-sha PLAN_SHA \
  --observation OBSERVATION_EVIDENCE_ID --cases examples/blackbox_cases.jsonl \
  --label replica-1 --model CODING_MODEL \
  --endpoint http://127.0.0.1:8000 --execute-llm

paperrev check-reconstruction --workspace runs/audit \
  --observation OBSERVATION_EVIDENCE_ID --candidate CANDIDATE_ID \
  --cases examples/blackbox_cases.jsonl \
  --image python:3.12-slim --execute-tests
~~~

All target API calls and LLM calls require opt-in; remote targets require HTTPS and explicit authorization. The generated code runs only inside the declared offline Docker container. A finite matching result does not establish the original private source, training method, or general model equivalence.

See [black-box model reconstruction and limits](docs/BLACKBOX_WORKFLOW.md).

**New: controlled differential analysis.** Observe the target on baseline and perturbed cases, then use `paperrev analyze-interventions` (also MCP `analyze_interventions`) to authenticate both captures, verify one-field input changes and test finite metamorphic predictions. Results include counterexample IDs, incomplete coverage and Evidence links. See [contrast research design](docs/CONTRAST_RESEARCH.md).

## Active black-box probing (v0.5)

After replaying two or more distinct candidate implementations in offline Docker on **the same JSONL pool**, use `paperrev active-select` to select inputs on which the candidates disagree. This uses the known Query-by-Committee idea, not a claim of novel information-theoretic optimization.

~~~bash
paperrev active-select --workspace runs/audit --pool pool.jsonl \
  --candidate-run ev_CANDIDATE_A --candidate-run ev_CANDIDATE_B --budget 5

# Only a separate authorized command can query the original model:
paperrev observe-api --workspace runs/audit --cases SELECTED_CASES_PATH \
  --origin http://127.0.0.1:8090 --path /v1/systemone \
  --output-pointer /answers --execute

paperrev active-evaluate --workspace runs/audit --pool pool.jsonl \
  --selection ev_SELECTION --observation ev_TARGET_CAPTURE
~~~

`active-select` makes **zero target API calls**. Candidate Docker replay files, chosen test IDs and captured responses are SHA256-bound to Evidence. Partial capture never establishes compatibility; if all candidates agree the system reports no discriminating questions, **not** proof that they are correct. See [Active Mechanism Probing](docs/ACTIVE_PROBING.md).
## Install

```bash
# Clone the current main branch
 git clone https://github.com/Yunbo-max/paper.git
 cd paper

python -m pip install -e '.[all]'
python -m unittest discover -s tests -v
paperrev --help
paperrev-mcp   # optional MCP stdio server, requires [mcp] extra
```

The core CLI and code-generation contract use Python's standard library; PDF reading needs PyMuPDF, MCP needs the official Python MCP SDK. The generated candidate's dependencies are *not* installed automatically. A local Docker image must be provisioned by the operator for code execution.

## Autonomous paper-to-code pipeline

With an **operator-approved model endpoint** implementing `/v1/chat/completions`, PaperRev can run the stages automatically:

```bash
paperrev reconstruct \
  --workspace runs/my_paper \
  --pdf ./my_paper.pdf \
  --title 'An independently reconstructed method' \
  --endpoint http://127.0.0.1:8000 \
  --model my-local-coding-model \
  --label implementation-v1 \
  --execute-llm

# To run the generated unit tests, add *both*:
#   --execute-tests --image <already-installed-docker-image>
```

Automatic execution requires explicit permission flags. A remote model additionally needs `--allow-remote` and HTTPS; credentials go in `PAPERREV_API_KEY`. The generated code is **not executed during generation** and is recorded as an untrusted candidate. Test execution uses Docker with no network and resource caps. No GitHub Actions or background work is involved.

### Agent-assisted/manual route (no model API needed)

```bash
paperrev init --workspace runs/demo --title 'Paper replication demo'
paperrev open-paper --workspace runs/demo --pdf ./paper.pdf
# Read its source_sha256 in the result, then author a paperrev/plan-v2 JSON
paperrev register-plan --workspace runs/demo --source-sha PAPER_SHA --plan plan.json
paperrev generate --workspace runs/demo --plan-sha PLAN_SHA --label candidate-a --proposal candidate.json
paperrev test --workspace runs/demo --candidate CANDIDATE_ID \
  --image python:3.12-slim --execute-tests
```

`plan.json` must include exact snippets from cited PDF pages; hallucinated quotes are rejected. `candidate.json` contains independent implementation files (`src/reproduction.py`, `tests/test_reproduction.py`). The code generation protocol checks paths, size and syntax before publishing the candidate. The only execution path is explicit `paperrev test` inside Docker.

### Verify behavior and export evidence

```bash
paperrev compare --workspace runs/demo --reference reference.jsonl \
  --candidate candidate_predictions.jsonl --atol 1e-6 --rtol 1e-5
paperrev verify --workspace runs/demo --plan-sha PLAN_SHA --bindings obligations.json
paperrev inspect --workspace runs/demo --kind candidate-unit-tests
paperrev export --workspace runs/demo --out /tmp/paperrev-evidence.json
paperrev verify-bundle --file /tmp/paperrev-evidence.json
```

The reference and candidate files use one JSON object per line: `{"id": "case1", "output": ...}`. A comparison only establishes agreement within the provided IDs, numeric tolerance and reference authority; it does not establish scientific or causal equivalence. Missing observations force partial coverage, not success.

## MCP integration

`paperrev-mcp` exposes `create_session`, `open_paper`, `record_plan`, `generate_code`, `inspect_candidate`, `run_candidate_tests`, `compare_behavior`, `verify_reconstruction`, `record_unknown`, `update_unknown`, `inspect_evidence`, and `export_evidence_bundle`. Agents should read [the installed Skill](.agents/skills/paper-reverse-engineering/SKILL.md) before using it. Installing the Skill alone does not register any server; MCP registration must point at the running `paperrev-mcp` command.

## Evidence semantics

- **Source-reported** — a claim was extracted from a page or the authors' released artifact; not experimentally validated.
- **Plan/inference** — one interpretation of the described method; not the original implementation.
- **Candidate code** — syntactically valid generated Python; never run during generation.
- **Observed test** — a recorded Docker execution; not proof of the paper's reported result.
- **Observed comparison** — matching output JSONL and declared coverage, subject to reference provenance.
- **Scientific reproduction** — requires validated reference authority, same evaluation conditions and independent experiments; **not automatically claimed by v0.3**.

The tool cannot uniquely infer original optimizer histories, proprietary datasets, or unpublished code from final weights alone. Code and source records are MIT; original paper/code/weights remain under their respective authors' licenses. No StartLux weights or protected data are vendored.

See [design and REA mapping](docs/REA_ARCHITECTURE.md), [verification contract](docs/VERIFICATION.md), and [StartLux study plan](docs/STARTLUX_PLAN.md).

# PaperRev — Evidence-First Paper Reverse Engineering

Reconstruct what a partially released ML paper actually makes verifiable. This repository is a **research workbench**, not an automated claim that a private training run can be recovered from a paper or checkpoint.

**First case:** [StartLux-Decision v1.1 AutoThink](https://arxiv.org/abs/2610.17309).
The [inference implementation](https://github.com/StartLuxLabs/StartLux-Decision) and [27B open weights](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink) are public. Full original v1.1 training scripts, intermediate checkpoints, and private data are not known to be completely released. We reuse the official inference server instead of reimplementing it needlessly.

## Scope and evidence ladder

| Level | What passes? | What it does NOT establish |
| --- | --- | --- |
| 0. Source audit | Claims cite exact public papers/files and unresolved details are recorded | Accuracy or code parity |
| 1. Executable reconstruction | Testable code, schemas, basic shapes and numerical contracts | The authors used that code |
| 2. Behavioral reproduction | Same data and runtime controls, matched outputs and measured metrics | Same original training process |
| 3. Mechanism test | Controlled interventions support/falsify a causal hypothesis | General proof or historical training attribution |

Model weights make white-box measurements possible, but final weights do not uniquely determine the training data, seed, optimizer trajectory, or objective.

## Install and smoke tests

Requires Python 3.10+; the standard command set has **zero mandatory third-party Python dependencies**. Optional PDF text extraction uses PyMuPDF.

~~~bash
git clone https://github.com/Yunbo-max/paper.git
cd paper
git switch feat/evidence-first-startlux-20261024
python -m pip install -e .
python -m unittest discover -s tests -v

# Start a research case: produces claim, unknown and manifest records.
paperrev init --case cases/startlux_v1_1/claims.json --out runs/startlux

# Inspect metadata in HF without downloading any tensor data.
paperrev inspect-hub \
  --repo startlux-models/StartLux-Decision-v1.1-27B-AutoThink \
  --out runs/startlux/model_hub.json

# Optional: after downloading only the small model config/index files
paperrev inspect-local --model-dir ./models/startlux_metadata \
  --out runs/startlux/model_local.json

# Optional: extract paper PDF text page by page; figures require visual review.
python -m pip install -e '.[pdf]'
paperrev ingest-pdf --pdf path/to/paper.pdf --out runs/startlux/paper_pages.json
~~~

## StartLux inference experiment (explicit opt-in)

Use the [official StartLux inference server](https://github.com/StartLuxLabs/StartLux-Decision) locally, ideally a **4B AutoThink** checkpoint for the first pilot. The 27B model can be used later on an adequate GPU. No official training recipe is required to run this experiment.

~~~bash
# This is ONLY a tiny 3-question plumbing smoke test, not a scientific benchmark.
paperrev probe-startlux \
  --dataset cases/startlux_v1_1/smoke.jsonl \
  --base-url http://127.0.0.1:8090 \
  --out runs/startlux/always.jsonl \
  --budget 256 --limit 3 --execute

paperrev analyze-startlux \
  --records runs/startlux/always.jsonl \
  --out runs/startlux/gates.json \
  --gates 0.6,0.9 --mix 0.75
~~~

To use a remote HTTPS service, provide --allow-remote and an API key via STARTLUX_API_KEY. The probe does not run without --execute, does not store credentials and does not request or store thought text. The local dataset remains the source of raw prompts.

**Interpretation:** The gate sweep uses *actual ALWAYS-mode first and after-thought probability distributions*, but **simulates** gate decisions offline. It is not a live AUTO-mode benchmark; no inference latency or training improvement is inferred. Missing comparable pairs are excluded and reported, never silently counted as successes.

## Documentation

- [Architecture and evidence contracts](docs/ARCHITECTURE.md)
- [StartLux research protocol and go/no-go gates](docs/STARTLUX_PLAN.md)
- [Detailed evaluation protocol](docs/EXPERIMENT_PROTOCOL.md)

## Comparators and sources

REA ([reverse-engineering skill](https://github.com/morluto/rea)) motivates source evidence, uncertainty registries, and scoped verification. Jacobian ([typed math vocabulary](https://github.com/morluto/jacobian)) motivates exact local mathematical checks. Neither is bundled or required to run the initial code. PaperBench, PaperCoder, ReproAgent, and other replication systems remain baselines to measure against, not claimed components of this repository.

The included research code is MIT-licensed. **StartLux model weights have a separate CC BY-NC 4.0 license**; downstream usage must respect that license. No source code or model weights from StartLux are vendored here.

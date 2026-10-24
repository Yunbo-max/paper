# Architecture: one evidence owner per scientific claim

## Why this is not PDF-to-code

An executable implementation can be useful while being *non-identical* to the unpublished original. A successful benchmark score can coexist with wrong mechanism assumptions. PaperRev preserves that distinction in machine-readable artifacts.

~~~text
Paper PDF + referenced code + optional released checkpoint
        |
        v
Source inventory and page/location anchors  [observed / reported]
        |
        v
Claim ledger + implementation obligations + unknown ledger
        |             |
        |             +-- Unknown training choice? Keep explicitly unresolved.
        v
Candidate implementation and minimal numerical/shape tests
        |
        v
White-box or black-box experiments against available model
        |
        v
Independent comparison and scoped verification
        |
        +-- verified within stated experimental scope
        +-- contradicted / missing data / unidentifiable: remain OPEN
~~~

## Planned interfaces

- **Source/Claim:** claim ID, exact source URL/location, source type (paper, implementation, repository metadata), evidence digest, hypothesis status and verification plan.
- **Unknown:** unknown ID, missing authority, candidate interpretations, experiment needed to distinguish them, dependencies and resolution history.
- **Artifact:** pinned model ID/revision, config/index digests, actual tensor/checkpoint availability, model/inference framework versions and license.
- **Experiment:** frozen dataset digest, runtime controls, model snapshot, observations, seeds, hardware, output metrics, exclusions and deviations.
- **Review:** comparison against source claims, independent implementation tests, conclusion scope and explicit residual uncertainty.

Never identify a generated code variant with historical author source unless public evidence actually proves it. An implementation compatible with a PDF is one element of an admissible hypothesis set, not necessarily the original implementation.

## v0.1 executable boundaries

| Module | Supported now | Not supported yet |
| --- | --- | --- |
| core.init_case | Freeze source case and seed claim/unknown ledgers | Automatic scientific claim extraction from text |
| core.ingest_pdf | Text by page + SHA256 + missing-text flags | Equation/figure correctness and OCR |
| core.inspect_hub_model | Model repo snapshot SHA, configs and weight-index inventory | Model download, tensor introspection |
| core.inspect_local_model | Local config/index read, file SHA256 | Weight hash audit or full module forward pass |
| startlux.probe | Real always-mode model outputs from official API | Training or model internals |
| startlux.analyze | Paired proper scores and post-hoc gate/cost estimates | Live-auto equivalence and causal attribution |

PDF figure and mathematical validation must be routed to separate visual review and exact symbolic/numeric tests. For future math checks, Jacobian could supply typed SAT/SMT, matrix, polynomial and invariant operations. Do not treat Jacobian's finite-result certificates as a proof of an arbitrary deep learning generalization claim.

## Proof obligations for the first case

- Source-bound: confirm model repo, checkpoint revision, config, pretrained family and supported inference controls.
- Numerical: response distributions are finite, normalized, with stable labels across first and after-thought readouts.
- Behavioral: reproduce baseline and thought-enabled predictions with pinned weights and a labeled held-out dataset.
- Comparative: compare gate thresholds using matched examples, independent seeds where sampling matters, proper scores and inference cost.
- Mechanistic: only after behavior is established, test isolated interventions and claims about benefit from thinking versus final probability mixing.
- Historical training: **not identifiable from final weights** without extra source/experiment evidence; remain unverified.

## Why this differs from simply running REA

REA focuses on shipped software artifacts, executable behavior, call graphs and native binaries. PaperRev targets empirical scientific claims, uncertainty about under-specified ML methods, white-box activations, paired model behavior and controlled mechanism experiments. The evidence discipline is borrowed; the investigative oracles and targets are different.

No remote worker, scheduled task, proprietary API, GPU experiment or paper claim verification is hidden behind a CLI import.

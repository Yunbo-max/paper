# Research plan: REA-style differential mechanism reconstruction

**Goal.** Given a scientific claim and only a model's input/output interface, find short, directly checkable counterexamples to candidate behavioral rules. Then use these finite observations to select the next code-reconstruction hypotheses. This is **not** automatic recovery of original training code.

## Research target (StartLux)

The official inference API returns typed decision probabilities. For a controlled batch of frozen gold-labelled questions, vary exactly one property at a time: numeric fact, available option, option order or available context. For each query, capture `think_mode=never` and the actual always/auto settings as distinct experiments. A generic metamorphic relation can check whether a **specified output probability** increased, remained unchanged or moved by a predicted delta; neither a probability shift nor an independent generated predictor alone establishes the internal mechanism.

**Example failure question:** Would inserting a distracting option change relative probabilities on the old choices? To test this, predeclare an input permutation/addition and the mathematical output relation, then compare using a frozen model and common request settings. Current `intervene` supports replacing an existing object/list or reversing a list, not inserting new dictionary keys. The relation evaluator requires original and probe captures be authenticated and paired; causal mechanism claims require extra controls, repeated seeds and intervention access inside the network.

## Traceable contract

1. Baseline observed output: `api-observation` Evidence ID and source case SHA256.
2. Intervention generated input: `intervention-plan` Evidence ID, source/spec/probe SHA256, parent case IDs.
3. Post-intervention observed output: second `api-observation` Evidence ID and its captured response SHA256.
4. Relation judgement: `intervention-differential` Evidence ID with parents 1/2/3, compared-pair count, contradiction witnesses, unknown cases and metadata-coverage status.
5. Only after independently reproduced behavior: candidate code and controlled forward/gradient interventions.

## Benchmark protocol

Use hidden implementation ground truth for synthetic targets, as well as at least one publicly released ML model. Compare random vs hypothesis-targeted perturbations at matched query budgets; primary metric is **unique independently reproducible behavioral counterexamples per 100 queries**, not quantity of generated code. Secondary metrics: false-positive rate under stochastic repeat sampling, hypothesis coverage, model cost, and reproduction under new unseen samples.

For an unknown model version, report **observed differences** without claiming they were caused by the intervention. A single counterexample defeats the corresponding *finite deterministic input/output prediction* for that tested instance; it does not reveal the author's historical optimizer, dataset or training trajectory.

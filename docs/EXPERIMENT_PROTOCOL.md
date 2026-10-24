# Experiment contract

## Registration

An experiment is not valid until dataset name/commit/digest, target model SHA, variant, precision, inference engine/version, evaluation split, prompt template, sampling controls, random seeds and success/failure metrics are recorded.

The default three-case smoke file is **not an independent benchmark**. Select a legitimate dataset with gold answers and document overlap risks. DecisionPivot and Decision Index have their own release and evaluation rules; do not invent access to private evaluation questions.

## Reproduction levels

- **L0:** source audit and claim extraction; no model output.
- **L1:** runnable forward path, tests and numerical invariants.
- **L2:** published or held-out metrics on the same checkpoint and comparable inputs.
- **L3:** controlled interventions demonstrate causal mechanism contributions within a stated tested scope.
- **L4:** independent training reconstruction approximates reported learning results; never relabel it historical source recovery.

Report failure counts, exclusions and uncertainty at every level. No study can be promoted solely because all its code exits with status 0.

## Proper scores

For label y among K options and distribution p:

- Negative log-likelihood NLL(p,y) = -log(max(p_y, 1e-15)), lower is better.
- Multi-class Brier(p,y) = sum_j (p_j - 1[j=y])^2, lower is better.
- Accuracy = 1[argmax p == y], where ties follow the model option order.

For an observed always-think pair, score both p0 and p1 and a convex mixture with frozen alpha. Keep all strategies on **the same paired test rows**. Missing first/post probabilities, missing gold or altered option ids must not be counted as successes.

## Causal tests worth running next

1. **Gate false positives:** cases with low initial confidence but negative post-thought log-score gain.
2. **Gate false negatives:** cases with high initial confidence but positive expected gain from thinking.
3. **Option-count invariance:** top-p gate vs entropy/margin-normalized gating across 2, 4, 8, 16 and 26 options.
4. **Thought quality vs mixing:** p0, p1, mixture alpha, random/noisy thought control and matched thought length.
5. **OOD and ambiguity:** adversarial minimal-pair changes that flip the correct decision under a nearly identical surface context.

Hypotheses must be frozen before looking at held-out outcomes. If optimizing gate/mixture on a development split, evaluate once on a separate untouched test split and retain all seeds, including unsuccessful ones.

## Baselines

- Official StartLux inference, think_mode=never.
- Official StartLux inference, think_mode=always.
- Official live think_mode=auto at 0.6 and 0.9.
- Offline gate estimator derived from always traces (must be identified as *estimated*).
- Simple entropy or margin threshold learned only on development data, and a matched random thought-budget control.

The initial code only implements capture and the offline gate estimator. Full source-model/metric parity and retraining remain future experiments, not features reported as completed.

## Security, provenance, cost

--execute is mandatory for HTTP requests. Remote inference requires --allow-remote and HTTPS, with credentials in STARTLUX_API_KEY rather than command line. Keep data provenance and ownership clear, and only send samples the endpoint is authorized to receive. Do not log secrets or private thought text. Do not silently expand GPU or API spending. The runner stops on failure with a partial capture so it cannot be mistaken for successful benchmark completion.

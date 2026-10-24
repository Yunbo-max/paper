# Typed reconstruction verification

## One claim, one scoped obligation

A claim must be bound to specific Evidence with sufficient authority. The `verify` command consumes `bindings.json`:

```json
{
  "C1": {
    "kind": "output-match",
    "evidence_id": "ev_EXACT_SHA256_FROM_COMPARE",
    "min_fraction": 0.99
  },
  "C2": {
    "kind": "unit-test",
    "evidence_id": "ev_EXACT_SHA256_FROM_TEST",
    "candidate_id": "candidate-a-SHA"
  }
}
```

`pass` says only that the **declared bounded check** was observed and passed. It is **not** a proof of the paper's central claim. For end-to-end scientific reproduction, one must additionally establish data fidelity, training/inference comparability, seed variability, and result stability.

`unknown` means missing Evidence, insufficient authority, an incomplete output comparison or an active implementation unknown scoped to the claim. `fail` requires an applicable observed check that actually failed; lack of a witness is never failure.

A `comparison` accepts two JSONL files with `id` and `output` fields. Names may be strings or integer IDs, must be unique within a file. Numeric leaves use `abs(x-y) <= atol + rtol * abs(y)` approximately; Python's standard `math.isclose` semantics apply. Nested dicts/lists must match their entire key/length structure. The engine records exact unmatched IDs and paired coverage.

## Replay boundaries

The Docker runner launches a preinstalled image with `--network=none`, `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--pids-limit=64`, a non-root UID, finite memory/CPU/timeout and a read-only source mount. It never performs `pip install`, external model downloads or implicit host execution.

The run records the candidate manifest digest, image identity chosen by the caller, exact command, elapsed time, exit, bounded stdout/stderr and container cleanup outcome when applicable. An unverified Docker image tag is not a byte-pinned runtime; for publication-grade runs supply a digest-pinned image and record its local identity separately. Docker is a containment boundary, not a guarantee against all host vulnerabilities; only run inputs you are authorized to inspect and execute.

## Trust limitations

- Model-generated PDF plans are proposals; quotes are checked literally against page text, but mathematical interpretation remains a separate task.
- A generated implementation may satisfy a paper's prose without matching authors' unreleased code.
- Reference outputs are treated as caller-supplied data, not authenticated author oracle data.
- Changes across two checkpoints do not reveal a unique historical training process.
- A single unit test suite does not establish a statistical claim; train/validation/test contamination and multiple-comparison effects matter.

The workbench retains unknowns for facts not identifiable from open artifacts. Passing numerical fixtures cannot resolve a question about historical training data or optimizer choices unless there is relevant observed evidence.

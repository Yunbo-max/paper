# Paper Reverse Engineering

A research-oriented toolkit for **evidence-grounded replication of partially released ML papers**.

The project separates four claims that should never be conflated:

1. **Source facts** — what a paper, published artifact, or repository actually states.
2. **Executable reconstruction** — what our implementation can run and verify.
3. **Behavioral reproduction** — what matches measured model behavior and published results.
4. **Mechanism identification** — what controlled interventions causally support.

The first target is [StartLux-Decision v1.1 AutoThink](https://arxiv.org/abs/2610.17309). Its inference code and model weights are public, but the complete original training pipeline is not. We will reuse the official inference implementation, audit the released artifacts, then conduct controlled inference experiments before attempting any independent RLCD² training reconstruction.

> Do not label a generated implementation as original source, or a test passing as a scientific replication.

See the development branch for the first executable pilot.

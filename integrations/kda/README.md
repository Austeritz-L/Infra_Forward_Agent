# KDA bridge contract (planned)

KDA handles a bounded kernel task after whole-forward analysis establishes why it matters.
This directory defines the handoff; v0.1 has no automated bridge and includes no KDA source.

Input packet:

- parent revision, forward evidence id, hotspot name, rank/device and raw trace references;
- exact mathematical inputs/outputs, dtype/layout/quantization and routing contract;
- representative shape/distribution matrix, reference and tolerances;
- architecture/toolchain constraints, validation and evaluation commands;
- isolation workspace, resource budget and stopping conditions.

Output packet:

- candidate revision/patch + parent link;
- correctness reports, kernel benchmark/profile artifacts and rejection history;
- compatibility constraints (Stream, graph, alignment, state and workspace lifecycle).

The forward Executor then applies the candidate in an isolated checkout, re-runs the complete
precision/benchmark/precision matrix and captures a new trace. Kernel-only gains cannot advance the
forward baseline. Workflow docs: [KDA agent flow](https://github.com/NVlabs/kda/blob/acdd79343c5e9d0c7cb96289f26fd3f1051ce785/docs/agent-flow.md).

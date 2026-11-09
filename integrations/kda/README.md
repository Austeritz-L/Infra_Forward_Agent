# KDA-style handoff and returned evidence (v0.6)

KDA handles a bounded kernel task after whole-forward analysis establishes why it matters.
The offline exporter and returned-evidence gate are implemented; automatic KDA invocation, code
generation, compilation and hardware execution remain unimplemented. No KDA source is included.

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

Run the independent exporter with:

~~~bash
ifa kernel-task workspaces/hotspot/contract.json --output runs/kernel-task-001
ifa kernel-evaluate workspaces/hotspot/bundle.json --output runs/kernel-eval-001
~~~

The task freezes the entire active service matrix, captured operator cases, retained performance
bundles and original deployment evidence. It hashes fixed sources into task_digest. Missing
instrumentation, all-rank timelines, matched resource models or counters/proxies produce
analysis_required; this packet is for investigation, not candidate implementation.

The output includes kernel.task.json, docs/draft.md, docs/plan.md and input snapshots. The commands
are reviewable data and are never executed. An external KDA/coding agent owns its implementation
workspace and must return parent/patch/binary identity, deep post-analysis, frozen-reference
precision before/after unprofiled operator/forward/service rounds, and preserved KV capacity.
Operator-only or forward-only gains cannot promote evidence. Synthetic packets never recommend a
real candidate. See the complete [kernel contract](../../docs/domains/kernel.md) for schema and limits.

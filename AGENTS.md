# Repository instructions

## Scope

Build an extensible multi-architecture agent for scenario testing, performance analysis, distributed
capacity/performance selection, operator optimization and optional trace comparison. Keep runtime imports
accelerator- and LLM-provider-independent. Read README.md and docs/architecture.md first.
Read docs/framework.md and docs/domains/testing.md for domain contracts. v0.3 adds a prefill-only
suite v2 with aligned service/forward/operator workloads to campaign planning and artifact replay;
neither mode is an autonomous live optimizer. Document capability status honestly.

## Working rules

- Keep reusable workflow in this repo, task patches and generated evidence in ignored workspaces/ and runs/.
- Fix workload, image digest, weights/layout/quantization, runtime and measurement boundaries before comparisons.
- Use precision -> unprofiled timing -> precision; require all ranks and all workload conditions.
- Never present summed kernel work or temporal exposure as a causal critical path or achieved speedup.
- Preserve immutable baselines, candidate parents, rejection reasons, raw evidence and noise estimates.
- Shared hardware requires live ownership checks; never stop somebody else's process. Respect paused tasks.
- No automatic production replacement. A selected artifact is not a deployed candidate.
- Never commit credentials, private host inventories, checkpoint data or large profiler binaries.
- Use the standard library in the control plane unless a dependency materially improves the implementation.
- Register architectures by explicit identity/capabilities/resources; never infer a verified ISA or
  resource limit from a vendor name. S5000, SM90 and SM103 have separate profiles.
- Keep service, forward and operator measurements separate. Cache-capacity selection needs measured
  per-rank memory; topology alternatives need fixed physical resources and global workload.
- Read task-relevant playbooks and knowledge cases when operating as a coding agent. The CLI planner
  only records knowledge references; do not claim automatic retrieval or policy enforcement.
- Current testing focus is prefill: service requests generate one token; decode/TPOT are out of scope.
  Keep actual cache tiers, query/chunk lengths and routing evidence. Never derive operator batch/shape
  from client concurrency times prompt length. Uncaptured forward/operator entries remain unbound.
- Testing-contract thresholds are design defaults, not implemented runtime verdicts. Require complete
  SSE and all submitted-request accounting; do not report a success-only subset as acceptance.

## Validation

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check .
python -m build
```

Test interval overlap, rank isolation, incomplete traces, evidence validity, gate rejection and loop
selection when modifying those contracts. A GPU integration must add a reproducible hardware acceptance
record; CPU fixtures cannot qualify a backend.

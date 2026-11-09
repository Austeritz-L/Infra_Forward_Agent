# Repository instructions

## Scope

Build an evidence-driven optimizer for whole-forward latency and throughput. Keep runtime imports
accelerator- and LLM-provider-independent. Read README.md and docs/architecture.md first.
v0.1 is artifact replay, not an autonomous optimizer. Document capability status honestly.

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

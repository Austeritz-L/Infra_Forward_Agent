# Forward optimization task starter

Read AGENTS.md, the task contract, current baseline evidence, the appropriate playbook and all prior
candidate decisions. Inspect the actual code and runtime before making architecture claims.

Objective: improve complete forward latency and effective-token throughput for the declared workload.
Keep image, weights, numerical semantics and undeclared framework/scheduler settings fixed.

1. Write `docs/draft.md` in the task workspace: facts, missing evidence, a falsifiable bottleneck
   hypothesis, expected metrics, candidate scope, resource needs and stop conditions.
2. Write `docs/plan.md`: exact implementation, correctness, timing, post-timing correctness and
   profiling commands. Check ownership immediately before any GPU job; respect all pause boundaries.
3. Before implementation, bind instrumented all-rank timelines, source-correlated counters/proxies,
   matched resource limits and a quantitative falsification test. Missing evidence means analysis_required.
   Implement one isolated candidate with a parent revision and patch. For an operator, use the
   integration contract in integrations/kda/README.md.
4. Run the declared reference on every rank/condition, then unprofiled forward samples, then reference
   again. Separately capture a representative all-rank trace. Preserve failures and raw outputs.
5. Reprofile the candidate and test the predicted counter change. Check complete operator wrapper,
   prefill and service separately, frozen first-token references and preserved long-context KV capacity.
   Produce an evidence bundle and inspect the deterministic gate. A rejected candidate is recorded,
   not silently removed. A selected candidate becomes a new analysis baseline, not a deployment.
6. Re-analyze the new trace. Continue until the task budget, user stop, no-gain threshold or a genuine
   external blocker is reached. Never fabricate a measurement to fill missing fields.

Current v0.6 exports analysis-gated tasks and imports validation artifacts. This prompt is for an external coding agent; the CLI does not
execute these steps, call a model or run a profiler automatically.

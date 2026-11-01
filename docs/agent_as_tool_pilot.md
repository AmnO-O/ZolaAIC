# Agent-as-Tool-inspired 20-sample ablation

This is an adaptation, not reproduction of arXiv:2507.01489. That paper uses
Qwen2.5-7B Planner, GPT-4o-mini search Toolcaller and GRPO. Here one Qwen3.5-4B
backbone handles both roles offline; there is no RL or external summarizer.

Three arms on the same 20 train IDs, rotating/reversing arm order across samples:

- app_plan: current main pipeline, one Planner decision per Caller invocation.
- receipt1: Caller limited to the current goal; Python returns a deterministic
  subtask_result containing exact call/result observations and provenance.
- receipt2: same receipt and scope, up to two Caller invocations per Planner goal.
  A local empty Caller response hands control back to Planner within the sample
  deadline. An additional empty handoff is bounded, not recursively retried.

Receipts preserve rejected/uncertain results, source IDs and raw data, including
empty lists. They never claim a goal completed because a call succeeded. Full
execution_history is still retained; this pilot does not test lossy compression.
Both receipt arms change Caller scope as well as Planner input; their contrast
with app_plan is not an observation-format-only ablation. receipt1 vs receipt2
isolates the bounded local loop more closely.

Each independent arm/sample has exactly one ToolEnv; no reset or retry is used
inside an episode. All actual calls are exported; plans and receipts do not execute
actions. Evaluation reads train labels only after each prediction. Raw call F1
and organiser-canonical-state-gated call F1 (timeout ignored) are reported, with
the same official-formula-parity caveat as other local scoring.

Configuration: Qwen3.5 base, no adapter, train reference transcripts (no ASR),
seed 20261101 for 20 random train IDs, 50-second shared sample budget, 8 outer
execution rounds, Planner/Caller output caps 384, planning cap 20 seconds.
The outer round limit is held constant; Planner counts may differ. Full prompt
and tool traces are saved privately, never printed live. The default main
pipeline and running 200-sample experiment are not switched to these variants.

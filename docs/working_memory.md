# Working-memory experiment

Variants: A `clean`, B `memory_full`, C `memory_compact`. Planning remains available as a separate legacy experiment. All use the same agent, full permitted tools, evidence validation, execution ledger and submission exporter. No second model or environment is created.

B and C have the same prompt and strict parser: exactly one nonempty `<memory>` (at most 1600 characters), then native tool calls or literal `[]`. Memory is natural-language progress, not a machine-verified plan. Tool-like text inside memory never executes. Syntax errors allow one bounded repair; invalid outputs cannot update memory or silently terminate successfully.

B accumulates full history including previous memories. C retains all initial messages (original request, supplied history and context), latest accepted memory, and the most recent assistant call/result batch. Repair feedback remains until a valid response. Old live observations are removed from model context, but raw evidence, execution ledger and detailed logs are preserved outside it. Lost facts may therefore require retrieval again. Compaction does not summarize initial supplied history.

Memory is accepted only after protocol validation and describes observations before the proposed actions. Each prompted result has a stable observation_N source_id and environment_result/controller_feedback source_type. Validation feedback is not app evidence. Successful/failed/uncertain execution status is governed by the existing ledger, never memory. Branch selection and derived argument correctness remain model responsibilities; no semantic gate or arithmetic guarantee is claimed. Result-dependent calls must wait for the result; batched calls execute sequentially and are not automatically transactional.

`finished` means the model requested termination, not verified task completion. All actual ToolEnv calls (including failures) and its real state hash are exported; memory stays in diagnostics only. Character limits are not token guarantees. Existing generation/context limits, deadline checks and duplicate-write protections remain active.

## Protocol

Use notebooks/kaggle_pipeline.ipynb, same explicit train ID list, first clean then memory_full. Only if justified test memory_compact. Keep checkpoint, FP16, greedy decoding, determinism, cache, budget, maximum turns and logging identical. Start with TRAIN_TRANSCRIPTS=True (train only), repeat with False to measure ASR effects. Run sequentially on GPU, not simultaneously. Each configuration starts a fresh environment once per sample.

Previously inspected IDs are development data, not holdout. Freeze a configuration before evaluating new train IDs. Do not tune on public/private test. Use existing diagnostic F1 and per-sample traces; separately annotate branch errors, missing/extra actions and information loss. Task completion requires an independent evaluator; normal termination is insufficient. Compare token, prefill, decode and total sample latency from logs; no speed or score improvement is established by synthetic tests. Model loading is excluded by the existing runner; session/ASR costs remain in the sample budget.

Public notebook default remains clean/27 seconds. Train notebook defaults clean/90 seconds for development. Memory modes are opt-in; the development budget is not competition compliant.

## Context serialization correction

Compact retains memory and native calls together in the latest accepted assistant message, followed by its results and any repair feedback. It no longer inserts a separate assistant memory message. Until an older live batch can be removed, full and compact receive identical messages. Format-error feedback explicitly says the entire rejected response executed no calls and supplied no accepted facts or memory. This improves the recovery protocol but does not guarantee the model obeys it.

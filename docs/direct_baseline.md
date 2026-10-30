# Direct baseline v2

The default runner and both Kaggle notebooks select `--prompt-variant direct`. Frozen Gipformer 1.5 CPU INT8 and Qwen3-4B-Instruct-2507 GPU FP16 stay pinned in configs/baseline.json. Seeds, deterministic greedy generation, full permitted tool schemas, 16K context ceiling and exact-prefix KV cache remain unchanged. No training, model improvement or A30 speedup is claimed.

## Runtime

Original request/context/history -> direct prompt -> complete native calls or [] -> permission, evidence and organiser schema validation -> sequential ToolEnv calls -> original results with execution/source labels -> next decision.

One supplied ToolEnv per sample. Arguments are passed unchanged: no casting, null deletion, default omission, train-prior lookup, routing, query rewriting, generated memory, planner or reviewer. No Steerability import is required for the direct prompt. Old variants are compatibility experiments; they do not execute in the direct path.

An empty string, prose-only answer, malformed JSON, duplicate JSON keys, non-finite constants, stringified arguments, truncated call or calls mixed with [] is a protocol failure, never a normal stop. One correction is permitted per sample. Raw rejected output remains in private logs, not assistant execution history. Feedback explicitly says no rejected proposal executed. This parser is post-generation validation, not grammar-constrained decoding.

Each tool result is labelled executed true/false and succeeded/failed/uncertain/rejected, preserving the raw result. Only successful real results update evidence. The shared evidence store currently scopes observed IDs by app, not a complete typed entity provenance graph; an observed ID can still be the wrong candidate. Model semantics remain responsible for branches, candidate matching, independent actions before clarification and whether the user task is fulfilled. `finished` is only a model stop; task_success_verified is false. There is no automatic semantic completion oracle.

Confirmed duplicate writes are blocked by the existing ledger (limitation: deliberately repeated identical actions may also be blocked). Uncertain mutations stop the batch to avoid blind retries. Explicit failures return to the model. Clarification stops when actually called. Actual failed calls remain in the environment export. Submission always comes from that same environment's get_tool_calls/get_state_hash, not proposed calls or controller memory.

Deadline includes session creation and ASR, excludes model loading. Checks surround generation, validation and execution; a native operation can overrun before returning. No hard real-time guarantee is claimed. Public defaults: 27 seconds, 8 decisions. Train notebook: 90 seconds, 16 decisions for diagnosis; this does not meet the competition time limit.

## Evaluation gate

1. Fixed train development IDs, TRAIN_TRANSCRIPTS=True: measure tool reasoning independently of ASR. Keep full logs for diagnosis.
2. Same IDs/configuration, TRAIN_TRANSCRIPTS=False: measure end-to-end ASR effects.
3. Freeze configuration; evaluate on new train IDs not used for prompt edits. Train labels are never in inference context.
4. Before public submission, measure real audio under 27-second budget and validate full coverage. Do not tune on public/private labels.

Record checkpoint/revision, prompt hash, variant, transcript source, GPU, determinism, cache, budget, output tokens, prefill/decode times, actual calls, errors and termination. Current F1 is a diagnostic and the optional state-gated metric is a proxy, not verified official scoring. Compare per-sample outcomes; do not infer correctness from name-only F1 or normal termination. At ~4 tokens/s on the observed T4, even short actions can exceed 30 seconds: runtime remains an open benchmark, not solved by this rewrite.

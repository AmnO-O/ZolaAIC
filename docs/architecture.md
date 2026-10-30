# Single agent

Audio/history -> Gipformer -> original transcript plus phone annotation -> fixed policy and full permitted schemas -> native calls or [] -> allowlist/evidence/organiser schema checks -> sequential ToolEnv calls -> labelled actual results -> next decision.

There is one controller (`agent.py`). No memory, plan, variants, router, reviewer, task-specific tool-name mapping or argument rewriting. Model weights and ASR are pinned in configs/baseline.json. The same backend can run the base control or a locally merged trained adapter. The optional adapter must carry the policy hash written by train_adapter.py.

Raw rejected outputs stay in logs, not execution history. A bounded protocol correction says explicitly that nothing in that proposal executed. Validation rejections are labelled controller feedback. Real outcomes are labelled succeeded/failed/uncertain, preserving the raw result. Only successful real results update evidence. Uncertain writes stop; confirmed duplicate writes are blocked. Native batches are executed sequentially, not transactionally; the model must respect result dependencies.

All allowed tools are present on each turn. There is no implicit semantic goal verifier. Normal stop, clarification, validation failure, execution failure, uncertainty, no progress, turn limit and deadline are distinct diagnostics. The submission comes exclusively from the same environment's actual call log and final hash.

The history is retained in full; context overflow fails instead of silently truncating. Qwen3.5 uses a fresh hybrid/recurrent state for every decision; native cache remains enabled within decoding. No cache cropping is applied to hybrid states. Every sample resets the cache. No guaranteed speedup is claimed. Training and inference use the same chat serialization, tool schemas, context formatting and result envelope.

Remaining limitations: app-scoped IDs can still reference a wrong same-app entity; state freshness and intentional duplicate actions need task reasoning; malformed assistant outputs are detected after generation, not grammar-constrained; deterministic GPU kernels and 30-second wall time require testing on scoring hardware.

Current trial: Qwen/Qwen3.5-4B pinned to 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. The full conditional-generation loader includes vision weights (unused for text) and counts them against the parameter limit. Thinking is explicitly disabled in the native template. Protocol XML parameter values are decoded from schema types; this is wire decoding, not a heuristic rewrite of JSON arguments. Invalid/truncated XML, duplicate parameters, nonempty thinking or unrelated prose is rejected.

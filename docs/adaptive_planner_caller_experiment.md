# Observation-driven Planner → BM25 → Caller: 10-sample pilot

Date: 31 October 2026. Source: `c272a59327d58fdd3295676dd21d3da48356f9ca`. Private Kaggle job: https://www.kaggle.com/code/amnpham/lifemate-adaptive-planner-caller . Job completed, with exactly 10 expected unique result rows and no sample exceptions or deadline terminations. No training was performed; default inference remains unchanged.

## Implementation

One Qwen3.5-4B base serves two role prompts, without adapters. Each controller round:

1. Planner reads original context/history and actual observations, and emits strict JSON `next`, `goal`, `query`. It can choose call or done; max 192 output tokens and 20 seconds within the sample budget.
2. For call, BM25 searches original user text plus the current query, taking top 8 seeds and expanding catalog-derived read dependencies. Schemas are selected again each round.
3. Caller receives original context/history, unverified current-step guidance and selected native schemas. Planner guidance is not inserted as user authorization or environment evidence. Old generated goals are not accumulated into persistent facts.
4. The unchanged execution harness validates against the full permitted catalog, grounds identifiers, blocks confirmed duplicate writes and executes through one ToolEnv per sample. Actual results return to Planner. Done skips Caller; no Summarizer is required for the submission.

Malformed/truncated plans fall back to original-query BM25 and Caller without extra planner repairs. None occurred in this run. Caller output remains strict native tool calls or literal []. All actual ToolEnv calls, including failures, are preserved. No environment files are inspected or copied.

This is a prompt-only adaptation inspired by α-UMi, **not a reproduction of its trained Planner/Caller checkpoints or GLPFT results**.

## Controls and limitations

The same familiar train development IDs as the earlier pilot: train_0001..train_0009 and train_0011. Reference transcripts, not ASR. T4 FP16, PyTorch 2.11.0+cu128, Transformers 5.16.1, greedy decoding, thinking off, cross-turn cache reuse off, Caller max 384 tokens, at most 8 controller rounds, 90 seconds per sample including planning. IDs and key runtime/model/policy settings were programmatically checked against the previous result metadata.

The earlier two methods are historical reference runs, not rerun/interleaved three-arm controls. Planner output budgets, schema refresh and caller guidance change intentionally with the architecture. New timing includes index creation/shared input preparation; earlier timing omitted some milliseconds of shared CPU work. Model loading is excluded in all sample times. The 90-second development budget does not meet the official 30-second requirement.

## Results

| Same 10 samples | Original BM25 | One initial Planner + BM25 | Adaptive Planner → BM25 → Caller |
| --- | ---: | ---: | ---: |
| Diagnostic macro tool F1 | 0.1640 | **0.2606** | 0.1515 |
| Mean sample seconds | **32.60** | 39.54 | 40.68 |
| Samples at least 30 seconds | **3/10** | 8/10 | 6/10 |
| Planner calls per sample | 0 | 1 | 4.0 |
| Caller calls per sample | 4.7 | 5.0 | 3.1 |
| Mean total LLM input tokens across all turns | **11,100.1** | 12,582.5 | 11,218.8 |
| Samples with termination=finished | 0/10 | 0/10 | 9/10 |
| Planner JSON errors | n/a | 2/10 | 0/40 calls |
| Validation rejections | 13 | 9 | 0 |

Adaptive planning consumed 19.54 seconds/sample on average; calling consumed 20.72 seconds/sample. There were 40 planner and 31 caller invocations, with 9 planner done decisions. The remaining sample ended in protocol_error. No executed task-state success was independently verified; `finished` only means the controller accepted the stop decision.

The union of schemas exposed over all adaptive rounds covered 95% of unique gold tools on average, with every gold tool exposed at some time in 8/10 samples. This cumulative measure is **not directly comparable** to the static-schema recall numbers from the earlier methods, and does not guarantee a needed tool was visible in the right round.

### Per-sample diagnostic F1

| ID | One initial Planner + BM25 | Adaptive Planner/Caller |
| --- | ---: | ---: |
| train_0001 | 0.1538 | 0.2222 |
| train_0002 | 0 | 0 |
| train_0003 | 0.1667 | 0.1818 |
| train_0004 | 0 | 0 |
| train_0005 | 0 | 0 |
| train_0006 | 1.0000 | 0.6667 |
| train_0007 | 0 | 0 |
| train_0008 | 0 | 0 |
| train_0009 | 1.0000 | 0 |
| train_0011 | 0.2857 | 0.4444 |

## Trace findings

- **False completion:** train_0001's planner declares the event updated after only a calendar lookup and subsequent messaging; no calendar.update_event ran. The caller also used current weather where a forecast was needed.
- **False completion of independent subtasks:** train_0011's planner claims payment, messaging and a reminder are complete, but the exported calls contain payment without the requested message/reminder. Its finished status is not evidence of task completion.
- **Wrong action type:** train_0009 performs a wallet transfer instead of mobile top-up, then the planner accepts the transfer as completion. This loses the main win from the earlier planner-assisted retrieval pilot.
- **Arguments still ungrounded semantically:** train_0003 still omits the bill lookup and uses an unobserved transfer amount. Identifier validation does not establish amount correctness.
- **Wrong conditional continuation:** train_0006 repeatedly reads the gold price and continues planning a conditional reminder although the observed price does not satisfy the threshold. Caller prose is rejected twice; this sample ends in protocol_error.
- **Caller can diverge from planner intent:** a planner instruction to read wallet balance leads to a bank balance call in train_0011. A natural-language goal is advisory, not an enforced semantic contract.

These findings support only a narrow conclusion: the prompt-only role split reduces some protocol/duplicate-action failures, but stopping often becomes confidently premature. It does not establish the trained α-UMi architecture is ineffective; it shows this untrained adaptation is not ready to replace the current policy.

## Decision

Do not promote this configuration. More planning calls did not improve diagnostic F1 in the pilot, and average runtime remains above 30 seconds on T4. Avoid further case-specific prompt patches or more agent layers based on these 10 familiar samples. Separate next-step semantic correctness, argument grounding and factual completion verification from surface output-format success before further scaling. Any revised policy needs fresh train validation and eventual real-ASR/A30 testing.

Private raw results: local `artifacts/kaggle_adaptive_pc_results/adaptive_pc/summary.json`, `samples.jsonl`, `analysis.json`. These contain train-derived data and remain ignored by Git. The tests passed: 65 total, including observation-driven replanning, done bypassing Caller, malformed-plan fallback and guidance isolation from original messages.

# BM25 versus planner-assisted retrieval pilot

This experiment is separate from the default inference runner. It does not train or promote a model. Source `191c75f`, Qwen3.5-4B pinned in `configs/baseline.json`, one T4, greedy decoding, thinking off, prompt-cache reuse off, 384 output tokens per executor turn, 8 executor turns, 90 seconds per sample including planning. Reference train transcripts isolate routing from ASR. Live query printing is off.

The private Kaggle job is https://www.kaggle.com/code/amnpham/lifemate-retrieval-planner-comparison . Version 1 used an invalid contiguous ID list including missing train_0010 and is not an evaluation result. Version 2 selects the first 10 actual IDs from train metadata: train_0001 through train_0009, then train_0011. These are familiar development samples, not untouched hold-out.

## Arms

- BM25: original user text (including previous user turns) queries a catalog-only index. Word features plus accent-normalized character trigrams, fixed weights 1 / 0.2; top 8 seed tools.
- Planner + BM25: the same model first produces up to six short conditional search goals, with app capability descriptions and the original messages as input. Search uses original text plus those goals; the same top-8 selection and dependency expansion applies. Planner gets up to 30 seconds within the existing 90-second total, and 384 output tokens. Invalid/truncated plans fall back to original-query BM25; failures are reported.

The planner's plan is NOT inserted into executor messages. This deliberately isolates its effect on tool retrieval. Executor always retains original request/context/history. A later experiment can separately test presenting a plan to the executor.

Exact catalog read-tool mentions and x-ref app references expand dependencies. sys.ask_missing_info remains visible. No lexical match falls back to full schemas. The dependency strategy is conservative and may still miss implicit cross-app prerequisites. A positive BM25 score is not calibrated confidence; no-match fallback does not protect against all retrieval failures.

Both arms use the full permitted catalog for execution validation/grounding, while the model sees only selected native schemas. Selection is fixed through an episode. All real ToolEnv calls are exported, including failures; rejected calls stay in the trace. Exactly one fresh ToolEnv per arm/sample. Starting state hashes must agree. Arm order alternates by sample. Gold calls enter only scoring/coverage evaluation; no labels or future results enter routing/planning/execution input.

## CPU audit

On the first 20 actual train rows, original-query BM25 reduced mean available tools from 40.7 to 10.25 after dependencies. Mean per-sample unique-gold-tool recall was 0.71; all gold tool names were visible in 12/20 samples. Mean local CPU time including index construction was about 0.004 seconds, with coarse Windows timer resolution; not a deployment latency guarantee. This is insufficient retrieval coverage for production, despite substantial schema reduction. Do not tune ad hoc rules from the listed failures.

## Reproduce

Install requirements, download the pinned base with `scripts/download_models.py --component llm`, then run:

```text
python scripts/compare_retrieval.py --train-dir TRAIN_DIRECTORY --model LOCAL_QWEN35_DIRECTORY --ids IDS_JSON --output NEW_PRIVATE_OUTPUT_DIRECTORY --seconds 90 --top-k 8
```

IDS_JSON must contain actual unique train IDs. Output creation is exclusive. Setup may use network; model execution is local. Raw results contain private train data and must remain private. No files under split_environment are inspected or copied.

## Interpretation

Compare full-vs-selected input token count, mean unique-tool recall and complete tool-set coverage, diagnostic argument-aware F1, planner failure rate, terminations and total runtime. Coverage is a proxy: alternative valid trajectories may use different tools. F1 is not the official score or verified task-state success. `finished` is not semantic success. The two arms share limits, but 90 seconds does not meet the competition's 30-second runtime.

Index construction and some shared CPU preparation are outside the per-arm timer in this first runner; these are milliseconds locally, but must be included before deployment latency claims. No claim about peak training VRAM follows from this inference experiment. Validate any winning method on new train samples and real ASR before further training.

## Completed GPU results — 31 October 2026

Kaggle version 2 completed successfully. Exactly 20 result rows cover both methods for all 10 expected IDs. Runtime: PyTorch 2.11.0+cu128, Transformers 5.16.1, T4 FP16, deterministic algorithms enabled, no adapter, no ASR model. No sample-level exception or deadline termination occurred. These are inference-only experiments; no training was performed.

| Metric (same 10 samples) | Original BM25 | Planner + BM25 |
| --- | ---: | ---: |
| Diagnostic macro tool F1 | 0.1640 | 0.2606 |
| Mean unique-gold-tool recall | 64% | 84% |
| Samples with every gold tool visible | 5/10 | 7/10 |
| Mean initial executor input tokens | 2,034.9 | 2,016.7 |
| Full-schema initial tokens, size-only reference | 5,542.5 | 5,542.5 |
| Initial token reduction | 63.3% | 63.6% |
| Mean selected tools after dependencies | 10.3 | 10.1 |
| Mean total LLM input tokens across all turns, including planner | 11,100.1 | 12,582.5 |
| Mean execution seconds, including planner | 32.60 | 39.54 |
| Median execution seconds | 28.03 | 35.92 |
| Mean planner seconds | 0 | 5.32 |
| Samples taking at least 30 seconds | 3/10 | 8/10 |
| Planner JSON failures, falling back to original query | n/a | 2/10 |
| Executor decisions per sample | 4.7 | 5.0 |
| Validation rejections | 13 | 9 |

Both methods substantially reduce initial input, but planner-assisted routing increases aggregate LLM input and total time. No executed call targeted a permitted tool outside the visible schemas in either arm.

### Paired outcomes

Planner + BM25 wins on one sample, loses on one and ties on eight:

- train_0009: diagnostic F1 0 to 1.0. The executor used mobile top-up with planner-assisted schemas; original BM25 led to a wallet balance read and transfer. The plan itself was not passed to the executor.
- train_0003: 0.2000 to 0.1667. Both miss the bill lookup and execute an unsupported amount inferred by the model; the planner arm adds repeated reads. Its search goals fail to preserve the amount-lookup dependency reliably.
- train_0002 and train_0008: the planner recovers the forecast schema, but F1 stays zero because argument/extra-action/termination errors remain.
- train_0001 and train_0011: planner JSON is malformed; routing falls back to original-query BM25 while still paying planner latency.

Thus the macro increase is concentrated almost entirely in one example, not broad evidence of improved robustness. A 10-sample familiar development pilot cannot establish statistical or out-of-distribution superiority.

### Termination remains a problem

Neither arm has a `finished` termination. Original BM25: 6 protocol_error, 3 no_progress, 1 uncertain_outcome. Planner + BM25: 7 protocol_error, 2 no_progress, 1 uncertain_outcome. Native output guard rejects prose, repair can lead to repeated reads, and duplicate writes are blocked. A diagnostic F1 of 1 does not imply clean completion: train_0009 in the planner arm still ends in no_progress after repeated top-up proposals were rejected. Only the first actual top-up appears in exported calls.

### Decision

Keep the current default inference path unchanged. BM25-only coverage is inadequate; planner-assisted coverage is better in this pilot but still misses required tools on 3/10 samples, has 2/10 planner-format failures and is slower. Do not promote either method based on these results.

The next useful experiments are a full-schema execution control on identical samples/settings, then improved semantic retrieval with catalog-derived dependency expansion and calibrated fallback, validated on fresh train data. Executor stopping and argument grounding also need improvement; tool retrieval alone does not solve them. Do not infer that reduced token count has already resolved training OOM, or extrapolate these T4 times directly to A30.

Private raw artifacts are kept locally in `artifacts/kaggle_retrieval_results/retrieval_ab/` (summary.json, samples.jsonl, analysis.json). They are ignored by Git; this document contains aggregate results only.

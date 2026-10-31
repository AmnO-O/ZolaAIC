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

# Kaggle notebook

For a full public-test submission, use [kaggle_public_test.ipynb](kaggle_public_test.ipynb). It fixes the split to `public_test`, defaults to `LIMIT=None` (all samples), audio inference, deterministic mode, full installed tools, a 27-second budget, and detailed logging off. It prints sample progress and saves logs/timing. There is no train evaluation or reference-transcript path. After checking complete IDs, JSON structure, permitted tool names, and SHA-256 hash format, it exports `/kaggle/working/result/submission.jsonl` for manual download/submission. A subset, duplicate IDs, missing rows, or failed inference process prevents final export. This validation does not prove state correctness or compute the official score. Existing final results are never overwritten with different content.

To run it: import into a private Kaggle notebook, attach `amnpham/zalochallenge`, enable GPU and Internet, then run all cells. Check `DATASET_ROOT` against the Input panel. Source/model setup uses Internet; inference loads local assets. Save a notebook version to retain output files. The notebook records the resolved source commit; set `REVISION` to a commit hash for repeatable source selection. It never submits to the leaderboard automatically.

The generic notebook below remains useful for small train/debug runs with full step inspection.

Both notebooks use `PROMPT_MODE="system"` and `PROMPT_VARIANT="clean"`: one general prompt, no demonstrations, all permitted tools, and a 27-second internal budget. `baseline` is available only to omit system instructions for comparison. Old prompt variants, lexical pruning and continuation nudges are removed. The generic notebook retains configurable train IDs and optional diagnostic logging; the public notebook runs all public samples by default.

Calls pass through schema/evidence validation without semantic argument rewrites. One ToolEnv is created per sample. If session creation/export fails, remaining samples continue, but the result is partial and the process exits nonzero. This new pipeline needs fresh GPU accuracy and timing measurements; previous Baseline A scores do not describe it.

Import [kaggle_pipeline.ipynb](kaggle_pipeline.ipynb) into a new **private** Kaggle notebook using **File → Import Notebook** (or upload it when creating a notebook).

1. Attach the private dataset `amnpham/zalochallenge` with **Add Input**.
2. Select a GPU and enable Internet for cloning, dependencies, and model downloads.
3. Edit the first code cell. The supplied dataset root is `/kaggle/input/datasets/amnpham/zalochallenge/train+public_test`; the notebook also handles Kaggle's shorter `/kaggle/input/zalochallenge/train+public_test` mount.
4. Run cells in order. Start with `LIMIT = 10`, then use `300` or `None` for all samples.
5. Download `submission.jsonl` and reports from the printed `/kaggle/working/runs/...` directory. Save a notebook version to retain outputs.

Use `SPLIT = "public_test"` for public-test inference. Train-only scoring is skipped automatically. Keep `TRAIN_TRANSCRIPTS = False` for actual audio inference. `REFERENCE_REPLAY` is an optional, separate train evaluator and defaults to false.

`REVISION` can be a Git commit hash for a reproducible source snapshot. A clean existing checkout is fetched and checked out at the requested revision without merging; local edits stop setup. Preserve edits and choose a new `REPO_DIR` if needed. Each inference run gets a separate output directory.

Model downloads can take several minutes and require disk space for the roughly 8 GB FP16 LLM plus cache and dependencies. To use pre-attached model assets, set `LLM_PATH` and `ASR_PATH` to their local model directories. Those assets must match the pinned configuration. Internet is still needed for source/dependency setup unless you prepare those separately.

The default sample budget is 27 seconds. A larger development budget does not satisfy the competition's 30-second requirement. Kaggle timings and diagnostic train metrics do not establish final A30 performance or the official score.

`PROMPT_CACHE = True` enables within-sample prompt KV reuse in the Transformers backend. Set it to false to compare full prompt processing. The cache resets for every query. Inspect `llm.cache` events for `cached_tokens` and `fresh_tokens`, and `llm.output` for `prefill_seconds` and `generation_seconds`.

Detailed query tracing is enabled by default with `STEP_LOGGING = True`. With `LIVE_STEPS = True`, live output and `inference.log` include each stage's full inputs and outputs as formatted JSON. `steps.jsonl` stores full inputs/outputs, and the final notebook cell displays expandable steps for a selected `QUERY_ID`. Logs include context, transcripts, the rendered LLM prompt, generated tool-call text, validation, tool results, and final predictions. They contain observed data and generated output, not hidden model reasoning. Keep them private. Disable logging for performance measurements because writing full prompts adds overhead.

After importing the updated notebook, restart the Kaggle session and run from the first cell. The setup supports `REVISION="main"` or an exact commit and rejects source versions that predate the clean pipeline.

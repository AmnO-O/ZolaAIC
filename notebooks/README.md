# Kaggle notebook

The updated notebook defaults to `AGENT_MODE="adaptive"`, `PROMPT_MODE="system"`, `TOOL_SELECTION="shortlist"`, and `PLANNING="off"`. After measuring this baseline, set `PLANNING="auto"` to test a short plan for conditional/sequential requests. `"always"` attempts planning whenever enough time remains. Planning consumes the same sample budget and falls back to direct execution on invalid output or timeout. See [ADAPTIVE_AGENT.md](../ADAPTIVE_AGENT.md) for behavior and limitations.

For the previous controller, set `AGENT_MODE="legacy"`; routing/planning options are then ignored. `TOOL_SELECTION="all"` isolates controller changes from tool routing. Keep `PROMPT_MODE` fixed across comparisons.

Import [kaggle_pipeline.ipynb](kaggle_pipeline.ipynb) into a new **private** Kaggle notebook using **File → Import Notebook** (or upload it when creating a notebook).

1. Attach the private dataset `amnpham/zalochallenge` with **Add Input**.
2. Select a GPU and enable Internet for cloning, dependencies, and model downloads.
3. Edit the first code cell. The supplied dataset root is `/kaggle/input/datasets/amnpham/zalochallenge/train+public_test`; the notebook also handles Kaggle's shorter `/kaggle/input/zalochallenge/train+public_test` mount.
4. Run cells in order. Start with `LIMIT = 10`, then use `300` or `None` for all samples.
5. Download `submission.jsonl` and reports from the printed `/kaggle/working/runs/...` directory. Save a notebook version to retain outputs.

Use `SPLIT = "public_test"` for public-test inference. Train-only scoring is skipped automatically. Keep `TRAIN_TRANSCRIPTS = False` for actual audio inference. `REFERENCE_REPLAY` is an optional, separate train evaluator and defaults to false.

`REVISION` can be a Git commit hash for a reproducible source snapshot. Existing checkouts are never overwritten; start a fresh session or change `REPO_DIR` to clone again. Each inference run gets a separate output directory.

Model downloads can take several minutes and require disk space for the roughly 8 GB FP16 LLM plus cache and dependencies. To use pre-attached model assets, set `LLM_PATH` and `ASR_PATH` to their local model directories. Those assets must match the pinned configuration. Internet is still needed for source/dependency setup unless you prepare those separately.

The default sample budget is 27 seconds. A larger development budget does not satisfy the competition's 30-second requirement. Kaggle timings and diagnostic train metrics do not establish final A30 performance or the official score.

`PROMPT_CACHE = True` enables within-sample prompt KV reuse in the Transformers backend. Set it to false to compare full prompt processing. The cache resets for every query. Inspect `llm.cache` events for `cached_tokens` and `fresh_tokens`, and `llm.output` for `prefill_seconds` and `generation_seconds`.

Detailed query tracing is enabled by default with `STEP_LOGGING = True`. With `LIVE_STEPS = True`, live output and `inference.log` include each stage's full inputs and outputs as formatted JSON. `steps.jsonl` stores full inputs/outputs, and the final notebook cell displays expandable steps for a selected `QUERY_ID`. Logs include context, transcripts, the rendered LLM prompt, generated tool-call text, validation, tool results, and final predictions. They contain observed data and generated output, not hidden model reasoning. Keep them private. Disable logging for performance measurements because writing full prompts adds overhead.

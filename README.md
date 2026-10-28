# ZolaAIC — offline Vietnamese tool agent

Source for a Vietnamese voice assistant using **Gipformer 1.5 68M RNN-T**, **Qwen3-4B-Instruct-2507**, and the actual Steerability SystemPrompt/FewShot controls. Model weights are frozen. No fine-tuning or RL is included.

Audio and conversation history → local ASR → controlled LLM tool calls → organiser ToolEnv → `submission.jsonl`.

For Kaggle, import [notebooks/kaggle_pipeline.ipynb](notebooks/kaggle_pipeline.ipynb) and follow the [notebook instructions](notebooks/README.md). It clones this repository, prepares models, and runs on your attached dataset.

The notebook now includes the [adaptive agent experiment](ADAPTIVE_AGENT.md): catalog-based tool selection with recovery, optional bounded planning, observed-ID checks, action tracking, and explicit completion. It defaults to system-only prompting and planning off so the changes can be compared independently. The command-line default remains the legacy controller.

## Setup

Use Python 3.11+ and a CUDA-capable environment for the LLM:

```bash
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -r requirements.txt
python scripts/download_models.py
```

Downloads happen during setup. Inference loads local assets only. Gipformer uses CPU INT8 greedy decoding with four threads; the LLM uses GPU FP16. For an ASR-only environment, install `requirements-asr.txt` and download with `--component asr`.

The unmodified Steerability Python source is included under `vendor/steerability`, with its LICENSE, NOTICE, and source hash manifest. No separate Steerability checkout or full toolkit installation is required. `scripts/prepare_steerability.py --source /path/to/steerability` can regenerate this vendor copy from a local checkout.

## Inference

Obtain the authorised competition package separately. Point `--split-dir` at a split containing its JSONL, tool catalog, audio, and supplied `zaic_tools`. The solution accesses app data through one ToolEnv per sample.

Set these environment variables (PowerShell uses `$env:NAME="value"`):

```bash
export LIFEMATE_LLM_PATH="$PWD/models/llm"
export LIFEMATE_ASR_PATH="$PWD/models/asr"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/infer.py --split-dir /path/to/public_test --backend lifemate.backend:create_backend --output result/submission.jsonl
```

The runner refuses to overwrite an existing output. Defaults: `LIFEMATE_ASR_KIND=gipformer`, `LIFEMATE_PROMPT_MODE=steerability`, 27-second sample budget, and a 16,384-token context limit. Prompt modes `system` and `baseline` support controlled comparisons. The competition target is 30 seconds per sample; A30 timing and deployment validation are pending.

For development, add `--step-log result/steps.jsonl --verbose-steps` to save every query's observed processing inputs/outputs and print full step inputs/outputs live. The notebook enables this by default and provides an expandable query viewer. Full prompts, tool responses, and context remain in this separate private diagnostic file; the submission schema stays unchanged. Disable detailed logging when measuring runtime.

## Development

```bash
python -B -m unittest discover -s tests -v
```

The local tests cover prompt integration, tool parsing, ASR configuration, controller behavior, prompt-cache correctness, and submission export with synthetic fixtures. Cache tests include greedy-output parity against uncached generation using a tiny, randomly initialized Qwen model. They do not require competition data or downloaded models. An actual offline CPU ASR smoke also passed; no new end-to-end accuracy improvement is claimed.

Prompt KV reuse is enabled within each sample and reset between samples. The backend checks exact token prefixes and crops mismatching suffixes before reuse. Set `LIFEMATE_PROMPT_CACHE=0` to disable it. Detailed traces record cache hits and prefill/generation wall times; GPU speedups remain to be measured.

`scripts/evaluate.py` provides train-only diagnostic evaluation; it is not an official scorer. Reference transcripts and labels must remain development-only.

See [PIPELINE.md](PIPELINE.md) for the implemented architecture. `configs/baseline.json` records pinned model revisions. Some generation limits are also enforced directly in the backend; this file is not a general runtime configuration loader.

This repository contains source, tests, dependencies, and vendor licences. Competition data, organiser runtime code, model weights, credentials, experiment outputs, and Docker packaging are excluded. Third-party source retains its own licence; this repository grants no rights to restricted competition data or model assets.

# LifeMate — Direct baseline v2

Offline Vietnamese tool agent: **Gipformer 1.5 68M CPU INT8 -> Qwen3-4B-Instruct-2507 GPU FP16 -> native actions -> evidence/schema validation -> organiser ToolEnv -> observations**.

The new default is `direct`, implemented in `src/lifemate/direct.py` with instructions in `src/lifemate/prompt_assets/direct.txt`. No generated memory, plan, reviewer, train-prior policy, keyword router, automatic parameter rewriting or semantic completion oracle. It retains full observed history and passes arguments unchanged. Historical variants remain selectable for comparisons, outside the default controller.

Model assets, seeds, greedy decoding and deterministic algorithms remain pinned. All permitted tools are exposed. Exact-prefix KV caching and the 16,384-token context ceiling are preserved. This is an untrained baseline: no accuracy improvement or A30 runtime compliance is claimed before GPU evaluation.

## Kaggle

Import [train/debug notebook](notebooks/kaggle_pipeline.ipynb), attach the authorised private dataset `amnpham/zalochallenge`, select GPU and enable Internet for setup. It clones this repository and downloads pinned models. Inference uses local assets and offline Hugging Face flags.

Default settings: `PROMPT_VARIANT="direct"`, train development IDs, 90-second budget, 16 decisions, detailed logs. Set `TRAIN_TRANSCRIPTS=True` for the first tool-reasoning diagnostic; then repeat with False for real Gipformer audio. Train reference transcripts must never be used for test splits. Run a frozen configuration on fresh train IDs before interpreting improvements.

[Public-test notebook](notebooks/kaggle_public_test.ipynb) defaults to real audio, all samples, 27 seconds and 8 decisions. It validates coverage/format and exports result/submission.jsonl for manual submission. It neither scores public accuracy nor submits automatically. A larger development budget does not satisfy the competition's 30-second limit.

See [execution and evaluation protocol](docs/direct_baseline.md).

## Local setup

Python 3.11+, CUDA GPU for LLM, organiser package obtained separately:

```bash
python -m pip install -r requirements.txt
python scripts/download_models.py
export PYTHONPATH="$PWD/src"
export LIFEMATE_LLM_PATH="$PWD/models/llm"
export LIFEMATE_ASR_PATH="$PWD/models/asr"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/infer.py --split-dir /data --backend lifemate.backend:create_backend --prompt-variant direct --output result/submission.jsonl
```

Output creation is exclusive: an existing run is never silently overwritten. Optional `--step-log result/steps.jsonl --verbose-steps` records and prints full processing inputs/outputs. Logs are separate from submissions and contain private context. Disable detailed logging for timing measurements.

## Guarantees and limits

One ToolEnv per sample, only organiser APIs for app data, actual calls and final hash exported from that same environment. No direct reads of environment files. Failed actual calls remain in submission. Proposed or rejected-before-execution calls stay in diagnostics only. Seeds/greedy decoding do not substitute for reproducibility checks on scoring hardware.

Only native calls or explicit [] are accepted; malformed output gets one bounded correction. Validation feedback explicitly marks nonexecution. Failed and uncertain writes are not completed. Uncertain mutations stop execution rather than being retried automatically. The ledger blocks identical successful writes; intentional identical repeats are a known limitation.

`finished` means the model requested a stop, not independently verified task success. Conditions, matching candidates, amounts, times, missing information and semantic completeness remain model responsibilities. Tools run sequentially; the harness does not infer a dependency graph. Generation, validation and execution have deadline checks, but blocking native operations may overrun before returning.

## Verification

```bash
python -B -m unittest discover -s tests -v
```

Synthetic tests exercise parsing, evidence boundaries, call/result pairing, failure recovery, actual-call export, CLI isolation and notebook compilation. They do not measure the model's decision quality. Train metrics in scripts/evaluate.py and scripts/evaluate_run.py are diagnostic; optional state-gated F1 is an unverified scoring proxy. Historical research notes are not organiser scoring specifications.

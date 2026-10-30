# LifeMate

One offline Vietnamese tool agent, one fixed policy, and a train-only supervised-learning path. Old memory/planning variants, Steerability, prompt switching, train priors, canonicalizers and experimental controllers have been removed. Previous implementations are in Git history.

The untrained Qwen checkpoint is a control, not a validated solution. Reorganizing source does not establish better task accuracy. The next model experiment is a tool-use adapter trained on causal train trajectories.

## Files

- `src/lifemate/agent.py`: native-call parser and the only execution loop.
- `src/lifemate/prompts.py`, `prompt_assets/system.txt`: fixed policy shared by training and inference.
- `backend.py`, `asr.py`, `prompt_cache.py`: pinned local Qwen/Gipformer, deterministic decoding, exact-prefix KV cache.
- `input_adapter.py`: public input metadata and one organiser ToolEnv per inference sample.
- `grounding.py`, `validation.py`, `execution.py`: evidence, schema handoff and execution facts. Arguments are never rewritten.
- `training.py`: train-only causal trajectories and assistant-only loss masks.
- `scripts/prepare_sft.py`, `scripts/train_adapter.py`: private training data and QLoRA adapter training.
- `scripts/infer.py`, `scripts/evaluate.py`, `scripts/evaluate_run.py`, `scripts/validate_submission.py`: inference, diagnostic evaluation and format/coverage checks.

## Kaggle

1. [Train adapter](notebooks/kaggle_train.ipynb): private train data, default **one-step GPU smoke test** using 8 train and 2 validation examples. Set `MAX_STEPS=-1` only after the smoke test succeeds to train one full epoch. T4 VRAM/runtime remain to be measured.
2. [Evaluate on train](notebooks/kaggle_pipeline.ipynb): set `ADAPTER_PATH` to the adapter and `IDS_FILE` to the prepared `manifest.json` to select validation IDs. First compare base vs adapter on reference transcripts; then repeat with real Gipformer audio. Default development budget is 90 seconds, not competition compliant.
3. [Public inference](notebooks/kaggle_public_test.ipynb): real audio, all samples, 27-second internal budget. Validate complete output before manual submission. No automatic leaderboard submission or test-label tuning.

Training dependencies are in `requirements-train.txt`; inference without an adapter uses `requirements.txt`. Model downloads happen only during setup. All inference assets must be local. Training output records the resolved packages. Keep restricted generated examples, logs, checkpoints and Kaggle outputs private.

## Local commands

```bash
python -m pip install -r requirements-train.txt
python scripts/download_models.py
python scripts/prepare_sft.py --train-dir /path/to/train --output artifacts/sft
python scripts/train_adapter.py --model models/llm --data artifacts/sft --output artifacts/smoke --smoke
```

For inference, set `LIFEMATE_LLM_PATH`, `LIFEMATE_ASR_PATH`, `PYTHONHASHSEED=42`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and optionally `LIFEMATE_ADAPTER_PATH`. The backend merges a local adapter into the base before inference and checks its policy hash. Then:

```bash
python scripts/infer.py --split-dir /data --backend lifemate.backend:create_backend --output result/submission.jsonl
python -B -m unittest discover -s tests -v
```

The runner refuses to overwrite outputs. `--step-log result/steps.jsonl --verbose-steps` saves/prints private processing details. Old `--prompt-variant` options have been removed.

## Limits

One ToolEnv per inference sample, app state accessed only through organiser APIs, all actual calls including failures exported with its real final hash. No direct environment-file reads. Training conversion reads train JSONL/catalog only and creates no ToolEnv. Prior assistant actions are never replayed by inference.

Seeds and greedy decoding are fixed. The model remains below the parameter limits and the total inference context is capped at 16,384 tokens. Deadline checks cannot interrupt every blocking native operation. A30 compliance, GPU training and model quality are not established by CPU tests.

`finished` is a model stop, not task-success verification. Branches, entity selection and semantic completeness remain model responsibilities. Evidence IDs are currently app-scoped, not a full typed provenance graph. Identical-write protection can also block intentionally repeated identical actions. Failed or uncertain writes are not claimed complete.

Current F1 is diagnostic, not verified official scoring. See [architecture](docs/architecture.md) and [training protocol](docs/training.md).

# LifeMate — Qwen3.5-4B trial

One offline Vietnamese tool agent, one fixed policy, and a train-only supervised-learning path. Old memory/planning variants, Steerability, prompt switching, train priors, canonicalizers and experimental controllers have been removed. Previous implementations are in Git history.

The current trial uses the official Qwen3.5-4B post-trained checkpoint without LifeMate fine-tuning, revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (2026-03-02), with thinking disabled and a 16K input/output cap. The published full checkpoint is 4,659,865,088 parameters; the loader still checks actual loaded counts plus ASR. Native XML is decoded with schema types. This is a control, not a validated solution. Reorganizing source does not establish better task accuracy. The next model experiment is a tool-use adapter trained on causal train trajectories.

Prompt context uses a single `installed_apps` dictionary mapping installed app IDs to descriptions from the split catalog. It emits no separate app-description list and adds no app-wide prefix to tool descriptions. Inference and SFT preparation use the same formatter. Regenerate prepared SFT examples after updating this prompt/context format.

## Files

- `src/lifemate/agent.py`: native-call parser and the only execution loop.
- `src/lifemate/prompts.py`, `prompt_assets/system.txt`: fixed policy shared by training and inference.
- `backend.py`, `asr.py`, `prompt_cache.py`: pinned local Qwen/Gipformer, deterministic decoding, exact-prefix KV cache.
- `input_adapter.py`: public input metadata and one organiser ToolEnv per inference sample.
- `grounding.py`, `validation.py`, `execution.py`: evidence, schema handoff and execution facts. Arguments are never rewritten.
- `training.py`: train-only causal trajectories and assistant-only loss masks.
- `scripts/prepare_sft.py`, `scripts/train_adapter.py`: private training data and QLoRA adapter training.
- `scripts/infer.py`, `scripts/evaluate.py`, `scripts/evaluate_run.py`, `scripts/validate_submission.py`: inference, diagnostic evaluation and format/coverage checks.

For a paired model control, `configs/qwen3_control.json` pins Qwen3-4B-Instruct-2507. Both models share the same policy and execution loop; the backend selects the checkpoint-native XML or JSON wire format. Compare with identical IDs, budgets, precision and cache settings.

## Kaggle

First run [the three-sample trial](notebooks/kaggle_pipeline.ipynb) with reference transcripts, then real audio. Fresh model files download to models/qwen3.5-4b; do not reuse the old models/llm directory. Cross-turn cache is disabled for hybrid recurrent state, but generation still uses its native cache within each decision. The runtime uses BF16 where supported, otherwise FP16 (T4). GPU quality and runtime must be measured before training.

1. [Train adapter](notebooks/kaggle_train.ipynb): private train data, default **one-step GPU smoke test** using 8 train and 2 validation examples. Set `MAX_STEPS=-1` only after the smoke test succeeds to train one full epoch. T4 VRAM/runtime remain to be measured.
2. [Evaluate on train](notebooks/kaggle_pipeline.ipynb): set `ADAPTER_PATH` to the adapter and `IDS_FILE` to the prepared `manifest.json` to select validation IDs. First compare base vs adapter on reference transcripts; then repeat with real Gipformer audio. Default development budget is 90 seconds, not competition compliant.
3. [Public inference](notebooks/kaggle_public_test.ipynb): real audio, all samples, 27-second internal budget. Validate complete output before manual submission. No automatic leaderboard submission or test-label tuning.

Training dependencies are in `requirements-train.txt`; inference without an adapter uses `requirements.txt`. Model downloads happen only during setup. All inference assets must be local. Training output records the resolved packages. Keep restricted generated examples, logs, checkpoints and Kaggle outputs private.

## Local commands

```bash
python -m pip install -r requirements-train.txt
python scripts/download_models.py
python scripts/prepare_sft.py --train-dir /path/to/train --output artifacts/sft
python scripts/train_adapter.py --model models/qwen3.5-4b --data artifacts/sft --output artifacts/smoke --smoke
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

The generation ceiling is configurable with `LIFEMATE_MAX_NEW_TOKENS` (default 1024), or `MAX_NEW_TOKENS` in the notebooks. It applies per decision, stops early at native EOS, and remains subject to the sample deadline and 16K total context cap. The public notebook explicitly retains a 384-token ceiling pending latency validation.

Inference accepts only consecutive native tool-call blocks or literal `[]`. A generation-time prefix guard stops text outside that envelope; the complete response is then parsed strictly before any calls execute. Plain text is never a successful stop, even at EOS. One protocol repair is allowed within the existing deadline/turn budget. The guard is not grammar-constrained decoding and does not verify task completion or tool-argument semantics.

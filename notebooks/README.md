# Qwen3.5-4B base-model trial

Start with kaggle_pipeline.ipynb: the same three train IDs as the previous run, reference transcripts, thinking disabled, 90-second development budget. Set TRAIN_TRANSCRIPTS=False for the second audio run. This downloads the pinned official checkpoint to models/qwen3.5-4b and keeps old weights separate. Cross-turn recurrent cache reuse is disabled; native generation caching stays enabled. No fine-tuning is performed in the trial.

# Kaggle workflows

- `kaggle_train.ipynb`: private train preparation and QLoRA. Default: one-step GPU smoke test, 8 train/2 validation examples. Full epoch requires MAX_STEPS=-1 after the smoke test works.
- `kaggle_pipeline.ipynb`: train evaluation. Defaults to reference transcripts and three trial development IDs, 90 seconds/16 decisions. For frozen evaluation, set IDS_FILE to prepared manifest.json and ADAPTER_PATH to the adapter. Compare base and adapter on the same IDs, then switch TRAIN_TRANSCRIPTS=False for audio.
- `kaggle_public_test.ipynb`: all public audio, 27 seconds/8 decisions, complete submission validation and manual export. No test scoring/tuning or automatic submission.

Attach authorised private dataset amnpham/zalochallenge. Setup uses network to clone source/install/download public models; inference is local. No Kaggle token is required in notebooks. Each run uses a fresh output directory and one ToolEnv per sample. Keep generated data, logs, training checkpoints and notebook outputs private.

The only policy is prompt_assets/system.txt. There are no prompt modes or variants. Optional adapters need requirements-train.txt and lifemate_policy.json emitted by train_adapter.py; they are merged locally into the base. Runtime reports record adapter path and policy hash. A one-step adapter is not a trained solution.

Use a clean checkout or fresh Kaggle session after the reset. REVISION can pin the commit. Model load time is recorded separately from sample timing. Full step logging is for diagnosis, not speed measurements. Larger development budgets do not meet the final 30-second constraint. Read ../docs/training.md before interpreting results.

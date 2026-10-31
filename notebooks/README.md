# Qwen3.5-4B strict tool-output notebooks

The train-evaluation and public-inference notebooks pin inference source to `4f7def6e1de13860f61d87b66d416fceabde62a4`. This includes the catalog-backed installed-app description dictionary and early output guard. Only consecutive native tool-call blocks or literal `[]` are accepted. Invalid prose is rejected, never treated as successful completion; at most one protocol repair is allowed within the sample budget. This is not grammar-constrained decoding or a guarantee of correct actions.

Upload `kaggle_pipeline.ipynb` to Kaggle for the next train test. Attach the authorised private dataset `amnpham/zalochallenge`, select a T4 GPU, enable internet for setup and Run All in a fresh session. Defaults: three train samples, reference transcripts, 90 seconds/sample, 16 turns, 1024 output tokens/turn, thinking disabled, private step logs on, live printing off. To test real Gipformer ASR, set `TRAIN_TRANSCRIPTS=False` before running inference. No fine-tuning or benchmark is started by downloading the notebook.

# Kaggle workflows

- `kaggle_sft_rl.ipynb`: private feasibility pilot, one SFT step then two interactive RL episodes. Live query printing off. Uses `main` until you pin the printed commit. **Current real-data configuration failed backward with OOM on one T4; no SFT/RL update completed.** Read `docs/sft_rl_pilot_report.md` before running. Reward is diagnostic, not verified task success, and a zero-signal group does not update the model.
- `kaggle_train.ipynb`: private train preparation and QLoRA. Default: one-step GPU smoke test, 8 train/2 validation examples. Full epoch requires MAX_STEPS=-1 after the smoke test works.
- `kaggle_pipeline.ipynb`: train evaluation. Defaults to reference transcripts and three trial development IDs, 90 seconds/16 decisions. For frozen evaluation, set IDS_FILE to prepared manifest.json and ADAPTER_PATH to the adapter. Compare base and adapter on the same IDs, then switch TRAIN_TRANSCRIPTS=False for audio.
- `kaggle_public_test.ipynb`: all public audio, 27 seconds/8 decisions, complete submission validation and manual export. No test scoring/tuning or automatic submission.

Attach authorised private dataset amnpham/zalochallenge. Setup uses network to clone source/install/download public models; inference is local. No Kaggle token is required in notebooks. Each run uses a fresh output directory and one ToolEnv per sample. Keep generated data, logs, training checkpoints and notebook outputs private.

The only policy is prompt_assets/system.txt. There are no prompt modes or variants. Optional adapters need requirements-train.txt and lifemate_policy.json emitted by train_adapter.py; they are merged locally into the base. Runtime reports record adapter path and policy hash. A one-step adapter is not a trained solution.

Use a clean checkout or fresh Kaggle session after the reset. REVISION can pin the commit. Model load time is recorded separately from sample timing. Full step logging is for diagnosis, not speed measurements. Larger development budgets do not meet the final 30-second constraint. Read ../docs/training.md before interpreting results.

# Top-8 Caller SFT run — 31 October 2026

The user selected a 4,096-token cap and explicitly excluding/reporting 266 long training targets. The run targets one epoch over 3,938 next-action/stop examples, with 540 teacher-forced validation-loss examples. No Planner or RL is trained.

## Configuration

- Qwen3.5-4B pinned base revision from configs/baseline.json; frozen NF4 base, FP16 compute and frozen vocabulary matrices, text-attention LoRA rank 16 / alpha 32, 3,145,728 trainable parameters.
- Top 8 BM25 seeds, dependency expansion, train-only target-schema repair. Validation never receives gold schema repair. The 107 missing-schema validation targets and 17 additional overlength covered targets do not contribute to validation loss; task evaluation must later include all validation requests with real retrieval.
- Batch 1, accumulation 8, 1 epoch, 493 optimizer steps, LR 5e-5, 25 warmup steps, checkpoint every 10 updates, validation loss every 100 updates. Keep two resumable checkpoints.
- A session saves and stops after 8 hours at an optimizer-step boundary, with at least two steps per session. The timer includes in-loop validation, but model/data setup precedes it. Platform job cap is 12 hours, leaving export/setup margin. A time-budget stop is a partial epoch.
- PEFT 0.20.0 and bitsandbytes 0.50.2 are pinned to the tested environment. Source is frozen at `48fd0581008e43e4063c353bb4c6fc4af07aa39e`.

## Verified checkpoint/resume

Private preflight: https://www.kaggle.com/code/amnpham/lifemate-caller-sft-k8-checkpoint-preflight . It used the entire retained dataset and the full 493-step LR schedule, stopping intentionally after two steps and then resuming for two more. These steps count toward the epoch; this is not restarting training from a separate smoke adapter.

| Stage | Start | End | Changed adapter tensors | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: |
| Initial session | 0 | 2 | 32 | 7.82 GiB |
| Resume test | 2 | 4 | 64 | 7.93 GiB |

Both stages retained exactly 3,938 train / 540 validation-loss examples, with no sample-size truncation to the old eight-example smoke set. Trainer state reports max_steps=493 and epoch about 0.00813 at step 4. The resume verifier requires matching model revision/config, source/data hashes, training settings and package versions, plus adapter, optimizer, scheduler, scaler, trainer and RNG state files. This verifies the resume path and continued weight updates; it is not an independent bit-for-bit comparison against an uninterrupted run.

## Long session started

Private training job: https://www.kaggle.com/code/amnpham/lifemate-caller-sft-top8 . It attaches the private preflight output and resumes checkpoint-4. The last checked platform status was RUNNING. The full epoch is **not yet complete**, and no full-training accuracy/F1 is available.

The job writes full_sft/checkpoints/progress.json, periodic checkpoint-N directories, training_report.json, skipped_length.json, validation_loss.json and adapter/. A checkpointed_time_budget result requires another session; it must not be presented as a completed epoch. A complete session report includes epoch_completed and the global step. No automatic future-session launch is configured; continue from the latest complete checkpoint until the epoch target is reached. Loading adapter weights alone would reset optimizer/scheduler/RNG and is not equivalent to resume.

Use `notebooks/kaggle_caller_sft.ipynb`, pinned to the frozen source, for manual reproduction or continuation. Attach the previous private notebook output and set RESUME_CHECKPOINT to its complete checkpoint directory. Preserve the training source, prepared data and settings. Do not overwrite a completed output directory or relax identity checks to force a resume.

All 68 CPU tests passed. New coverage includes step-boundary saving and rejection of mismatched/incomplete checkpoint state. Preflight GPU training and resume both succeeded. See [the input/target/loss diagram](caller_sft_flow.md) for how examples and masked causal loss are constructed. Training outputs contain private train-derived information and stay private; only source, notebooks and aggregate reports are pushed to GitHub.

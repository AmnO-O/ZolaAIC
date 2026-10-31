# SFT / RL feasibility report — 31 October 2026

## Result

Implementation is available; real-data GPU training is **not yet feasible on the tested single T4 configuration**. Three bounded private Kaggle attempts reached the failures below. No SFT optimizer step completed, no SFT adapter was produced, and the notebook correctly did not start RL. There is no accuracy improvement to report. No further GPU job was launched after attempt 3.

| Attempt | Source | Result |
| --- | --- | --- |
| 1 | `b79b5cc` | Model loaded in NF4; pinned Transformers rejected obsolete `warmup_ratio` before training. |
| 2 | `1a9c19b` | API fixed; assistant-position sparse loss; backward OOM requesting 1.95 GiB on T4 with 14.56 GiB available capacity. No optimizer update. |
| 3 | `d883999` | Frozen vocabulary matrices explicitly FP16, expandable allocator enabled; backward OOM requesting 3.04 GiB. No optimizer update. |

Private job: https://www.kaggle.com/code/amnpham/lifemate-sft-rl-feasibility . The job is a feasibility check, not a leaderboard submission or benchmark. Outputs contain private train-derived data and must stay private.

## Implemented

- Existing causal, train-only QLoRA preparation/trainer reused and corrected for Transformers 5.16.1. Input/future observation separation and assistant-only loss remain enforced.
- A native Qwen3.5 multi-turn rollout actor uses actual ToolEnv observations. Exactly one ToolEnv is created per training episode. Initial state hashes must agree across a sample's group, without reading/copying environment files.
- A bounded group-relative on-policy training pilot loads an SFT adapter, samples two episodes, computes standardized rewards and makes one policy-gradient update per nonconstant group. It has no replay or reference-model KL and must not be scaled unattended.
- Diagnostic reward checks typed tool arguments, unmatched/duplicate writes, failed writes, protocol/validation failures and clean stopping. Efficiency reward applies only to clean exact call matches. This is not an official state-success verifier.
- Gold calls remain scorer-side; future results/final state never enter actor messages. RL reads only SFT-manifest train IDs; validation is excluded.
- `notebooks/kaggle_sft_rl.ipynb` defaults to one SFT step followed by one RL group. Live query logging is off. Failure at SFT prevents RL from running.

## Data and validation

Local preparation retained 1,332 train samples (4,204 targets) and 213 validation samples (664 targets); no trajectories were rejected. Previously inspected IDs through train_0084 and their normalized transcript groups are excluded from validation. This is conservative separation, not proof that paraphrase templates cannot cross partitions.

CPU tests verify reward failure cases, label separation, one-session behavior, group normalization, policy-gradient direction, frozen-matrix precision, and sparse assistant loss/log-prob alignment and backward on a tiny real Qwen3.5 architecture. They do not establish full-checkpoint CUDA compatibility or accuracy. Long examples are not silently truncated to make the GPU smoke pass.

## Next gate

Test the same bounded SFT smoke on a GPU with more VRAM, or implement and validate a memory-offload strategy without removing task context. A larger GPU has not yet been tested and is not guaranteed to solve every issue. Before a long RL run, add a validated task/state reward and SFT-reference KL control; the current exact-call proxy can penalize valid alternate trajectories.

Once SFT actually trains, compare frozen base versus SFT on the manifest validation IDs with identical settings. Only then compare a bounded RL checkpoint against SFT. Keep inference greedy/offline and measure the official 30-second budget separately from these development training experiments.

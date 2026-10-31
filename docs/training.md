# Causal tool-use learning

This replaces prompt-variant experiments with one policy and a measured base-versus-adapter comparison. Model quality is not fixed by repository cleanup.

## Preparation

prepare_sft.py accepts only the train catalog/JSONL. It does not read final_state, environment files or call ToolEnv. Each target is one gold action, with only previous observations in its input. IDs and amounts from the current or future tool result cannot appear through future-result leakage. Source user/context/history facts remain allowed. Empty trajectories teach []; terminal clarification does not get an extra stop target. Unpaired, unavailable or malformed trajectories are reported rather than repaired.

The split groups casefolded transcripts with numeric runs masked, using a fixed hash seed. The known development IDs train_0001..train_0061 and exact text groups are excluded from validation. This is conservative bookkeeping, not proof of complete template separation; related paraphrases may cross groups. If other IDs were already inspected, add them to configs/development_ids.json before preparing data. All steps of a sample stay in one partition. The manifest records IDs, source/catalog/policy hashes and rejected trajectories. No priors derived from validation are injected into inference.

Assistant-only labels mask context, schemas, user text and observed results. The actual template must demonstrate an exact text-prefix match; template mismatch fails. Prefix and completion are encoded separately at the generation boundary, because BPE can merge trailing prompt whitespace with the first completion token. Long examples are recorded and skipped, never silently truncated. Training defaults to max length 8192, while inference supports 16384. Inspect retained/skipped distributions before a full run.

## Training

The script uses local pinned Qwen weights with NF4 4-bit QLoRA, FP16 compute for T4, rank 16 attention adapters, gradient checkpointing, batch size 1 and accumulation 8. Seeds and deterministic algorithms are set. The notebook starts with one optimizer step on at most 8 train and 2 validation examples to test GPU compatibility, not produce a competitive model. A full run is one initial epoch; this is an experiment, not a recommended optimal hyperparameter set. OOM/unsupported kernels must be reported rather than silently switching precision or devices.

Adapters save policy metadata. Inference loads base and adapter locally, verifies policy hash and merges before execution. No teacher/second model is used at inference. Base and merged parameter counts are checked. Training dependencies and resolved versions are recorded; full reproducibility requires freezing that environment and testing reruns.

QLoRA implementation follows the official PEFT quantization workflow: https://huggingface.co/docs/peft/developer_guides/quantization . These references describe APIs, not evidence of performance on this task.

## Evaluation

Use manifest validation IDs for both frozen base and adapter, with the same policy, schemas, decoding, logging, deadlines and hardware. Start with train reference text; then repeat with ASR audio. Compare per-sample diagnostic tool F1, exact complete call matches, errors, call count, premature stops and latency. Validation loss is not task accuracy. Do not repeatedly edit prompts after inspecting validation failures and still call it untouched holdout.

Public/private test must not be labelled or used for training/tuning. Exported train examples and all outputs remain private. The current diagnostic matcher and optional state-gated proxy are not a verified official scorer. No GPU accuracy, runtime or compliance claim is supported by CPU tests alone.

Old adapters/data prepared under the JSON prompt must be regenerated; policy hashes differ. The official template emits XML, with enable_thinking=False in both inference and supervised encoding.

## SFT then interactive RL pilot

Use `notebooks/kaggle_sft_rl.ipynb` in a private Kaggle session. Default: one SFT optimizer step, then one RL group of two episodes. This tests training compatibility and memory; it does not produce a competitive checkpoint. Inference notebooks and their pinned revisions remain unchanged. Live per-query logging is off; private episode traces are saved to disk.

The prepared split now excludes previously inspected IDs through train_0084 from validation, including their exact normalized transcript groups. Local preparation on 31 October 2026 retained 1,332 train samples / 4,204 targets and 213 validation samples / 664 targets, with no rejected trajectories. These counts concern data preparation, not model accuracy.

`scripts/train_rl.py` requires an SFT adapter and the same source/catalog/policy manifest. Only manifest train IDs are eligible. Gold calls go to the scorer; initial actor messages contain context, history and the latest train transcript. Gold future tool results, gold calls and final state are never passed to the actor. Reference transcripts isolate tool policy first; real ASR evaluation is still required afterward.

Every sampled episode creates exactly one ToolEnv via the existing organizer API. There is no reset implemented by reading or copying environment files. All actual calls remain in the prediction, including failed attempts. Initial hashes must agree across a sample's rollout group; otherwise training aborts. Training repeats episodes from the same train sample, unlike one-episode inference. This is development-only and does not change the submission runner.

The pilot uses Transformers + PEFT directly, without installing ToolRL's incompatible old dependencies. It is a **single-update on-policy group-relative policy-gradient variant**: population-standardized episode rewards, average over generated assistant tokens per episode, one optimizer update per group, no replay/PPO epochs and KL coefficient zero. It is not the original ToolRL training implementation. Identical rewards skip the update and increment `zero_signal_groups`; a completed job with zero updates is not evidence that RL learned. Lack of KL regularization makes this unsuitable for an unattended long run; add and validate an SFT-reference KL objective before scaling.

Sampling uses temperature 1, top-p 1, top-k 0 with seed 42; inference stays greedy. Native calls and literal `[]` use the same parser and execution controller. The training sampler captures exact prompt/response token IDs, including EOS and rejected generated output; only response tokens receive policy gradient. Observations are recomputed from ToolEnv, never generated by the model. Dropout is disabled for sampling and scoring. Long prompts fail rather than being silently truncated. Quantized Qwen3.5 gradients, memory use and generation/scoring consistency still require the GPU pilot; CPU tests do not establish compatibility.

Reward v1 is explicitly diagnostic: `2 * clean_exact_call_proxy + 0.5 * tool_F1 - penalties + efficiency`. Wrong or duplicate writes cost 1 each (capped at 2); failed writes cost 0.5 each (capped at 2); protocol and validation failures cost 0.1 each (each capped at 2). Efficiency is `0.1/(1+decisions)` only for clean exact matches. Empty gold correctly rewards a clean `[]`; an empty answer for a nonempty task receives no completion bonus. Exact call matching does **not** prove state/task success, and valid alternate trajectories can be unfairly penalized. Do not optimize this proxy for a long run or report it as official accuracy. A validated organizer-supported state/task verifier is needed before that claim; this implementation does not inspect `final_state` or environment files.

After GPU feasibility, run a bounded SFT epoch and compare frozen base/SFT checkpoints on the manifest validation IDs. Only if SFT is usable, test a short RL pilot against that SFT checkpoint on the same untouched validation partition. Retain SFT if RL does not improve task outcomes. Never promote a checkpoint based on training reward or three familiar examples.

Reference interfaces: [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), and [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer). TRL documents custom rollout/reward interfaces, but this bounded pilot does not depend on TRL or claim API compatibility with its full trainer.

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

Training is paused while evaluating the unadapted Qwen3.5-4B base. Old adapters/data prepared under the JSON prompt must be regenerated; policy hashes differ. The official template now emits XML, with enable_thinking=False in both inference and supervised encoding.

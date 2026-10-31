# Compact Caller SFT: successful one-step GPU check

Date: 31 October 2026. Private Kaggle job: https://www.kaggle.com/code/amnpham/lifemate-compact-caller-sft . Tested source `dc19faa82e7ff997821a8d2e0ffecda8fc9d2bb6`. The job completed successfully. This establishes feasibility for the tested short examples, **not improved tool-use accuracy or readiness to submit**. No long training run, Planner training, RL or F1 evaluation was started.

## Data contract

Prepare with `scripts/prepare_sft.py --retrieval-top-k 8`. Each causal target receives the original user text/history/context, previous observed results and BM25-selected native schemas, with catalog-derived read dependencies. Retrieval queries use original user text plus the latest observed tool result. Future results and target arguments never enter input.

If the target tool's schema is absent, **train only** adds that schema plus dependencies. This is supervised schema selection, not inference-time routing performance. Validation never receives this gold repair: missing-target examples are recorded and excluded from validation loss. All validation samples must still be included in later live pipeline evaluation, with genuine retrieval and no gold repair.

The existing grouped split and development-ID exclusion are unchanged. Preparation retained 1,332 train samples / 4,204 targets. 728 train targets needed schema repair. There are 213 validation samples; 107 of the original 664 validation targets were excluded from loss because their tools were missing, leaving 557 targets. No raw trajectory was rejected. Validation loss on this covered subset is biased relative to end-to-end pipeline quality and cannot be reported as task accuracy.

## Training configuration

- Pinned Qwen3.5-4B base from configs/baseline.json, one NVIDIA T4, NF4 double-quantized weights, FP16 compute, frozen vocabulary matrices FP16, norms/adapters retain training dtype.
- LoRA rank 16, alpha 32, dropout 0, text q_proj/k_proj/v_proj/o_proj only: 32 projection modules, 3,145,728 trainable parameters. Vision, embeddings, output head, base weights, MLP and hybrid in_proj_* projections are not trained.
- Batch 1, accumulation 8, learning rate 1e-4, no smoke warmup, non-reentrant gradient checkpointing, sparse assistant-position vocabulary logits, deterministic algorithms and fixed seed 42. Explicit smoke GradScaler initial scale 256.
- One optimizer step using the first 8 train examples and 2 covered validation examples. Length cap 4,096; actual train lengths 1,760–3,247 tokens; validation lengths 2,817 and 3,106. All 8/2 examples retained, with no truncation or length skips. This is not a full-epoch training run on the 4,204 targets.

## Verified result

| Measurement | Result |
| --- | ---: |
| Global optimizer step | 1 |
| Changed adapter tensors | 32 |
| Maximum absolute adapter-weight change | ~0.0001 |
| Trainable parameters | 3,145,728 |
| Peak PyTorch allocated CUDA memory | 6.41 GiB |
| Trainer runtime, including in-loop validation | 128.52 s |
| Training loss | 1.5299 |
| Validation loss on 2 covered examples | 0.05845 |

Peak allocated memory is not total device usage including allocator reservation and CUDA/driver allocations. This configuration fits the observed short batches on T4; it is not proof that every 4K example or the full dataset fits. The earlier OOM runs used different input lengths/configurations, so this is not an isolated causal attribution to token reduction alone.

The adapter was downloaded privately and independently checked on CPU: 64 tensors, all finite, 32 nonzero LoRA-B tensors, 3,145,728 parameters, 12,592,424-byte safetensors file. Combined with the before/after snapshot in the training report, this verifies a real weight update rather than just an incremented step counter. No claim of improved F1 follows from these measurements.

## Artifacts and next gate

`notebooks/kaggle_compact_caller_sft.ipynb` reproduces the one-step check with live query logging off. It pins the tested source revision. Private local results are in `artifacts/kaggle_compact_sft_results/training/4ec68578b6c7/training/`: training_report.json, skipped_length.json, validation_loss.json and adapter/. They are ignored by Git. Do not publish train-derived data or the private checkpoint inadvertently.

Before a longer run, audit retained/skipped length distributions and expected cost across the full data. Before claiming model improvement, evaluate frozen base and adapter on the same validation samples with the **same compact query/schema contract**, without gold target-schema repair. The default full-schema inference notebook has not yet been adapted to this contract. Eventually repeat with real ASR and the official runtime constraints. The one-step adapter should not replace the default inference model.

All 66 CPU tests passed, including causal data separation, train-only schema repair, missing validation targets, loss masks, tiny-model gradients and existing execution checks. Notebook code cells were syntax-checked. No external API or environment files were used to label competition test data.

## Top-10 follow-up

At the user's request, the same source, data split, seed and QLoRA settings were tested again with 10 retrieval seeds instead of 8. Private job: https://www.kaggle.com/code/amnpham/lifemate-compact-k10-caller-sft . The same 8 train / 2 validation examples, messages and targets were verified; K8 schema sets are subsets of their K10 counterparts. Final schema counts can exceed 10 because dependencies, clarification and train-only target repair are included.

| Metric | Top 8 | Top 10 |
| --- | ---: | ---: |
| Train targets requiring gold schema repair, out of 4,204 | 728 | 544 |
| Validation targets missing schema, out of 664 | 107 | 90 |
| Validation targets retained before length filtering | 557 | 574 |
| Train smoke sequence length | 1,760–3,247 | 2,101–3,563 |
| Actual train smoke schemas per example | 10–13 | 11–16 |
| Peak PyTorch allocated GPU memory | 6.41 GiB | 7.00 GiB |
| Trainer runtime including in-loop validation | 128.52 s | 144.98 s |
| Actual optimizer steps / changed adapter tensors | 1 / 32 | 1 / 32 |

K10 completed with all 8/2 examples retained and no truncation. Train loss was 1.5424, validation loss 0.05984 on only two retrieval-covered examples; neither value establishes task accuracy. Adapter delta was approximately 0.0001. The independently downloaded K10 adapter contains 64 finite tensors, including 32 nonzero LoRA-B tensors, and remains private in `artifacts/kaggle_compact_k10_sft_results/training/50baebe2d4bc/training/adapter/`. No long training or F1 evaluation was run.

The full K10 tokenization audit had no template/encoding errors. Of 4,204 train targets, 407 exceed the current 4,096-token cap (133 are stop targets); median length 2,925.5, p95 4,468, maximum 6,262. Of the 574 schema-covered validation targets, 35 exceed the cap; maximum 4,993. A full run at the current cap would retain 3,797 train and 539 validation targets after length filtering. This is a separate limitation from the 90 validation targets whose tools were missed. Long examples must be reported rather than silently truncated, and their removal can bias the task mix.

The notebook now defaults to `RETRIEVAL_TOP_K = 10`, still pinned to the tested source. Set it to 8 to reproduce the original check. Top 10 reduces missing schemas but does not guarantee complete retrieval or better F1. Address the remaining missing schemas and long targets before treating this as a full training recipe. Batch size remains **1 per GPU, gradient accumulation 8, effective batch size 8**; an example is one next-action/stop prediction, not an entire request trajectory.

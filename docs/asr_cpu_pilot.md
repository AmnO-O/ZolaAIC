# Three-model local CPU ASR pilot

10 random train latest-request audio files, seed 42, eligible duration <=25s.
Only metadata and audio were accessed; no ToolEnv or environment state was used.
Inference ran locally offline, 4 CPU threads, greedy deterministic decoding.
References were used only for scoring and the explicitly oracle candidate metric.
This pilot is not downstream agent accuracy or a protected held-out evaluation.

| Model / selector | Corpus WER | Mean inference seconds/audio |
|---|---:|---:|
| Gipformer 1.5 INT8 | 15.95% | 0.23 |
| PhoWhisper-small FP32 | 31.03% | 5.36 |
| PhoWhisper-base FP32 | 46.55% | 2.17 |
| Simple aligned majority vote | 19.83% | — |
| Whole-candidate consensus medoid | 20.26% | — |
| Oracle selection using reference (evaluation only) | 15.52% | — |

Serial model inference totals 7.76 seconds/audio on average, excluding model
loading, downloads and fusion. These laptop CPU timings are not A30 estimates.
The two PhoWhisper models share a family and may have correlated errors; these
numbers do not establish a general failure of ASR ensembles. The small oracle
gain is one fewer word error over this 10-audio sample, so these candidates show
little complementary value under this metric.

WER uses Unicode NFC, lowercasing, punctuation removal and whitespace tokenization.
Digit strings are not equated with spoken number words, so numeric formatting can
inflate errors without a semantic difference. No claim of slot or task accuracy
is made. Voting aligns candidates against the first nonempty Gipformer hypothesis
with deterministic majority decisions and anchor tie-breaking. It is inspired by
ROVER, not an implementation of NIST's reference algorithm. Medoid chooses the
candidate with minimum summed pairwise normalized edit distance. Neither selector
reads reference text. Fusion is not yet enabled in the main inference pipeline.

Manifest: configs/asr_cpu_models.json pins all three public checkpoints. The
benchmark requires torch, transformers, numpy, scipy, soundfile, sherpa-onnx and
huggingface-hub. The measured environment used torch 2.14.0+cpu and transformers
5.19.0; production inference dependencies were not modified. Optional ASR
dependencies were installed into a private artifact directory, not system packages.

```sh
python experiments/asr_ensemble_cpu.py --train-dir /path/to/train \
  --manifest configs/asr_cpu_models.json --output artifacts/asr-new-run \
  --samples 10 --threads 4
```

Download pinned public models before offline inference. Output contains ids.json,
candidates.jsonl (each model's original prediction/timing), progress.json and
report.json (references, predictions, fusion, parameters and errors). Keep these
competition-derived outputs private. Existing output directories are not overwritten.
Whisper EOS is observed during decoding because this runtime removes it from the
returned cleaned sequence. A token-limit truncation is rejected rather than scored
as a complete transcript. A failed preflight preceded the successful run after
this EOS handling was corrected.

Model references:
- https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt
- https://huggingface.co/vinai/PhoWhisper-small
- https://huggingface.co/vinai/PhoWhisper-base
Fusion reference: https://www.nist.gov/publications/post-processing-system-yield-reduced-word-error-rates-recognizer-output-voting-error

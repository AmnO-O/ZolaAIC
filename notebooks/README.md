# Kaggle notebooks

- kaggle_pipeline.ipynb: main App Planner/Caller train evaluation. One shared beyoru
  model, Gipformer optional via TRAIN_TRANSCRIPTS=False. Default 3 train samples,
  90 seconds, 8 rounds, 384 Caller tokens, live logging off.
- kaggle_public_test.ipynb: main App Planner/Caller on all public audio, 27-second
  internal budget, full output validation and manual submission export.
- kaggle_caller_sft.ipynb: retained frozen Qwen3.5 Caller SFT, pinned revision
  48fd058. This is a separate training workflow; its adapter cannot load into
  beyoru. Planner distillation has not been implemented or run.

Inference notebooks pin source commit 5977801 and record the resolved checkout.
Update REVISION deliberately for repeatability. Keep notebook outputs and datasets private. Internet is
needed for setup only; inference uses local models. Attach amnpham/zalochallenge.
No notebook reads environment files directly. No automatic leaderboard submission.

Archived comparison runners are in ../experiments. Removed RL/obsolete smoke
notebooks are recoverable from Git history. Checkpoints and local artifacts remain.

Planner kind is read/write/mixed/meta for calls, none for done (apps=[]). It is
passed to Caller as unverified guidance and does not filter schemas or authorize
writes. Planner generation ceiling is 384 tokens, configurable via
LIFEMATE_PLANNER_MAX_NEW_TOKENS (notebooks: PLANNER_MAX_NEW_TOKENS). The existing
20-second per-plan and overall sample deadlines still apply.

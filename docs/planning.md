# Planning development

The model reads the request, emits a brief tentative action plan with its first
native tool calls, observes results, then revises only the remaining work when
needed. There is no separate planner LLM, no automatic tool lookup, and no
long reasoning transcript. The clean prompt remains an explicit comparison.

## Run

The train/debug notebook defaults to planning, a 90-second whole-sample budget,
16 model decisions and detailed logging. The public notebook and CLI remain
clean/27 seconds/8 decisions. For an A/B comparison use identical train IDs,
budget, maximum decisions, model assets, determinism and logging in both runs.
Neither this experiment nor a longer budget establishes competition compliance.

```
python scripts/infer.py --split-dir /path/to/train --backend lifemate.backend:create_backend --ids configs/ids_25.json --prompt-variant planning --budget-seconds 90 --max-turns 16 --step-log result/planning_steps.jsonl --output result/planning.jsonl
```

Use a new output path for every run. Change only `--prompt-variant clean` to
compare the direct caller under the same harness and budget. Historical clean
scores are not directly comparable: execution guards and diagnostics also changed.
The LLM context stays capped at 16,384 tokens and greedy decoding is unchanged.

## Output protocol

The first actionable response includes `<plan>...</plan>` followed by native
`<tool_call>` blocks. A plan contains up to 600 characters describing 1-3 short
action steps, dependencies and branches. Direct actions do not require a read.
Later responses omit the plan unless its remaining work needs revision.
A stop must be explicit `[]`, optionally preceded by a final plan update.
Plan-only, malformed, or oversized responses execute nothing and get one
format correction; a second format error stops with `protocol_error`.

The parser removes the plan before parsing tool calls, so tool-like text inside
a plan cannot execute. Accepted plans remain assistant content in conversation
history; newer revisions supersede earlier plans. They never enter user evidence
or become organiser calls. Only actual ToolEnv calls/hash enter submission output.

## Execution and stopping

* `succeeded`: `ok=true` and no explicit `done=false`. Only these observations
  establish evidence. This is an execution indicator, not proof of user intent.
* `failed`: explicit `ok=false`. A corrected action may follow; unrelated
  successes do not erase failures for other targets.
* `uncertain`: missing/ambiguous status, `ok=true, done=false`, or a raised call.
  An uncertain write/unknown mutation stops its batch and the sample with
  `uncertain_outcome`. No generic safe verification/reconciliation is implemented.
  An uncertain read may be corrected by the model within the same budget.
* Identical writes already confirmed successful are rejected rather than replayed.
  Unknown tool kinds are treated conservatively as potential mutations.
* Three equivalent call/result batches without new execution observations stop
  with `no_progress`; changing plan prose does not reset this guard.
* Explicit `[]` stops with `finished`, `validation_failed`, or `execution_failed`
  according to observed state. Clarification stops the batch. Deadlines and the
  configurable decision limit remain enforced. No automatic sample retries occur.

`finished` means the model stopped without a recorded unresolved execution error;
it does not certify that the plan, requested branch or final app state is correct.
A non-empty narrative plan is not a machine-verifiable list of remaining tasks.
The model is responsible for branch polarity, preserving unconditional actions,
revising its plan and recognizing whether more work is needed.

## Diagnostics and limits

`planning.updated` and `planning.protocol_error` events are printed/saved with
step logging. The sample trace includes plan revisions, decision count,
validation rejections and an execution ledger independent of plan text.
Timing reports distinguish elapsed thresholds from actual deadline timeouts;
rejection turns group multiple rejected calls from the same model decision.

The harness does not execute plans or rewrite arguments. It cannot prove
semantic task completion, interpret every app's pending/error conventions,
or reconcile uncertain writes. Identical-write suppression can reject an
intentionally repeated identical action; repeated-read detection can stop valid
polling workflows. Such cases need explicit protocol/schema support, not
sample-specific keyword rules. Plan revisions and extra tokens may hurt speed
or accuracy. Evaluate with train data and a fresh hold-out; do not label test data.

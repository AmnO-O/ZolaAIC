# App-level Planner pilot

Planner outputs `{"next":"call","goal":"immediate task","apps":["calendar"]}`.
For a valid plan Python supplies all sample-permitted schemas in the selected
namespaces, plus `sys.ask_missing_info` if permitted. There is no top-k pruning
or automatic expansion to other apps in this branch. Caller selects functions
and arguments using the goal, original request, context and observed history.
After observations Planner chooses a new step. `done` requires an empty apps list.
Invalid plans fall back to original-query BM25 without forwarding invalid guidance.

Compare against the existing observation-driven Planner+BM25, not a static
initial plan. Both use the same model, alternate arm order per sample, fresh
ToolEnv per independent sample episode, 8 controller rounds, greedy generation,
192 planner tokens/20 seconds per planning call and 384 Caller tokens. The
90-second development sample budget includes all planning and execution.
Live query logging is off; completed decisions and observations remain in the
private output artifact for diagnosis. No environment files are read by our code.

Model: `beyoru/Qwen3-4B-I-1209`, revision
`2ee10771b4488c9979642afb0c27f2a114f95cb2`. No adapter or ASR in this pilot.
IDs: train_0001 through train_0009, then train_0011, the same ten familiar
development samples as previous pilots. These are not a holdout.

```sh
python scripts/compare_adaptive_caller.py --train-dir /path/to/train \
  --model /path/to/beyoru --ids /path/to/ids.json --output /path/to/fresh/output \
  --seconds 90 --top-k 8 --methods adaptive_plan app_plan
```

Compare diagnostic F1, latency, actual termination reasons, planner errors,
initial and cumulative gold-name schema recall. Gold is evaluator-only.
Cumulative recall means a schema appeared at least once; it does not establish
that it was available at the required moment. `finished` is not verified task
success. The official deadline remains 30 seconds.

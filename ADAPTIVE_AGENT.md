# Adaptive agent experiment

Implemented 28 October 2026. Local tests pass; GPU latency and task-score improvements have not been established. The previous controller remains available with `--agent-mode legacy`.

## Running

```bash
python scripts/infer.py --split-dir /path/to/train --backend lifemate.backend:create_backend \
  --agent-mode adaptive --planning off --tool-selection shortlist \
  --budget-seconds 27 --limit 10 --output result/submission.jsonl \
  --step-log result/steps.jsonl --verbose-steps
```

Set `LIFEMATE_PROMPT_MODE=system` for the initial comparison. The notebook chooses this mode, `AGENT_MODE="adaptive"`, `TOOL_SELECTION="shortlist"`, and `PLANNING="off"`. These are experiment settings, not a validated final competition configuration. `--agent-mode` defaults to legacy on the command line for backward compatibility.

## Tool selection and recovery

The router groups permitted tools by app using the supplied catalog. It uses accent-normalized lexical matches against tool names, descriptions, and app descriptions from all user turns. Strong matches shortlist up to four initial apps; `x-ref` dependencies can add more. Weak matches keep the full catalog. Schemas are preserved without dropping argument constraints.

The executor sees the selected full schemas and a compact inventory of omitted capabilities. It can request the full permitted catalog once with `agent.expand_tools`. A proposed omitted call causes expansion and feedback before any execution. Before accepting unsupported/no-action completion from a reduced catalog, the controller expands it. Previously unseen app names are supported through catalog metadata; the router has no fixed app whitelist beyond the sample permissions.

This is a conservative lexical baseline, not a proven semantic retriever. It can miss synonyms, overselect irrelevant tools, or fall back to all tools. Compare routing coverage and task success against `TOOL_SELECTION="all"` before adopting it.

## Bounded planning

Set `PLANNING="auto"` to plan requests containing simple conditional/sequential markers; `"always"` attempts planning whenever the minimum remaining budget permits. The marker rule is a heuristic, not a reliable complexity classifier. `"off"` skips the planning call.

The same Qwen instance receives the conversation and a compact app/function inventory. It must return JSON containing `apps`, up to six requested `actions` (goal, target tool, conditional flag), and unresolved `unknown` information. Invalid apps/functions, malformed/truncated JSON, or timeout cause direct-execution fallback. No planner tool calls execute.

The planner gets at most `PLANNER_BUDGET_SECONDS` (default 4 seconds), capped at 20% of remaining sample time, and at most `PLANNER_MAX_TOKENS` (default 128). It is skipped if less than 12 seconds remain. These conservative development limits may produce frequent fallbacks at the observed T4 speed; measure before increasing them. Its time is charged to the same sample budget. Planner KV is cleared before execution because it uses a different prompt.

A valid plan selects app groups and provides provisional action tracking. A write to an operation outside the plan is rejected before execution. The model may revise its plan once with `agent.revise_plan`; revision validates the same schema and expands permitted tools. A plan is not independently verified user intent, and revision is not proof of correctness.

## Working memory and completion

Per-sample memory retains actual tool observations, their source app, observed identifier fields, and provisional action status. Historical tool results can ground identifiers without replaying historical actions. For a top-level argument with `x-ref`, an identifier must occur in observed results from that reference's app or be explicitly present in a user message. These checks are deliberately narrow: they do not establish recipient identity, amount correctness, arbitrary nested references, or complete semantic validity.

Each actual tool response is appended with its observation number and pending plan-action indexes. Successful calls (`ok=true`, with `done` not false) provide structural support for matching planned tools, counted separately for repeated actions. This is not a guarantee that their arguments fulfilled the natural-language request. All original tool results remain available; there is no lossy result summarizer or cross-sample memory.

The model must end through `agent.finish` with status `done`, `unsupported`, or `condition_false`. Empty calls/prose receive bounded correction rather than being treated as completion. With a plan, done is rejected while target operations lack successful-call evidence. A false-condition finish must cite actual successful read observations and cannot skip pending unconditional planned actions. The controller does not automatically prove the natural-language condition from those observations. Without a plan, completion relies more heavily on the executor; no independent semantic judge is claimed.

`agent.finish`, `agent.expand_tools`, and `agent.revise_plan` are local protocol signals. They never call ToolEnv and never appear in the submission. Only actual app calls are exported. `sys.ask_missing_info` remains a real permitted tool and stops the batch after execution.

There are at most 12 decision turns, one full-catalog expansion, one plan revision, and two repairs before the next repair ends the query. Native backend deadlines remain cooperative. No second prediction ToolEnv is created.

## Tracing and experiments

New trace events: `planner.input/output/cache`, `plan.accepted/fallback/skipped/revised`, `tools.selected/expanded`, `memory.updated`, and `controller.feedback`. `sample.input.runtime` records actual device, dtype, GPU, versions, deterministic mode, and configured attention implementation; this does not identify the actual low-level GPU kernel.

Run paired comparisons on the same train IDs:

1. Legacy/system/full tools versus adaptive/system/full tools with planning off.
2. Adaptive/full tools versus adaptive/shortlist, planning off.
3. Adaptive planning off versus auto under the same whole-sample deadline.
4. Repeat with train reference transcripts to isolate ASR effects.

Keep failures in evaluation. `done` and empty pending lists are control signals, not official accuracy scores. Keep detailed logs private and disable them for timing comparisons. No new GPU benchmark, ASR decoder change, quantization, fine-tuning, RL, vLLM, or Docker work is included in this change.

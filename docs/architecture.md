# Clean runtime architecture

Audio and observed history → Gipformer ASR → general system instructions and
full permitted tool schemas → Qwen native tool calls → allowlist, evidence and
organiser schema validation → the same ToolEnv → observed results → next decision.

An empty decision stops immediately. Clarification stops its batch. Deadlines
and a turn limit bound execution. Failed executions remain visible; the harness never automatically retries writes.
Planning format errors get one bounded correction opportunity, not a tool call.
An uncertain mutation stops the sample; repeated observations terminate without progress.

| Module | Responsibility |
| --- | --- |
| `input_adapter`, `scripts/infer.py` | Permitted metadata/audio, one environment per sample, export actual calls/hash, failure isolation |
| `asr`, `backend`, `prompt_cache` | Local inference, parameter/context limits, sample-scoped KV cache |
| `messages` | Native conversation serialization and call parsing |
| `prompts`, `prompt_assets/` | General system instruction and optional planning overlay via Steerability SystemPrompt |
| `planning` | Parse brief action plans alongside native calls, with explicit stop/protocol checks |
| `execution` | Actual success/failure/uncertainty ledger, duplicate-write and repeated-observation guards |
| `context` | Remove redundant installed-app schema metadata from prompt context |
| `agent` | Bounded validate/execute/observe loop |
| `grounding` | Successful current/historical tool evidence, explicit user data and trusted context |
| `validation` | Evidence and clarification checks; no domain intent interpretation |
| `catalog` | Unique explicit read-tool mentions in schema descriptions, restricted to allowed tools |
| `values` | Pure Unicode, phone and ID parsing |
| `step_log`, evaluation scripts | Private diagnostics; diagnostic F1 is not an official score |
| `scripts/validate_submission.py` | Coverage/structure validation without app state access |

## Removed

* A/B prompt variants and static few-shot examples with lexical selection.
* Handwritten app keywords, tool pruning and recovery expansion.
* Handwritten entity-to-lookup mapping. Ambiguous catalog references have no hint.
* Keyword nudges for notifications or conditional actions after a model stop.
* Forced bill lookup rules inferred from Vietnamese words.
* Automatic rewriting of addresses/event destinations, appended calendar end
  times, and clearing bill filters.

The model owns semantic decisions. Code validates explicit structural/evidence
constraints. Spoken-digit phone parsing remains as deterministic annotation of
user text; assistant prose and failed tool results do not supply trusted phones.
Tool schemas and organiser validation remain authoritative.

## Migration and limitations

Use `PROMPT_MODE="system"`. The public notebook uses `clean` with 27 seconds;
the generic notebook uses `planning` with 90 seconds and 16 decisions for development.
See [planning.md](planning.md) for the protocol and limitations. The old `continue_after_read` and `full_tools` Python
arguments and `lifemate.routing` module are removed. Old modes/variants fail
explicitly rather than silently selecting different experiments. Previous
versions remain in Git history; previous measured F1 is not inherited.

This cleanup improves separation of responsibilities, not proven model accuracy.
The model may still choose wrong branches, stop prematurely, or select incorrect
but grounded values. Evidence IDs are currently scoped to source app, not a
complete typed model of every entity collection. Explicit catalog tool mentions
are hints, not proof of return semantics. Full GPU evaluation, final-state
verification and A30 timing remain necessary. Never inspect environment files;
all app data access stays inside the single ToolEnv.

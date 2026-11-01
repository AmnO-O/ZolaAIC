# App Planner / Caller architecture

The production entry point is scripts/infer.py. It constructs PlannerCaller with
app routing for each sample, sharing one LocalBackend across samples and roles.
The execution loop remains run_agent; Planner never executes tools itself.

Planner sees app IDs/descriptions derived from the sample-permitted catalog,
context, original request and all observed conversation turns. Its strict JSON
contract is `next`, `goal`, `apps`, `kind`. Names must be permitted app namespaces, not
function names. A call requires at least one app; done requires apps=[].

Selection uses exact namespace matching against the permitted schema list. All
functions of selected apps are supplied, plus sys.ask_missing_info if permitted.
No app names or domain dependencies are hardcoded. A syntactically valid but
semantically wrong app plan can still omit necessary tools. On invalid plans,
original-request BM25 is a bounded fallback; there is no extra planning repair.

Caller receives original messages plus a temporary system planner_guidance object.
It does not replace user instructions or observations and is not persisted as
facts. The harness checks evidence from original messages and actual results,
not generated plans. The next Planner call sees actual outcomes, not a stored plan.

The backend handles checkpoint-native JSON tool calls for Qwen3/beyoru and XML
for the historical Qwen3.5 controls. NativeOutputGuard rejects prose early;
full calls must still pass strict parsing and schema/grounding validation.
No arguments are rewritten. Successful writes are not repeated automatically.

Inference uses exactly one ToolEnv per sample. All real calls and final hash come
from that environment. Read/write success, failed/uncertain outcomes and model
stops are distinct. `finished` is not semantic completion verification.

The default model config is configs/baseline.json (beyoru + Gipformer). Historical
Qwen3.5 assets are described in configs/qwen35_training.json. Historical SFT remains
pinned to its original source and does not train this Planner. Experiment runners
live under experiments/ and never feed gold to either inference role.

Planner kind is read/write/mixed/meta for calls, none for done (apps=[]). It is
passed to Caller as unverified guidance and does not filter schemas or authorize
writes. Planner generation ceiling is 384 tokens, configurable via
LIFEMATE_PLANNER_MAX_NEW_TOKENS (notebooks: PLANNER_MAX_NEW_TOKENS). The existing
20-second per-plan and overall sample deadlines still apply.

## Planner input projection

Planner receives five explicit fields: context, available_apps, prior_conversation,
current_request and execution_history. App descriptions appear once in available_apps;
installed_apps/app_descriptions are removed only from the Planner context copy.
The current-request boundary is captured on the first decision and remains fixed
when protocol feedback is appended as a user message. Earlier assistant actions
remain in prior_conversation; current calls are paired with their observations,
including executed/source_type/execution_status from the harness. Rejections and
unobserved proposals are not labelled successful observations. No data is obtained
from environment files and no execution behaviour or Caller input is rewritten.

Goal covers the immediate action, retaining any directly relevant guard. It must
not list speculative later tasks. Independent lookups within the step remain
possible. Previous generated goals are not appended: a rolling last-plan memory
would be a separate experiment, not a verified observation or proof of CoT.

Structural replay checked 319 saved Planner inputs across 100 samples. Mean JSON
user-payload length fell from 3875.85 to 3319.74 characters; these are not token
counts, runtime or accuracy results. GPU evaluation of the new format is pending.
Native chat templates remain unchanged; the official Qwen model card demonstrates
apply_chat_template: https://huggingface.co/Qwen/Qwen3.5-4B . This input organization
is our engineering hypothesis, not a prescribed or proven optimal Qwen format.

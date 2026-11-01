# Main inference flow

1. Load the pinned local beyoru checkpoint and Gipformer once per process.
2. For each sample create exactly one ToolEnv and start the sample deadline.
3. Transcribe current/history audio and retain original context and history.
4. Planner receives permitted app descriptions and the observed conversation;
   it emits `next`, `goal`, `apps`, `kind`. A valid stop uses `next=done`, `apps=[]`.
5. For a valid app plan, select every permitted function schema in those apps.
   Invalid plans fall back to original-query BM25; no invalid goal is forwarded.
6. Caller receives the current goal as unverified guidance, selected schemas,
   original request, context and observations. It emits native calls or `[]`.
7. Validate calls and grounding, execute through the same ToolEnv, append actual
   results, then return to Planner. Only independent calls may be batched.
8. Stop on no remaining work, clarification, uncertain write outcome, loop bound
   or deadline. Export all actual calls and the real final hash; retain diagnostics.

Planner: at most 384 output tokens and 20 seconds per invocation, within the shared
sample budget. Caller notebook ceiling: 384 tokens. Both roles use greedy decoding.
Eight controller rounds by default. Main runner budget 27 seconds; train notebook
uses 90 seconds for development. Cooperative deadlines can overrun native operations.

See README.md for source layout and notebooks. experiments/ contains comparison
scripts and is not used by inference. Frozen Qwen3.5 Caller SFT is a separate
historical training workflow, not a trained app Planner or a beyoru adapter.

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

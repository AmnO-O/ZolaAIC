# Main inference flow

1. Load the pinned local beyoru checkpoint and Gipformer once per process.
2. For each sample create exactly one ToolEnv and start the sample deadline.
3. Transcribe current/history audio and retain original context and history.
4. Planner receives permitted app descriptions and the observed conversation;
   it emits `next`, `goal`, `apps`. A valid stop uses `next=done`, `apps=[]`.
5. For a valid app plan, select every permitted function schema in those apps.
   Invalid plans fall back to original-query BM25; no invalid goal is forwarded.
6. Caller receives the current goal as unverified guidance, selected schemas,
   original request, context and observations. It emits native calls or `[]`.
7. Validate calls and grounding, execute through the same ToolEnv, append actual
   results, then return to Planner. Only independent calls may be batched.
8. Stop on no remaining work, clarification, uncertain write outcome, loop bound
   or deadline. Export all actual calls and the real final hash; retain diagnostics.

Planner: at most 192 output tokens and 20 seconds per invocation, within the shared
sample budget. Caller notebook ceiling: 384 tokens. Both roles use greedy decoding.
Eight controller rounds by default. Main runner budget 27 seconds; train notebook
uses 90 seconds for development. Cooperative deadlines can overrun native operations.

See README.md for source layout and notebooks. experiments/ contains comparison
scripts and is not used by inference. Frozen Qwen3.5 Caller SFT is a separate
historical training workflow, not a trained app Planner or a beyoru adapter.

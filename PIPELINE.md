# Current pipeline

See [Direct baseline v2](docs/direct_baseline.md) for the current execution contract. Default entry point: scripts/infer.py --prompt-variant direct. Prompt: src/lifemate/prompt_assets/direct.txt. Controller/parser: src/lifemate/direct.py. Shared infrastructure: backend.py, asr.py, prompt_cache.py, grounding.py, validation.py (evidence checks only), execution.py and input_adapter.py.

Legacy agent.py, planning.py and memory.py are used only by explicitly selected historical variants. safe_canonicalize is not called by direct_v2. No mined train priors are loaded at inference.

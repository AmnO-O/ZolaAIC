"""One general system prompt; no sample-specific examples or lexical selection."""
import copy
import os
from pathlib import Path
import sys

INSTRUCTION = (Path(__file__).parent / 'prompt_assets/system.txt').read_text(encoding='utf-8')


class PromptControls:
    def __init__(self, mode='system'):
        self.variant = os.environ.get('LIFEMATE_PROMPT_VARIANT', 'clean')
        if self.variant not in ('clean', 'planning', 'memory_full', 'memory_compact'):
            raise ValueError('Use clean, planning, memory_full or memory_compact; legacy prompt variants were removed')
        if mode not in ('baseline', 'system'):
            raise ValueError('Prompt mode must be baseline or system; legacy few-shot mode was removed')
        self.mode = mode
        if mode == 'baseline':
            if self.variant != 'clean':
                raise ValueError('Planning and memory variants require system prompt controls')
            return
        source = os.environ.get('LIFEMATE_STEERABILITY_PATH')
        if source is None:
            source = str(Path(__file__).resolve().parents[2] / 'vendor/steerability')
        if Path(source, 'steerability/__init__.py').is_file():
            sys.path.insert(0, str(Path(source).resolve()))
        from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
        self.instruction = INSTRUCTION
        if self.variant == "planning":
            self.instruction += (Path(__file__).parent / "prompt_assets/planning.txt").read_text(encoding="utf-8")
        if self.variant.startswith('memory_'):
            self.instruction += (Path(__file__).parent / 'prompt_assets/memory.txt').read_text(encoding='utf-8')
        self.system = SystemPrompt(text=self.instruction, mode='prepend')
        self.system.steer()

    def apply(self, messages, tools):
        batch = [copy.deepcopy(messages)]
        if self.mode == 'baseline':
            return batch[0]
        return self.system.adapt_messages(batch)[0]

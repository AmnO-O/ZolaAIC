"""Local Transformers baseline. All inference assets must already exist locally."""
import os
import json
import hashlib
from pathlib import Path
import random
import time


from .messages import chat_messages, native_tools
from .prompts import INSTRUCTION, apply_policy
from .agent import parse_decision


class LocalBackend:
    def __init__(self, llm_path, asr_path=None, device='cuda:0', asr_kind='gipformer'):
        if asr_kind != 'gipformer':
            raise ValueError('Only the configured Gipformer ASR is supported')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import numpy as np
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Deterministic algorithms are the default. The two diagnostic settings
        # below exist only to measure how much latency the guarantee costs on
        # the deployment device; they are not a recommended submission mode.
        deterministic = os.environ.get('LIFEMATE_DETERMINISTIC', '1') == '1'
        if deterministic:
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        config = AutoConfig.from_pretrained(llm_path, local_files_only=True)
        if config.model_type != 'qwen3_5':
            raise ValueError('This trial requires the pinned Qwen3.5 checkpoint, not old Qwen3 assets')
        self.is_hybrid = True
        loader = Qwen3_5ForConditionalGeneration
        dtype = torch.bfloat16 if self.is_hybrid and str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float16
        self.model = loader.from_pretrained(
            llm_path, local_files_only=True, torch_dtype=dtype,
            attn_implementation=os.environ.get('LIFEMATE_ATTN', 'sdpa')).to(device).eval()
        self.tokenizer.model_max_length = 16384
        self.model.config.max_position_embeddings = 16384
        if self.is_hybrid:
            self.model.config.text_config.max_position_embeddings = 16384
        adapter_path = os.environ.get('LIFEMATE_ADAPTER_PATH')
        if adapter_path:
            metadata = json.loads((Path(adapter_path)/'lifemate_policy.json').read_text())
            if metadata['prompt_sha256'] != hashlib.sha256(INSTRUCTION.encode()).hexdigest():
                raise ValueError('Adapter policy differs from inference policy')
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, local_files_only=True).merge_and_unload().eval()
        self.llm_parameters = sum(p.numel() for p in self.model.parameters())
        if self.llm_parameters >= 5_000_000_000:
            raise ValueError('LLM exceeds competition parameter limit')
        self.asr = None
        self.asr_parameters = 0
        if asr_path:
            from .asr import GipformerASR
            self.asr = GipformerASR(
                asr_path, precision=os.environ.get('LIFEMATE_ASR_PRECISION', 'int8'),
                num_threads=int(os.environ.get('LIFEMATE_ASR_THREADS', '4')))
            self.asr_parameters = self.asr.parameters
        if self.llm_parameters + self.asr_parameters >= 6_000_000_000:
            raise ValueError('Combined models exceed parameter limit')
        self.stats = []
        from .prompt_cache import PromptCache
        self.prompt_cache = PromptCache()
        self.reuse_prompt_cache = not self.is_hybrid and os.environ.get('LIFEMATE_PROMPT_CACHE', '1') == '1'
        import transformers
        self.runtime_info = {'torch':torch.__version__,'transformers':transformers.__version__,
            'cuda':torch.version.cuda,'device':str(next(self.model.parameters()).device),
            'dtype':str(next(self.model.parameters()).dtype),
            'configured_attention':self.model.config._attn_implementation,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
        self.runtime_info['model_type'] = config.model_type
        self.runtime_info['llm_parameters'] = self.llm_parameters
        self.runtime_info['total_parameters'] = self.llm_parameters + self.asr_parameters
        self.runtime_info['thinking'] = False
        self.runtime_info['cross_turn_cache'] = self.reuse_prompt_cache
        self.runtime_info['adapter_path'] = adapter_path
        self.runtime_info['policy'] = 'single'
        self.runtime_info['prompt_sha256'] = hashlib.sha256(
            INSTRUCTION.encode('utf-8')).hexdigest()
        if str(device).startswith('cuda'):
            self.runtime_info['gpu'] = torch.cuda.get_device_name(device)

    def reset_sample(self):
        self.prompt_cache.reset()

    def decide(self, messages, tools, remaining_seconds):
        text = self._generate(messages, tools, remaining_seconds)
        return parse_decision(text, tools)

    def _generate(self, messages, tools, remaining_seconds, max_tokens=384, controls=True, phase='llm'):
        if remaining_seconds <= 0:
            raise TimeoutError('No generation budget')
        from transformers import StoppingCriteria, StoppingCriteriaList
        end = time.monotonic() + remaining_seconds

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= end

        rendered_tools = native_tools(tools)
        turns = apply_policy(chat_messages(messages)) if controls else messages
        encoded = self.tokenizer.apply_chat_template(
            turns, tools=rendered_tools, add_generation_prompt=True,
            tokenize=True, return_tensors='pt', return_dict=True, enable_thinking=False)
        tokens = encoded['input_ids'].shape[-1]
        token_ids = encoded['input_ids'][0].tolist()
        budget = min(max_tokens, 16384-tokens)
        if budget <= 0:
            raise ValueError('Context exceeds 16K; refusing silent truncation')
        emit = getattr(self, 'trace_event', None)
        if emit:
            emit(phase + '.input', input={'messages': turns, 'tools': rendered_tools,
                 'rendered_prompt': self.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=False)},
                 input_tokens=tokens, max_new_tokens=budget)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        if getattr(self, 'is_hybrid', False):
            return self._generate_hybrid(encoded, tokens, budget, end, emit, phase, Deadline)
        cache, reused = self.prompt_cache.take(token_ids) if self.reuse_prompt_cache else (None, 0)
        if emit:
            emit(phase + '.cache', input_tokens=tokens, cached_tokens=reused,
                 prefill_tokens=tokens-1-reused, fresh_tokens=tokens-reused,
                 enabled=self.reuse_prompt_cache)
        prefill_started = time.monotonic()
        with self.torch.inference_mode():
            chunk_size = int(os.environ.get('LIFEMATE_PREFILL_CHUNK', '1024'))
            for offset in range(reused, max(0, tokens-1), chunk_size):
                if time.monotonic() >= end:
                    raise TimeoutError('Prompt processing deadline reached')
                stop = min(offset+chunk_size, tokens-1)
                prefill = self.model(
                    input_ids=encoded['input_ids'][:, offset:stop],
                    attention_mask=encoded['attention_mask'][:, :stop],
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = prefill.past_key_values
                del prefill
            if str(self.device).startswith('cuda'):
                self.torch.cuda.synchronize(self.device)
            if time.monotonic() >= end:
                raise TimeoutError('Prompt processing deadline reached')
            prefill_seconds = time.monotonic() - prefill_started
            generation_started = time.monotonic()
            generated = self.model.generate(
                **encoded, max_new_tokens=budget, do_sample=False,
                past_key_values=cache,
                return_dict_in_generate=True,
                stopping_criteria=StoppingCriteriaList([Deadline()]),
                pad_token_id=self.tokenizer.eos_token_id)
        sequences = generated.sequences
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize(self.device)
        generation_seconds = time.monotonic() - generation_started
        text = self.tokenizer.decode(sequences[0, tokens:], skip_special_tokens=True)
        if emit:
            emit(phase + '.output', output=text, output_tokens=sequences.shape[-1]-tokens,
                 cached_tokens=reused, prefill_seconds=prefill_seconds, generation_seconds=generation_seconds)
        self.stats.append({'input_tokens': tokens, 'output_tokens': sequences.shape[-1]-tokens,
                          'cached_tokens': reused, 'prefill_seconds': prefill_seconds,
                          'generation_seconds': generation_seconds})
        if time.monotonic() >= end:
            raise TimeoutError('Model generation deadline reached')
        if self.reuse_prompt_cache:
            self.prompt_cache.store(generated.past_key_values, token_ids[:-1])
        return text

    def _generate_hybrid(self, encoded, tokens, budget, end, emit, phase, deadline_class):
        """Fresh hybrid state per decision; native cache remains enabled within decoding."""
        from transformers import StoppingCriteriaList
        started = time.monotonic()
        first = []
        def prefill_finished(module, inputs, outputs):
            if not first:
                if str(self.device).startswith('cuda'):
                    self.torch.cuda.synchronize(self.device)
                if not self.torch.isfinite(outputs.logits[:, -1, :]).all():
                    raise FloatingPointError('Non-finite Qwen3.5 prefill logits; inspect dtype/runtime')
                first.append(time.monotonic()-started)
        if emit:
            emit(phase + '.cache', input_tokens=tokens, cached_tokens=0,
                 prefill_tokens=tokens, fresh_tokens=tokens, enabled=False,
                 reason='Hybrid recurrent cache is fresh for each decision')
        handle = self.model.register_forward_hook(prefill_finished)
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(**encoded, max_new_tokens=budget, do_sample=False,
                    use_cache=True, logits_to_keep=1, return_dict_in_generate=True,
                    stopping_criteria=StoppingCriteriaList([deadline_class()]),
                    pad_token_id=self.tokenizer.eos_token_id)
        finally:
            handle.remove()
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize(self.device)
        total = time.monotonic()-started
        prefill = first[0] if first else total
        output = generated.sequences[0, tokens:]
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        stats = {'input_tokens': tokens, 'output_tokens': len(output), 'cached_tokens': 0,
                 'prefill_seconds': prefill, 'generation_seconds': max(0, total-prefill)}
        self.stats.append(stats)
        if emit:
            emit(phase + '.output', output=text, **{k:v for k,v in stats.items() if k!='input_tokens'})
        if time.monotonic() >= end:
            raise TimeoutError('Model generation deadline reached')
        return text

    def transcribe(self, path, remaining_seconds):
        if self.asr is None:
            raise RuntimeError('No local ASR model configured')
        return self.asr.transcribe(path, remaining_seconds)


def create_backend():
    return LocalBackend(os.environ['LIFEMATE_LLM_PATH'], os.environ.get('LIFEMATE_ASR_PATH'),
        asr_kind=os.environ.get('LIFEMATE_ASR_KIND','gipformer'))

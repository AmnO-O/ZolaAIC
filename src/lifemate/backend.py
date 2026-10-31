"""Local Transformers baseline. All inference assets must already exist locally."""
import os
import json
import hashlib
from pathlib import Path
import random
import time


from .messages import chat_messages, native_tools
from .prompts import INSTRUCTION, apply_policy
from .agent import Decision, parse_decision
from .output_protocol import NativeOutputGuard


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
        self.max_new_tokens = int(os.environ.get('LIFEMATE_MAX_NEW_TOKENS', '384'))
        if not 1 <= self.max_new_tokens <= 16384:
            raise ValueError('LIFEMATE_MAX_NEW_TOKENS must be between 1 and 16384')
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        config = AutoConfig.from_pretrained(llm_path, local_files_only=True)
        if config.model_type not in ('qwen3', 'qwen3_5'):
            raise ValueError('Supported checkpoints: Qwen3 and Qwen3.5')
        self.is_hybrid = config.model_type == 'qwen3_5'
        self.protocol = 'xml' if self.is_hybrid else 'json'
        loader = Qwen3_5ForConditionalGeneration if self.is_hybrid else AutoModelForCausalLM
        dtype = torch.bfloat16 if self.is_hybrid and str(device).startswith('cuda') and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
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
        self.runtime_info['max_new_tokens'] = self.max_new_tokens
        self.runtime_info['thinking'] = False
        self.runtime_info['strict_tool_output'] = True
        self.runtime_info['cross_turn_cache'] = self.reuse_prompt_cache
        self.runtime_info['adapter_path'] = adapter_path
        self.runtime_info['policy'] = 'planner_caller_apps'
        self.runtime_info['prompt_sha256'] = hashlib.sha256(
            INSTRUCTION.encode('utf-8')).hexdigest()
        if str(device).startswith('cuda'):
            self.runtime_info['gpu'] = torch.cuda.get_device_name(device)

    def reset_sample(self):
        self.prompt_cache.reset()

    def decide(self, messages, tools, remaining_seconds):
        text = self._generate(messages, tools, remaining_seconds)
        error = getattr(getattr(self, '_output_guard', None), 'error', None)
        if error:
            return Decision(protocol_error=error, raw_text=text)
        if not getattr(self, '_generation_complete', True):
            return Decision(protocol_error='Token limit reached before assistant turn ended', raw_text=text)
        return parse_decision(text, tools, protocol=getattr(self, 'protocol', 'xml'))

    def _generate(self, messages, tools, remaining_seconds, max_tokens=None, controls=True, phase='llm'):
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
        if max_tokens is None:
            max_tokens = getattr(self, 'max_new_tokens', 1024)
        budget = min(max_tokens, 16384-tokens)
        if budget <= 0:
            raise ValueError('Context exceeds 16K; refusing silent truncation')
        emit = getattr(self, 'trace_event', None)
        if emit:
            emit(phase + '.input', input={'messages': turns, 'tools': rendered_tools,
                 'rendered_prompt': self.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=False)},
                 input_tokens=tokens, max_new_tokens=budget)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        self._output_guard = NativeOutputGuard(self.tokenizer, tokens, self._assistant_stop_ids()) if controls else None
        stoppers = StoppingCriteriaList([Deadline()] + ([self._output_guard] if controls else []))
        if getattr(self, 'is_hybrid', False):
            return self._generate_hybrid(encoded, tokens, budget, end, emit, phase, stoppers)
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
                eos_token_id=self._assistant_stop_ids(),
                return_dict_in_generate=True,
                stopping_criteria=stoppers,
                pad_token_id=self.tokenizer.eos_token_id)
        sequences = generated.sequences
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize(self.device)
        generation_seconds = time.monotonic() - generation_started
        self._generation_complete = (sequences.shape[-1] > tokens and
                                     int(sequences[0, -1]) in self._assistant_stop_ids())
        text = self.tokenizer.decode(sequences[0, tokens:], skip_special_tokens=True)
        guard_error = getattr(self._output_guard, 'error', None)
        if emit:
            emit(phase + '.output', output=text, output_tokens=sequences.shape[-1]-tokens,
                 cached_tokens=reused, prefill_seconds=prefill_seconds, generation_seconds=generation_seconds,
                 output_guard_error=guard_error, assistant_turn_complete=self._generation_complete)
        self.stats.append({'input_tokens': tokens, 'output_tokens': sequences.shape[-1]-tokens,
                          'cached_tokens': reused, 'prefill_seconds': prefill_seconds,
                          'generation_seconds': generation_seconds, 'output_guard_error': guard_error})
        if time.monotonic() >= end:
            raise TimeoutError('Model generation deadline reached')
        if self.reuse_prompt_cache:
            self.prompt_cache.store(generated.past_key_values, token_ids[:-1])
        return text

    def _assistant_stop_ids(self):
        values = [self.tokenizer.eos_token_id]
        configured = self.model.generation_config.eos_token_id
        values.extend(configured if isinstance(configured, list) else [configured])
        return sorted({value for value in values if isinstance(value, int)})

    def _generate_hybrid(self, encoded, tokens, budget, end, emit, phase, stoppers):
        """Fresh recurrent state per decision, memory-bounded chunked prefill."""
        started = time.monotonic()
        cache = None
        stop_ids = self._assistant_stop_ids()
        # This task is text-only. The conditional-generation wrapper keeps
        # rope_deltas after generate(); clear it before a fresh prompt.
        if hasattr(self.model.model, 'rope_deltas'):
            self.model.model.rope_deltas = None
        chunk_size = int(os.environ.get('LIFEMATE_PREFILL_CHUNK', '256'))
        if chunk_size <= 0:
            raise ValueError('Prefill chunk size must be positive')
        if emit:
            emit(phase + '.cache', input_tokens=tokens, cached_tokens=0,
                 prefill_tokens=tokens-1, fresh_tokens=tokens, enabled=False,
                 prefill_chunk=chunk_size,
                 reason='Fresh hybrid state per decision; no recurrent cache cropping')
        with self.torch.inference_mode():
            for offset in range(0, max(0, tokens-1), chunk_size):
                if time.monotonic() >= end:
                    raise TimeoutError('Prompt processing deadline reached')
                stop = min(offset+chunk_size, tokens-1)
                positions = self.torch.arange(offset, stop, device=self.device).view(1, 1, -1)
                positions = positions.expand(3, encoded['input_ids'].shape[0], -1)
                result = self.model(input_ids=encoded['input_ids'][:, offset:stop],
                    position_ids=positions,
                    attention_mask=encoded['attention_mask'][:, :stop],
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                if not self.torch.isfinite(result.logits[:, -1, :]).all():
                    raise FloatingPointError('Non-finite Qwen3.5 logits; inspect dtype/runtime')
                cache = result.past_key_values
                del result
            if str(self.device).startswith('cuda'):
                self.torch.cuda.synchronize(self.device)
            if time.monotonic() >= end:
                raise TimeoutError('Prompt processing deadline reached')
            prefill = time.monotonic()-started
            generation_started = time.monotonic()
            generated = self.model.generate(**encoded, max_new_tokens=budget, do_sample=False,
                past_key_values=cache, use_cache=True, logits_to_keep=1,
                eos_token_id=stop_ids,
                return_dict_in_generate=True,
                stopping_criteria=stoppers,
                pad_token_id=self.tokenizer.eos_token_id)
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize(self.device)
        generation = time.monotonic()-generation_started
        output = generated.sequences[0, tokens:]
        self._generation_complete = bool(len(output) and int(output[-1]) in stop_ids)
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        stats = {'input_tokens': tokens, 'output_tokens': len(output), 'cached_tokens': 0,
                 'prefill_seconds': prefill, 'generation_seconds': generation,
                 'assistant_turn_complete': self._generation_complete,
                 'output_guard_error': getattr(self._output_guard, 'error', None),
                 'eos_token_id': self.tokenizer.eos_token_id, 'stop_token_ids': stop_ids}
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

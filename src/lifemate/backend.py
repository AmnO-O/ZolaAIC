"""Local Transformers baseline. All inference assets must already exist locally."""
import json
import os
import random
import re
import time


def parse_calls(text):
    calls = []
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S)
    if text.count('<tool_call>')!=text.count('</tool_call>') or ('<tool_call>' in text and not blocks):
        raise ValueError('Truncated tool call')
    for block in blocks:
        call = json.loads(block)
        args = call.get('arguments', {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(call.get('name'), str) or not isinstance(args, dict):
            raise ValueError('Malformed model tool call')
        calls.append({'name': call['name'], 'arguments': args})
    return calls


def chat_messages(messages):
    """Normalize internal and historical turns into the native Qwen template."""
    output = []
    for original in messages:
        role = original['role']
        if role == 'assistant':
            msg = {'role': role, 'content': original.get('text', original.get('content', ''))}
            if original.get('tool_calls'):
                msg['tool_calls'] = [{'type': 'function', 'function': c}
                                     for c in original['tool_calls']]
            output.append(msg)
            for result in original.get('tool_results', []):
                output.append({'role': 'tool', 'content': json.dumps(result, ensure_ascii=False)})
        else:
            content = original.get('content', '')
            output.append({'role': role, 'content': content if isinstance(content, str)
                           else json.dumps(content, ensure_ascii=False)})
    return output


class LocalBackend:
    def __init__(self, llm_path, asr_path=None, device='cuda:0', prompt_mode='steerability', asr_kind='gipformer'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_path, local_files_only=True, torch_dtype=torch.float16,
            attn_implementation='sdpa').to(device).eval()
        self.tokenizer.model_max_length = 16384
        self.model.config.max_position_embeddings = 16384
        self.llm_parameters = sum(p.numel() for p in self.model.parameters())
        if self.llm_parameters >= 5_000_000_000:
            raise ValueError('LLM exceeds competition parameter limit')
        self.asr = self.processor = None
        self.asr_parameters = 0
        if asr_path:
            if asr_kind=='gipformer':
                from .asr import GipformerASR
                self.asr=GipformerASR(asr_path,precision=os.environ.get('LIFEMATE_ASR_PRECISION','int8'),
                    num_threads=int(os.environ.get('LIFEMATE_ASR_THREADS','4')))
                self.asr_parameters=self.asr.parameters
            elif asr_kind=='whisper':
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                self.processor = AutoProcessor.from_pretrained(asr_path, local_files_only=True)
                self.asr = AutoModelForSpeechSeq2Seq.from_pretrained(
                    asr_path, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
                self.asr_parameters = sum(p.numel() for p in self.asr.parameters())
            else:raise ValueError('Unknown ASR kind')
        if self.llm_parameters + self.asr_parameters >= 6_000_000_000:
            raise ValueError('Combined models exceed parameter limit')
        self.stats = []
        from .prompts import PromptControls
        self.prompt_controls=PromptControls(prompt_mode)
        self.asr_kind=asr_kind
        from .prompt_cache import PromptCache
        self.prompt_cache = PromptCache()
        self.reuse_prompt_cache = os.environ.get('LIFEMATE_PROMPT_CACHE', '1') == '1'

    def reset_sample(self):
        self.prompt_cache.reset()

    def decide(self, messages, tools, remaining_seconds):
        if remaining_seconds <= 0:
            raise TimeoutError('No generation budget')
        from transformers import StoppingCriteria, StoppingCriteriaList
        end = time.monotonic() + remaining_seconds

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= end

        native_tools = [{'type': 'function', 'function': {
            'name': t['name'], 'description': t.get('description', ''),
            'parameters': t['parameters']}} for t in tools]
        turns = self.prompt_controls.apply(chat_messages(messages),tools)
        encoded = self.tokenizer.apply_chat_template(
            turns, tools=native_tools, add_generation_prompt=True,
            tokenize=True, return_tensors='pt', return_dict=True)
        tokens = encoded['input_ids'].shape[-1]
        token_ids = encoded['input_ids'][0].tolist()
        budget = min(384, 16384-tokens)
        if budget <= 0:
            raise ValueError('Context exceeds 16K; refusing silent truncation')
        emit = getattr(self, 'trace_event', None)
        if emit:
            emit('llm.input', input={'messages': turns, 'tools': native_tools,
                 'rendered_prompt': self.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=False)},
                 input_tokens=tokens, max_new_tokens=budget)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        cache, reused = self.prompt_cache.take(token_ids) if self.reuse_prompt_cache else (None, 0)
        if emit:
            emit('llm.cache', input_tokens=tokens, cached_tokens=reused,
                 prefill_tokens=tokens-1-reused, fresh_tokens=tokens-reused,
                 enabled=self.reuse_prompt_cache)
        prefill_started = time.monotonic()
        with self.torch.inference_mode():
            # Deterministic SDPA on T4 can fall back to quadratic-memory math
            # attention. Prefill in bounded query chunks while retaining KV.
            for offset in range(reused, max(0, tokens-1), 256):
                if time.monotonic() >= end:
                    raise TimeoutError('Prompt processing deadline reached')
                stop = min(offset+256, tokens-1)
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
            emit('llm.output', output=text, output_tokens=sequences.shape[-1]-tokens,
                 cached_tokens=reused, prefill_seconds=prefill_seconds, generation_seconds=generation_seconds)
        self.stats.append({'input_tokens': tokens, 'output_tokens': sequences.shape[-1]-tokens,
                          'cached_tokens': reused, 'prefill_seconds': prefill_seconds,
                          'generation_seconds': generation_seconds})
        if time.monotonic() >= end:
            raise TimeoutError('Model generation deadline reached')
        calls = parse_calls(text)
        if self.reuse_prompt_cache:
            self.prompt_cache.store(generated.past_key_values, token_ids[:-1])
        return calls

    def transcribe(self, path, remaining_seconds):
        if self.asr is None:
            raise RuntimeError('No local ASR model configured')
        if self.asr_kind=='gipformer':
            return self.asr.transcribe(path,remaining_seconds)
        from transformers import StoppingCriteria, StoppingCriteriaList
        import soundfile as sf
        import numpy as np
        from scipy.signal import resample_poly
        from math import gcd
        if remaining_seconds <= 0:
            raise TimeoutError('No ASR budget')
        end = time.monotonic() + remaining_seconds

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= end

        audio, rate = sf.read(path, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if rate != 16000:
            factor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000//factor, rate//factor)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Invalid audio')
        # Exact digital silence contains no speech. Avoid Whisper hallucinations
        # without introducing a learned VAD or rejecting quiet real recordings.
        if np.max(np.abs(audio)) == 0:
            return ''
        texts = []
        for start in range(0, len(audio), 30*16000):
            if time.monotonic() >= end:
                raise TimeoutError('ASR deadline reached')
            batch = self.processor(audio[start:start+30*16000], sampling_rate=16000,
                                   return_tensors='pt')
            features = batch.input_features.to(self.device, dtype=self.torch.float16)
            with self.torch.inference_mode():
                ids = self.asr.generate(input_features=features, language='vi', task='transcribe',
                                        do_sample=False, max_new_tokens=256,
                                        stopping_criteria=StoppingCriteriaList([Deadline()]))
            texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0])
        if time.monotonic() >= end:
            raise TimeoutError('ASR deadline reached')
        return ' '.join(texts).strip()


def create_backend():
    return LocalBackend(os.environ['LIFEMATE_LLM_PATH'], os.environ.get('LIFEMATE_ASR_PATH'),
        prompt_mode=os.environ.get('LIFEMATE_PROMPT_MODE','steerability'),
        asr_kind=os.environ.get('LIFEMATE_ASR_KIND','gipformer'))

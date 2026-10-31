"""Small Transformers-only RL pilot. Sampling is training-only, never inference."""
import time

from .agent import Decision, parse_decision
from .messages import chat_messages, native_tools
from .output_protocol import NativeOutputGuard
from .prompts import apply_policy


class RolloutActor:
    def __init__(self, model, tokenizer, max_new_tokens=384, max_length=8192):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.max_length = max_new_tokens, max_length
        self.protocol = 'xml' if model.config.model_type == 'qwen3_5' else 'json'
        values = [tokenizer.eos_token_id]
        eos = model.generation_config.eos_token_id
        values += eos if isinstance(eos, list) else [eos]
        self.stop_ids = sorted({v for v in values if isinstance(v, int)})
        self.reset_episode()

    def reset_episode(self):
        self.segments = []

    def reset_rope(self):
        # ConditionalGeneration retains this across generation calls.
        base = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        if hasattr(base.model, 'rope_deltas'):
            base.model.rope_deltas = None

    def decide(self, messages, tools, remaining_seconds):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        end = time.monotonic() + remaining_seconds

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= end

        inputs = self.tokenizer.apply_chat_template(apply_policy(chat_messages(messages)),
                    tools=native_tools(tools), tokenize=True, return_dict=True,
                    return_tensors='pt', add_generation_prompt=True, enable_thinking=False)
        prompt = inputs['input_ids'][0].tolist()
        budget = min(self.max_new_tokens, self.max_length-len(prompt))
        if budget <= 0:
            raise ValueError('RL context exceeds configured limit; no silent truncation')
        guard = NativeOutputGuard(self.tokenizer, len(prompt), self.stop_ids)
        self.reset_rope()
        self.model.eval()  # Dropout off during both sampling and log-prob scoring.
        with torch.no_grad():
            output = self.model.generate(**{k:v.to(self.model.device) for k,v in inputs.items()},
                do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, max_new_tokens=budget, use_cache=True,
                eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([Deadline(), guard]))
        response = output[0, len(prompt):].tolist()
        if response:
            self.segments.append({'prompt_ids': prompt, 'response_ids': response})
        text = self.tokenizer.decode(response, skip_special_tokens=True)
        if time.monotonic() >= end:
            raise TimeoutError('RL rollout generation deadline')
        if guard.error:
            return Decision(protocol_error=guard.error, raw_text=text)
        if not response or response[-1] not in self.stop_ids:
            return Decision(protocol_error='Truncated assistant turn', raw_text=text)
        return parse_decision(text, tools, self.protocol)

    def logps(self, segment):
        """Only sampled assistant tokens, including EOS; no re-tokenization."""
        import torch
        self.reset_rope()
        prompt, response = segment['prompt_ids'], segment['response_ids']
        ids = torch.tensor([prompt+response], device=self.model.device)
        output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=False, logits_to_keep=len(response)+1)
        logits = output.logits[0, -len(response)-1:-1].float()
        labels = ids[0, -len(response):]
        return logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)

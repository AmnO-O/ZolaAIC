import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.prompt_cache import PromptCache


class FakeCache:
    def __init__(self, length): self.length = length
    def get_seq_length(self): return self.length
    def crop(self, count):
        assert count < 0
        self.length += count


class PromptCacheTests(unittest.TestCase):
    def test_changed_suffix_crops_to_exact_prefix_and_transfers_ownership(self):
        holder = PromptCache()
        holder.store(FakeCache(8), [1, 2, 3, 4])
        cache, reused = holder.take([1, 2, 9, 8])
        self.assertEqual((reused, cache.get_seq_length()), (2, 2))
        self.assertEqual(holder.take([1, 2, 9]), (None, 0))

    def test_identical_or_shorter_prompt_leaves_one_token_for_logits(self):
        holder = PromptCache()
        holder.store(FakeCache(4), [1, 2, 3, 4])
        cache, reused = holder.take([1, 2, 3])
        self.assertEqual((reused, cache.get_seq_length()), (2, 2))

    def test_reset_and_prefix_mismatch_recompute(self):
        holder = PromptCache()
        holder.store(FakeCache(3), [1, 2, 3])
        self.assertEqual(holder.take([8, 2, 3]), (None, 0))
        holder.store(FakeCache(3), [1, 2, 3])
        holder.reset()
        self.assertEqual(holder.take([1, 2, 3, 4]), (None, 0))

    def test_real_qwen_cached_and_fresh_generation_match(self):
        # Tiny random Qwen, created locally: no pretrained assets or GPU needed.
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16,
                            max_position_embeddings=128, eos_token_id=None,
                            pad_token_id=0, bos_token_id=1)
        config._attn_implementation = 'eager'
        model = Qwen3ForCausalLM(config).eval()
        from lifemate.backend import LocalBackend
        class Tokenizer:
            eos_token_id=0
            ids=[]
            def apply_chat_template(self,*args,**kwargs):
                tensor=torch.tensor([self.ids])
                return {'input_ids':tensor,'attention_mask':torch.ones_like(tensor)}
            def decode(self,tokens,**kwargs):return ','.join(map(str,tokens.tolist()))
        backend=LocalBackend.__new__(LocalBackend)
        backend.torch,backend.device,backend.model=torch,'cpu',model
        backend.tokenizer=Tokenizer()
        backend.prompt_cache=PromptCache()
        backend.reuse_prompt_cache=True
        backend.stats=[]
        prompts = [[1, 5, 8, 9, 10], [1, 5, 8, 9, 10, 11, 12],
                   [1, 5, 8, 20, 21], [1, 5]]
        hits = []
        with torch.inference_mode():
            for ids in prompts:
                tensor = torch.tensor([ids])
                mask = torch.ones_like(tensor)
                backend.tokenizer.ids=ids
                reused_output=backend._generate([],[],30,max_tokens=3,controls=False)
                hits.append(backend.stats[-1]['cached_tokens'])
                fresh_output = model.generate(input_ids=tensor, attention_mask=mask,
                    max_new_tokens=3, do_sample=False, pad_token_id=0)
                self.assertEqual(reused_output,backend.tokenizer.decode(fresh_output[0,len(ids):]))
        self.assertEqual(hits, [0, 4, 3, 1])


if __name__ == '__main__': unittest.main()

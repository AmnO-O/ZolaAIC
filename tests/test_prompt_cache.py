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
        self.assert_real_model_guard(model, backend.tokenizer, torch)

    def assert_real_model_guard(self, model, tokenizer, torch):
        from transformers import StoppingCriteriaList, LogitsProcessorList
        from lifemate.output_protocol import NativeOutputGuard
        ids=torch.tensor([[1,5,8,9]])
        guard=NativeOutputGuard(tokenizer,4,[0])
        def force_non_tool(ids,scores):
            scores.fill_(-float('inf'));scores[:,6]=0
            return scores
        with torch.inference_mode():
            result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                max_new_tokens=3,do_sample=False,pad_token_id=0,eos_token_id=0,
                logits_processor=LogitsProcessorList([force_non_tool]),
                stopping_criteria=StoppingCriteriaList([guard]))
        self.assertEqual(result.shape[-1],5)
        self.assertIsNotNone(guard.error)

    def test_hybrid_generation_uses_fresh_recurrent_state(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from lifemate.backend import LocalBackend
        torch.manual_seed(42)
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
            layer_types=['linear_attention']*3+['full_attention'], max_position_embeddings=128,
            rope_parameters={'rope_type':'default','rope_theta':10000,'partial_rotary_factor':0.5,'mrope_section':[1,1,2]},
            eos_token_id=None, pad_token_id=0)
        config._attn_implementation='eager'
        vision=Qwen3_5VisionConfig(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
            num_position_embeddings=16,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=32,deepstack_visual_indexes=[])
        full=Qwen3_5Config(text_config=config.to_dict(),vision_config=vision.to_dict(),
            image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
        full._attn_implementation='eager'
        model=Qwen3_5ForConditionalGeneration(full).eval()
        class Tokenizer:
            eos_token_id=0
            def apply_chat_template(self,*args,**kwargs):
                self.thinking=kwargs['enable_thinking']
                ids=torch.tensor([[1,5,8,9]])
                return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
            def decode(self,tokens,**kwargs):return ','.join(map(str,tokens.tolist()))
        backend=LocalBackend.__new__(LocalBackend)
        backend.model,backend.tokenizer=model,Tokenizer()
        backend.device,backend.torch,backend.is_hybrid='cpu',torch,True
        backend.stats=[]
        from unittest.mock import patch
        with patch.dict('os.environ', {'LIFEMATE_PREFILL_CHUNK':'2'}):
            first=backend._generate([],[],30,max_tokens=2,controls=False)
            second=backend._generate([],[],30,max_tokens=2,controls=False)
        self.assertEqual(first,second)
        ids=torch.tensor([[1,5,8,9]])
        with torch.inference_mode():
            fresh=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                                 max_new_tokens=2,do_sample=False,pad_token_id=0,eos_token_id=0)
        self.assertEqual(first,backend.tokenizer.decode(fresh[0,4:]))
        self.assert_real_model_guard(model, backend.tokenizer, torch)
        self.assertFalse(backend.tokenizer.thinking)
        self.assertEqual([s['cached_tokens'] for s in backend.stats],[0,0])


if __name__ == '__main__': unittest.main()

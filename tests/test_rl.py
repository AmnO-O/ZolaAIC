"""Reward failure modes, actor/label boundary and on-policy gradient checks."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.agent import Decision
from lifemate.rl import (train_task, rollout, score_episode, group_advantages, policy_loss)

TOOLS = [{'name':'wallet.transfer', 'kind':'write', 'parameters':{
    'type':'object', 'properties':{'recipient':{'type':'string'}, 'amount':{'type':'integer'}}}}]
GOLD = [{'name':'wallet.transfer','arguments':{'recipient':'person_A','amount':100}}]


def episode(calls, termination='finished', errors=0, status='succeeded'):
    return {'prediction':{'tool_calls':calls}, 'trace':{
        'termination':termination, 'protocol_errors':errors, 'validation_rejections':0,
        'decisions':2, 'trace':[{'call':call,'executed':True,'status':status} for call in calls]}}


class RLTests(unittest.TestCase):
    def test_wrong_amount_recipient_and_extra_writes(self):
        correct = score_episode(episode(GOLD), GOLD, TOOLS)
        self.assertTrue(correct['exact_call_proxy'])
        self.assertFalse(correct['task_success_verified'])
        for args in ({'recipient':'person_A','amount':999},
                     {'recipient':'person_B','amount':100},
                     {'recipient':'person_A','amount':True}):
            reward = score_episode(episode([{'name':'wallet.transfer','arguments':args}]), GOLD, TOOLS)
            self.assertLess(reward['total'], 0)
            self.assertEqual(reward['efficiency'], 0)
        extra = score_episode(episode(GOLD*2), GOLD, TOOLS)
        self.assertEqual(extra['wrong_writes'], 1)
        self.assertLess(extra['total'], correct['total'])

    def test_empty_stop_and_failed_execution(self):
        self.assertGreater(score_episode(episode([]), [], TOOLS)['total'], 2)
        self.assertEqual(score_episode(episode([]), GOLD, TOOLS)['total'], 0)
        self.assertLess(score_episode(episode(GOLD), [], TOOLS)['total'], 0)
        for ep in (episode(GOLD, status='failed'), episode(GOLD, termination='deadline'),
                   episode(GOLD, errors=1)):
            reward = score_episode(ep, GOLD, TOOLS)
            self.assertFalse(reward['exact_call_proxy'])
            self.assertEqual(reward['efficiency'], 0)

    def test_label_boundary(self):
        row = {'id':'train_x','audio_text':'Pay the requested amount', 'context':{
            'installed_apps':{'wallet':{'tools':['wallet.transfer']}}}, 'history':[],
            'label':{'tool_calls':GOLD,'tool_results':[{'ok':True, 'receipt':'FUTURE_SECRET'}],
                     'final_state':{'secret':'NEVER_USE'}}}
        original = copy.deepcopy(row)
        task = train_task(row, TOOLS, {'apps':[]})
        self.assertNotIn('person_A', str(task['messages']))
        self.assertNotIn('FUTURE_SECRET', str(task))
        self.assertNotIn('NEVER_USE', str(task))
        self.assertEqual(row, original)
        row['id'] = 'public_test_x'
        with self.assertRaises(ValueError): train_task(row, TOOLS, {'apps':[]})

    def test_one_session_per_rollout_and_no_gold_to_actor(self):
        class Actor:
            def reset_episode(self): self.segments=[]
            def decide(self, messages, tools, remaining):
                assert 'SECRET' not in str(messages)
                return Decision()
        class Env:
            def get_state_hash(self): return 'same'
            def get_tool_calls(self): return []
        sessions=[]
        def factory(identifier):
            sessions.append(identifier)
            return Env()
        result=rollout({'id':'train_x','messages':[], 'tools':[], 'gold_calls':['SECRET']},
                       Actor(), factory)
        self.assertEqual(sessions, ['train_x'])
        self.assertEqual(result['initial_hash'], 'same')
        self.assertEqual(result['trace']['termination'], 'finished')

    def test_group_signal_and_policy_gradient_direction(self):
        import torch
        self.assertEqual(group_advantages([1,1]), [0,0])
        self.assertEqual(group_advantages([0,2]), [-1,1])
        for bad in ([1], [float('nan'), 1]):
            with self.assertRaises(ValueError): group_advantages(bad)
        value = torch.tensor([-2.,-3.], requires_grad=True)
        policy_loss(value, 1, 2, 2).backward()
        self.assertTrue((value.grad < 0).all())
        # A gradient descent update increases likelihood for positive advantage.
        self.assertTrue(torch.allclose(value.grad, torch.tensor([-.25,-.25])))

    def test_hybrid_scoring_matches_sampled_tokens_and_backpropagates(self):
        import torch
        from transformers import (Qwen3_5TextConfig, Qwen3_5VisionConfig,
                                   Qwen3_5Config, Qwen3_5ForConditionalGeneration)
        from lifemate.rl_actor import RolloutActor
        torch.manual_seed(42)
        text = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4,
            layer_types=['linear_attention','full_attention'], max_position_embeddings=128,
            rope_parameters={'rope_type':'default','rope_theta':10000,
                             'partial_rotary_factor':0.5,'mrope_section':[1,1,2]})
        vision = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64,
            num_heads=2, num_position_embeddings=16, patch_size=2, spatial_merge_size=2,
            temporal_patch_size=2, out_hidden_size=32, deepstack_visual_indexes=[])
        config = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
            image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
        config._attn_implementation='eager'
        model = Qwen3_5ForConditionalGeneration(config).eval()
        class Tokenizer: eos_token_id=1
        actor = RolloutActor(model, Tokenizer())
        segment={'prompt_ids':[2,3,4], 'response_ids':[5,1]}
        actual=actor.logps(segment)
        actor.reset_rope()
        ids=torch.tensor([[2,3,4,5,1]])
        expected=model(input_ids=ids,use_cache=False).logits[0,2:4].float().log_softmax(-1)
        expected=expected.gather(-1,torch.tensor([[5],[1]])).squeeze(-1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        policy_loss(actual,1,2,2).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        from lifemate.training import assistant_loss
        sparse, _ = assistant_loss(model, {'input_ids':ids,
            'attention_mask':torch.ones_like(ids), 'labels':torch.tensor([[-100,-100,-100,5,1]])})
        self.assertTrue(torch.allclose(sparse, -expected.mean(), atol=1e-6))


if __name__ == '__main__': unittest.main()

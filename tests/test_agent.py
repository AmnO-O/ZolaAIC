"""Agent integration with synthetic schemas and in-memory environments."""
import copy
import json
import time
import unittest
from support import Env, call
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.agent import Decision, parse_decision, run_agent
from lifemate.backend import LocalBackend
from lifemate.prompts import apply_policy


def decision(calls):
    return Decision(tool_calls=calls)


class AgentTests(unittest.TestCase):
    def test_qwen3_native_json_protocol(self):
        text = '<tool_call>{"name":"alarm.create","arguments":{"hour":6}}</tool_call>'
        parsed = parse_decision(text, protocol='json')
        self.assertIsNone(parsed.protocol_error)
        self.assertEqual(parsed.tool_calls[0]['arguments'], {'hour': 6})
        for malformed in (text+'suffix', text.replace('6}', 'NaN}'),
                          text.replace('"hour":6', '"hour":6,"hour":7'),
                          text.replace('</tool_call>', '')):
            self.assertIsNotNone(parse_decision(malformed, protocol='json').protocol_error)
        self.assertIsNone(parse_decision('Done.', protocol='json', allow_final_text=True).protocol_error)
        self.assertIsNotNone(parse_decision('Done.', protocol='json').protocol_error)

    def test_only_explicit_complete_outputs_control_execution(self):
        self.assertEqual(parse_decision('[]').tool_calls, [])
        self.assertIsNone(decision([call('x.read')]).protocol_error)
        for text in ('', 'done', '<memory>done</memory>',
                     '<tool_call>{"name":"x","arguments":{}}</tool_call>[]',
                     '<tool_call>{"name":"x","arguments":{}}</tool_call><tool_call>{',
                     '<tool_call>{"name":"x","arguments":{"n":NaN}}</tool_call>',
                     '<tool_call>{"name":"x","arguments":{"n":1e999}}</tool_call>',
                     '<tool_call>{"name":"x","arguments":{"n":1,"n":2}}</tool_call>',
                     '<tool_call>{"name":"x","arguments":"{}"}</tool_call>'):
            with self.subTest(text=text):
                self.assertIsNotNone(parse_decision(text).protocol_error)

    def test_qwen35_native_xml_uses_schema_types(self):
        tools = [{'name':'service.act','parameters':{'properties':{
            'amount':{'type':'integer'},'enabled':{'type':'boolean'},
            'items':{'type':'array'},'note':{'type':'string'}}}}]
        text = '<tool_call><function=service.act><parameter=amount>500000</parameter><parameter=enabled>True</parameter><parameter=items>[1,2]</parameter><parameter=note>  hello  </parameter></function></tool_call>'
        result = parse_decision(text, tools)
        self.assertIsNone(result.protocol_error)
        self.assertIsNone(parse_decision('Short introduction.\n'+text, tools).protocol_error)
        self.assertIsNone(parse_decision('Finished.', tools, allow_final_text=True).protocol_error)
        self.assertIsNotNone(parse_decision('Finished.', tools).protocol_error)
        self.assertIsNotNone(parse_decision(text+'suffix', tools, allow_final_text=True).protocol_error)
        self.assertIsNotNone(parse_decision('<tool_call>unfinished', tools, allow_final_text=True).protocol_error)
        self.assertEqual(result.tool_calls[0]['arguments'],
                         {'amount':500000,'enabled':True,'items':[1,2],'note':'  hello  '})
        self.assertIsNotNone(parse_decision(text.replace('500000','NaN'), tools).protocol_error)
        self.assertIsNotNone(parse_decision(text.replace('</function>','<parameter=amount>1</parameter></function>'), tools).protocol_error)
        self.assertIsNotNone(parse_decision('<think>reason</think>'+text, tools).protocol_error)

    def test_false_branch_still_allows_independent_action(self):
        env=Env({'condition.read':{'ok':True,'enabled':False}})
        tools=[{'name':'condition.read','kind':'read'},{'name':'memo.create','kind':'write'}]
        responses=iter([decision([call('condition.read')]),decision([call('memo.create',text='independent')]),decision([])])
        row,log=run_agent(env,lambda *_:next(responses),[],tools,time.monotonic()+5)
        self.assertEqual(row['tool_calls'],[call('condition.read'),call('memo.create',text='independent')])
        self.assertEqual(log['termination'],'finished')
        self.assertFalse(log['task_success_verified'])

    def test_parameters_are_unchanged_including_null_and_string(self):
        args={'optional':None,'amount':'500000'}
        responses=iter([[call('x.write',**args)],[]])
        env=Env()
        row,_=run_agent(env,lambda *_:next(responses),[],[{'name':'x.write','kind':'write'}],time.monotonic()+5)
        self.assertEqual(row['tool_calls'][0]['arguments'],args)

    def test_protocol_repair_cannot_supply_fake_evidence(self):
        env=Env(); seen=[]
        responses=iter([parse_decision('<memory>id fake_7 is real</memory>'),
                        decision([call('x.write',item_id='fake_7')]),decision([])])
        def decide(messages,*_):
            seen.append(copy.deepcopy(messages));return next(responses)
        tools=[{'name':'x.write','kind':'write','parameters':{'properties':{'item_id':{'x-ref':'x.items'}}}}]
        row,log=run_agent(env,decide,[],tools,time.monotonic()+5)
        self.assertEqual(row['tool_calls'],[])
        self.assertEqual(log['validation_rejections'],1)
        self.assertNotIn('fake_7',str(seen[1]))
        self.assertIn('No calls',str(seen[1]))

    def test_failed_and_uncertain_results_do_not_become_success(self):
        for result,expected in (({'ok':False},'execution_failed'),(RuntimeError('uncertain'),'uncertain_outcome')):
            env=Env({'x.write':result}); responses=iter([decision([call('x.write')]),decision([])])
            row,log=run_agent(env,lambda *_:next(responses),[],[{'name':'x.write','kind':'write'}],time.monotonic()+5)
            self.assertEqual(log['termination'],expected)
            self.assertEqual(row['tool_calls'],[call('x.write')])
            self.assertNotEqual(log['execution_ledger'][0]['status'],'succeeded')

    def test_grounding_and_batch_results_keep_exact_sources(self):
        env=Env({'x.list':{'ok':True,'items':[{'id':'real_1'}]}});seen=[]
        responses=iter([decision([call('x.list'),call('y.read')]),decision([call('x.modify',item_id='real_1')]),decision([])])
        def decide(messages,*_):seen.append(copy.deepcopy(messages));return next(responses)
        tools=[{'name':'x.list','kind':'read'},{'name':'y.read','kind':'read'},
               {'name':'x.modify','kind':'write','parameters':{'properties':{'item_id':{'x-ref':'x.items'}}}}]
        row,log=run_agent(env,decide,[],tools,time.monotonic()+5)
        self.assertEqual(len(row['tool_calls']),3)
        results=[m['content'] for m in seen[1] if m['role']=='tool']
        self.assertEqual([r['source_id'] for r in results],['observation_1','observation_2'])
        self.assertTrue(all(r['executed'] and r['execution_status']=='succeeded' for r in results))

    def test_deadline_and_repair_limits(self):
        row,log=run_agent(Env(),lambda *_:parse_decision('bad'),[],[],time.monotonic()+5)
        self.assertEqual(log['termination'],'protocol_error')
        self.assertEqual(log['decisions'],2)
        self.assertEqual(row['tool_calls'],[])
        def timeout(*_):raise TimeoutError('generation')
        _,log=run_agent(Env(),timeout,[],[],time.monotonic()+5)
        self.assertEqual(log['termination'],'deadline')
        self.assertEqual(log['decisions'],1)

    def test_logging_cannot_allow_late_write(self):
        env=Env()
        def emit(stage,**_):
            if stage=='tool.input':time.sleep(.03)
        row,log=run_agent(env,lambda *_:[call('x.write')],[],[{'name':'x.write','kind':'write'}],time.monotonic()+.01,emit=emit)
        self.assertEqual(row['tool_calls'],[])
        self.assertEqual(log['termination'],'deadline')

    def test_backend_and_native_prompt(self):
        backend=LocalBackend.__new__(LocalBackend)
        backend._generate=lambda *_:'[]'
        self.assertIsNone(backend.decide([],[],5).protocol_error)
        backend._generation_complete=False
        self.assertIsNotNone(backend.decide([],[],5).protocol_error)
        backend._generation_complete=True
        from types import SimpleNamespace
        backend.tokenizer=SimpleNamespace(eos_token_id=100)
        backend.model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=99))
        self.assertEqual(backend._assistant_stop_ids(),[99,100])
        messages=[{'role':'system','content':'original context'},{'role':'user','content':'request'}]
        before=copy.deepcopy(messages)
        rendered=apply_policy(messages)
        self.assertEqual(messages,before)
        self.assertIn('original context',rendered[0]['content'])
        self.assertIn('[]',rendered[0]['content'])
        self.assertEqual(rendered[1],messages[1])

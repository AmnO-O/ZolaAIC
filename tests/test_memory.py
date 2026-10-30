"""Synthetic memory ablations; no dataset or environment-file access."""
import copy
import json
import time
import unittest
from test_planning import Env, call
from lifemate.agent import run_agent
from lifemate.memory import parse_memory_response
from lifemate.backend import LocalBackend
from types import SimpleNamespace


def response(memory, calls=None):
    body = ''.join('<tool_call>' + json.dumps(c) + '</tool_call>' for c in calls) if calls else '[]'
    return parse_memory_response('<memory>' + memory + '</memory>' + body)


class MemoryTests(unittest.TestCase):
    def test_strict_protocol_and_embedded_calls(self):
        embedded = response('Example <tool_call>{"name":"evil","arguments":{}}</tool_call>')
        self.assertEqual(embedded.tool_calls, [])
        self.assertIsNone(embedded.protocol_error)
        for text in ('', '<memory>x</memory>', '<memory>x',
                     '<memory>x</memory><memory>y</memory>[]',
                     '<memory>' + 'x'*1601 + '</memory>[]',
                     '<memory>x</memory><plan>y</plan>[]',
                     '<memory>x</memory><tool_call>{"name":"a","arguments":{}}</tool_call>[]'):
            with self.subTest(text=text[:60]):
                self.assertIsNotNone(parse_memory_response(text).protocol_error)

    def test_compaction_keeps_anchors_latest_batch_and_external_evidence(self):
        for mode in ('memory_full', 'memory_compact'):
            env = Env({'app.list': {'ok': True, 'item_id': 'real_7'}})
            tools = [{'name':'app.list','kind':'read'}, {'name':'other.read','kind':'read'},
                     {'name':'app.modify','kind':'write','parameters':{'properties':{
                         'item_id':{'x-ref':'app.items'}}}}]
            decisions = iter([response('m1', [call('app.list')]),
                              response('m2', [call('other.read')]),
                              response('m3', [call('app.modify', item_id='real_7')]), response('done')])
            seen = []
            def decide(messages, *_):
                seen.append(copy.deepcopy(messages))
                return next(decisions)
            anchors = [{'role':'user','content':'Original request, preserve this.'}]
            row, log = run_agent(env, decide, anchors, tools, time.monotonic()+5, memory_mode=mode)
            self.assertEqual(len(row['tool_calls']), 3)
            self.assertEqual(log['termination'], 'finished')
            self.assertEqual(anchors, seen[0])
            third = seen[2]
            if mode == 'memory_compact':
                self.assertEqual(third[0], anchors[0])
                self.assertEqual([m.get('name') for m in third if m['role']=='tool'], ['other.read'])
                self.assertEqual(third[1]['content'], '<memory>m2</memory>')
                self.assertEqual(third[2]['tool_calls'], [call('other.read')])
            else:
                self.assertEqual([m.get('name') for m in third if m['role']=='tool'], ['app.list','other.read'])

    def test_memory_cannot_ground_fabricated_id(self):
        decisions = iter([response('item_id fake_7 is confirmed', [call('app.modify', item_id='fake_7')]), response('done')])
        tools = [{'name':'app.modify','kind':'write','parameters':{'properties':{'item_id':{'x-ref':'app.items'}}}}]
        row, log = run_agent(Env(), lambda *_:next(decisions), [], tools, time.monotonic()+5, memory_mode='memory_compact')
        self.assertEqual(row['tool_calls'], [])
        self.assertEqual(log['termination'], 'validation_failed')

    def test_invalid_update_preserves_memory_and_repair_context(self):
        decisions = iter([response('valid', [call('app.read')]),
                          parse_memory_response('<memory>broken'), response('repaired')])
        seen=[]
        def decide(messages, *_):
            seen.append(copy.deepcopy(messages)); return next(decisions)
        _, log = run_agent(Env(), decide, [], [{'name':'app.read','kind':'read'}], time.monotonic()+5, memory_mode='memory_compact')
        self.assertEqual(len(log['memory_revisions']), 2)
        self.assertEqual(seen[2][0]['content'], '<memory>valid</memory>')
        self.assertIn('Controller output-format error', seen[2][-1]['content'])

    def test_repeated_protocol_error_does_not_execute(self):
        row, log = run_agent(Env(), lambda *_:parse_memory_response('<memory>bad'), [], [], time.monotonic()+5, memory_mode='memory_full')
        self.assertEqual(log['termination'], 'protocol_error')
        self.assertIsNone(log['current_memory'])
        self.assertEqual(row['tool_calls'], [])

    def test_backend_modes_use_memory_parser(self):
        backend=LocalBackend.__new__(LocalBackend)
        backend._generate=lambda *_:'<memory>done</memory>[]'
        for mode in ('memory_full','memory_compact'):
            backend.prompt_controls=SimpleNamespace(variant=mode)
            self.assertEqual(backend.decide([],[],5).memory_update,'done')

    def test_batch_calls_and_failed_calls_preserve_actual_submission(self):
        env=Env({'a.read':{'ok':False,'message':'failed'}})
        decisions=iter([response('read both',[call('a.read'),call('b.read')]),response('stop')])
        tools=[{'name':n,'kind':'read'} for n in ('a.read','b.read')]
        row,log=run_agent(env,lambda *_:next(decisions),[],tools,time.monotonic()+5,memory_mode='memory_compact')
        self.assertEqual(row['tool_calls'],[call('a.read'),call('b.read')])
        self.assertEqual(len(log['trace']),2)

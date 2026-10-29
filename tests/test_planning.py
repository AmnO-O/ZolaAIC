"""Synthetic planning, evidence boundaries, and execution failure regressions."""
import copy
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.agent import run_agent
from lifemate.backend import LocalBackend
from lifemate.execution import ExecutionLedger, outcome
from lifemate.messages import chat_messages
from lifemate.planning import parse_planning_response
from lifemate.prompts import PromptControls


def call(name, **arguments):
    return {'name': name, 'arguments': arguments}


def response(plan=None, calls=None):
    text = '<plan>' + plan + '</plan>' if plan else ''
    text += ''.join('<tool_call>' + json.dumps(c) + '</tool_call>' for c in calls) if calls else '[]'
    return parse_planning_response(text)


class Env:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def validate(self, name, arguments):
        return {'ok': True}

    def call(self, name, arguments):
        self.calls.append(call(name, **copy.deepcopy(arguments)))
        result = self.results.get(name, {'ok': True})
        if isinstance(result, Exception):
            raise result
        return result

    def get_tool_calls(self):
        return copy.deepcopy(self.calls)

    def get_state_hash(self):
        return 'synthetic'


class PlanningTests(unittest.TestCase):
    def test_native_calls_and_brief_plan_are_parsed(self):
        decision = response('Read the condition; act only in the selected branch.', [call('condition.read')])
        self.assertIsNone(decision.protocol_error)
        self.assertEqual(decision.tool_calls, [call('condition.read')])
        self.assertIn('selected branch', decision.plan_update)

    def test_plan_text_cannot_execute_embedded_tool_calls(self):
        text = '<plan>Example <tool_call>{"name":"evil.write","arguments":{}}</tool_call></plan>[]'
        decision = parse_planning_response(text)
        self.assertEqual(decision.tool_calls, [])
        self.assertIsNone(decision.protocol_error)

    def test_plan_only_malformed_or_oversized_output_is_not_completion(self):
        for text in ('<plan>Read then act.</plan>', '<plan>unfinished',
                     '<plan>' + 'x' * 601 + '</plan>[]',
                     '<tool_call>[]</tool_call>',
                     '<plan>One</plan><plan>Two</plan>[]',
                     '<tool_call>{"name":"a","arguments":{}}</tool_call>[]'):
            with self.subTest(text=text[:50]):
                self.assertIsNotNone(parse_planning_response(text).protocol_error)

    def test_condition_can_cancel_branch_but_preserve_independent_action(self):
        env = Env({'condition.read': {'ok': True, 'enabled': False}})
        tools = [{'name': 'condition.read', 'kind': 'read'}, {'name': 'memo.create', 'kind': 'write'}]
        decisions = iter([
            response('Check condition; if true perform its action. Independently save the memo.', [call('condition.read')]),
            response('The conditional action is unnecessary; save the independent memo.', [call('memo.create', text='hello')]),
            response('No remaining supported work.'),
        ])
        seen = []
        def decide(messages, *args):
            seen.append(copy.deepcopy(messages))
            return next(decisions)
        row, log = run_agent(env, decide, [{'role': 'user', 'content': 'Conditional action and independent memo'}],
                             tools, time.monotonic() + 5, planning=True)
        self.assertEqual([c['name'] for c in row['tool_calls']], ['condition.read', 'memo.create'])
        self.assertEqual(log['termination'], 'finished')
        self.assertEqual(len(log['plan_revisions']), 3)
        self.assertIn('<plan>', seen[1][1]['content'])
        self.assertEqual(set(row), {'tool_calls', 'state_hash'})
        json.dumps(log)  # Diagnostic decision/ledger data is serializable.

    def test_plan_does_not_ground_invented_id(self):
        decisions = iter([response('Use entity fake_17.', [call('app.modify', item_id='fake_17')]), response()])
        tools = [{'name': 'app.modify', 'kind': 'write', 'parameters': {
            'properties': {'item_id': {'x-ref': 'app.items'}}}}]
        row, log = run_agent(Env(), lambda *_: next(decisions), [], tools, time.monotonic() + 5, planning=True)
        self.assertEqual(row['tool_calls'], [])
        self.assertEqual(log['termination'], 'validation_failed')

    def test_plan_only_gets_one_protocol_correction_not_a_successful_stop(self):
        decision = parse_planning_response('<plan>Read then act.</plan>')
        row, log = run_agent(Env(), lambda *_: decision, [], [], time.monotonic() + 5, planning=True)
        self.assertEqual(log['termination'], 'protocol_error')
        self.assertEqual(log['decisions'], 2)
        self.assertEqual(row['tool_calls'], [])

    def test_repeated_reads_stop_without_progress(self):
        env = Env()
        row, log = run_agent(env, lambda *_: [call('app.read')], [],
                             [{'name': 'app.read', 'kind': 'read'}], time.monotonic() + 5, max_turns=16)
        self.assertEqual(log['termination'], 'no_progress')
        self.assertEqual(len(row['tool_calls']), 3)

    def test_plan_revisions_do_not_reset_stagnation(self):
        env = Env(); counter = [0]
        def decide(*args):
            counter[0] += 1
            return response(f'Remaining read, revision {counter[0]}', [call('app.read')])
        _, log = run_agent(env, decide, [], [{'name': 'app.read', 'kind': 'read'}],
                           time.monotonic() + 5, planning=True)
        self.assertEqual(log['termination'], 'no_progress')
        self.assertEqual(log['decisions'], 3)

    def test_successful_write_is_not_executed_twice(self):
        env = Env(); decisions = iter([[call('memo.create', text='x')]] * 2 + [[]])
        row, log = run_agent(env, lambda *_: next(decisions), [],
                             [{'name': 'memo.create', 'kind': 'write'}], time.monotonic() + 5)
        self.assertEqual(len(row['tool_calls']), 1)
        self.assertEqual(log['validation_rejections'], 1)

    def test_uncertain_write_stops_batch_and_retains_actual_call(self):
        for result in ({'ok': True, 'done': False}, RuntimeError('after possible mutation')):
            with self.subTest(result=str(result)):
                env = Env({'memo.create': result})
                row, log = run_agent(env, lambda *_: [call('memo.create', text='x'), call('other.write')], [],
                    [{'name': 'memo.create', 'kind': 'write'}, {'name': 'other.write', 'kind': 'write'}],
                    time.monotonic() + 5)
                self.assertEqual([c['name'] for c in row['tool_calls']], ['memo.create'])
                self.assertEqual(log['termination'], 'uncertain_outcome')
                self.assertEqual(log['execution_ledger'][0]['status'], 'uncertain')

    def test_new_creation_cannot_clear_failure_of_another_creation(self):
        ledger = ExecutionLedger()
        ledger.observe(call('memo.create', text='first'), {'ok': False}, 'write')
        ledger.observe(call('memo.create', text='second'), {'ok': True}, 'write')
        self.assertTrue(ledger.failed_targets)
        self.assertEqual(outcome({'ok': False}), 'failed')
        self.assertEqual(outcome({'message': 'maybe'}), 'uncertain')

    def test_backend_returns_structured_decision_only_in_planning_mode(self):
        from types import SimpleNamespace
        backend = LocalBackend.__new__(LocalBackend)
        backend.prompt_controls = SimpleNamespace(variant='planning')
        backend._generate = lambda *_: '<plan>Read</plan><tool_call>{"name":"app.read","arguments":{}}</tool_call>'
        decision = backend.decide([], [], 5)
        self.assertEqual(decision.plan_update, 'Read')
        backend.prompt_controls.variant = 'clean'
        self.assertEqual(backend.decide([], [], 5), [call('app.read')])

    def test_native_history_wrapper_and_plan_are_preserved(self):
        messages = [{'role': 'assistant', 'content': '<plan>Pending</plan>',
                     'tool_calls': [{'type': 'function', 'function': call('app.read')}]}]
        serialized = chat_messages(messages)
        self.assertEqual(serialized[0]['content'], '<plan>Pending</plan>')
        self.assertEqual(serialized[0]['tool_calls'][0]['function'], call('app.read'))

    def test_planning_instructions_are_opt_in_and_general(self):
        with patch.dict('os.environ', {'LIFEMATE_PROMPT_VARIANT': 'planning'}):
            planning = PromptControls('system')
            self.assertIn('ACTION PLANNING MODE', planning.instruction)
            with self.assertRaises(ValueError):
                PromptControls('baseline')
        with patch.dict('os.environ', {'LIFEMATE_PROMPT_VARIANT': 'clean'}):
            self.assertNotIn('ACTION PLANNING MODE', PromptControls('system').instruction)


if __name__ == '__main__':
    unittest.main()

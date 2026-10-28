import sys
import json
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.agent import run_agent
from lifemate.input_adapter import SplitInput


class FakeEnv:
    def __init__(self):
        self.calls = []

    def validate(self, name, args):
        return {'ok': name != 'bad', 'error': 'invalid'}

    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': args})
        return {'ok': True}

    def get_tool_calls(self):
        return self.calls.copy()

    def get_state_hash(self):
        return 'fake-hash'


TOOLS = [{'name': 'weather.get_forecast', 'kind': 'read'},
         {'name': 'calendar.list_events', 'kind': 'read'},
         {'name': 'calendar.update_event', 'kind': 'write'},
         {'name': 'sms.send', 'kind': 'write'},
         {'name': 'sys.ask_missing_info', 'kind': 'meta'}]


def read():
    return [{'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-29'}}]


def multi_read():
    return [{'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-29'}},
            {'name': 'calendar.list_events', 'arguments': {}}]


def write():
    return [{'name': 'calendar.update_event', 'arguments': {'event_id': 'evt_01'}}]


def multi_write():
    return [{'name': 'calendar.update_event', 'arguments': {'event_id': 'evt_01'}},
            {'name': 'sms.send', 'arguments': {'contact_id': 'c_08', 'content': 'ok'}}]


class PipelineTests(unittest.TestCase):
    def test_validation_repair_does_not_execute_invalid_call(self):
        env = FakeEnv()
        decisions = iter([[{'name': 'bad', 'arguments': {}}],
                          [{'name': 'valid', 'arguments': {}}], []])
        row, log = run_agent(env, lambda *_: next(decisions), [], [], time.monotonic()+5)
        self.assertEqual([c['name'] for c in row['tool_calls']], ['valid'])
        self.assertEqual(log['termination'], 'finished')

    def test_deadline_executes_nothing(self):
        row, log = run_agent(FakeEnv(), lambda *_: self.fail('backend called'), [], [], 0)
        self.assertEqual(row['tool_calls'], [])
        self.assertEqual(log['termination'], 'deadline')

    def test_continue_after_read_reminds_when_empty_after_read(self):
        env = FakeEnv()
        decisions = iter([read(), [], write(), []])
        row, log = run_agent(env, lambda *_: next(decisions), [], TOOLS, time.monotonic() + 5, continue_after_read=True)
        self.assertEqual([c['name'] for c in row['tool_calls']], ['weather.get_forecast', 'calendar.update_event'])
        self.assertTrue(log['continued_after_read'])
        self.assertEqual(log['termination'], 'finished')

    def test_batched_reads_execute_and_count_as_read_executed(self):
        """Both reads in a multi-read batch should execute, and write follows on next turn."""
        env = FakeEnv()
        decisions = iter([multi_read(), multi_write(), []])
        row, log = run_agent(env, lambda *_: next(decisions), [], TOOLS, time.monotonic() + 5)
        names = [c['name'] for c in row['tool_calls']]
        self.assertIn('weather.get_forecast', names)
        self.assertIn('calendar.list_events', names)
        self.assertIn('calendar.update_event', names)
        self.assertIn('sms.send', names)
        self.assertEqual(log['termination'], 'finished')
        # No spurious reminder when write did execute
        self.assertFalse(log['continued_after_read'])

    def test_batched_reads_trigger_remind_if_no_write_follows(self):
        """Multi-read batch with no subsequent write should trigger the continue reminder."""
        env = FakeEnv()
        decisions = iter([multi_read(), [], write(), []])
        row, log = run_agent(env, lambda *_: next(decisions), [], TOOLS, time.monotonic() + 5, continue_after_read=True)
        self.assertTrue(log['continued_after_read'])
        self.assertIn('calendar.update_event', [c['name'] for c in row['tool_calls']])

    def test_clarification_stops_remaining_batch(self):
        env=FakeEnv()
        row,log=run_agent(env,lambda *_:[{'name':'sys.ask_missing_info','arguments':{'tool_name':'a.b','params':['id']}},
                                       {'name':'must_not_execute','arguments':{}}],[],[],time.monotonic()+5)
        self.assertEqual([c['name'] for c in row['tool_calls']],['sys.ask_missing_info'])
        self.assertEqual(log['termination'],'clarification')

    def test_train_labels_and_transcripts_are_removed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as temporary:
            root = Path(temporary)
            (root / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': [
                {'name': 'sys.ask_missing_info'}, {'name': 'uninstalled.action'}]}]}), encoding='utf-8')
            (root / 'train.jsonl').write_text(json.dumps({
                'id': 'synthetic_1', 'audio': 'audio/example.wav',
                'context': {'installed_apps': {}}, 'label': {'synthetic': True},
                'audio_text': 'REFERENCE_ONLY',
                'history': [{'role': 'user', 'audio': 'audio/history.wav', 'audio_text': 'HISTORY_REFERENCE'}]
            }), encoding='utf-8')
            split = SplitInput(root)
            item = next(split.records())
            self.assertNotIn('label', item)
            self.assertNotIn('audio_text', item)
            self.assertTrue(all('audio_text' not in turn for turn in item['history']))
            self.assertEqual([t['name'] for t in split.tools(item)], ['sys.ask_missing_info'])


if __name__ == '__main__':
    unittest.main()

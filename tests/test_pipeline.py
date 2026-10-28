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

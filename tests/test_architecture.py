"""Boundary regressions using synthetic schemas and messages only."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.catalog import lookup_for
from lifemate.grounding import Evidence
from lifemate.agent import run_agent
from lifemate.values import annotate_spoken_phones
import time
import json


class BoundaryTests(unittest.TestCase):
    def test_phone_annotation_preserves_one_utterance(self):
        raw = 'Nạp cho số không chín một hai ba bốn năm sáu bảy tám'
        annotated = annotate_spoken_phones(raw)
        self.assertTrue(annotated.startswith(raw + '\n'))
        self.assertIn('recognized_phone: 0912345678', annotated)
        self.assertEqual(annotate_spoken_phones('không chín một hai'), 'không chín một hai')

    def test_assistant_and_failed_tool_text_cannot_ground_spoken_phone(self):
        spoken = 'không chín một hai ba bốn năm sáu bảy tám'
        for message in (
            {'role': 'assistant', 'content': spoken},
            {'role': 'tool', 'name': 'contacts.search',
             'content': {'ok': False, 'message': spoken}},
        ):
            evidence = Evidence([message])
            self.assertNotIn('0912345678', evidence.phones)
            self.assertIsNotNone(evidence.check(
                {'name': 'sms.send'}, {'phone': '0912345678'}))

    def test_user_spoken_phone_and_successful_observation_are_trusted(self):
        for message in (
            {'role': 'user', 'content': 'không chín một hai ba bốn năm sáu bảy tám'},
            {'role': 'tool', 'name': 'contacts.search',
             'content': {'ok': True, 'contacts': [{'phone': '0912345678'}]}},
        ):
            self.assertIsNone(Evidence([message]).check(
                {'name': 'sms.send'}, {'phone': '0912345678'}))

    def test_explicit_catalog_description_supports_unseen_app(self):
        available = {'hotel.find_rooms': {'kind': 'read'}}
        self.assertEqual(lookup_for('hotel.rooms',
            {'description': 'ID lấy từ hotel.find_rooms'}, available), 'hotel.find_rooms')
        self.assertIsNone(lookup_for('hotel.rooms',
            {'description': 'ID lấy từ hotel.find_rooms'}, {}))

    def test_ambiguous_or_unrelated_reads_are_not_guessed(self):
        available = {'hotel.find_rooms': {'kind': 'read'},
                     'hotel.list_rooms': {'kind': 'read'}}
        self.assertIsNone(lookup_for('hotel.rooms', {}, available))
        self.assertIsNone(lookup_for('hotel.rooms', {'description':
            'hotel.find_rooms or hotel.list_rooms'}, available))

    def test_empty_allowlist_never_executes_a_proposed_tool(self):
        class Env:
            def validate(self, *args):
                raise AssertionError('Disallowed call reached environment')
            def call(self, *args):
                raise AssertionError('Disallowed call executed')
            def get_tool_calls(self):
                return []
            def get_state_hash(self):
                return 'synthetic'
        decisions = iter([[{'name': 'unknown.act', 'arguments': {}}], []])
        row, log = run_agent(Env(), lambda *_: next(decisions), [], [], time.monotonic() + 5)
        self.assertEqual(row['tool_calls'], [])
        self.assertIn('not permitted', log['trace'][0]['result']['validation_error'])

    def test_clarification_is_structurally_validated(self):
        evidence = Evidence([])
        tools = {'sys.ask_missing_info': {}, 'app.create': {'parameters': {
            'properties': {'content': {'type': 'string'}}}}}
        for target, params in [('unavailable.act', ['content']),
                               ('app.create', ['invented']), ('app.create', [{}])]:
            with self.subTest(target=target, params=params):
                self.assertIsNotNone(evidence.check({'name': 'sys.ask_missing_info'},
                    {'tool_name': target, 'params': params}, tools))

    def test_notebooks_compile_and_use_clean_configuration(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'notebooks').glob('*.ipynb'):
            notebook = json.loads(path.read_text(encoding='utf-8'))
            source = ''
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    code = ''.join(cell['source'])
                    compile(code, f'{path.name}:{index}', 'exec')
                    source += code
            self.assertIn('PROMPT_MODE = "system"', source)
            self.assertIn('PROMPT_VARIANT = "clean"', source)
            self.assertNotIn('TOOL_PRUNING', source)
            self.assertNotIn('b_cond', source)

if __name__ == '__main__':
    unittest.main()

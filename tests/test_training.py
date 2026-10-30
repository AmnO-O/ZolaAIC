"""Causal data boundaries, split isolation and loss masks; no app state access."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.training import trajectory_examples, encode_example, validation_group, split_group


TOOLS = [
    {'name': 'app.find', 'kind': 'read', 'parameters': {'type': 'object', 'properties': {}}},
    {'name': 'app.modify', 'kind': 'write', 'parameters': {'type': 'object', 'properties': {'item_id': {'type': 'string'}}}},
    {'name': 'sys.ask_missing_info', 'kind': 'meta', 'parameters': {'type': 'object', 'properties': {}}},
]


CATALOG = {'apps': [{'id': 'app', 'description': 'Find and modify entries', 'tools': TOOLS}]}


def sample(identifier='train_x', text='Change the item'):
    return {'id': identifier, 'audio_text': text, 'history': [],
            'context': {'installed_apps': {'app': {'tools': ['app.find', 'app.modify']}}},
            'label': {'tool_calls': [
                {'name': 'app.find', 'arguments': {}},
                {'name': 'app.modify', 'arguments': {'item_id': 'UNIQUE_FUTURE_ID'}}],
                'tool_results': [{'ok': True, 'item': {'id': 'UNIQUE_FUTURE_ID'}},
                                 {'ok': True, 'receipt': 'UNIQUE_FUTURE_RECEIPT'}],
                'final_state': {'secret': 'STATE_MUST_NOT_LEAK'}}}


class Tokenizer:
    """Deterministic template stub makes the prefix/target boundary observable."""
    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, **kwargs):
        text = json.dumps(messages[:-1] if not add_generation_prompt else messages, sort_keys=True)
        text += '<assistant>'
        if not add_generation_prompt:
            text += json.dumps(messages[-1], sort_keys=True) + '<eos>'
        return list(text.encode()) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class TrainingTests(unittest.TestCase):
    def test_no_target_or_future_result_in_input(self):
        row = sample(); original = copy.deepcopy(row)
        examples = trajectory_examples(row, TOOLS, CATALOG)
        self.assertEqual(examples[0]['messages'][0]['content']['context']['installed_apps'],
                         {'app': 'Find and modify entries'})
        self.assertEqual(len(examples), 3)
        self.assertNotIn('UNIQUE_FUTURE_ID', str(examples[0]['messages']))
        self.assertIn('UNIQUE_FUTURE_ID', str(examples[1]['messages']))
        self.assertNotIn('UNIQUE_FUTURE_RECEIPT', str(examples[1]['messages']))
        self.assertNotIn('STATE_MUST_NOT_LEAK', str(examples))
        self.assertEqual(row, original)
        self.assertEqual(examples[-1]['target']['content'], '[]')

    def test_empty_and_clarification_trajectories(self):
        row = sample(); row['label'] = {'tool_calls': [], 'tool_results': []}
        self.assertEqual(trajectory_examples(row, TOOLS, CATALOG)[0]['target']['content'], '[]')
        row['label'] = {'tool_calls': [{'name': 'sys.ask_missing_info', 'arguments': {}}],
                        'tool_results': [{'ok': True, 'waiting_for_user': True}]}
        self.assertEqual(len(trajectory_examples(row, TOOLS, CATALOG)), 1)

    def test_unpaired_and_unavailable_gold_is_rejected(self):
        row = sample(); row['label']['tool_results'].pop()
        with self.assertRaises(ValueError): trajectory_examples(row, TOOLS, CATALOG)
        with self.assertRaises(ValueError): trajectory_examples(sample(), TOOLS[:1], CATALOG)

    def test_assistant_only_loss_no_truncation(self):
        example = trajectory_examples(sample(), TOOLS, CATALOG)[0]
        encoded = encode_example(example, Tokenizer())
        self.assertEqual(len(encoded['input_ids']), len(encoded['labels']))
        first = next(i for i, value in enumerate(encoded['labels']) if value != -100)
        self.assertGreater(first, 0)
        self.assertTrue(all(v == -100 for v in encoded['labels'][:first]))
        self.assertEqual(encoded['labels'][first:], encoded['input_ids'][first:])
        self.assertIsNone(encode_example(example, Tokenizer(), max_length=2))

    def test_grouping_and_known_development_exclusion(self):
        self.assertEqual(split_group('Change 123'), split_group('change 456'))
        self.assertEqual(validation_group('change 123'), validation_group('change 456'))

    def test_export_only_train_and_fresh_output(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('prepare_sft_test', root/'scripts/prepare_sft.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as temp:
            directory = Path(temp)/'train'; directory.mkdir()
            (directory/'train_tools_catalog.json').write_text(json.dumps(CATALOG))
            row = sample('known')
            # Select a text whose hash normally places it in validation.
            for i in range(500):
                text = 'different text ' + chr(0x41+i)
                if validation_group(text): break
            row['audio_text'] = text
            (directory/'train.jsonl').write_text(json.dumps(row))
            manifest = module.prepare(directory, Path(temp)/'out', development_ids=['known'])
            self.assertEqual(manifest['train'], ['known'])
            self.assertEqual(manifest['validation'], [])
            self.assertEqual(manifest['rejected'], [])
            with self.assertRaises(FileExistsError): module.prepare(directory, Path(temp)/'out')
            other = Path(temp)/'public_test'; other.mkdir()
            (other/'public_test_tools_catalog.json').write_text(json.dumps({'apps': []}))
            with self.assertRaises(ValueError): module.prepare(other, Path(temp)/'bad')

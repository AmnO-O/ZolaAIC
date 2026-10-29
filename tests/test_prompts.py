"""General instructions preserve facts without inserting demonstrations."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.prompts import INSTRUCTION, PromptControls


class PromptTests(unittest.TestCase):
    def test_context_history_and_tools_are_not_mutated(self):
        messages = [{'role': 'system', 'content': 'SAMPLE_TIME_AND_PROFILE'},
                    {'role': 'user', 'content': 'Yêu cầu hiện tại'},
                    {'role': 'tool', 'name': 'service.read', 'content': 'OBSERVED_ID'}]
        original = copy.deepcopy(messages)
        controls = PromptControls('system')
        output = controls.apply(messages, [{'name': 'service.act'}])
        self.assertEqual(messages, original)
        self.assertEqual(output[1:], original[1:])
        self.assertIn(INSTRUCTION, output[0]['content'])
        self.assertIn('SAMPLE_TIME_AND_PROFILE', output[0]['content'])
        self.assertEqual(output[0]['content'].count(INSTRUCTION), 1)

    def test_no_examples_or_domain_specific_tool_names(self):
        for text in ('Positive example', 'evt_123', 'calendar.update_event',
                     'Sếp Dũng', 'phòng ngủ nhỏ', 'banking.transfer'):
            self.assertNotIn(text, INSTRUCTION)
        self.assertLess(len(INSTRUCTION.split()), 400)

    def test_system_prefix_is_stable_across_observations(self):
        controls = PromptControls('system')
        messages = [{'role': 'system', 'content': 'context'}, {'role': 'user', 'content': 'request'}]
        later = messages + [{'role': 'tool', 'content': 'new observation'}]
        self.assertEqual(controls.apply(messages, [])[0], controls.apply(later, [])[0])

    def test_baseline_is_explicit_noop(self):
        messages = [{'role': 'user', 'content': 'task'}]
        self.assertEqual(PromptControls('baseline').apply(messages, []), messages)

    def test_legacy_mode_fails_instead_of_silently_changing_experiment(self):
        with self.assertRaises(ValueError):
            PromptControls('steerability')


if __name__ == '__main__':
    unittest.main()

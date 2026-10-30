"""General instructions preserve facts without inserting demonstrations."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.prompts import INSTRUCTION, apply_policy


class PromptTests(unittest.TestCase):
    def test_context_history_and_tools_are_not_mutated(self):
        messages = [{'role': 'system', 'content': 'SAMPLE_TIME_AND_PROFILE'},
                    {'role': 'user', 'content': 'Yêu cầu hiện tại'},
                    {'role': 'tool', 'name': 'service.read', 'content': 'OBSERVED_ID'}]
        original = copy.deepcopy(messages)
        output = apply_policy(messages)
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
        messages = [{'role': 'system', 'content': 'context'}, {'role': 'user', 'content': 'request'}]
        later = messages + [{'role': 'tool', 'content': 'new observation'}]
        self.assertEqual(apply_policy(messages)[0], apply_policy(later)[0])


if __name__ == '__main__': unittest.main()

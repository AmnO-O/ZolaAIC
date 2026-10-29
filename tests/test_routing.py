"""Context compaction does not select tools or infer user intent."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.context import clean_context_for_prompt


class ContextTests(unittest.TestCase):
    def test_compaction_preserves_context_and_input(self):
        context = {'current_time': '2030-01-01', 'user_profile': {'name': 'Example'},
                   'installed_apps': {'new_app': {'tools': ['new_app.read']}}}
        original = copy.deepcopy(context)
        cleaned = clean_context_for_prompt(context)
        self.assertEqual(context, original)
        self.assertEqual(cleaned['installed_apps'], ['new_app'])
        self.assertEqual(cleaned['user_profile'], context['user_profile'])
        self.assertEqual(cleaned['current_time'], context['current_time'])

    def test_missing_or_non_mapping_context(self):
        self.assertEqual(clean_context_for_prompt({}), {})
        self.assertIsNone(clean_context_for_prompt(None))


if __name__ == '__main__':
    unittest.main()

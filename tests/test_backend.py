import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.backend import parse_calls, chat_messages
from lifemate.evaluation import call_metrics, token_f1


class BackendTests(unittest.TestCase):
    def test_native_calls_and_string_arguments(self):
        calls = parse_calls('<tool_call>{"name":"a.b","arguments":"{\\"id\\":1}"}</tool_call>')
        self.assertEqual(calls, [{'name':'a.b','arguments':{'id':1}}])

    def test_truncated_call_rejected(self):
        with self.assertRaises(ValueError):
            parse_calls('<tool_call>{"name":"a.b"')

    def test_partial_second_call_does_not_silently_drop(self):
        with self.assertRaises(ValueError):
            parse_calls('<tool_call>{"name":"a.b","arguments":{}}</tool_call><tool_call>{')

    def test_history_results_preserved(self):
        turns = chat_messages([{'role':'assistant','text':'ok','tool_calls':[{'name':'a.b','arguments':{}}], 'tool_results':[{'ok':True}]}])
        self.assertEqual(turns[0]['tool_calls'][0]['function']['name'], 'a.b')
        self.assertEqual(turns[1]['role'], 'tool')

    def test_duplicate_predictions_cannot_match_same_gold(self):
        call = {'name':'a.b','arguments':{}}
        metrics = call_metrics([call,call],[call],{})
        self.assertEqual(metrics['matched'], 1)
        self.assertEqual(metrics['precision'], .5)

    def test_fuzzy_threshold_is_strict(self):
        specs={'a.b':{'parameters':{'properties':{'text':{'x-match':'fuzzy'}}}}}
        p={'name':'a.b','arguments':{'text':'a b c'}}
        g={'name':'a.b','arguments':{'text':'a b c d e f g'}}
        self.assertAlmostEqual(token_f1(p['arguments']['text'],g['arguments']['text']), .6)
        self.assertEqual(call_metrics([p],[g],specs)['matched'], 0)


if __name__ == '__main__': unittest.main()

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.prompts import PromptControls, INSTRUCTION, EXAMPLES
from lifemate.backend import parse_calls


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.controls=PromptControls('steerability')

    def test_context_tools_and_results_preserved_without_mutation(self):
        messages=[{'role':'system','content':'SAMPLE_CONTEXT_TIME'},
                  {'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'contacts.search','arguments':{'query':'Lan'}}}]},
                  {'role':'tool','content':'OBSERVED_ID_FROM_TOOL'}]
        original=copy.deepcopy(messages)
        output=self.controls.apply(messages,[{'name':'notes.create'},{'name':'sys.ask_missing_info'}])
        self.assertEqual(messages,original)
        self.assertEqual(output[1:],original[1:])
        self.assertEqual(sum(m['role']=='system' for m in output),1)
        self.assertIn('SAMPLE_CONTEXT_TIME',output[0]['content'])
        self.assertEqual(output[0]['content'].count(INSTRUCTION),1)
        self.assertIn('Positive example',output[0]['content'])
        self.assertNotIn('"name":"sms.send"',output[0]['content'])

    def test_fixed_examples_stay_identical_across_decisions(self):
        tools=[{'name':'notes.create'},{'name':'sys.ask_missing_info'}]
        first=[{'role':'system','content':'context'},{'role':'user','content':'request'}]
        second=first+[{'role':'tool','content':'new observation'}]
        self.assertEqual(self.controls.apply(first,tools)[0],self.controls.apply(second,tools)[0])

    def test_baseline_is_an_explicit_noop(self):
        messages=[{'role':'user','content':'task'}]
        self.assertEqual(PromptControls('baseline').apply(messages,[]),messages)

    def test_demonstrations_use_valid_native_calls_without_fabricated_ids(self):
        for required,example in EXAMPLES:
            calls=parse_calls(example['behavior'])
            self.assertTrue(calls)
            for call in calls:
                self.assertIn(call['name'],required)
                self.assertNotIn('contact_id',call['arguments'])
                self.assertNotIn('event_id',call['arguments'])


if __name__=='__main__':unittest.main()

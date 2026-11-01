import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.planner_input import PlannerInput


class PlannerInputTests(unittest.TestCase):
    def messages(self):
        return [
            {'role':'system','content':{'context':{'current_time':'2026-11-01',
                'user_profile':{'home_address':'HOME'},'installed_apps':{'a':'DESCRIPTION'},
                'app_descriptions':{'a':'DESCRIPTION'}}}},
            {'role':'user','content':'OLD_REQUEST'},
            {'role':'assistant','text':'already applied','tool_calls':[{'name':'a.write','arguments':{}}],
                'tool_results':[{'ok':True}]},
            {'role':'user','content':'CURRENT_REQUEST'}]

    def test_five_fields_preserve_context_and_separate_prior_actions(self):
        messages=self.messages();before=copy.deepcopy(messages)
        payload=PlannerInput(messages,{'a':'DESCRIPTION'}).payload(messages)
        self.assertEqual(set(payload),{'context','available_apps','prior_conversation',
                                       'current_request','execution_history'})
        self.assertNotIn('installed_apps',payload['context'])
        self.assertNotIn('app_descriptions',payload['context'])
        self.assertEqual(payload['context']['user_profile']['home_address'],'HOME')
        self.assertEqual(payload['prior_conversation'][0]['text'],'OLD_REQUEST')
        self.assertEqual(payload['prior_conversation'][1]['tool_results'],[{'ok':True}])
        self.assertEqual(payload['execution_history'],[])
        self.assertEqual(json.dumps(payload).count('CURRENT_REQUEST'),1)
        self.assertEqual(json.dumps(payload).count('DESCRIPTION'),1)
        self.assertEqual(messages,before)

    def test_controller_feedback_cannot_replace_original_request(self):
        messages=self.messages();builder=PlannerInput(messages,{'a':'DESCRIPTION'})
        feedback={'controller_feedback':'output_rejected','executed':False,'error':'bad output'}
        messages.append({'role':'user','content':json.dumps(feedback)})
        payload=builder.payload(messages)
        self.assertEqual(payload['current_request'],'CURRENT_REQUEST')
        self.assertEqual(payload['execution_history'],[{'type':'controller_feedback','feedback':feedback}])

    def test_batch_pairs_calls_and_preserves_rejected_vs_executed(self):
        messages=self.messages();builder=PlannerInput(messages,{'a':'DESCRIPTION'})
        calls=[{'name':'a.read','arguments':{'q':'one'}},
               {'name':'a.write','arguments':{'id':'invented'}}]
        messages.append({'role':'assistant','tool_calls':calls})
        first={'source_id':'observation_1','source_type':'environment_result','executed':True,
               'execution_status':'succeeded','result':{'items':[1]}}
        second={'source_id':'observation_2','source_type':'controller_feedback','executed':False,
                'execution_status':'rejected','result':{'validation_error':'unobserved ID'}}
        messages.extend([{'role':'tool','name':'a.read','content':first},
                         {'role':'tool','name':'a.write','content':second}])
        events=builder.payload(messages)['execution_history']
        self.assertEqual([e['call'] for e in events],calls)
        self.assertEqual(events[0]['result'],{'items':[1]})
        self.assertTrue(events[0]['executed'])
        self.assertFalse(events[1]['executed'])
        self.assertEqual(events[1]['source_type'],'controller_feedback')

    def test_unobserved_proposal_never_implies_success(self):
        messages=self.messages();builder=PlannerInput(messages,{'a':'DESCRIPTION'})
        call={'name':'a.write','arguments':{}}
        messages.append({'role':'assistant','tool_calls':[call]})
        self.assertEqual(builder.payload(messages)['execution_history'],
                         [{'type':'unobserved_proposal','call':call}])

    def test_payload_copies_cannot_mutate_next_turn(self):
        messages=self.messages();apps={'a':'DESCRIPTION'}
        builder=PlannerInput(messages,apps)
        payload=builder.payload(messages)
        payload['context']['user_profile']['home_address']='MODIFIED'
        payload['available_apps']['a']='MODIFIED'
        self.assertEqual(builder.payload(messages)['context']['user_profile']['home_address'],'HOME')
        self.assertEqual(apps,{'a':'DESCRIPTION'})

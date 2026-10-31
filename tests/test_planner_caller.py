import copy
import json
from pathlib import Path
import sys
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.agent import Decision,run_agent
from lifemate.planner_caller import PlannerCaller,parse_step

TOOLS=[{'name':'a.read','kind':'read','description':'đọc thông tin','parameters':{}}]
CATALOG={'apps':[{'id':'a','description':'ứng dụng thông tin'}]}


class Backend:
    def __init__(self,outputs):self.outputs=iter(outputs);self.stats=[];self.caller_inputs=[];self.planner_inputs=[]
    def _generate(self,messages,*args,**kwargs):
        self.planner_inputs.append(copy.deepcopy(messages));self._generation_complete=True
        self.stats.append({'input_tokens':10});return next(self.outputs)
    def decide(self,messages,tools,seconds):
        self.caller_inputs.append(copy.deepcopy(messages));self.stats.append({'input_tokens':20})
        return Decision([{'name':'a.read','arguments':{}}])


class PlannerCallerTests(unittest.TestCase):
    def test_observation_drives_replanning_and_done_skips_caller(self):
        backend=Backend([json.dumps({'next':'call','goal':'read','query':'đọc thông tin'}),
                         json.dumps({'next':'done','goal':'enough','query':''})])
        controller=PlannerCaller(backend,TOOLS,CATALOG)
        class Env:
            def __init__(self):self.calls=[]
            def validate(self,*args):return {'ok':True}
            def call(self,name,args):self.calls.append({'name':name,'arguments':args});return {'ok':True,'value':'OBSERVED_ONLY'}
            def get_tool_calls(self):return self.calls
            def get_state_hash(self):return 'hash'
        messages=[{'role':'system','content':{'context':{}}},{'role':'user','content':'read'}]
        original=copy.deepcopy(messages)
        pred,trace=run_agent(Env(),controller.decide,messages,TOOLS,time.monotonic()+10)
        self.assertEqual(trace['termination'],'finished');self.assertEqual(len(pred['tool_calls']),1)
        self.assertEqual(len(backend.caller_inputs),1);self.assertEqual(len(backend.planner_inputs),2)
        self.assertIn('OBSERVED_ONLY',str(backend.planner_inputs[1]))
        self.assertEqual(messages,original)
        self.assertIn('unverified_planner_guidance',str(backend.caller_inputs[0]))
        self.assertNotIn('planner_guidance',str(messages))

    def test_malformed_plan_falls_back_without_extra_planning(self):
        backend=Backend(['not JSON']);controller=PlannerCaller(backend,TOOLS,CATALOG)
        controller.decide([{'role':'user','content':'đọc thông tin'}],TOOLS,10)
        self.assertTrue(controller.steps[0]['planner_error'])
        self.assertEqual(len(backend.planner_inputs),1);self.assertEqual(len(backend.caller_inputs),1)
        self.assertNotIn('planner_guidance',str(backend.caller_inputs))

    def test_strict_route_parsing(self):
        for raw in ['{}','{"next":"done","next":"call","goal":"x","query":"x"}',
                    '{"next":"call","goal":"x","query":""}',
                    '{"next":"unknown","goal":"x","query":"x"}']:
            with self.assertRaises(ValueError):parse_step(raw)


if __name__=='__main__':unittest.main()

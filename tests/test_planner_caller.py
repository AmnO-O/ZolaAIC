import copy
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.agent import Decision,run_agent
from lifemate.planner_caller import PlannerCaller,parse_step

TOOLS=[{'name':'a.read','kind':'read','description':'đọc thông tin','parameters':{}}]
CATALOG={'apps':[{'id':'a','description':'ứng dụng thông tin'}]}


class Backend:
    def __init__(self,outputs):self.outputs=iter(outputs);self.stats=[];self.caller_inputs=[];self.planner_inputs=[];self.generation_options=[]
    def _generate(self,messages,*args,**kwargs):
        self.planner_inputs.append(copy.deepcopy(messages));self.generation_options.append(kwargs);self._generation_complete=True
        self.stats.append({'input_tokens':10});return next(self.outputs)
    def decide(self,messages,tools,seconds):
        self.caller_inputs.append(copy.deepcopy(messages));self.stats.append({'input_tokens':20})
        return Decision([{'name':'a.read','arguments':{}}])


class PlannerCallerTests(unittest.TestCase):
    def test_observation_drives_replanning_and_done_skips_caller(self):
        backend=Backend([json.dumps({'next':'call','goal':'read','query':'đọc thông tin'}),
                         json.dumps({'next':'done','goal':'enough','query':''})])
        controller=PlannerCaller(backend,TOOLS,CATALOG,routing='bm25')
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
        backend=Backend(['not JSON']);controller=PlannerCaller(backend,TOOLS,CATALOG,routing='bm25')
        controller.decide([{'role':'user','content':'đọc thông tin'}],TOOLS,10)
        self.assertTrue(controller.steps[0]['planner_error'])
        self.assertEqual(len(backend.planner_inputs),1);self.assertEqual(len(backend.caller_inputs),1)
        self.assertNotIn('planner_guidance',str(backend.caller_inputs))

    def test_strict_route_parsing(self):
        for raw in ['{}','{"next":"done","next":"call","goal":"x","query":"x"}',
                    '{"next":"call","goal":"x","query":""}',
                    '{"next":"unknown","goal":"x","query":"x"}']:
            with self.assertRaises(ValueError):parse_step(raw)

    def test_app_route_selects_whole_permitted_group_and_keeps_history(self):
        tools=TOOLS+[
            {'name':'a.write','kind':'write','parameters':{}},
            {'name':'b.read','kind':'read','parameters':{}},
            {'name':'sys.ask_missing_info','kind':'meta','parameters':{}}]
        backend=Backend([json.dumps({'next':'call','goal':'read app a','apps':['a'],'kind':'read'})])
        selected=[]
        original_decide=backend.decide
        def decide(messages,visible,seconds):
            selected.extend(t['name'] for t in visible)
            return original_decide(messages,visible,seconds)
        backend.decide=decide
        controller=PlannerCaller(backend,tools,CATALOG,routing='apps')
        messages=[{'role':'system','content':{'context':{'marker':'CONTEXT'}}},
            {'role':'user','content':'read'},
            {'role':'tool','content':{'result':'REAL_OBSERVATION'}}]
        original=copy.deepcopy(messages)
        controller.decide(messages,tools,10)
        self.assertEqual(set(selected),{'a.read','a.write','sys.ask_missing_info'})
        self.assertEqual(messages,original)
        self.assertIn('REAL_OBSERVATION',str(backend.caller_inputs))
        self.assertIn('CONTEXT',str(backend.caller_inputs))
        self.assertEqual(controller.steps[0]['routing']['mode'],'apps')
        self.assertEqual(backend.caller_inputs[0][0]['content']['planner_guidance']['kind'],'read')
        self.assertEqual(backend.generation_options[0]['max_tokens'],controller.planner_max_tokens)

    def test_invalid_app_plan_falls_back_and_cannot_expand_allowlist(self):
        for apps in [['unknown'],['a.read'],['a','a'],[],[1]]:
            backend=Backend([json.dumps({'next':'call','goal':'read','apps':apps,'kind':'read'})])
            controller=PlannerCaller(backend,TOOLS,CATALOG,routing='apps')
            controller.decide([{'role':'user','content':'đọc thông tin'}],TOOLS,10)
            self.assertTrue(controller.steps[0]['planner_error'])
            self.assertEqual(controller.steps[0]['routing']['mode'],'bm25_fallback')
            self.assertNotIn('planner_guidance',str(backend.caller_inputs))
        with self.assertRaises(ValueError):
            parse_step('{"next":"done","goal":"done","apps":["a"]}',{'a'})

    def test_kind_contract_and_app_done_skips_caller(self):
        for kind in ('read','write','mixed','meta'):
            step={'next':'call','goal':'task','apps':['a'],'kind':kind}
            self.assertEqual(parse_step(json.dumps(step),{'a'})['kind'],kind)
        for next_value,apps,kind in [('call',['a'],'none'),('call',['a'],'unknown'),
                                     ('done',[],'read'),('call',['a'],['read'])]:
            with self.assertRaises(ValueError):
                parse_step(json.dumps({'next':next_value,'goal':'task','apps':apps,'kind':kind}),{'a'})
        backend=Backend([json.dumps({'next':'done','goal':'complete','apps':[],'kind':'none'})])
        controller=PlannerCaller(backend,TOOLS,CATALOG)
        result=controller.decide([{'role':'user','content':'read'}],TOOLS,10)
        self.assertEqual(result.raw_text,'[]')
        self.assertFalse(backend.caller_inputs)

    def test_planner_token_budget_configuration(self):
        with patch.dict('os.environ',{'LIFEMATE_PLANNER_MAX_NEW_TOKENS':'768'}):
            backend=Backend([json.dumps({'next':'done','goal':'complete','apps':[],'kind':'none'})])
            PlannerCaller(backend,TOOLS,CATALOG).decide([{'role':'user','content':'read'}],TOOLS,10)
            self.assertEqual(backend.generation_options[0]['max_tokens'],768)
        for value in ('0','16385','bad'):
            with patch.dict('os.environ',{'LIFEMATE_PLANNER_MAX_NEW_TOKENS':value}),self.assertRaises(ValueError):
                PlannerCaller(Backend([]),TOOLS,CATALOG)


if __name__=='__main__':unittest.main()

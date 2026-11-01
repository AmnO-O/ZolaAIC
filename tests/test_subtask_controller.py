import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from subtask_controller import SubtaskController
from lifemate.agent import Decision,run_agent
from support import Env,call


class SubtaskTests(unittest.TestCase):
    def test_empty_local_caller_replans_without_ending_request(self):
        class Backend:
            def __init__(self):
                self.stats=[];self.planners=[];self.callers=0
            def _generate(self,messages,*args,**kwargs):
                self.planners.append(json.loads(messages[1]['content']))
                self._generation_complete=True
                return json.dumps({'next':'call','goal':'read','apps':['x'],'kind':'read'}
                    if len(self.planners)==1 else {'next':'done','goal':'enough','apps':[],'kind':'none'})
            def decide(self,*args):
                self.callers+=1
                return Decision([call('x.read')]) if self.callers==1 else Decision(raw_text='[]')
        backend=Backend();tools=[{'name':'x.read','kind':'read','parameters':{}}]
        controller=SubtaskController(backend,tools,{'apps':[{'id':'x','description':'read'}]},caller_turns=2)
        prediction,trace=run_agent(Env({'x.read':{'ok':True,'items':[]}}),controller.decide,
            [{'role':'user','content':'read'}],tools,time.monotonic()+10)
        self.assertEqual(prediction['tool_calls'],[call('x.read')])
        self.assertEqual(len(backend.planners),2)
        self.assertEqual(backend.callers,2)
        receipt=backend.planners[1]['subtask_result']
        self.assertTrue(receipt['caller_returned_empty'])
        self.assertEqual(receipt['observations'][0]['result'],{'ok':True,'items':[]})
        self.assertEqual(receipt['goal_completion'],'not_verified')
        self.assertEqual(trace['termination'],'finished')

    def test_receipt_preserves_rejected_call_as_nonexecution(self):
        class Backend:
            def __init__(self):self.stats=[];self.payloads=[]
            def _generate(self,messages,*args,**kwargs):
                self.payloads.append(json.loads(messages[1]['content']))
                self._generation_complete=True
                return json.dumps({'next':'call','goal':'write','apps':['x'],'kind':'write'}
                    if len(self.payloads)==1 else {'next':'done','goal':'stop','apps':[],'kind':'none'})
            def decide(self,*args):return Decision([call('x.write',item_id='invented')])
        backend=Backend();tools=[{'name':'x.write','kind':'write','parameters':{'properties':{
            'item_id':{'type':'string','x-ref':'x.items'}}}}]
        controller=SubtaskController(backend,tools,{'apps':[{'id':'x'}]},caller_turns=1)
        prediction,_=run_agent(Env(),controller.decide,[{'role':'user','content':'write'}],tools,time.monotonic()+10)
        self.assertEqual(prediction['tool_calls'],[])
        receipt=backend.payloads[1]['subtask_result']
        self.assertEqual(receipt['status'],'blocked')
        self.assertFalse(receipt['observations'][0]['executed'])
        self.assertEqual(receipt['observations'][0]['source_type'],'controller_feedback')

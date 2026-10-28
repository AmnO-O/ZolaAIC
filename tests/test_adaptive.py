import copy
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.adaptive import run_adaptive, validate_plan, WorkingMemory
from lifemate.routing import ToolRouter, compact_tools


def spec(name,kind='read',props=None,description=''):
    return {'name':name,'kind':kind,'description':description,'parameters':{
        'type':'object','properties':props or {},'required':list(props or {})}}


TOOLS = [spec('tasks.search'),spec('tasks.move','write',{'task_id':{'type':'string','x-ref':'tasks.items'}}),
         spec('tasks.cancel','write',{'task_id':{'type':'string','x-ref':'tasks.items'}}),
         spec('people.search'),spec('messages.send','write',{'contact_id':{'type':'string','x-ref':'people.contacts'}}),
         spec('sys.ask_missing_info','meta')]
MESSAGES = [{'role':'system','content':{'context':{'installed_apps':{'tasks':{}}}}},
            {'role':'user','content':'Nếu cần, dời việc và báo người nhận.'}]
PLAN = {'apps':['tasks','people','messages'], 'actions':[
    {'goal':'move task','tool':'tasks.move','conditional':True},
    {'goal':'notify recipient','tool':'messages.send','conditional':True}], 'unknown':[]}


def call(name,**args):return {'name':name,'arguments':args}
def finish(status='done',evidence=None):return [call('agent.finish',status=status,evidence=evidence or [])]


class Env:
    def __init__(self): self.calls=[]
    def validate(self,name,args):
        assert not name.startswith('agent.')
        return {'ok':True}
    def call(self,name,args):
        self.calls.append(call(name,**args))
        if name=='tasks.search': return {'ok':True,'items':[{'id':'task_a'}]}
        if name=='people.search': return {'ok':True,'contacts':[{'id':'person_b'}]}
        return {'ok':True,'done':True}
    def get_tool_calls(self):return self.calls.copy()
    def get_state_hash(self):return 'synthetic-hash'


class Backend:
    def __init__(self,decisions,plan=PLAN):self.decisions=iter(decisions);self.proposal=plan;self.seen=[]
    def plan(self,*args,**kwargs):
        if isinstance(self.proposal,Exception):raise self.proposal
        return self.proposal
    def decide(self,messages,tools,remaining):
        self.seen.append(copy.deepcopy((messages,tools)))
        return next(self.decisions)


class AdaptiveTests(unittest.TestCase):
    def run_case(self,backend,**kwargs):
        env=Env();events=[]
        row,trace=run_adaptive(env,backend,MESSAGES,TOOLS,time.monotonic()+30,
                              emit=lambda stage,**payload:events.append((stage,payload)),**kwargs)
        return row,trace,events

    def test_plan_execution_and_completion_controls_never_reach_environment(self):
        backend=Backend([[call('tasks.search'),call('people.search')],
                         [call('tasks.move',task_id='task_a'),call('messages.send',contact_id='person_b')],finish()])
        original=copy.deepcopy(MESSAGES)
        row,trace,events=self.run_case(backend)
        self.assertEqual(trace['termination'],'done')
        self.assertEqual(len(row['tool_calls']),4)
        self.assertEqual(trace['pending_plan_actions'],[])
        self.assertEqual(MESSAGES,original)
        self.assertTrue(any(s=='memory.updated' for s,_ in events))

    def test_invented_id_and_wrong_operation_are_rejected_before_execution(self):
        backend=Backend([[call('tasks.move',task_id='invented')],[call('tasks.search')],
                         [call('tasks.cancel',task_id='task_a')],
                         [call('sys.ask_missing_info')]])
        row,trace,_=self.run_case(backend)
        self.assertEqual([c['name'] for c in row['tool_calls']],['tasks.search','sys.ask_missing_info'])
        self.assertEqual(trace['termination'],'clarification')

    def test_premature_done_is_rejected_and_can_continue(self):
        backend=Backend([finish(),[call('tasks.search'),call('people.search')],
            [call('tasks.move',task_id='task_a'),call('messages.send',contact_id='person_b')],finish()])
        row,trace,events=self.run_case(backend)
        self.assertEqual(trace['termination'],'done')
        self.assertTrue(any(s=='controller.feedback' for s,_ in events))

    def test_invalid_or_timed_out_planner_falls_back_without_app_calls(self):
        for proposal in ({'apps':['uninstalled']},TimeoutError('budget')):
            backend=Backend([finish('unsupported')],proposal)
            row,trace,events=self.run_case(backend,selection='all')
            self.assertEqual(row['tool_calls'],[])
            self.assertIsNone(trace['plan'])
            self.assertTrue(any(s=='plan.fallback' for s,_ in events))

    def test_unavailable_tool_and_unstructured_prose_do_not_execute(self):
        backend=Backend([[],[call('not_installed.erase')],finish('unsupported')],None)
        row,trace,_=self.run_case(backend,planning='off',selection='all')
        self.assertEqual(row['tool_calls'],[])
        self.assertEqual(trace['termination'],'unsupported')

    def test_plan_can_be_revised_once_from_new_information(self):
        revised={'apps':['tasks'],'actions':[{'goal':'cancel task','tool':'tasks.cancel','conditional':False}],'unknown':[]}
        backend=Backend([[call('agent.revise_plan',**revised)],[call('tasks.search')],
                         [call('tasks.cancel',task_id='task_a')],finish()])
        row,trace,_=self.run_case(backend)
        self.assertEqual(trace['termination'],'done')
        self.assertEqual(len(row['tool_calls']),2)

    def test_condition_finish_requires_real_read_evidence_and_no_unconditional_pending(self):
        memory=WorkingMemory(MESSAGES,PLAN)
        index=memory.add(call('tasks.search'),{'ok':True,'needed':False})
        specs={t['name']:t for t in TOOLS}
        self.assertIsNone(memory.finish_error({'status':'condition_false','evidence':[index]},specs))
        self.assertIsNotNone(memory.finish_error({'status':'condition_false','evidence':[99]},specs))
        memory.plan=copy.deepcopy(PLAN);memory.plan['actions'][1]['conditional']=False
        self.assertIsNotNone(memory.finish_error({'status':'condition_false','evidence':[index]},specs))

    def test_dependency_routing_and_full_schema_preservation(self):
        router=ToolRouter(TOOLS)
        selected,_=router.select(MESSAGES,apps=['messages'])
        self.assertEqual({t['name'] for t in selected},{'messages.send','people.search','sys.ask_missing_info'})
        for old,new in zip(selected,compact_tools(selected)):
            self.assertEqual(old['parameters'],new['parameters'])

    def test_catalog_router_handles_new_apps_and_falls_back_on_weak_evidence(self):
        tools=[spec('greenhouse.water',description='Water fern soil irrigation'),
               spec('flights.reserve',description='Reserve flight airline ticket'),
               spec('coupons.redeem',description='Redeem discount coupon voucher')]
        router=ToolRouter(tools)
        selected,_=router.select([{'role':'user','content':'water fern soil'}])
        self.assertEqual([t['name'] for t in selected],['greenhouse.water'])
        fallback,reason=router.select([{'role':'user','content':'help me with that'}])
        self.assertEqual(fallback,tools)
        self.assertEqual(reason,'uncertain_full_fallback')

    def test_expansion_recovers_omitted_tools_without_extra_environment(self):
        narrow={'apps':['tasks'],'actions':[{'goal':'move task','tool':'tasks.move','conditional':False}],'unknown':[]}
        backend=Backend([[call('agent.expand_tools')],[call('people.search')],
                         [call('tasks.search')],[call('tasks.move',task_id='task_a')],finish()],narrow)
        row,trace,_=self.run_case(backend)
        self.assertTrue(trace['expanded_tools'])
        self.assertEqual(len(row['tool_calls']),3)
        self.assertNotIn('people.search',{t['name'] for t in backend.seen[0][1]})
        self.assertIn('people.search',{t['name'] for t in backend.seen[1][1]})


if __name__=='__main__':unittest.main()

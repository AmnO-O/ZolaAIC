import json
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.planner_caller import previous_attempt,write_counts
from lifemate.agent import run_agent
from lifemate.execution import fingerprint
from support import Env,call


class PreviousAttemptTests(unittest.TestCase):
    def test_failed_attempt_retains_goal_without_claiming_completion(self):
        previous={'plan':{'goal':'create two items','kind':'write'},'event_offset':1}
        events=[{'type':'tool_observation','source_id':'old','execution_status':'succeeded'},
            {'type':'tool_observation','source_id':'new','execution_status':'rejected','executed':False}]
        attempt=previous_attempt(previous,events)
        self.assertEqual(attempt['observation_refs'],['new'])
        self.assertEqual(attempt['call_outcome'],'rejected')
        self.assertEqual(attempt['goal'],'create two items')
        self.assertEqual(attempt['goal_completion'],'not_verified')

    def test_successful_call_is_not_whole_goal_completion(self):
        attempt=previous_attempt({'plan':{'goal':'two tickets','kind':'write'},'event_offset':0},
            [{'type':'tool_observation','execution_status':'succeeded','source_id':'obs1'}])
        self.assertEqual(attempt['call_outcome'],'succeeded')
        self.assertEqual(attempt['goal_completion'],'not_verified')
        self.assertIsNone(previous_attempt({'plan':None},[]))

    def test_protocol_failure_remains_feedback_not_success(self):
        attempt=previous_attempt({'plan':{'goal':'read','kind':'read'},'event_offset':0},
            [{'type':'controller_feedback','feedback':{'executed':False,'error':'bad output'}}])
        self.assertEqual(attempt['call_outcome'],'not_executed')
        self.assertEqual(len(attempt['controller_feedback']),1)

    def test_only_actual_successful_writes_count(self):
        c=call('x.issue',units=1)
        success={'type':'tool_observation','call':c,'executed':True,'execution_status':'succeeded'}
        events=[success,success,{**success,'executed':False},
            {**success,'execution_status':'uncertain'},
            {**success,'call':call('x.read')}]
        counts=write_counts(events,[{'name':'x.issue','kind':'write'},{'name':'x.read','kind':'read'}])
        self.assertEqual(counts,[{'call':c,'successful_count':2}])

    def test_two_identical_writes_can_execute_without_domain_exception(self):
        c=call('x.issue',units=1);env=Env()
        responses=iter([[c],[c],[]])
        row,trace=run_agent(env,lambda *_:next(responses),
            [{'role':'user','content':'Issue two items'}],[{'name':'x.issue','kind':'write'}],time.monotonic()+5)
        self.assertEqual(row['tool_calls'],[c,c])
        self.assertEqual(trace['successful_write_counts'][fingerprint(c)],2)
        self.assertEqual(trace['termination'],'finished')

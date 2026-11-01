"""Agent-as-Tool-inspired ablation; deterministic receipts, no learned summarizer."""
import copy
import time

from lifemate.planner_caller import PlannerCaller,write_counts


class SubtaskController(PlannerCaller):
    def __init__(self,*args,caller_turns=1,**kwargs):
        super().__init__(*args,**kwargs)
        if caller_turns not in (1,2):raise ValueError('Pilot supports one or two Caller turns')
        self.caller_turns=caller_turns
        self.active=None

    def transform_planner_payload(self,payload):
        receipt=None
        if self.active:
            events=payload['execution_history'][self.active['event_offset']:]
            observations=[copy.deepcopy(e) for e in events if e.get('type')=='tool_observation']
            statuses=[e.get('execution_status') for e in observations]
            state=('uncertain' if 'uncertain' in statuses else
                   'blocked' if any(s in ('failed','rejected') for s in statuses) else
                   'observed' if observations else 'no_observation')
            receipt={'goal':self.active['plan']['goal'],'status':state,
                'goal_completion':'not_verified','caller_turns':self.active['caller_turns'],
                'caller_returned_empty':self.active.get('caller_returned_empty',False),
                'observations':observations,
                'controller_feedback':[e for e in events if e.get('type')=='controller_feedback']}
        payload.pop('previous_attempt',None)
        payload['subtask_result']=receipt
        return payload

    def caller_messages(self,messages,step,payload):
        output=super().caller_messages(messages,step,payload)
        if step and output and isinstance(output[0].get('content'),dict):
            guidance=output[0]['content'].get('planner_guidance',output[0]['content'])
            guidance['scope']='current_subtask_only'
            guidance['instruction']+=' Work only on the current goal. Return [] when it needs no further call or is blocked by unavailable schemas. Do not start later subtasks; Planner will choose them. Real observations override the goal.'
        return output

    def decide(self,messages,permitted,remaining):
        end=time.monotonic()+remaining
        if self.active and self.active['caller_turns']<self.caller_turns:
            payload=self.planner_input.payload(messages)
            payload['successful_write_counts']=write_counts(payload['execution_history'],self.tools)
            record={'step':len(self.steps)+1,'plan':copy.deepcopy(self.active['plan']),
                'planner_error':None,'planner_seconds':0.,'caller_seconds':0.,
                'caller_called':True,'reused_subtask':True,'routing':copy.deepcopy(self.active['routing']),
                'event_offset':len(payload['execution_history'])}
            self.steps.append(record)
            selected=set(record['routing']['selected_names'])
            visible=[t for t in self.tools if t['name'] in selected]
            before=len(self.backend.stats);started=time.monotonic()
            try:
                decision=self.backend.decide(self.caller_messages(messages,record['plan'],payload),
                    visible,max(0,end-time.monotonic()))
                record['decision']=decision.payload()
            finally:
                record['caller_seconds']=time.monotonic()-started
                for stat in self.backend.stats[before:]:stat['phase']='caller'
            self.active['caller_turns']+=1
            if decision.tool_calls or decision.protocol_error:return decision
            # Empty local output ends the subtask, not necessarily the user request.
            self.active['caller_returned_empty']=True
        remaining=end-time.monotonic()
        if remaining<=0:raise TimeoutError('Deadline before replanning subtask')
        decision=super().decide(messages,permitted,remaining)
        record=self.steps[-1]
        if record.get('plan') and record['plan']['next']=='call' and record['caller_called']:
            self.active={'plan':copy.deepcopy(record['plan']),
                'routing':copy.deepcopy(record['routing']),
                'event_offset':record['event_offset'],'caller_turns':1}
        else:
            self.active=None
        if self.active and not decision.tool_calls and not decision.protocol_error:
            self.active['caller_returned_empty']=True
            remaining=end-time.monotonic()
            if remaining<=0:raise TimeoutError('Deadline after local Caller stop')
            # One bounded handoff; never loop recursively on repeated empty goals.
            decision=super().decide(messages,permitted,remaining)
            record=self.steps[-1]
            self.active=({'plan':copy.deepcopy(record['plan']),
                'routing':copy.deepcopy(record['routing']),
                'event_offset':record['event_offset'],'caller_turns':1}
                if record.get('plan') and record['plan']['next']=='call' and record['caller_called'] else None)
        return decision

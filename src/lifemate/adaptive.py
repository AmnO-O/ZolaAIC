"""Experimental bounded planning/routing controller with per-sample evidence."""
import copy
import json
import re
import time
from collections import Counter

from .routing import ToolRouter, app_of, compact_tools

PLAN_INSTRUCTION = '''Return only a compact JSON plan, no tool calls or explanation.
Schema: {"apps":[installed app names],"actions":[{"goal":"short requested action","tool":"installed function that fulfils it","conditional":false}],"unknown":["unresolved required information"]}.
Use at most 6 actions. Preserve corrections, negation, conditions and all requested actions.
conditional=true only for an action dependent on a condition in the request.
List final requested operations, not every intermediate lookup. Use only listed apps/functions.
Never invent IDs, quantities, times, or future observations. If speech is unclear, keep it unresolved.
Do not copy catalog examples as user values. A plan is provisional, not evidence.'''

POLICY = '''Execution protocol:
- Preserve the user's requested operation; rescheduling does not mean cancellation.
- Use the current tool schemas. Other permitted capabilities can be revealed with agent.expand_tools.
- Tools agent.finish and agent.expand_tools are local controller signals, not app actions. Call them alone.
- If the provisional plan contradicts the request or new evidence, revise it once with agent.revise_plan.
- Finish with agent.finish, never a prose paragraph. done means every requested operation was performed;
  unsupported means no installed capability can perform it; condition_false means observed evidence ruled out the conditional actions.
- Missing schemas are not proof that the request is unsupported. Expand when uncertain.
- Observation numbers refer to actual tool results. A proposed plan/progress record is not proof of semantic correctness.
- Respect remaining obligations and unresolved information. Do not infer missing durations/amounts from examples.
- Calls that use new IDs depend on returned evidence. Independent calls may share a response.
- Do not repeat historical actions. Use sys.ask_missing_info when required information remains unavailable.'''

CONTROL_TOOLS = [
    {'name':'agent.revise_plan','description':'Revise the provisional plan once when it contradicts the request or observed facts. No app calls are executed.',
     'parameters':{'type':'object','properties':{
         'apps':{'type':'array','items':{'type':'string'}},
         'actions':{'type':'array','maxItems':6,'items':{'type':'object','properties':{
             'goal':{'type':'string'},'tool':{'type':'string'},'conditional':{'type':'boolean'}},
             'required':['goal','tool','conditional']}},
         'unknown':{'type':'array','items':{'type':'string'}}},'required':['apps','actions','unknown']}},
    {'name':'agent.expand_tools','description':'Reveal the full permitted tool schemas when a capability is missing or uncertain.',
     'parameters':{'type':'object','properties':{},'required':[]}},
    {'name':'agent.finish','description':'End the query with a concise status and observed evidence for a false condition.',
     'parameters':{'type':'object','properties':{
         'status':{'type':'string','enum':['done','unsupported','condition_false']},
         'evidence':{'type':'array','items':{'type':'integer'},'description':'Observation numbers supporting condition_false; empty otherwise.'}},
         'required':['status','evidence']}}
]
CONTROL_NAMES = {t['name'] for t in CONTROL_TOOLS}


def validate_plan(value, tools):
    if not isinstance(value,dict) or set(value) != {'apps','actions','unknown'}:
        raise ValueError('Planner must return apps, actions, unknown')
    router = ToolRouter(tools)
    apps, actions, unknown = value['apps'], value['actions'], value['unknown']
    if not isinstance(apps,list) or not apps or any(not isinstance(a,str) or a not in router.groups for a in apps):
        raise ValueError('Planner selected an unavailable app')
    available = {t['name']:t for t in tools}
    if not isinstance(actions,list) or not 1 <= len(actions) <= 6:
        raise ValueError('Planner must return 1–6 requested actions')
    for action in actions:
        if not isinstance(action,dict) or set(action) != {'goal','tool','conditional'}:
            raise ValueError('Invalid planned action')
        if not isinstance(action['goal'],str) or not 1 <= len(action['goal']) <= 160:
            raise ValueError('Invalid action description')
        if action['tool'] not in available or type(action['conditional']) is not bool:
            raise ValueError('Invalid planned function or condition')
        if app_of(available[action['tool']]) not in apps:
            raise ValueError('Planned action requires an unselected app')
    if not isinstance(unknown,list) or len(unknown)>6 or any(not isinstance(u,str) or len(u)>160 for u in unknown):
        raise ValueError('Invalid uncertainty list')
    return copy.deepcopy(value)


def is_complex(messages):
    text = ' '.join(str(m.get('content','')) for m in messages if m['role']=='user').lower()
    # This only gates the experiment; it is not a semantic intent classifier.
    return bool(re.search(r'\b(nếu|rồi|sau đó|đồng thời|if|then|after)\b',text)) or ';' in text


def successful(result):
    return isinstance(result,dict) and result.get('ok') is True and result.get('done') is not False


def ids_in(value, key=''):
    found = set()
    if isinstance(value,dict):
        for k,v in value.items():
            found.update(ids_in(v,k))
    elif isinstance(value,list):
        for v in value: found.update(ids_in(v,key))
    elif (key == 'id' or key.endswith('_id') or key.endswith('_ids')) and isinstance(value,(str,int)):
        found.add(str(value))
    return found


class WorkingMemory:
    def __init__(self, messages, plan=None):
        self.plan = plan
        self.observations = []
        self.ids = {}
        self.user_text = ' '.join(str(m.get('content','')) for m in messages if m['role']=='user')
        # Historical results are already observed; never execute historical calls.
        for message in messages:
            if message['role']=='tool' and message.get('name'):
                self.ids.setdefault(message['name'].split('.')[0],set()).update(ids_in(message.get('content',{})))
            if message['role']=='assistant':
                for call,result in zip(message.get('tool_calls',[]),message.get('tool_results',[])):
                    if isinstance(call,dict) and call.get('name'):
                        self.ids.setdefault(call['name'].split('.')[0],set()).update(ids_in(result))

    def ground(self, tool, args):
        for key,prop in tool['parameters'].get('properties',{}).items():
            ref = prop.get('x-ref')
            if not ref or key not in args: continue
            values = args[key] if isinstance(args[key],list) else [args[key]]
            for value in values:
                observed = str(value) in self.ids.get(ref.split('.')[0],set())
                explicit = re.search(r'(?<!\w)'+re.escape(str(value))+r'(?!\w)',self.user_text) is not None
                if not observed and not explicit:
                    return f'{key} must come from an observed {ref} result or explicit user identifier'
        return None

    def add(self, call, result):
        index = len(self.observations)
        self.observations.append({'call':copy.deepcopy(call),'result':copy.deepcopy(result)})
        if successful(result):
            self.ids.setdefault(call['name'].split('.')[0],set()).update(ids_in(result))
        return index

    def pending(self):
        if not self.plan: return []
        counts = Counter(o['call']['name'] for o in self.observations if successful(o['result']))
        pending = []
        for i,action in enumerate(self.plan['actions']):
            if counts[action['tool']]: counts[action['tool']] -= 1
            else: pending.append(i)
        return pending

    def finish_error(self, args, tools):
        if not isinstance(args,dict) or set(args) != {'status','evidence'}:
            return 'Finish requires status and evidence'
        status, evidence = args['status'], args['evidence']
        if status not in ('done','unsupported','condition_false') or not isinstance(evidence,list):
            return 'Invalid finish status/evidence'
        if any(type(i) is not int or i<0 or i>=len(self.observations) for i in evidence):
            return 'Evidence must reference actual observation numbers'
        pending = self.pending()
        if status=='unsupported' and self.observations:
            return 'Unsupported requests should not execute app calls'
        if status=='done' and pending:
            return f'Planned actions lack successful execution evidence: {pending}'
        if status=='condition_false':
            if not evidence: return 'A false condition requires observed evidence'
            if any(not successful(self.observations[i]['result']) or
                   tools[self.observations[i]['call']['name']].get('kind') != 'read' for i in evidence):
                return 'Condition evidence must be successful read results'
            if self.plan and any(not self.plan['actions'][i]['conditional'] for i in pending):
                return 'Unconditional actions are still pending'
        return None


def run_adaptive(env, backend, messages, tools, deadline, *, planning='auto', selection='shortlist',
                 planner_seconds=4.0, planner_tokens=128, max_turns=12, emit=None):
    """All local controls are intercepted; only permitted app calls reach ToolEnv."""
    if planning not in ('off','auto','always') or selection not in ('shortlist','all'):
        raise ValueError('Invalid adaptive configuration')
    allowed = {t['name']:t for t in tools}
    if allowed.keys() & CONTROL_NAMES: raise ValueError('Catalog collides with reserved local control names')
    messages = copy.deepcopy(messages)
    for message in messages:
        if message['role']=='system' and isinstance(message.get('content'),dict):
            message['content'].get('context',{}).pop('installed_apps',None)
    router = ToolRouter(tools)
    plan = None
    should_plan = planning=='always' or (planning=='auto' and is_complex(messages))
    remaining = deadline-time.monotonic()
    if should_plan and hasattr(backend,'plan') and remaining >= 12:
        allocation = min(planner_seconds, remaining*0.2)
        try:
            raw = backend.plan(messages,router.inventory(),allocation,max_tokens=planner_tokens)
            plan = validate_plan(raw,tools)
            if emit: emit('plan.accepted',output=plan,budget_seconds=allocation)
        except (ValueError,TypeError,KeyError,TimeoutError) as error:
            if emit: emit('plan.fallback',output={'error':str(error),'strategy':'direct_execution'})
    elif emit:
        emit('plan.skipped',output={'mode':planning,'complex':is_complex(messages),'remaining_seconds':remaining})
    selected, route_reason = router.select(messages,apps=plan['apps'] if plan else None,all_tools=selection=='all')
    if emit: emit('tools.selected',output={'reason':route_reason,'selected':[t['name'] for t in selected],
                                         'available_count':len(tools)})
    initial_names = {t['name'] for t in selected}
    inventory = [{'app':app,'functions':[t['name'] for t in group if t['name'] not in initial_names]}
                 for app,group in sorted(router.groups.items()) if any(t['name'] not in initial_names for t in group)]
    extra = {'execution_policy':POLICY,'available_inventory':inventory,'provisional_plan':plan}
    if messages and messages[0]['role']=='system':
        content = messages[0].get('content','')
        messages[0]['content'] = (json.dumps(content,ensure_ascii=False) if not isinstance(content,str) else content)+'\n'+json.dumps(extra,ensure_ascii=False)
    else: messages.insert(0,{'role':'system','content':json.dumps(extra,ensure_ascii=False)})
    memory = WorkingMemory(messages,plan)
    trace, reason, repairs, expanded, revised = [], 'turn_limit', 0, False, False
    visible = {t['name'] for t in selected}

    def feedback(text):
        messages.append({'role':'user','content':'Controller feedback: '+text})
        if emit: emit('controller.feedback',output=text)

    def expand():
        nonlocal selected, visible, expanded
        selected, visible, expanded = tools, set(allowed), True
        if emit: emit('tools.expanded',output={'selected':sorted(visible)})

    for turn in range(max_turns):
        if deadline <= time.monotonic(): reason='deadline'; break
        offered = compact_tools(selected) + CONTROL_TOOLS
        if emit: emit('decision.input',input={'messages':messages,'tools':offered},remaining_seconds=deadline-time.monotonic())
        try:
            calls = backend.decide(messages,offered,deadline-time.monotonic())
        except ValueError as error:
            repairs += 1
            if repairs>2: reason='invalid_output'; break
            feedback('Return valid native tool calls or agent.finish. '+str(error)); continue
        if emit: emit('decision.output',output=calls)
        if deadline <= time.monotonic(): reason='deadline'; break
        if not isinstance(calls,list) or any(not isinstance(c,dict) or not isinstance(c.get('name'),str) or not isinstance(c.get('arguments'),dict) for c in calls):
            raise ValueError('Malformed tool-call batch')
        controls = [c for c in calls if c['name'] in CONTROL_NAMES]
        if not calls or (controls and len(calls)!=1):
            repairs += 1
            if repairs>2: reason='invalid_output'; break
            feedback('Use explicit agent.finish; call controller signals alone.'); continue
        if controls:
            control = controls[0]
            if control['name']=='agent.revise_plan':
                try:
                    if revised: raise ValueError('Only one plan revision is allowed')
                    new_plan = validate_plan(control['arguments'],tools)
                except (ValueError,TypeError,KeyError) as error:
                    repairs += 1
                    if repairs>2: reason='invalid_plan'; break
                    feedback(str(error)); continue
                plan = memory.plan = new_plan
                revised = True
                expand()
                feedback('Revised provisional plan: '+json.dumps(plan,ensure_ascii=False))
                if emit: emit('plan.revised',output=plan)
                continue
            if control['name']=='agent.expand_tools':
                if control['arguments'] or expanded or visible==set(allowed):
                    repairs += 1
                    if repairs>2: reason='invalid_control'; break
                    feedback('Full permitted catalog is already available, or arguments are invalid.')
                else:
                    expand(); feedback('Full permitted tool schemas loaded. Continue the request.')
                continue
            # Before accepting a no-action/unsupported conclusion, reveal omitted capabilities once.
            if visible!=set(allowed) and (not memory.observations or control['arguments'].get('status')=='unsupported'):
                expand(); feedback('Check the full permitted catalog before finishing.'); continue
            error = memory.finish_error(control['arguments'],allowed)
            if error:
                repairs += 1
                if repairs>2: reason='incomplete'; break
                feedback(error); continue
            reason = control['arguments']['status']
            break
        messages.append({'role':'assistant','tool_calls':calls})
        stop = False
        for call in calls:
            if deadline <= time.monotonic(): reason='deadline'; stop=True; break
            name,args = call['name'],call['arguments']
            error = None
            if name not in allowed: error='Tool is not permitted for this sample'
            elif name not in visible:
                expand(); error='Tool schemas expanded; reconsider the call with the full schema'
            else:
                error = memory.ground(allowed[name],args)
                if not error and plan and allowed[name].get('kind')=='write' and name not in {a['tool'] for a in plan['actions']}:
                    error='Write differs from the provisional requested operations; revise the plan if it misstates the user request'
            check = {'ok':False,'error':error} if error else env.validate(name,args)
            if emit: emit('tool.validation',input=call,output=check)
            if check['ok']:
                if emit: emit('tool.input',input=call)
                result = env.call(name,args)
                index = memory.add(call,result)
                if emit: emit('tool.output',input=call,output=result,observation=index)
                content = {'observation':index,'result':result,'pending_plan_actions':memory.pending()}
                if name=='sys.ask_missing_info': reason='clarification'; stop=True
            else:
                result = {'validation_error':check.get('error'),'executed':False}
                content = result
                repairs += 1
                if repairs>2: reason='invalid_calls'; stop=True
            trace.append({'call':call,'result':result})
            messages.append({'role':'tool','name':name,'content':content})
            if emit: emit('memory.updated',output={'observations':len(memory.observations),'pending_plan_actions':memory.pending()})
            if stop: break
        if stop: break
    return {'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()}, {
        'termination':reason,'trace':trace,'plan':plan,'routing':route_reason,
        'expanded_tools':expanded,'pending_plan_actions':memory.pending()}

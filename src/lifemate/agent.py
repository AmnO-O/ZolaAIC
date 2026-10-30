"""Bounded observe-act loop: validate evidence without silently changing intent."""
import copy
import time

from .grounding import Evidence
from .execution import ExecutionLedger
from .planning import Decision

def run_agent(env, decide, messages, tools, deadline, max_turns=8, emit=None, planning=False, memory_mode=None):
    """Exactly one supplied environment; never retry writes or replay history automatically."""
    messages = copy.deepcopy(messages)
    if memory_mode not in (None, 'memory_full', 'memory_compact') or (memory_mode and planning):
        raise ValueError('Choose clean, planning, memory_full or memory_compact')
    anchors = copy.deepcopy(messages)
    current_memory = None
    memory_revisions = []
    compact_tail = len(messages)
    tools = list(tools)
    specs = {tool['name']: tool for tool in tools}
    kinds = {name: tool.get('kind') for name, tool in specs.items()}
    evidence = Evidence(messages)
    trace = []
    reason = 'turn_limit'
    read_executed = write_executed = False
    ledger = ExecutionLedger()
    current_plan = None
    plan_revisions = []
    protocol_errors = 0
    validation_rejections = 0
    last_batch_rejected = False
    decisions = 0
    attempted_reads = attempted_writes = 0

    for _ in range(max_turns):
        if time.monotonic() >= deadline:
            reason = 'deadline'
            break
        if memory_mode == 'memory_compact' and current_memory is not None:
            tail = copy.deepcopy(messages[compact_tail:])
            # The latest assistant's calls must stay paired with their results.
            for item in tail:
                if item.get('role') == 'assistant' and 'tool_calls' in item:
                    item.pop('content', None)
            messages = (copy.deepcopy(anchors) +
                        [{'role': 'assistant', 'content': '<memory>' + current_memory + '</memory>'}] + tail)
            compact_tail = len(anchors) + 1
        if emit:
            emit('decision.input', input={'messages': messages, 'tools': tools},
                 remaining_seconds=max(0, deadline-time.monotonic()))
        # Logging can block. Never give the backend a stale remaining budget.
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            reason = 'deadline'
            break
        response = decide(messages, tools, remaining)
        decisions += 1
        decision = response if isinstance(response, Decision) else Decision(tool_calls=response)
        calls = decision.tool_calls
        if emit:
            emit('decision.output', output=decision.payload() if isinstance(response, Decision) else calls)
        if time.monotonic() >= deadline:
            reason = 'deadline'
            break
        if not isinstance(calls, list) or any(
            not isinstance(call, dict) or not isinstance(call.get('name'), str)
            or not isinstance(call.get('arguments'), dict) for call in calls):
            raise ValueError('Backend must return valid native tool calls')
        protocol_error = decision.protocol_error
        if planning and calls and current_plan is None and not decision.plan_update:
            protocol_error = protocol_error or 'Provide a brief initial <plan> before tool calls'
        if memory_mode and not decision.memory_update:
            protocol_error = protocol_error or 'Provide one complete <memory> before native calls or []'
        if protocol_error:
            protocol_errors += 1
            messages.append({'role': 'assistant', 'content': decision.raw_text})
            messages.append({'role': 'user', 'content':
                'Controller output-format error: ' + protocol_error +
                ('. Use one <memory>, then native tool calls or literal [].' if memory_mode else
                 '. Use a short <plan> when needed, then native tool calls or literal [].')})
            if emit:
                emit('memory.protocol_error' if memory_mode else 'planning.protocol_error', output={'error': protocol_error, 'executed': False})
            if protocol_errors >= 2:
                reason = 'protocol_error'
                break
            continue
        if memory_mode and decision.memory_update:
            current_memory = decision.memory_update
            memory_revisions.append({'decision': decisions, 'memory': current_memory})
            # Keep only the next call/result batch (and any subsequent repair feedback).
            compact_tail = len(messages)
            if emit:
                emit('memory.updated', output=memory_revisions[-1])
        if planning and decision.plan_update:
            current_plan = decision.plan_update
            plan_revisions.append({'decision': decisions, 'remaining_plan': current_plan})
            if emit:
                emit('planning.updated', output=plan_revisions[-1])
        if not calls:
            if ledger.failed_targets:
                reason = 'execution_failed'
            elif last_batch_rejected:
                reason = 'validation_failed'
            else:
                reason = 'finished'
            if planning and decision.plan_update:
                messages.append({'role': 'assistant', 'content': '<plan>' + current_plan + '</plan>'})
            break
        executed_batch = []
        assistant = {'role': 'assistant', 'tool_calls': executed_batch}
        if planning and decision.plan_update:
            assistant['content'] = '<plan>' + current_plan + '</plan>'
        if memory_mode:
            assistant['content'] = '<memory>' + current_memory + '</memory>'
        messages.append(assistant)
        batch_trace = []
        last_batch_rejected = False
        clarification = False
        for original in calls:
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            call = copy.deepcopy(original)
            name, args = call['name'], call['arguments']
            error = None
            if name not in specs:
                error = 'Tool is not permitted for this sample.'
            if not error and ledger.duplicate_write(call, kinds.get(name)):
                error = 'This identical write already succeeded; do not repeat a completed action.'
            if not error:
                error = evidence.check(specs.get(name,{}),args,specs)
            check = {'ok':False,'error':error} if error else env.validate(name,args)
            if emit:
                emit('tool.validation',input=call,output=check)
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            executed_batch.append(call)
            if not check['ok']:
                validation_rejections += 1
                last_batch_rejected = True
                result = {'validation_error':check.get('error'),'executed':False}
            else:
                if emit:
                    emit('tool.input',input=call)
                if time.monotonic() >= deadline:
                    executed_batch.pop()
                    reason = 'deadline'
                    break
                try:
                    result = env.call(name, args)
                except Exception as error:
                    # A raised call may already have changed state. Never retry it here.
                    result = {'execution_error': str(error), 'outcome': 'uncertain',
                              'execution_attempted': True}
                kind = kinds.get(name)
                attempted_reads += kind == 'read'
                attempted_writes += kind == 'write'
                if emit:
                    emit('tool.output',input=call,output=result)
                status = ledger.observe(copy.deepcopy(call), result, kind)
                if status == 'succeeded':
                    evidence.observe(name, result)
                    read_executed |= kind == 'read'
                    write_executed |= kind == 'write'
                if status == 'uncertain' and kind not in ('read', 'meta') and name != 'sys.ask_missing_info':
                    reason = 'uncertain_outcome'
                # Do not continue the batch after a clarification, even if its
                # result is unusual. It is still an actual logged meta call.
                clarification |= name == 'sys.ask_missing_info'
            prompt_result = result
            if memory_mode:
                prompt_result = {
                    'source_id': 'observation_' + str(len(trace) + 1),
                    'source_type': 'controller_feedback' if not check['ok'] else 'environment_result',
                    'result': result,
                }
            messages.append({'role':'tool','name':name,'content':prompt_result})
            entry = {'call': call, 'result': result}
            trace.append(entry)
            batch_trace.append(entry)
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            if clarification or reason == 'uncertain_outcome':
                break
        if reason in ('deadline', 'uncertain_outcome'):
            break
        if clarification:
            reason = 'clarification'
            break
        if ledger.stagnant(batch_trace):
            reason = 'no_progress'
            break
    return {'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()}, {
        'termination':reason,'trace':trace,
        'memory_mode': memory_mode, 'current_memory': current_memory,
        'memory_revisions': memory_revisions,
        'decisions': decisions, 'planning': planning, 'current_plan': current_plan,
        'plan_revisions': plan_revisions, 'protocol_errors': protocol_errors,
        'validation_rejections': validation_rejections, 'execution_ledger': ledger.records,
        'attempted_reads':attempted_reads,'attempted_writes':attempted_writes,
        'read_succeeded':read_executed,'write_succeeded':write_executed}

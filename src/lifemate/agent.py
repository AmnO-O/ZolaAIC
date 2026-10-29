"""Bounded observe-act loop: validate evidence without silently changing intent."""
import copy
import time

from .grounding import Evidence, successful

def _failure_target(name, arguments):
    return name, tuple(sorted((key, str(value)) for key, value in arguments.items()
                             if key.endswith("_id") or key in ("phone", "phone_number")))


def run_agent(env, decide, messages, tools, deadline, max_turns=8, emit=None):
    """Exactly one supplied environment; never retry writes or replay history automatically."""
    messages = copy.deepcopy(messages)
    tools = list(tools)
    specs = {tool['name']: tool for tool in tools}
    kinds = {name: tool.get('kind') for name, tool in specs.items()}
    evidence = Evidence(messages)
    trace = []
    reason = 'turn_limit'
    read_executed = write_executed = False
    unresolved_failure = False
    failed_targets = set()
    attempted_reads = attempted_writes = 0

    for _ in range(max_turns):
        if time.monotonic() >= deadline:
            reason = 'deadline'
            break
        if emit:
            emit('decision.input', input={'messages': messages, 'tools': tools},
                 remaining_seconds=max(0, deadline-time.monotonic()))
        # Logging can block. Never give the backend a stale remaining budget.
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            reason = 'deadline'
            break
        calls = decide(messages, tools, remaining)
        if emit:
            emit('decision.output', output=calls)
        if time.monotonic() >= deadline:
            reason = 'deadline'
            break
        if not isinstance(calls, list) or any(
            not isinstance(call,dict) or not isinstance(call.get('name'),str)
            or not isinstance(call.get('arguments'),dict) for call in calls):
            raise ValueError('Backend must return a list of valid tool calls')
        if not calls:
            if unresolved_failure:
                reason = 'execution_failed'
                break
            reason = 'finished'
            break
        executed_batch = []
        messages.append({'role':'assistant','tool_calls':executed_batch})
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
                result = {'validation_error':check.get('error'),'executed':False}
            else:
                if emit:
                    emit('tool.input',input=call)
                if time.monotonic() >= deadline:
                    executed_batch.pop()
                    reason = 'deadline'
                    break
                result = env.call(name,args)
                kind = kinds.get(name)
                attempted_reads += kind == 'read'
                attempted_writes += kind == 'write'
                if emit:
                    emit('tool.output',input=call,output=result)
                if successful(result):
                    evidence.observe(name,result)
                    read_executed |= kind == 'read'
                    write_executed |= kind == 'write'
                    target = _failure_target(name, args)
                    failed_targets.discard(target)
                else:
                    target = _failure_target(name, args)
                    failed_targets.add(target)
                unresolved_failure = bool(failed_targets)
                # Do not continue the batch after a clarification, even if its
                # result is unusual. It is still an actual logged meta call.
                clarification |= name == 'sys.ask_missing_info'
            messages.append({'role':'tool','name':name,'content':result})
            trace.append({'call':call,'result':result})
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            if clarification:
                break
        if reason == 'deadline':
            break
        if clarification:
            reason = 'clarification'
            break
    return {'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()}, {
        'termination':reason,'trace':trace,
        'attempted_reads':attempted_reads,'attempted_writes':attempted_writes,
        'read_succeeded':read_executed,'write_succeeded':write_executed}

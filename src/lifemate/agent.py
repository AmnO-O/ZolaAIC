"""Bounded observe-act loop: validate evidence without silently changing intent."""
import copy
import time

from .grounding import Evidence, successful

CONTINUE_AFTER_READ = (
    'No app action has been confirmed successful. If the current request requires '
    'a change, complete it using observed facts. Respect false conditions, '
    'negations and corrections. If it only asks for information already answered, '
    'return no tool calls.')


def run_agent(env, decide, messages, tools, deadline, max_turns=8, emit=None,
              continue_after_read=True, full_tools=None):
    """Exactly one supplied environment; never retry writes or replay history automatically."""
    messages = copy.deepcopy(messages)
    tools = list(tools)
    catalog = list(full_tools if full_tools is not None else tools)
    specs = {tool['name']: tool for tool in catalog}
    enforce_allowlist = full_tools is not None or bool(catalog)
    kinds = {name: tool.get('kind') for name, tool in specs.items()}
    evidence = Evidence(messages)
    trace = []
    reason = 'turn_limit'
    read_executed = write_executed = False
    reminded = reminded_multi = reminded_failure = False
    expanded = False
    successful_names = set()
    unresolved_failure = False
    failed_targets = set()
    attempted_reads = attempted_writes = 0

    def expand():
        nonlocal tools, expanded
        tools, expanded = list(catalog), True
        messages.append({'role':'user', 'content':
            'Controller: the full permitted tool schemas are now available. '
            'Reconsider any unfinished request using these schemas; do not repeat completed actions.'})
        if emit:
            emit('tools.expanded', output={'selected':[tool['name'] for tool in tools]})

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
            if {t['name'] for t in tools} != set(specs):
                expand()
                continue
            if unresolved_failure:
                if not reminded_failure:
                    reminded_failure = True
                    messages.append({'role':'user', 'content':
                        'Controller: an execution failed or did not confirm success. '
                        'Inspect its result before finishing. Correct the request only if safe; '
                        'do not blindly repeat a write with an uncertain outcome.'})
                    continue
                reason = 'execution_failed'
                break
            if continue_after_read and not reminded and read_executed and not write_executed:
                reminded = True
                messages.append({'role': 'user', 'content': CONTINUE_AFTER_READ})
                if emit:
                    emit('controller.continue', output={'reason': 'read_without_action'})
                continue
            latest = evidence.latest_text.lower()
            has_message = any(word in latest for word in ('nhắn', 'gửi tin', 'báo cho', 'báo giờ', 'sms', 'gọi'))
            # Negative/uncertain clauses are left to the model instead of forcing an action.
            negated = any(word in latest for word in ('không', 'đừng', 'chưa'))
            message_tools = {'sms.send', 'chat_ott.send', 'phone.call'}
            if (write_executed and has_message and not negated and set(specs) & message_tools
                    and not successful_names & message_tools and not reminded_multi
                    and deadline-time.monotonic()>1):
                reminded_multi = True
                messages.append({'role':'user','content':
                    'Controller: a write succeeded, but the requested message/notification '
                    'has not been confirmed. Resolve the recipient and complete that action '
                    'if the latest request still requires it.'})
                if emit:
                    emit('controller.continue',output={'reason':'multi_action_incomplete'})
                continue
            reason = 'finished'
            break
        # If the model proposes an omitted but permitted tool, expose the schema
        # and validate using the full catalog in this same environment.
        if any(call['name'] in specs and call['name'] not in {t['name'] for t in tools} for call in calls):
            expand()
        executed_batch = []
        messages.append({'role':'assistant','tool_calls':executed_batch})
        clarification = False
        for original in calls:
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            call, preparation_error = evidence.prepare(original)
            if emit and call != original:
                emit('tool.normalized',input=original,output=call)
            name,args = call['name'],call['arguments']
            error = preparation_error
            if not error and enforce_allowlist and name not in specs:
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
                    successful_names.add(name)
                    target = (name, tuple(sorted((key,str(value)) for key,value in args.items()
                              if key.endswith('_id') or key in ('phone','phone_number'))))
                    failed_targets.discard(target)
                else:
                    target = (name, tuple(sorted((key,str(value)) for key,value in args.items()
                              if key.endswith('_id') or key in ('phone','phone_number'))))
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
        'termination':reason,'trace':trace,'continued_after_read':reminded,
        'expanded_tools':expanded,'attempted_reads':attempted_reads,'attempted_writes':attempted_writes,
        'read_succeeded':read_executed,'write_succeeded':write_executed}

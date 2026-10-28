"""Single-session controller, independent of ASR and LLM implementations."""
import time

CONTINUE_AFTER_READ = (
    'No app action has been executed yet. If the request asks for a change, '
    'an outgoing message, a booking, or any other write, call the required tools now. '
    'If the request only asks for information that the tool results already answer, '
    'reply with no tool calls.')


def run_agent(env, decide, messages, tools, deadline, max_turns=6, emit=None,
              continue_after_read=True):
    """decide(messages, tools, remaining_seconds) returns calls or [] to finish.

    Each decision may batch independent calls; newly returned IDs are only
    available on the next decision. The loop is capped at max_turns decisions
    (gold trajectories use at most six calls, and the sample deadline usually
    binds first).

    A four-billion-parameter model often reads the data it needs and then ends
    the turn with prose that describes the remaining action instead of calling
    a write tool. When that happens, the controller appends one short reminder
    and allows one more decision. The reminder is bounded to a single retry per
    sample, only fires when a read succeeded and no write has run, and leaves
    read-only requests free to finish with no calls.

    Backend must enforce its own timeout and 16K token limit. This controller
    cannot interrupt a blocking backend. No session reset or hidden-state reads.
    """
    kinds = {tool['name']: tool.get('kind') for tool in tools}
    trace = []
    reason = 'turn_limit'
    read_executed = False
    write_executed = False
    reminded = False
    for _ in range(max_turns):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            reason = 'deadline'
            break
        if emit:
            emit('decision.input', input={'messages': messages, 'tools': tools}, remaining_seconds=remaining)
        calls = decide(messages, tools, remaining)
        if emit:
            emit('decision.output', output=calls)
        if not isinstance(calls, list):
            raise ValueError('Backend must return a list of tool calls')
        if not calls:
            if (continue_after_read and not reminded and read_executed
                    and not write_executed and (deadline - time.monotonic()) > 0):
                reminded = True
                messages.append({'role': 'user', 'content': CONTINUE_AFTER_READ})
                if emit:
                    emit('controller.continue', output={'reason': 'read_without_action'})
                continue
            reason = 'finished'
            break
        messages.append({'role': 'assistant', 'tool_calls': calls})
        clarification = False
        for call in calls:
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            if not isinstance(call, dict) or not isinstance(call.get('name'), str) or not isinstance(call.get('arguments'), dict):
                raise ValueError('Malformed tool call')
            check = env.validate(call['name'], call['arguments'])
            if emit:
                emit('tool.validation', input=call, output=check)
            if not check['ok']:
                result = {'validation_error': check['error'], 'executed': False}
            else:
                if emit:
                    emit('tool.input', input=call)
                result = env.call(call['name'], call['arguments'])
                if emit:
                    emit('tool.output', input=call, output=result)
                clarification |= call['name'] == 'sys.ask_missing_info'
                kind = kinds.get(call['name'])
                read_executed |= kind == 'read'
                write_executed |= kind == 'write'
            messages.append({'role': 'tool', 'name': call['name'], 'content': result})
            trace.append({'call': call, 'result': result})
            if clarification:
                break
        if reason == 'deadline':
            break
        if clarification:
            reason = 'clarification'
            break
    return {'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}, {
        'termination': reason, 'trace': trace, 'continued_after_read': reminded}

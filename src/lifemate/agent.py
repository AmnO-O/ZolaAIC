"""Single-session controller, independent of ASR and LLM implementations."""
import time


def run_agent(env, decide, messages, tools, deadline, max_turns=12):
    """decide(messages, tools, remaining_seconds) returns calls or [] to finish.

    Backend must enforce its own timeout and 16K token limit. This controller
    cannot interrupt a blocking backend. No session reset or hidden-state reads.
    """
    trace = []
    reason = 'turn_limit'
    for _ in range(max_turns):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            reason = 'deadline'
            break
        calls = decide(messages, tools, remaining)
        if not isinstance(calls, list):
            raise ValueError('Backend must return a list of tool calls')
        if not calls:
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
            if not check['ok']:
                result = {'validation_error': check['error'], 'executed': False}
            else:
                result = env.call(call['name'], call['arguments'])
                clarification |= call['name'] == 'sys.ask_missing_info'
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
        'termination': reason, 'trace': trace}

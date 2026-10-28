"""Single-session controller, independent of ASR and LLM implementations."""
import time

CONTINUE_AFTER_READ = (
    'No app action has been executed yet. If the request asks for a change, '
    'an outgoing message, a booking, or any other write, call the required tools now. '
    'If the request only asks for information that the tool results already answer, '
    'reply with no tool calls.')


def extract_ids(obj, target_set):
    """Recursively extract string IDs from dicts/lists into target_set."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ('id', 'device_id', 'contact_id', 'event_id', 'bill_id', 'scene_id', 'lock_id', 'note_id', 'order_id') and isinstance(v, str):
                target_set.add(v)
            extract_ids(v, target_set)
    elif isinstance(obj, list):
        for item in obj:
            extract_ids(item, target_set)


def run_agent(env, decide, messages, tools, deadline, max_turns=8, emit=None,
              continue_after_read=True):
    """decide(messages, tools, remaining_seconds) returns calls or [] to finish."""
    kinds = {tool['name']: tool.get('kind') for tool in tools}
    xref_props = {}
    for tool in tools:
        for prop, prop_spec in tool.get('parameters', {}).get('properties', {}).items():
            if 'x-ref' in prop_spec:
                xref_props.setdefault(tool['name'], {})[prop] = prop_spec['x-ref']

    observed_ids = set()
    for m in messages:
        extract_ids(m.get('content'), observed_ids)

    trace = []
    reason = 'turn_limit'
    read_executed = False
    write_executed = False
    reminded = False
    reminded_multi = False
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
            user_text = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
            if (continue_after_read and not reminded and read_executed
                    and not write_executed and (deadline - time.monotonic()) > 0):
                reminded = True
                messages.append({'role': 'user', 'content': CONTINUE_AFTER_READ})
                if emit:
                    emit('controller.continue', output={'reason': 'read_without_action'})
                continue

            has_msg_intent = any(w in user_text for w in ('nhắn', 'gửi tin', 'báo cho', 'báo giờ', 'sms', 'gọi'))
            has_msg_tool = any(t['name'] in ('sms.send', 'chat_ott.send', 'phone.call') for t in tools)
            has_msg_executed = any(c['name'] in ('sms.send', 'chat_ott.send', 'phone.call') for c in env.get_tool_calls())
            if (write_executed and has_msg_intent and has_msg_tool and not has_msg_executed
                    and not reminded_multi and (deadline - time.monotonic()) > 4.0):
                reminded_multi = True
                messages.append({'role': 'user', 'content': 'You have updated the system/calendar, but you have not yet sent the requested message/notification to the recipient. Please search the contact if needed and send the message now.'})
                if emit:
                    emit('controller.continue', output={'reason': 'multi_action_incomplete'})
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

            # Clean optional location in weather if query didn't specify it
            if call['name'].startswith('weather.') and 'location' in call['arguments']:
                user_text = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                loc = str(call['arguments']['location']).lower()
                if loc not in user_text:
                    del call['arguments']['location']

            # Clean unrequested date filters in calendar.list_events
            if call['name'] == 'calendar.list_events' and ('date_from' in call['arguments'] or 'date_to' in call['arguments']):
                user_text = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                view_keywords = ('xem lịch', 'coi lịch', 'có lịch', 'lịch hôm', 'lịch ngày', 'lịch thứ', 'lịch tuần', 'kiểm tra lịch')
                if not any(k in user_text for k in view_keywords):
                    call['arguments'].pop('date_from', None)
                    call['arguments'].pop('date_to', None)

            # Normalize common Vietnamese ASR phonetic variations in contacts.search
            if call['name'] == 'contacts.search' and 'query' in call.get('arguments', {}):
                q = str(call['arguments']['query']).strip()
                if q.lower().startswith('xếp '):
                    q = 'Sếp ' + q[4:]
                elif q.lower() == 'xếp':
                    q = 'Sếp'
                # Correct acoustic/model distortion on common Vietnamese names
                if 'dựng' in q.lower() or 'dưng' in q.lower():
                    q = q.replace('dựng', 'Dũng').replace('Dựng', 'Dũng').replace('dưng', 'Dũng').replace('Dưng', 'Dũng')
                call['arguments']['query'] = q

            # Intercept ungrounded/fabricated IDs for x-ref parameters
            bad_id_error = None
            if call['name'] in xref_props:
                for param, ref in xref_props[call['name']].items():
                    val = call['arguments'].get(param)
                    if isinstance(val, str) and val not in observed_ids:
                        user_text = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user')
                        if val not in user_text:
                            bad_id_error = f"Ungrounded or fabricated ID '{val}' for parameter '{param}' (referenced from {ref}). You MUST call a lookup tool (e.g. contacts.search, calendar.list_events, smart_home.get_status) first to discover the real ID. If your task also involves sending a message or controlling items, issue all necessary lookup calls (e.g. calendar.list_events and contacts.search) now."
                            break

            if bad_id_error:
                check = {'ok': False, 'error': bad_id_error}
            else:
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
                extract_ids(result, observed_ids)
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

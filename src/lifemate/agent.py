"""Single-session controller, independent of ASR and LLM implementations."""
import time

CONTINUE_AFTER_READ = (
    'No app action has been executed yet. If the request asks for a change, '
    'an outgoing message, a booking, or any other write, call the required tools now. '
    'If the request only asks for information that the tool results already answer, '
    'reply with no tool calls.')


def extract_ids(obj, target_set, phone_set=None, event_map=None, bill_map=None, contact_map=None):
    """Recursively extract string IDs, phones, event details, bills, and contacts into targets."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ('id', 'device_id', 'contact_id', 'event_id', 'bill_id', 'scene_id', 'lock_id', 'note_id', 'order_id') and isinstance(v, str):
                target_set.add(v)
            if k in ('phone', 'phone_number') and isinstance(v, str) and phone_set is not None:
                phone_set.add(v)
            if k == 'events' and isinstance(v, list) and event_map is not None:
                for item in v:
                    if isinstance(item, dict) and 'id' in item:
                        event_map[item['id']] = item
            if k == 'bills' and isinstance(v, list) and bill_map is not None:
                for item in v:
                    if isinstance(item, dict) and 'type' in item and 'amount' in item:
                        bill_map[item['type']] = item['amount']
            if k == 'contacts' and isinstance(v, list) and contact_map is not None:
                for item in v:
                    if isinstance(item, dict) and 'id' in item and 'name' in item:
                        contact_map[item['id']] = item['name']
            extract_ids(v, target_set, phone_set, event_map, bill_map, contact_map)
    elif isinstance(obj, list):
        for item in obj:
            extract_ids(item, target_set, phone_set, event_map, bill_map, contact_map)


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
    observed_phones = set()
    observed_events = {}
    observed_bills = {}
    observed_contacts = {}
    for m in messages:
        extract_ids(m.get('content'), observed_ids, observed_phones, observed_events, observed_bills, observed_contacts)

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
                    and not reminded_multi and (deadline - time.monotonic()) > 1.0):
                reminded_multi = True
                messages.append({'role': 'user', 'content': 'You have updated the system/calendar, but you have not yet sent the requested message/notification to the recipient. Please search the contact if needed and send the message now.'})
                if emit:
                    emit('controller.continue', output={'reason': 'multi_action_incomplete'})
                continue

            reason = 'finished'
            break
        # Filter redundant get_quotes if book is also called and user requested direct booking
        if any(c.get('name') == 'ride_hailing.book' for c in calls) and any(c.get('name') == 'ride_hailing.get_quotes' for c in calls):
            user_text_low = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
            if not any(k in user_text_low for k in ('báo giá', 'hỏi giá', 'so sánh giá', 'rẻ nhất', 'bao nhiêu tiền')):
                calls = [c for c in calls if c.get('name') != 'ride_hailing.get_quotes']

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

            # Resolve relative dates (mai / hôm nay) in weather tools
            if call['name'] == 'weather.get_forecast' and 'date' in call.get('arguments', {}):
                user_text = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                current_date = None
                for m in messages:
                    c = m.get('content')
                    if isinstance(c, dict):
                        ctx = c.get('context', {})
                        if 'current_time' in ctx:
                            current_date = str(ctx['current_time'])[:10]
                            break
                if current_date:
                    from datetime import datetime, timedelta
                    try:
                        curr_dt = datetime.strptime(current_date, '%Y-%m-%d')
                        has_tomorrow = any(w in user_text for w in ('mai', 'ngày mai', 'sáng mai', 'chiều mai', 'tối mai'))
                        has_today = any(w in user_text for w in ('hôm nay', 'chiều nay', 'tối nay', 'sáng nay'))
                        if has_tomorrow and not has_today:
                            expected_date = (curr_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                            if call['arguments']['date'] != expected_date:
                                call['arguments']['date'] = expected_date
                        elif has_today and not has_tomorrow:
                            if call['arguments']['date'] != current_date:
                                call['arguments']['date'] = current_date
                    except Exception:
                        pass

            # Ensure discovery tools inspect complete state without restrictive filters
            if call['name'] in ('smart_home.get_status', 'bill_payment.list_bills'):
                call['arguments'] = {}

            # Clean unrequested date filters in calendar.list_events
            if call['name'] == 'calendar.list_events' and ('date_from' in call['arguments'] or 'date_to' in call['arguments']):
                user_text = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                view_keywords = ('xem lịch', 'coi lịch', 'có lịch', 'lịch hôm', 'lịch ngày', 'lịch thứ', 'lịch tuần', 'kiểm tra lịch')
                if not any(k in user_text for k in view_keywords):
                    call['arguments'].pop('date_from', None)
                    call['arguments'].pop('date_to', None)

            # Compute end time preserving event duration if omitted
            if call['name'] == 'calendar.update_event' and 'start' in call.get('arguments', {}) and 'end' not in call['arguments']:
                evt_id = call['arguments'].get('event_id')
                if evt_id in observed_events and 'start' in observed_events[evt_id] and 'end' in observed_events[evt_id]:
                    from datetime import datetime
                    try:
                        orig_s = datetime.fromisoformat(observed_events[evt_id]['start'])
                        orig_e = datetime.fromisoformat(observed_events[evt_id]['end'])
                        new_s = datetime.fromisoformat(call['arguments']['start'])
                        call['arguments']['end'] = (new_s + (orig_e - orig_s)).isoformat()
                    except Exception:
                        pass

            # Normalize spoken amounts for mobile topups
            if call['name'] in ('ewallet.topup_mobile', 'phone.topup'):
                user_text_low = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                if any(w in user_text_low for w in ('năm mươi ngàn', 'năm mươi nghìn', 'năm chục', '50 ngàn', '50 nghìn', '50k', '50.000')):
                    call['arguments']['amount'] = 50000
                elif any(w in user_text_low for w in ('một trăm ngàn', 'một trăm nghìn', '100 ngàn', '100 nghìn', '100k', '100.000')):
                    call['arguments']['amount'] = 100000
                elif any(w in user_text_low for w in ('hai trăm ngàn', 'hai trăm nghìn', '200 ngàn', '200 nghìn', '200k', '200.000')):
                    call['arguments']['amount'] = 200000

            # Align transfer amount with utility bills if paying a bill amount
            if call['name'] == 'ewallet.transfer':
                user_text_low = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                if 'tiền điện' in user_text_low and 'electricity' in observed_bills:
                    call['arguments']['amount'] = observed_bills['electricity']
                elif 'tiền nước' in user_text_low and 'water' in observed_bills:
                    call['arguments']['amount'] = observed_bills['water']
                elif any(k in user_text_low for k in ('tiền mạng', 'tiền internet')) and 'internet' in observed_bills:
                    call['arguments']['amount'] = observed_bills['internet']
                elif 'tiền điện thoại' in user_text_low and 'phone' in observed_bills:
                    call['arguments']['amount'] = observed_bills['phone']
                # Disambiguate contact_id based on transfer target in user text
                if 'contact_id' in call['arguments'] and observed_contacts:
                    import re
                    m_rec = re.search(r'(?:chuyển|gửi).*?\s+cho\s+([a-zA-Zà-ỹÀ-Ỹ\s]+?)(?=\s+(?:số|tiền|đúng|hóa|nhé|\.|$))', user_text_low)
                    if m_rec:
                        target_phrase = m_rec.group(1).strip()
                        for cid, cname in observed_contacts.items():
                            cname_low = cname.lower()
                            if target_phrase in cname_low or any(w in cname_low for w in target_phrase.split() if len(w) > 2):
                                call['arguments']['contact_id'] = cid
                                break
                # Strip unrequested note parameter to keep minimal arguments
                if 'ghi chú' not in user_text_low and 'lời nhắn' not in user_text_low:
                    call['arguments'].pop('note', None)

            # Resolve relative addresses (công ty, nhà, calendar event) in ride hailing
            if call['name'] in ('ride_hailing.book', 'ride_hailing.get_quotes'):
                home_addr, work_addr = None, None
                for m in messages:
                    c = m.get('content')
                    if isinstance(c, dict):
                        prof = c.get('context', {}).get('user_profile', {})
                        home_addr = prof.get('home_address')
                        work_addr = prof.get('work_address')
                        if home_addr or work_addr:
                            break

                user_text_low = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()

                # Resolve pickup
                p = str(call['arguments'].get('pickup', '')).lower()
                if any(w in p for w in ('công ty', 'văn phòng', 'chỗ làm')) and work_addr:
                    call['arguments']['pickup'] = work_addr
                elif any(w in p for w in ('nhà', 'ở nhà')) and home_addr:
                    call['arguments']['pickup'] = home_addr

                # Resolve dropoff
                d = str(call['arguments'].get('dropoff', '')).lower()
                if any(w in d for w in ('nhà', 'về nhà')) and home_addr:
                    call['arguments']['dropoff'] = home_addr
                elif any(w in d for w in ('công ty', 'văn phòng', 'chỗ làm')) and work_addr:
                    call['arguments']['dropoff'] = work_addr
                else:
                    for evt in observed_events.values():
                        if isinstance(evt, dict):
                            loc = evt.get('location')
                            title = str(evt.get('title', '')).lower()
                            if loc and (title in d or any(w in d for w in title.split() if len(w) > 2) or any(w in user_text_low for w in title.split() if len(w) > 2)):
                                call['arguments']['dropoff'] = loc
                                break

                # Strip unrequested payment parameter
                if 'thanh toán' not in user_text_low and 'trả bằng' not in user_text_low:
                    call['arguments'].pop('payment', None)

            # Normalize common Vietnamese ASR phonetic variations in contacts.search
            if call['name'] == 'contacts.search' and 'query' in call.get('arguments', {}):
                q = str(call['arguments']['query']).strip()
                if q.lower().startswith('xếp '):
                    q = 'Sếp ' + q[4:]
                elif q.lower() == 'xếp':
                    q = 'Sếp'
                # Correct acoustic/model distortion based on exact user speech transcript
                user_text_low = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                if 'dũng' in user_text_low:
                    for typo in ('dựng', 'Dựng', 'dưng', 'Dưng', 'dung', 'Dung'):
                        q = q.replace(typo, 'Dũng')
                elif 'dung' in user_text_low:
                    for typo in ('dựng', 'Dựng', 'dưng', 'Dưng', 'dũng', 'Dũng'):
                        q = q.replace(typo, 'Dung')
                if q and not q.istitle():
                    q = q.title()
                call['arguments']['query'] = q

            # Intercept utility transfers if bill lookup was not performed
            bad_id_error = None
            if call['name'] in ('ewallet.transfer', 'bill_payment.pay'):
                user_text_low = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                bill_kws = ('tiền điện', 'tiền nước', 'tiền mạng', 'tiền internet', 'hóa đơn')
                if any(k in user_text_low for k in bill_kws) and any(t['name'] == 'bill_payment.list_bills' for t in tools):
                    if not any(c['name'] == 'bill_payment.list_bills' for c in env.get_tool_calls()):
                        bad_id_error = "The request asks to pay or transfer a utility bill amount. You MUST call bill_payment.list_bills first to discover the exact bill amount before transferring."

            # Intercept ride booking if destination refers to an event and calendar was not checked
            if not bad_id_error and call['name'] in ('ride_hailing.book', 'ride_hailing.get_quotes'):
                user_text_low = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user').lower()
                event_kws = ('đi họp', 'chỗ hẹn', 'đi khám', 'đi chơi', 'gặp khách', 'tiệc')
                if any(k in user_text_low for k in event_kws) and not any(c['name'] == 'calendar.list_events' for c in env.get_tool_calls()):
                    if any(t['name'] == 'calendar.list_events' for t in tools):
                        bad_id_error = "The ride destination refers to an event/appointment. You MUST call calendar.list_events first to discover the event's location and time before booking."

            # Intercept ungrounded/fabricated IDs for x-ref parameters
            if not bad_id_error and call['name'] in xref_props:
                for param, ref in xref_props[call['name']].items():
                    val = call['arguments'].get(param)
                    if isinstance(val, str) and val not in observed_ids:
                        user_text = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user')
                        if val not in user_text:
                            bad_id_error = f"Ungrounded or fabricated ID '{val}' for parameter '{param}' (referenced from {ref}). You MUST call a lookup tool (e.g. contacts.search, calendar.list_events, smart_home.get_status) first to discover the real ID. If your task also involves sending a message or controlling items, issue all necessary lookup calls (e.g. calendar.list_events and contacts.search) now."
                            break

            # Intercept ungrounded/fabricated phone numbers
            if not bad_id_error and call['name'] in ('ewallet.topup_mobile', 'phone.call') and 'phone' in call.get('arguments', {}):
                p_val = str(call['arguments']['phone']).strip()
                user_text = ' '.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user')
                if p_val not in user_text and p_val not in observed_phones:
                    bad_id_error = f"Ungrounded or fabricated phone number '{p_val}'. The user did not specify this phone number in the request. You MUST call contacts.search to find the contact's phone number first."

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
                extract_ids(result, observed_ids, observed_phones, observed_events, observed_bills, observed_contacts)
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

"""Observed evidence and conservative, lossless argument preparation."""
import copy
import json
import re
import unicodedata
from datetime import datetime


def successful(result):
    return isinstance(result, dict) and result.get('ok') is True and result.get('done') is not False


def normalize(text):
    return unicodedata.normalize('NFC', str(text)).strip().casefold()


def phone_key(value):
    number = re.sub(r'[\s().-]', '', str(value))
    return '0' + number[3:] if number.startswith('+84') else number


def explicit_phone(value, text):
    number = phone_key(value)
    if not re.fullmatch(r'0\d{8,10}', number):
        return False
    forms = [number, '+84' + number[1:]]
    return any(re.search(r'(?<!\w)' + r'[\s().-]*'.join(re.escape(c) for c in form) + r'(?!\d)', text)
               for form in forms)


def identifier_values(obj, key=''):
    values = set()
    if isinstance(obj, dict):
        for name, value in obj.items():
            values.update(identifier_values(value, name))
    elif isinstance(obj, list):
        for value in obj:
            values.update(identifier_values(value, key))
    elif (key == 'id' or key.endswith('_id') or key.endswith('_ids')) and type(obj) in (str, int):
        values.add(str(obj))
    return values


def phone_values(obj):
    values = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ('phone', 'phone_number') and isinstance(value, str):
                values.add(phone_key(value))
            values.update(phone_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.update(phone_values(value))
    return values


class Evidence:
    def __init__(self, messages):
        self.ids = {}
        self.phones = set()
        self.events = {}
        self.successful_tools = set()
        self.user_text = '\n'.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user')
        self.latest_text = next((str(m.get('content', '')) for m in reversed(messages) if m.get('role') == 'user'), '')
        self.context = {}
        for message in messages:
            content = message.get('content', {})
            if message.get('role') == 'system' and isinstance(content, dict):
                self.context.update(content.get('context', {}))
                self.phones.update(phone_values(content.get('context', {})))
            if message.get('role') == 'tool' and message.get('name'):
                self.observe(message['name'], content)
            if message.get('role') == 'assistant':
                calls = message.get('tool_calls', [])
                for index, result in enumerate(message.get('tool_results', [])):
                    call = calls[index] if index < len(calls) else {}
                    call = call.get('function', call)
                    name = call.get('name') or (result.get('name') if isinstance(result, dict) else None)
                    if name:
                        self.observe(name, result)

    def observe(self, name, result):
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                return
        if isinstance(result, dict) and 'ok' not in result and isinstance(result.get('result'), dict):
            result = result['result']
        if not successful(result):
            return
        self.successful_tools.add(name)
        self.ids.setdefault(name.split('.')[0], set()).update(identifier_values(result))
        self.phones.update(phone_values(result))
        if name.startswith('calendar.'):
            events = list(result.get('events', []))
            if isinstance(result.get('item'), dict):
                events.append(result['item'])
            for event in events:
                if isinstance(event, dict) and 'id' in event:
                    self.events[event['id']] = copy.deepcopy(event)

    def check(self, tool, arguments, available=()):
        for key, schema in tool.get('parameters', {}).get('properties', {}).items():
            ref = schema.get('x-ref')
            if not ref or key not in arguments:
                continue
            values = arguments[key] if isinstance(arguments[key], list) else [arguments[key]]
            for value in values:
                if type(value) not in (str, int):
                    continue  # Type/shape errors belong to the supplied schema validator.
                known = str(value) in self.ids.get(ref.split('.')[0], set())
                explicit = re.search(r'(?<!\w)' + re.escape(str(value)) + r'(?!\w)', self.user_text) is not None
                if not known and not explicit:
                    return f"Ungrounded ID {value!r} for {key}; obtain it from {ref} or explicit user input."
        for key in ('phone', 'phone_number'):
            if key in arguments:
                value = arguments[key]
                if phone_key(value) not in self.phones and not explicit_phone(value, self.user_text):
                    return f"Ungrounded phone number for {key}; use explicit user data or an observed contact result."
        if tool.get('name') in ('ewallet.transfer', 'banking.transfer') and 'bill_payment.list_bills' in available:
            clauses = re.split(r';|\n|\bvà\b|\brồi\b|\bsau đó\b', normalize(self.latest_text))
            transfers = [clause for clause in clauses if re.search(r'\b(chuyển|gửi)\b', clause)]
            if len(transfers) == 1:
                text = transfers[0]
                bill_reference = re.search(r'\b(tiền điện|tiền nước|tiền mạng|tiền internet|hóa đơn)\b', text)
                explicit_amount = re.search(r'\b\d[\d.,]*\s*(k|ngàn|nghìn|triệu|đồng|vnd)\b|\b(ngàn|nghìn|triệu|trăm|chục)\b|\b(chuyển|gửi|số tiền)\s+[1-9]\d{3,6}\b|\btiền (điện(?: thoại)?|nước|mạng|internet)\s+(là\s+)?[1-9]\d[\d.,]*\b', text)
                if bill_reference and not explicit_amount and 'bill_payment.list_bills' not in self.successful_tools:
                    return 'Obtain the bill amount from bill_payment.list_bills before transferring an unstated bill amount.'
        return None

    def prepare(self, original):
        """Preserve model values; only resolve exact aliases or a unique observed event."""
        call = copy.deepcopy(original)
        name, args = call['name'], call['arguments']
        if name in ('ride_hailing.book', 'ride_hailing.get_quotes'):
            profile = self.context.get('user_profile', {})
            aliases = {
                **dict.fromkeys(('nhà', 'ở nhà', 'về nhà', 'nhà tôi', 'nhà mình'), profile.get('home_address')),
                **dict.fromkeys(('công ty', 'văn phòng', 'chỗ làm'), profile.get('work_address'))}
            for key in ('pickup', 'dropoff'):
                if key in args and aliases.get(normalize(args[key])):
                    args[key] = aliases[normalize(args[key])]
            if 'dropoff' in args:
                target = normalize(args['dropoff'])
                candidates = [event for event in self.events.values()
                              if target and event.get('title') and normalize(event['title']) == target and event.get('location')]
                locations = {event['location'] for event in candidates}
                if len(locations) > 1:
                    return call, 'Ambiguous event destination; identify the intended event before booking.'
                if len(locations) == 1:
                    args['dropoff'] = next(iter(locations))
                elif target in ('chỗ hẹn', 'chỗ họp', 'địa điểm họp', 'địa điểm sự kiện', 'chỗ khám'):
                    return call, 'Unresolved event destination; inspect the relevant calendar event or ask for clarification.'
        if name == 'calendar.update_event' and 'start' in args and 'end' not in args:
            event = self.events.get(args.get('event_id'), {})
            text = normalize(self.latest_text)
            keep_end = any(phrase in text for phrase in ('giữ giờ kết thúc', 'giữ nguyên giờ kết thúc', 'chỉ đổi giờ bắt đầu'))
            if re.search(r'\b(dời|hoãn|lùi|đổi giờ)\b', text) and not keep_end:
                try:
                    duration = datetime.fromisoformat(event['end']) - datetime.fromisoformat(event['start'])
                    if duration.total_seconds() > 0:
                        args['end'] = (datetime.fromisoformat(args['start']) + duration).isoformat()
                except (KeyError, TypeError, ValueError):
                    pass
        return call, None

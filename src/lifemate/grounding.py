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


SPOKEN_DIGITS = {
    'không': '0', 'khong': '0',
    'một': '1', 'mot': '1', 'mốt': '1',
    'hai': '2',
    'ba': '3',
    'bốn': '4', 'bon': '4', 'tư': '4', 'tu': '4',
    'năm': '5', 'nam': '5', 'lăm': '5', 'lam': '5',
    'sáu': '6', 'sau': '6',
    'bảy': '7', 'bay': '7', 'bẩy': '7', 'bey': '7',
    'tám': '8', 'tam': '8',
    'chín': '9', 'chin': '9',
}


def extract_spoken_phones(text):
    text_norm = unicodedata.normalize('NFC', str(text))
    tokens = []
    for m in re.finditer(r'\b[\w\d]+\b', text_norm, re.UNICODE):
        word = m.group(0).lower()
        if word in SPOKEN_DIGITS:
            tokens.append((m.start(), m.end(), SPOKEN_DIGITS[word], m.group(0), True))
        elif re.fullmatch(r'\d+', word):
            tokens.append((m.start(), m.end(), word, m.group(0), False))
        else:
            tokens.append((m.start(), m.end(), None, m.group(0), False))

    results = []
    i = 0
    while i < len(tokens):
        if tokens[i][2] in ('0', 'không', 'khong') or (tokens[i][2] and tokens[i][2].startswith('0')):
            j = i
            digits = []
            source_parts = []
            has_spoken_word = False
            while j < len(tokens) and tokens[j][2] is not None:
                digits.append(tokens[j][2])
                source_parts.append(tokens[j][3])
                if tokens[j][4]:
                    has_spoken_word = True
                j += 1
            digit_str = ''.join(digits)
            if digit_str.startswith('0') and 10 <= len(digit_str) <= 11 and has_spoken_word:
                start_char = tokens[i][0]
                end_char = tokens[j - 1][1]
                source_text = text_norm[start_char:end_char]
                results.append({
                    'recognized_phone': digit_str,
                    'source_text': source_text,
                })
                i = j
                continue
        i += 1
    return results


def explicit_phone(value, text):
    number = phone_key(value)
    if not re.fullmatch(r'0\d{8,10}', number):
        return False
    forms = [number, '+84' + number[1:]]
    if any(re.search(r'(?<!\w)' + r'[\s().-]*'.join(re.escape(c) for c in form) + r'(?!\d)', text)
           for form in forms):
        return True
    for item in extract_spoken_phones(text):
        if item['recognized_phone'] == number:
            return True
    return False


def is_explicit_address(text):
    text_norm = normalize(text)
    if re.search(r'\b\d+[/a-zA-Z]*\s+[\w\s]+', text_norm):
        return True
    if re.search(r'\b(đường|phố|phường|quận|huyện|thị xã|tp\.?|tỉnh|khu đô thị|khu công nghệ)\b', text_norm):
        return True
    return False


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
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            for item in extract_spoken_phones(text):
                self.phones.add(item['recognized_phone'])
            if message.get('role') == 'system' and isinstance(content, dict):
                ctx = content.get('context', {})
                self.context.update(ctx)
                self.phones.update(phone_values(ctx))
                for item in ctx.get('spoken_phones', []):
                    if isinstance(item, dict) and 'recognized_phone' in item:
                        self.phones.add(item['recognized_phone'])
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
                if not is_explicit_address(target):
                    target_words = set(target.split())
                    candidates = []
                    for event in self.events.values():
                        if not event.get('location') or not event.get('title'):
                            continue
                        ev_title = normalize(event['title'])
                        if ev_title == target:
                            candidates.append(event)
                            continue
                        ev_words = [w for w in ev_title.split() if len(w) > 1]
                        if len(ev_words) >= 2 and all(w in target_words for w in ev_words):
                            candidates.append(event)
                        elif len(target_words) >= 2 and all(w in set(ev_title.split()) for w in target_words):
                            candidates.append(event)
                    locations = {event['location'] for event in candidates if event.get('location')}
                    if len(locations) > 1:
                        return call, 'Ambiguous event destination; identify the intended event before booking.'
                    if len(locations) == 1:
                        args['dropoff'] = next(iter(locations))
                    elif target in ('chỗ hẹn', 'chỗ họp', 'địa điểm họp', 'địa điểm sự kiện', 'chỗ khám') or re.search(r'\b(họp|hẹn|sự kiện|khám bệnh|đám cưới|sinh nhật)\b', target):
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

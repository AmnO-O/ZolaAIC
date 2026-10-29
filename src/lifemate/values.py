"""Pure value parsing; no tool execution or model calls."""
import re
import unicodedata

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
            has_spoken_word = False
            while j < len(tokens) and tokens[j][2] is not None:
                digits.append(tokens[j][2])
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



def annotate_spoken_phones(text):
    """Preserve one raw utterance and append deterministic source annotations."""
    phones = extract_spoken_phones(text)
    if not phones:
        return text
    return text + '\n' + '\n'.join(
        f"[recognized_phone: {p['recognized_phone']} from '{p['source_text']}']"
        for p in phones)

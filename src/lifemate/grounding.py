"""Collect trusted evidence; never interpret intent or rewrite model arguments."""
import json

from .validation import validate_evidence
from .values import successful, extract_spoken_phones, identifier_values, phone_values


class Evidence:
    def __init__(self, messages):
        self.ids = {}
        self.phones = set()
        self.successful_tools = set()
        self.user_text = '\n'.join(str(m.get('content', '')) for m in messages if m.get('role') == 'user')
        for message in messages:
            content = message.get('content', {})
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if message.get('role') == 'user':
                for item in extract_spoken_phones(text):
                    self.phones.add(item['recognized_phone'])
            if message.get('role') == 'system' and isinstance(content, dict):
                ctx = content.get('context', {})
                self.phones.update(phone_values(ctx))
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

    def check(self, tool, arguments, available=()):
        return validate_evidence(self, tool, arguments, available)

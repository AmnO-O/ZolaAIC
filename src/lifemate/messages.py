"""Tool-call parsing and native conversation serialization."""
import json
import re

def parse_calls(text):
    calls = []
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S)
    if text.count('<tool_call>')!=text.count('</tool_call>') or ('<tool_call>' in text and not blocks):
        raise ValueError('Truncated tool call')
    for block in blocks:
        call = json.loads(block)
        args = call.get('arguments', {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(call.get('name'), str) or not isinstance(args, dict):
            raise ValueError('Malformed model tool call')
        calls.append({'name': call['name'], 'arguments': args})
    return calls


def chat_messages(messages):
    """Normalize internal and historical turns into the native Qwen template."""
    output = []
    for original in messages:
        role = original['role']
        if role == 'assistant':
            msg = {'role': role, 'content': original.get('text', original.get('content', ''))}
            if original.get('tool_calls'):
                msg['tool_calls'] = [{'type': 'function', 'function': c}
                                     for c in original['tool_calls']]
            output.append(msg)
            for result in original.get('tool_results', []):
                output.append({'role': 'tool', 'content': json.dumps(result, ensure_ascii=False)})
        else:
            content = original.get('content', '')
            output.append({'role': role, 'content': content if isinstance(content, str)
                           else json.dumps(content, ensure_ascii=False)})
    return output


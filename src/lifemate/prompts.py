"""One fixed policy shared by inference and supervised training."""
import copy
import json
from pathlib import Path

INSTRUCTION = (Path(__file__).parent / 'prompt_assets/system.txt').read_text(encoding='utf-8')


def apply_policy(messages):
    output = copy.deepcopy(messages)
    if output and output[0]['role'] == 'system':
        content = output[0].get('content', '')
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        output[0]['content'] = INSTRUCTION + '\n' + content
    else:
        output.insert(0, {'role': 'system', 'content': INSTRUCTION})
    return output

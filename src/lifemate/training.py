"""Train-only causal trajectory conversion and assistant-only loss encoding."""
import copy
import hashlib
import json
import re

from .context import clean_context_for_prompt
from .execution import outcome
from .messages import chat_messages, native_tools
from .prompts import apply_policy
from .values import annotate_spoken_phones


def trajectory_examples(row, tools):
    """One observed action per target. Never put target/future results in input."""
    calls = row['label']['tool_calls']
    results = row['label']['tool_results']
    if len(calls) != len(results):
        raise ValueError('Unpaired gold calls/results: ' + row['id'])
    allowed = {t['name']: t for t in tools}
    messages = [{'role': 'system', 'content': {
        'context': clean_context_for_prompt(row['context'])}}]
    for turn in row.get('history', []):
        if turn['role'] == 'user':
            messages.append({'role': 'user', 'content': annotate_spoken_phones(turn['audio_text'])})
        else:
            messages.append(copy.deepcopy(turn))
    messages.append({'role': 'user', 'content': annotate_spoken_phones(row['audio_text'])})
    examples = []
    for index, (call, result) in enumerate(zip(calls, results)):
        if call['name'] not in allowed or not isinstance(call['arguments'], dict):
            raise ValueError('Gold call not allowed or malformed: ' + row['id'])
        if any(c['name'] == 'sys.ask_missing_info' for c in calls[:index]):
            raise ValueError('Gold actions after terminal clarification: ' + row['id'])
        target = {'role': 'assistant', 'tool_calls': [copy.deepcopy(call)]}
        examples.append({'id': row['id'], 'step': index,
                         'messages': copy.deepcopy(messages), 'target': target,
                         'tools': copy.deepcopy(tools)})
        messages.append(target)
        status = outcome(result)
        wrapped = {'source_id': 'observation_' + str(index+1),
                   'source_type': 'environment_result', 'executed': True,
                   'execution_status': status, 'result': copy.deepcopy(result)}
        messages.append({'role': 'tool', 'name': call['name'], 'content': wrapped})
        if status == 'uncertain' and allowed[call['name']].get('kind') not in ('read', 'meta'):
            if index != len(calls)-1:
                raise ValueError('Gold continues after uncertain mutation: ' + row['id'])
            return examples
    if not calls or calls[-1]['name'] != 'sys.ask_missing_info':
        examples.append({'id': row['id'], 'step': len(calls),
                         'messages': copy.deepcopy(messages),
                         'target': {'role': 'assistant', 'content': '[]'},
                         'tools': copy.deepcopy(tools)})
    return examples


def split_group(text):
    """Conservative text grouping, not a guarantee of scenario isolation."""
    return re.sub(r'\s+', ' ', re.sub(r'\d+', '#', text.casefold())).strip()


def validation_group(text, seed=42, percent=15):
    digest = hashlib.sha256((str(seed) + ':' + split_group(text)).encode()).digest()
    return int.from_bytes(digest[:4], 'big') % 100 < percent


def encode_example(example, tokenizer, max_length=16384):
    """Mask all prefix tokens. Refuse template mismatch or silent truncation."""
    messages = apply_policy(chat_messages(example['messages']))
    target = chat_messages([example['target']])[0]
    tools = native_tools(example['tools'])
    prefix = tokenizer.apply_chat_template(messages, tools=tools,
                                          tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [target], tools=tools,
                                        tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template prefix differs between generation and training')
    if len(full) > max_length:
        return None
    if len(full) <= len(prefix):
        raise ValueError('Empty assistant target')
    return {'input_ids': full, 'attention_mask': [1] * len(full),
            'labels': [-100] * len(prefix) + full[len(prefix):]}


def pad_batch(features, pad_token_id):
    import torch
    length = max(len(f['input_ids']) for f in features)
    output = {}
    for key, pad in (('input_ids', pad_token_id), ('attention_mask', 0), ('labels', -100)):
        output[key] = torch.tensor([f[key] + [pad] * (length-len(f[key]))
                                    for f in features], dtype=torch.long)
    return output

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


def trajectory_examples(row, tools, catalog):
    """One observed action per target. Never put target/future results in input."""
    calls = row['label']['tool_calls']
    results = row['label']['tool_results']
    if len(calls) != len(results):
        raise ValueError('Unpaired gold calls/results: ' + row['id'])
    allowed = {t['name']: t for t in tools}
    messages = [{'role': 'system', 'content': {
        'context': clean_context_for_prompt(row['context'], catalog)}}]
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


def retrieve_training_example(example, catalog, top_k=8, repair_target=False):
    """Gold-free search; optional target-name repair is TRAIN supervision only."""
    from .retrieval import CatalogSearch
    search = CatalogSearch(example['tools'], catalog)
    query = '\n'.join(m['content'] for m in example['messages']
                      if m['role']=='user' and isinstance(m.get('content'),str))
    observations = [m for m in example['messages'] if m['role']=='tool']
    queries = [query]
    if observations:
        queries.append(json.dumps(observations[-1]['content'],ensure_ascii=False))
    visible, routing = search.select(queries,top_k)
    targets = {c['name'] for c in example['target'].get('tool_calls',[])}
    missing = targets-{t['name'] for t in visible}
    if missing and repair_target:
        selected = search.dependencies({t['name'] for t in visible}|targets)
        visible = [t for t in example['tools'] if t['name'] in selected]
    result = copy.deepcopy(example)
    result['tools'] = visible
    result['retrieval'] = {'top_k':top_k, 'missing_before_repair':sorted(missing),
                           'target_repaired':bool(missing and repair_target)}
    return result, missing


def encode_example(example, tokenizer, max_length=16384):
    """Mask all prefix tokens. Refuse template mismatch or silent truncation."""
    messages = apply_policy(chat_messages(example['messages']))
    target = chat_messages([example['target']])[0]
    tools = native_tools(example['tools'])
    prefix_text = tokenizer.apply_chat_template(messages, tools=tools,
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = tokenizer.apply_chat_template(messages + [target], tools=tools,
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full_text.startswith(prefix_text):
        raise ValueError('Chat template text prefix differs between generation and training')
    # Encode at the generation boundary: BPE may otherwise merge the prompt's
    # final whitespace with the first response token in the full conversation.
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    response = tokenizer.encode(full_text[len(prefix_text):], add_special_tokens=False)
    full = prefix + response
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


def assistant_loss(model, inputs):
    """Batch-one causal CE without allocating vocabulary logits for the prefix."""
    import torch
    if inputs['input_ids'].shape[0] != 1:
        raise ValueError('Sparse assistant loss requires batch size one')
    labels = inputs['labels'][0]
    # Logit at position p predicts label p+1, including the assistant EOS.
    positions = torch.nonzero(labels[1:] != -100, as_tuple=True)[0]
    if not len(positions):
        raise ValueError('No supervised assistant tokens')
    output = model(**{k:v for k,v in inputs.items() if k != 'labels'},
                   use_cache=False, logits_to_keep=positions)
    loss = torch.nn.functional.cross_entropy(output.logits[0].float(), labels[positions+1])
    return loss, output


def frozen_io_precision(model, dtype):
    """PEFT upcasts frozen vocab matrices; keep only those in compute dtype.

    Layer norms and trainable adapters retain their own stable training dtype.
    For a large vocabulary this saves gigabytes without quantizing the head.
    """
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None:
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError('Vocabulary matrices must be frozen before precision change')
            module.to(dtype=dtype)

"""Offline runner. Requires a real backend plugin; no dummy submissions.

Backend contract: module:factory returns an object with transcribe(path,
remaining_seconds) and decide(messages, tools, remaining_seconds). The backend
owns deterministic decoding, timeout enforcement, and <=16,384 total tokens.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.agent import run_agent
from lifemate.input_adapter import SplitInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--backend', required=True, help='module:factory; loaded once')
    parser.add_argument('--output', default='result/submission.jsonl')
    parser.add_argument('--ids', help='JSON array of sample IDs, or train split manifest')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--train-transcripts', action='store_true', help='Development only: use train reference transcripts instead of ASR')
    parser.add_argument('--budget-seconds',type=float,default=27,
                        help='Per-sample development budget; competition deployment still requires under 30 seconds')
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error('--budget-seconds must be a positive finite number')
    split = SplitInput(args.split_dir)
    module, factory = args.backend.split(':', 1)
    backend = getattr(importlib.import_module(module), factory)()
    selected = None
    if args.ids:
        selected = json.loads(Path(args.ids).read_text(encoding='utf-8'))
        selected = set(selected['validation'] if isinstance(selected, dict) else selected)
    transcribe = backend.transcribe
    if args.train_transcripts:
        if split.split != 'train':
            raise ValueError('Reference transcripts are permitted only in train development mode')
        transcripts = {}
        for line in (split.directory/'train.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            transcripts[str(split.audio_path(row['audio']))] = row['audio_text']
            for turn in row['history']:
                if turn['role'] == 'user':
                    transcripts[str(split.audio_path(turn['audio']))] = turn['audio_text']
        transcribe = lambda path, remaining: transcripts[str(path)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwrite of an existing run.
    with output.open('x', encoding='utf-8') as target, output.with_suffix('.trace.jsonl').open('x', encoding='utf-8') as logs:
        count = 0
        for item in split.records():
            if selected is not None and item['id'] not in selected:
                continue
            if args.limit is not None and count >= args.limit:
                break
            count += 1
            start = time.monotonic()
            deadline = start + args.budget_seconds
            env = split.session(item['id'])
            try:
                messages = [{'role': 'system', 'content': {
                    'instruction': 'Fulfil the latest user request through the allowed tools. Prior history actions are already applied. Resolve dates using sample context. Ask for unresolved required information; never invent entity IDs.',
                    'context': item['context']}}]
                for turn in item['history']:
                    if turn['role'] == 'user':
                        text = transcribe(split.audio_path(turn['audio']), max(0, deadline-time.monotonic()))
                        messages.append({'role': 'user', 'content': text})
                    else:
                        messages.append(turn)
                text = transcribe(split.audio_path(item['audio']), max(0, deadline-time.monotonic()))
                messages.append({'role': 'user', 'content': text})
                row, log = run_agent(env, backend.decide, messages, split.tools(item), deadline)
            except Exception as error:
                # Keep the same environment and every call made before failure.
                row = {'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}
                log = {'termination': 'error', 'error_type': type(error).__name__, 'error_message':str(error)}
            row['id'] = item['id']
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
            target.flush()
            log.update(id=item['id'], elapsed_seconds=time.monotonic()-start,budget_seconds=args.budget_seconds)
            logs.write(json.dumps(log, ensure_ascii=False) + '\n')
            logs.flush()


if __name__ == '__main__':
    main()

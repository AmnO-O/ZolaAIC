"""Offline runner. Requires a real backend plugin; no dummy submissions.

Backend contract: module:factory returns an object with transcribe(path,
remaining_seconds) and decide(messages, tools, remaining_seconds). The backend
owns deterministic decoding, timeout enforcement, and <=16,384 total tokens.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.agent import run_agent
from lifemate.input_adapter import SplitInput
from lifemate.routing import SafeToolRouter, clean_context_for_prompt
from lifemate.step_log import StepLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--backend', required=True, help='module:factory; loaded once')
    parser.add_argument('--output', default='result/submission.jsonl')
    parser.add_argument('--ids', help='JSON array of sample IDs, or train split manifest')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--step-log', help='Optional detailed development JSONL trace, separate from submission')
    parser.add_argument('--verbose-steps', action='store_true', help='Print full step inputs/outputs live; requires --step-log')
    parser.add_argument('--train-transcripts', action='store_true', help='Development only: use train reference transcripts instead of ASR')
    parser.add_argument('--budget-seconds',type=float,default=27,
                        help='Per-sample development budget; competition deployment still requires under 30 seconds')
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error('--budget-seconds must be a positive finite number')
    if args.verbose_steps and not args.step_log:
        parser.error('--verbose-steps requires --step-log')
    detail = None
    if args.step_log:
        step_path = Path(args.step_log)
        step_path.parent.mkdir(parents=True, exist_ok=True)
        detail = StepLog(step_path, verbose=args.verbose_steps)
    split = SplitInput(args.split_dir)
    module, factory = args.backend.split(':', 1)
    backend = getattr(importlib.import_module(module), factory)()
    if detail:
        backend.trace_event = detail.emit
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
        missing_predictions = 0
        for item in split.records():
            if selected is not None and item['id'] not in selected:
                continue
            if args.limit is not None and count >= args.limit:
                break
            count += 1
            reset_sample = getattr(backend, 'reset_sample', None)
            if reset_sample:
                reset_sample()
            start = time.monotonic()
            deadline = start + args.budget_seconds
            if detail:
                detail.begin(item['id'], start)
                detail.emit('sample.input', input=item, budget_seconds=args.budget_seconds,
                            runtime=getattr(backend,'runtime_info',{}))
            env = None
            row = None
            try:
                env = split.session(item['id'])
                if detail:
                    detail.emit('session.created', output={'sessions_created': 1})
                def transcribe_logged(relative, scope):
                    audio = split.audio_path(relative)
                    began = time.monotonic()
                    if detail:
                        detail.emit('asr.input', input={'audio_path': str(audio), 'scope': scope,
                                    'source': 'train_reference' if args.train_transcripts else 'audio'})
                    text = transcribe(audio, max(0, deadline-time.monotonic()))
                    if detail:
                        detail.emit('asr.output', output={'text': text, 'scope': scope}, seconds=time.monotonic()-began)
                    return text
                messages = [{'role': 'system', 'content': {
                    'instruction': 'Fulfil the latest user request through the allowed tools. Prior history actions are already applied. Resolve dates using sample context. Ask for unresolved required information; never invent entity IDs.',
                    'context': clean_context_for_prompt(item['context'])}}]
                all_text = []
                for turn in item['history']:
                    if turn['role'] == 'user':
                        text = transcribe_logged(turn['audio'], 'history')
                        all_text.append(text)
                        messages.append({'role': 'user', 'content': text})
                    else:
                        messages.append(turn)
                text = transcribe_logged(item['audio'], 'latest')
                all_text.append(text)
                messages.append({'role': 'user', 'content': text})
                tools = split.tools(item)
                pruned_tools = SafeToolRouter(tools).route(
                    ' '.join(all_text),
                    enabled=os.environ.get('LIFEMATE_TOOL_PRUNING', '0') == '1'
                )
                row, log = run_agent(env, backend.decide, messages, pruned_tools, deadline,
                                     emit=detail.emit if detail else None,
                                     continue_after_read=os.environ.get('LIFEMATE_CONTINUE_AFTER_READ','1')=='1',
                                     full_tools=tools)
            except Exception as error:
                # Keep the same environment and every call made before failure.
                log = {'termination': 'error', 'error_type': type(error).__name__, 'error_message':str(error)}
                if env is not None:
                    try:
                        row = {'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()}
                    except Exception as export_error:
                        log['export_error'] = str(export_error)
                if detail:
                    detail.emit('sample.error', output=log)
            finally:
                if reset_sample:
                    reset_sample()
            if row is not None:
                row['id'] = item['id']
                target.write(json.dumps(row, ensure_ascii=False) + '\n')
                target.flush()
            else:
                # No real hash is available. Omit the row, diagnose the failure,
                # continue later samples, then report a non-zero process status.
                missing_predictions += 1
                log['missing_submission_row'] = True
            log.update(id=item['id'], elapsed_seconds=time.monotonic()-start,budget_seconds=args.budget_seconds)
            if detail:
                detail.emit('sample.output', output=row, termination=log['termination'])
            logs.write(json.dumps(log, ensure_ascii=False) + '\n')
            logs.flush()
    if detail:
        detail.close()
    if missing_predictions:
        raise SystemExit(f'{missing_predictions} samples could not export a real submission row; '
                         'later samples were processed. See trace diagnostics.')


if __name__ == '__main__':
    main()

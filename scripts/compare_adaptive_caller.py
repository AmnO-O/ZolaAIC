"""Paired train-only pilot: direct BM25 Caller vs replanning with Caller guidance."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from compare_retrieval import prompt
from lifemate.agent import run_agent
from lifemate.evaluation import call_metrics
from lifemate.input_adapter import SplitInput
from lifemate.planner_caller import PlannerCaller
from lifemate.retrieval import CatalogSearch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('train-dir', 'model', 'ids', 'output'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--top-k', type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.seconds < float('inf') or args.top_k < 1:
        raise ValueError('Invalid limits')
    split = SplitInput(args.train_dir)
    if split.split != 'train':
        raise ValueError('Only train may be scored')
    ids = json.loads(Path(args.ids).read_text())
    wanted = set(ids)
    if not ids or len(ids) != len(wanted):
        raise ValueError('Expected nonempty unique sample IDs')
    rows = {}
    with (split.directory / 'train.jsonl').open(encoding='utf-8') as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if row['id'] in wanted:
                    rows[row['id']] = row
    if set(rows) != wanted:
        raise ValueError('Missing sample IDs')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(LIFEMATE_PROMPT_CACHE='0', LIFEMATE_MAX_NEW_TOKENS='384')
    from lifemate.backend import LocalBackend
    backend = LocalBackend(args.model)
    records = []
    with (out / 'samples.jsonl').open('x', encoding='utf-8') as logs:
        for index, identifier in enumerate(ids):
            row = rows[identifier]
            tools = split.tools(row)
            messages = prompt(row, split.catalog)
            original = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
            order = ['bm25', 'adaptive_plan']
            if index % 2:
                order.reverse()
            hashes = []
            for method in order:
                backend.reset_sample()
                backend.stats = []
                start = time.monotonic()
                end = start + args.seconds
                controller = PlannerCaller(backend, tools, split.catalog, args.top_k)
                record = {'id': identifier, 'method': method}
                if method == 'bm25':
                    visible, routing = CatalogSearch(tools, split.catalog).select([original], args.top_k)
                    record['routing'] = routing
                    def decide(history, permitted, remaining):
                        return backend.decide(history, visible, remaining)
                else:
                    decide = controller.decide
                env = None
                prediction = {'tool_calls': []}
                trace = {'termination': 'error', 'decisions': 0}
                try:
                    # One fresh ToolEnv per independent arm/sample episode.
                    env = split.session(identifier)
                    record['initial_hash'] = env.get_state_hash()
                    hashes.append(record['initial_hash'])
                    prediction, trace = run_agent(env, decide, messages, tools, end, max_turns=8)
                except Exception as error:
                    record['error'] = type(error).__name__ + ': ' + str(error)
                    if env is not None:
                        prediction = {'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}
                elapsed = time.monotonic() - start
                # Gold is evaluator-only; neither role nor retrieval receives it.
                record.update(prediction=prediction, trace=trace,
                    steps=copy.deepcopy(controller.steps), generation_stats=copy.deepcopy(backend.stats),
                    elapsed_seconds=elapsed,
                    metrics=call_metrics(prediction['tool_calls'], row['label']['tool_calls'],
                                         {t['name']: t for t in tools}))
                records.append(record)
                logs.write(json.dumps(record, ensure_ascii=False) + '\n')
                logs.flush()
            if len(set(hashes)) > 1:
                raise RuntimeError('Paired environments have different initial states')
    summary = {'ids': ids, 'top_k': args.top_k, 'seconds': args.seconds,
        'runtime': backend.runtime_info, 'methods': {},
        'limitations': ['Diagnostic F1, not official score or verified state success',
            'Familiar train development samples; reference transcripts, no ASR',
            '90-second development budget includes planning; official limit remains 30 seconds',
            'Both arms retain original context and complete observed history',
            'Adaptive arm replans every round and sends the current goal to Caller',
            'Contrast changes both retrieval updates and Caller guidance; not a guidance-only ablation']}
    for method in ('bm25', 'adaptive_plan'):
        selected = [r for r in records if r['method'] == method]
        summary['methods'][method] = {
            'samples': len(selected),
            'diagnostic_f1': statistics.mean(r['metrics']['f1'] for r in selected),
            'mean_seconds': statistics.mean(r['elapsed_seconds'] for r in selected),
            'samples_ge_30s': sum(r['elapsed_seconds'] >= 30 for r in selected),
            'termination_counts': dict(Counter(r['trace']['termination'] for r in selected)),
            'planner_errors': sum(bool(s['planner_error']) for r in selected for s in r['steps']),
            'mean_planner_calls': statistics.mean(len(r['steps']) for r in selected),
            'mean_planner_seconds': statistics.mean(sum(s['planner_seconds'] for s in r['steps']) for r in selected),
            'sample_errors': sum('error' in r for r in selected)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

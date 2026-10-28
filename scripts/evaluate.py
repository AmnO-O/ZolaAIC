"""Evaluate train predictions; labels never enter the prediction backend.

Optional reference replay is a separate evaluator operation, not an agent retry.
No environment file or hidden session state is read by this script.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.input_adapter import SplitInput
from lifemate.evaluation import call_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--train-dir', default='train')
    parser.add_argument('--reference-replay', action='store_true', help='Separate gold session per train sample to derive reference hashes')
    parser.add_argument('--output', default='artifacts/evaluation.json')
    args = parser.parse_args()
    split = SplitInput(args.train_dir)
    if split.split != 'train':
        raise ValueError('Evaluation labels must come from train only')
    specs = {t['name']: t for app in split.catalog['apps'] for t in app['tools']}
    gold = {r['id']: r for r in map(json.loads, (split.directory/'train.jsonl').read_text(encoding='utf-8').splitlines())}
    seen = set()
    results = []
    for line in Path(args.predictions).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['id'] in seen or row['id'] not in gold:
            raise ValueError('Duplicate or unknown prediction ID')
        seen.add(row['id'])
        label = gold[row['id']]['label']
        metrics = call_metrics(row['tool_calls'], label['tool_calls'], specs)
        metrics['id'] = row['id']
        if args.reference_replay:
            reference = split.session(row['id'])
            for call in label['tool_calls']:
                reference.call(call['name'], call['arguments'])
            metrics['state_match'] = row['state_hash'] == reference.get_state_hash()
        results.append(metrics)
    report = {'samples': len(results), 'missing_train_predictions': len(gold)-len(seen),
              'macro_tool_f1': sum(r['f1'] for r in results)/len(results) if results else None,
              'limitation': 'Diagnostic F1; official score equation/tokenizer not yet verified',
              'results': results}
    if args.reference_replay and results:
        report['state_match_rate'] = sum(r['state_match'] for r in results)/len(results)
        report['state_gated_macro_f1_proxy'] = sum(r['f1']*r['state_match'] for r in results)/len(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__': main()

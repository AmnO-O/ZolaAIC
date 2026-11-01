"""Compare saved train predictions using published state-gated call matching.

Evaluator only: no model invocation, no tool-call replay, no direct environment
file access. One evaluation ToolEnv per sample supplies initial state via API;
train label.final_state contains complete replacements for changed apps.
The organiser's state_hash function hashes that reconstructed gold snapshot.
Timeouts are deliberately ignored. This is not the organiser's scorer binary.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.evaluation import call_metrics
from lifemate.input_adapter import SplitInput


def expected_hash(env, final_state, hash_state):
    initial=copy.deepcopy(env.get_state())
    if hash_state(initial)!=env.get_state_hash():
        raise ValueError('Runtime snapshot and canonical hash disagree')
    # Pure evaluator construction; neither prediction nor execution state is changed.
    initial.update(copy.deepcopy(final_state))
    return hash_state(initial)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',required=True)
    p.add_argument('--predictions',required=True)
    p.add_argument('--ids',required=True,help='Evaluation denominator: JSON list of sample IDs')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    split=SplitInput(args.train_dir)
    if split.split!='train':raise ValueError('Train-only label evaluation')
    ids=json.loads(Path(args.ids).read_text(encoding='utf-8'))
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Expected unique nonempty IDs')
    wanted=set(ids)
    gold={}
    for line in (split.directory/'train.jsonl').open(encoding='utf-8'):
        if line.strip():
            row=json.loads(line)
            if row['id'] in wanted:gold[row['id']]=row
    if set(gold)!=wanted:raise ValueError('Unknown evaluation IDs')
    predictions={}
    for line in Path(args.predictions).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row['id'] in predictions or row['id'] not in wanted:
            raise ValueError('Duplicate/unexpected prediction ID')
        predictions[row['id']]=row
    runtime=split.runtime()
    results=[]
    for identifier in ids:
        label=gold[identifier]['label']
        env=split.session(identifier)  # Exactly one fresh evaluator session; no calls/replay.
        reference=expected_hash(env,label['final_state'],runtime.state_hash)
        pred=predictions.get(identifier)
        metrics=call_metrics(pred['tool_calls'] if pred else [],label['tool_calls'],
                             {t['name']:t for t in split.tools(gold[identifier])})
        match=bool(pred and pred['state_hash']==reference)
        if env.get_tool_calls():raise RuntimeError('Evaluator unexpectedly executed tools')
        results.append({'id':identifier,'missing_prediction':pred is None,
            'predicted_state_hash':pred['state_hash'] if pred else None,'gold_state_hash':reference,
            'state_match':match,'call_metrics':metrics,
            'unmatched_submitted':metrics['predicted']-metrics['matched'],
            'score':metrics['f1'] if match else 0.})
    report={'samples':len(results),'timeout_applied':False,
        'method':'State-gated one-to-one call F1; exact arguments and fuzzy token-F1 > 0.6',
        'formula':'state_match * 2*M/(|G|+M+U); empty gold scores 1 only for empty prediction and matching state',
        'mean_score':sum(r['score'] for r in results)/len(results),
        'state_match_count':sum(r['state_match'] for r in results),
        'state_match_rate':sum(r['state_match'] for r in results)/len(results),
        'ungated_macro_call_f1':sum(r['call_metrics']['f1'] for r in results)/len(results),
        'missing_predictions':sum(r['missing_prediction'] for r in results),
        'limitations':['Reimplementation of published rules, not the organiser scorer',
            'Extracted local description omits displayed equation; exact official formula parity unverified',
            'No runtime penalty or ASR assessment; saved predictions are reused without replay'],
        'results':results}
    with Path(args.output).open('x',encoding='utf-8') as target:
        json.dump(report,target,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

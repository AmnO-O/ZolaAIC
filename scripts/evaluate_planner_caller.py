"""Private train pilot: adaptive Planner -> BM25 -> native Caller -> ToolEnv."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.agent import run_agent
from lifemate.evaluation import call_metrics
from lifemate.input_adapter import SplitInput
from lifemate.planner_caller import PlannerCaller
from compare_retrieval import prompt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('train-dir','model','ids','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--seconds',type=float,default=90);p.add_argument('--top-k',type=int,default=8)
    args=p.parse_args()
    if not 0<args.seconds<float('inf') or args.top_k<1:raise ValueError('Invalid limits')
    split=SplitInput(args.train_dir)
    if split.split!='train':raise ValueError('Only train may be scored')
    ids=json.loads(Path(args.ids).read_text());wanted=set(ids)
    if len(ids)!=len(wanted):raise ValueError('Duplicate IDs')
    rows={}
    with (split.directory/'train.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['id'] in wanted:rows[row['id']]=row
    if set(rows)!=wanted:raise ValueError('Missing IDs')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    os.environ.update(LIFEMATE_PROMPT_CACHE='0',LIFEMATE_MAX_NEW_TOKENS='384')
    from lifemate.backend import LocalBackend
    backend=LocalBackend(args.model)
    records=[]
    with (out/'samples.jsonl').open('x',encoding='utf-8') as logs:
        for identifier in ids:
            row=rows[identifier];backend.reset_sample();backend.stats=[]
            start=time.monotonic();end=start+args.seconds
            tools=split.tools(row);messages=prompt(row,split.catalog)
            controller=PlannerCaller(backend,tools,split.catalog,args.top_k)
            env=None;prediction={'tool_calls':[]};trace={'termination':'error','decisions':0}
            record={'id':identifier}
            try:
                env=split.session(identifier)
                record['initial_hash']=env.get_state_hash()
                prediction,trace=run_agent(env,controller.decide,messages,tools,end,max_turns=8)
            except Exception as error:
                record['error']=type(error).__name__+': '+str(error)
                if env is not None:prediction={'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()}
            elapsed=time.monotonic()-start
            # Gold remains evaluator-side. It never enters prompt or controller.
            specs={t['name']:t for t in tools};gold=row['label']['tool_calls']
            shown=set(n for step in controller.steps for n in step.get('routing',{}).get('selected_names',[]))
            gold_names={c['name'] for c in gold}
            record.update(prediction=prediction,trace=trace,steps=controller.steps,
                generation_stats=backend.stats,elapsed_seconds=elapsed,
                metrics=call_metrics(prediction['tool_calls'],gold,specs),
                cumulative_tool_recall=len(shown&gold_names)/len(gold_names) if gold_names else 1.,
                all_gold_tools_ever_visible=gold_names<=shown)
            records.append(record);logs.write(json.dumps(record,ensure_ascii=False)+'\n');logs.flush()
    summary={'samples':len(records),'ids':ids,'runtime':backend.runtime_info,
        'diagnostic_f1':statistics.mean(r['metrics']['f1'] for r in records),
        'mean_seconds':statistics.mean(r['elapsed_seconds'] for r in records),
        'samples_ge_30s':sum(r['elapsed_seconds']>=30 for r in records),
        'mean_planner_calls':statistics.mean(len(r['steps']) for r in records),
        'mean_caller_calls':statistics.mean(sum(s['caller_called'] for s in r['steps']) for r in records),
        'planner_errors':sum(bool(s['planner_error']) for r in records for s in r['steps']),
        'termination_counts':dict(Counter(r['trace']['termination'] for r in records)),
        'sample_errors':sum('error' in r for r in records),
        'limitations':['Untrained same-base role prompts, not a reproduction of trained alpha-UMi',
            'Familiar train development samples, reference transcripts, no ASR',
            '90s development budget includes all planning and calling; not official score/state verification',
            'Compared to historical same-settings runs, not an interleaved three-arm rerun',
            'Cumulative schema recall is not comparable to static per-episode recall as a like-for-like metric']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

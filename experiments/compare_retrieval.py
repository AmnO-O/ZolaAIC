"""Private train-only paired pilot: original BM25 vs planner-expanded BM25."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.agent import run_agent
from lifemate.context import clean_context_for_prompt
from lifemate.evaluation import call_metrics
from lifemate.input_adapter import SplitInput
from lifemate.messages import chat_messages, native_tools
from lifemate.prompts import apply_policy
from lifemate.retrieval import CatalogSearch
from lifemate.values import annotate_spoken_phones


def prompt(row,catalog):
    messages=[{'role':'system','content':{'context':clean_context_for_prompt(row['context'],catalog)}}]
    for turn in row.get('history',[]):
        messages.append({'role':'user','content':annotate_spoken_phones(turn['audio_text'])}
                        if turn['role']=='user' else copy.deepcopy(turn))
    messages.append({'role':'user','content':annotate_spoken_phones(row['audio_text'])})
    return messages


def make_plan(backend,messages,tools,catalog,seconds):
    names={t['name'].split('.')[0] for t in tools}
    capabilities={a['id']:a.get('description','') for a in catalog['apps'] if a['id'] in names}
    request=[{'role':'system','content':
        'Expand the user request into short Vietnamese search goals for finding tools. '
        'Preserve corrections, negations, independent actions and conditional dependencies. '
        'Do not execute anything or invent IDs, amounts, observations or tool names. '
        'Return only JSON: {"goals":[{"query":"search phrase","when":"condition or always"}]}. '
        'At most 6 goals. This is tentative search guidance, not evidence.'},
        {'role':'user','content':json.dumps({'capabilities':capabilities,'conversation':messages},ensure_ascii=False)}]
    text=backend._generate(request,[],seconds,max_tokens=384,controls=False,phase='planner')
    if not backend._generation_complete: raise ValueError('Planner output truncated')
    plan=json.loads(text)
    goals=plan.get('goals')
    if (not isinstance(goals,list) or not 1<=len(goals)<=6 or any(
        not isinstance(g,dict) or set(g)!={'query','when'} or
        not all(isinstance(g[k],str) and g[k].strip() for k in ('query','when')) for g in goals)):
        raise ValueError('Invalid planner goals')
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',required=True);p.add_argument('--model',required=True)
    p.add_argument('--output',required=True);p.add_argument('--ids',required=True)
    p.add_argument('--seconds',type=float,default=90);p.add_argument('--top-k',type=int,default=8)
    args=p.parse_args()
    split=SplitInput(args.train_dir)
    if split.split!='train': raise ValueError('Only train may be scored')
    if not 0<args.seconds<float('inf') or args.top_k<1: raise ValueError('Invalid limits')
    ids=json.loads(Path(args.ids).read_text()); selected=set(ids)
    rows={}
    with (split.directory/'train.jsonl').open(encoding='utf-8') as source:
        for line in source:
            row=json.loads(line)
            if row['id'] in selected: rows[row['id']]=row
    if len(rows)!=len(ids): raise ValueError('Missing/duplicate sample IDs')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    os.environ.update(LIFEMATE_PROMPT_CACHE='0',LIFEMATE_MAX_NEW_TOKENS='384')
    from lifemate.backend import LocalBackend
    backend=LocalBackend(args.model)
    def tokens(messages,tools):
        encoded=backend.tokenizer.apply_chat_template(apply_policy(chat_messages(messages)),
            tools=native_tools(tools),add_generation_prompt=True,enable_thinking=False,tokenize=True)
        return len(encoded['input_ids'] if hasattr(encoded,'keys') else encoded)
    records=[]
    with (out/'samples.jsonl').open('x',encoding='utf-8') as logs:
        for i,identifier in enumerate(ids):
            row=rows[identifier];tools=split.tools(row);messages=prompt(row,split.catalog)
            search=CatalogSearch(tools,split.catalog)
            # Gold is used only below by the evaluator, never by routing or the actor.
            original='\n'.join(m['content'] for m in messages if m['role']=='user')
            order=['bm25','planner_bm25'] if i%2==0 else ['planner_bm25','bm25']
            hashes=[]
            for method in order:
                backend.reset_sample();backend.stats=[];start=time.monotonic();end=start+args.seconds
                record={'id':identifier,'method':method,'plan':None,'planner_error':None}
                queries=[original];planner_seconds=0
                if method=='planner_bm25':
                    began=time.monotonic()
                    try:
                        plan=make_plan(backend,messages,tools,split.catalog,min(30,max(.01,end-time.monotonic())))
                        record['plan']=plan;queries += [g['query'] for g in plan['goals']]
                    except (ValueError,TimeoutError) as error:
                        record['planner_error']=str(error)
                    planner_seconds=time.monotonic()-began
                before=time.monotonic();visible,routing=search.select(queries,args.top_k)
                retrieval_seconds=time.monotonic()-before
                full_tokens=tokens(messages,tools);visible_tokens=tokens(messages,visible)
                prediction={'tool_calls':[]};trace={'termination':'error','decisions':0}
                env=None
                try:
                    env=split.session(identifier);hashes.append(env.get_state_hash())
                    def decide(history,permitted,remaining):
                        return backend.decide(history,visible,remaining)
                    # Full permitted schemas remain authoritative for execution/grounding.
                    prediction,trace=run_agent(env,decide,messages,tools,end,max_turns=8)
                except Exception as error:
                    record['error']=type(error).__name__+': '+str(error)
                    if env is not None:
                        prediction={'tool_calls':env.get_tool_calls(),'state_hash':env.get_state_hash()}
                gold=row['label']['tool_calls'];specs={t['name']:t for t in tools}
                gold_names={c['name'] for c in gold};shown={t['name'] for t in visible}
                record.update(routing=routing,planner_seconds=planner_seconds,retrieval_seconds=retrieval_seconds,
                    full_input_tokens=full_tokens,selected_input_tokens=visible_tokens,
                    all_gold_tools_visible=gold_names<=shown,
                    gold_tool_recall=len(gold_names&shown)/len(gold_names) if gold_names else 1.,
                    metrics=call_metrics(prediction['tool_calls'],gold,specs),
                    elapsed_seconds=time.monotonic()-start,prediction=prediction,trace=trace,
                    generation_stats=copy.deepcopy(backend.stats))
                records.append(record);logs.write(json.dumps(record,ensure_ascii=False)+'\n');logs.flush()
            if len(set(hashes))>1: raise RuntimeError('Paired environments have different starting hashes')
    summary={'limitations':['Diagnostic F1, not official score or verified state success',
        '10 familiar development samples; train reference transcripts, no ASR',
        '90s development deadline includes planner; not competition-compliant',
        'Planner guides retrieval only; plan not added to executor, isolating retrieval effect'],
        'ids':ids,'top_k':args.top_k,'runtime':backend.runtime_info,'methods':{}}
    for method in ('bm25','planner_bm25'):
        rs=[r for r in records if r['method']==method]
        summary['methods'][method]={k:statistics.mean(r[k] for r in rs) for k in
            ('elapsed_seconds','planner_seconds','retrieval_seconds','full_input_tokens',
             'selected_input_tokens','gold_tool_recall','all_gold_tools_visible')}
        summary['methods'][method].update(diagnostic_f1=statistics.mean(r['metrics']['f1'] for r in rs),
            deadlines=sum(r['trace']['termination']=='deadline' for r in rs),
            planner_errors=sum(r['planner_error'] is not None for r in rs))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

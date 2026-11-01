"""Local train-audio CPU benchmark; fusion never receives reference transcripts."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import gc
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def tokens(text):
    text=unicodedata.normalize('NFC',text).lower()
    return ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split()


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=current
    return row[-1]


def fuse(candidates):
    """Simple anchor-aligned majority; not a reproduction of NIST ROVER."""
    sequences=[tokens(c) for c in candidates]
    base=next((s for s in sequences if s),[])
    aligned=[]
    for seq in sequences:
        words=['']*len(base);insertions=['']*(len(base)+1)
        for op,i1,i2,j1,j2 in SequenceMatcher(None,base,seq,autojunk=False).get_opcodes():
            if op in ('equal','replace'):
                for offset,word in enumerate(seq[j1:j1+min(i2-i1,j2-j1)]):words[i1+offset]=word
                if j2-j1>i2-i1:insertions[i2]=' '.join(seq[j1+i2-i1:j2])
            elif op=='insert':insertions[i1]=' '.join(seq[j1:j2])
        aligned.append((words,insertions))
    def vote(values,anchor):
        counts=Counter(values);best=max(counts.values())
        return anchor if counts[anchor]==best else next(v for v in values if counts[v]==best)
    output=[]
    for i in range(len(base)+1):
        output.append(vote([ins[i] for _,ins in aligned],''))
        if i<len(base):output.append(vote([ws[i] for ws,_ in aligned],base[i]))
    return ' '.join(x for x in output if x)


def medoid(candidates):
    sequences=[tokens(c) for c in candidates]
    scores=[sum(distance(s,t)/max(1,len(s),len(t)) for t in sequences) for s in sequences]
    return candidates[min(range(len(scores)),key=lambda i:scores[i])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=10)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS=str(args.threads))
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor,AutoModelForSpeechSeq2Seq,StoppingCriteria,StoppingCriteriaList
    from lifemate.asr import GipformerASR
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.use_deterministic_algorithms(True)
    class ObserveEOS(StoppingCriteria):
        # Whisper strips EOS from returned text sequences in this runtime.
        # Observe completion during decoding rather than checking cleaned output.
        def __init__(self,eos):
            self.ids=eos if isinstance(eos,list) else [eos]
            self.saw_eos=False
        def __call__(self,input_ids,scores,**kwargs):
            self.saw_eos|=int(input_ids[0,-1]) in self.ids
            return False
    train=Path(args.train_dir).resolve();out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for line in (train/'train.jsonl').open(encoding='utf-8'):
        if not line.strip():continue
        row=json.loads(line);path=(train/row['audio']).resolve()
        if not path.is_relative_to(train/'audio'):raise ValueError('Audio path outside train/audio')
        info=sf.info(path)
        if 0<info.duration<=25:candidates.append((row,path,info.duration))
    selected=random.Random(42).sample(sorted(candidates,key=lambda x:x[0]['id']),args.samples)
    (out/'ids.json').write_text(json.dumps([r['id'] for r,_,_ in selected],indent=2))
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    report={'status':'running','samples':args.samples,'threads':args.threads,'seed':42,
        'selection':'Random train latest-audio samples of duration <=25 seconds; no histories',
        'models':manifest,'results':[],'load_seconds':{},'parameters':{},
        'limitations':['Small diagnostic sample, no downstream agent execution',
            'WER normalizes NFC/case/punctuation but does not equate digits with spoken numbers',
            'Both PhoWhisper models share a family; errors may be correlated',
            'Voting is a simple ROVER-inspired prototype, not the reference ROVER implementation',
            'CPU inference times exclude model loading and should not be extrapolated to A30']}
    records={r['id']:{'id':r['id'],'reference':r['audio_text'],'duration':duration,
        'predictions':{},'seconds':{}} for r,_,duration in selected}
    for spec in manifest:
        name=spec['id'];root=spec['path']
        if 'gipformer' not in name:
            wait_end=time.monotonic()+1200
            while not (Path(root)/'pytorch_model.bin').is_file():
                (out/'progress.json').write_text(json.dumps({'stage':'waiting_for_public_weights','model':name}))
                if time.monotonic()>wait_end:raise TimeoutError('Public model download not ready after 20 minutes')
                time.sleep(5)
        began=time.monotonic()
        if 'gipformer' in name:
            model=GipformerASR(root,num_threads=args.threads);processor=None
            report['parameters'][name]=model.parameters
        else:
            processor=AutoProcessor.from_pretrained(root,local_files_only=True)
            model=AutoModelForSpeechSeq2Seq.from_pretrained(root,local_files_only=True,
                torch_dtype=torch.float32).eval().to('cpu')
            model.generation_config.forced_decoder_ids=None
            report['parameters'][name]=sum(p.numel() for p in model.parameters())
        report['load_seconds'][name]=time.monotonic()-began
        for row,path,duration in selected:
            started=time.monotonic()
            if processor is None:text=model.transcribe(path,remaining_seconds=300)
            else:
                samples,rate=sf.read(path,dtype='float32')
                if samples.ndim>1:samples=samples.mean(axis=1)
                if rate!=16000:
                    from scipy.signal import resample_poly
                    from math import gcd
                    divisor=gcd(rate,16000);samples=resample_poly(samples,16000//divisor,rate//divisor)
                inputs=processor(samples,sampling_rate=16000,return_tensors='pt')
                completion=ObserveEOS(model.generation_config.eos_token_id)
                with torch.inference_mode():
                    output=model.generate(**inputs,language='vi',task='transcribe',do_sample=False,
                        num_beams=1,max_new_tokens=256,return_timestamps=False,
                        stopping_criteria=StoppingCriteriaList([completion]))
                if not completion.saw_eos:
                    raise ValueError('ASR generation truncated; refusing to score incomplete output')
                text=processor.batch_decode(output,skip_special_tokens=True)[0].strip()
            records[row['id']]['predictions'][name]=text
            records[row['id']]['seconds'][name]=time.monotonic()-started
            (out/'progress.json').write_text(json.dumps({'model':name,'last_id':row['id'],
                'completed_in_model':sum(name in r['predictions'] for r in records.values())}))
            with (out/'candidates.jsonl').open('a',encoding='utf-8') as target:
                target.write(json.dumps({'id':row['id'],'model':name,'prediction':text,
                    'seconds':records[row['id']]['seconds'][name]},ensure_ascii=False)+'\n')
        del model,processor;gc.collect()
    names=[s['id'] for s in manifest]
    for row in records.values():
        texts=[row['predictions'][n] for n in names]
        row['predictions']['aligned_vote']=fuse(texts)
        row['predictions']['medoid']=medoid(texts)
        # Oracle selection is evaluation-only and is never used by fusion.
        reference=tokens(row['reference'])
        row['predictions']['oracle_candidate']=min(texts,key=lambda x:distance(reference,tokens(x)))
        row['word_errors']={n:distance(reference,tokens(t)) for n,t in row['predictions'].items()}
        row['reference_words']=len(reference)
        report['results'].append(row)
    total=sum(r['reference_words'] for r in records.values())
    report['summary']={name:{'corpus_wer':sum(r['word_errors'][name] for r in records.values())/total,
        'mean_seconds':statistics.mean(r['seconds'][name] for r in records.values()) if name in names else None}
        for name in [*names,'aligned_vote','medoid','oracle_candidate']}
    report['mean_serial_ensemble_seconds']=statistics.mean(sum(r['seconds'].values()) for r in records.values())
    report['status']='completed'
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['results','models']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Runner integration using invented inputs and an in-memory runtime stub."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RunnerTests(unittest.TestCase):
    def test_runner_exports_actual_stub_log_and_excludes_labels(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'artifacts', prefix='runner-test-') as temporary:
            base=Path(temporary)
            self.assertTrue(base.resolve().is_relative_to((root/'artifacts').resolve()))
            split=base/'train'
            runtime=split/'zaic_tools'
            runtime.mkdir(parents=True)
            (split/'train_tools_catalog.json').write_text(json.dumps({'apps':[{'tools':[
                {'name':'memo.create','parameters':{'type':'object','properties':{'text':{'type':'string'}}}}]}]}))
            (split/'train.jsonl').write_text(json.dumps({'id':'invented_1','audio':'audio/a.wav',
                'audio_text':'REFERENCE_MUST_NOT_LEAK','label':{'secret':'LABEL_MUST_NOT_LEAK'},
                'history':[],'context':{'installed_apps':{'memo':{'tools':['memo.create']}}}}))
            (runtime/'__init__.py').write_text('''
created=set()
class Session:
 def __init__(self):self.calls=[]
 def validate(self,name,args):return {'ok':True,'error':None}
 def call(self,name,args):
  assert not name.startswith('agent.')
  self.calls.append({'name':name,'arguments':args});return {'ok':True}
 def get_tool_calls(self):return self.calls.copy()
 def get_state_hash(self):return '0'*64
def ToolEnv(split,sample_id,root):
 if sample_id in created:raise RuntimeError('Session created twice')
 created.add(sample_id)
 import os
 if os.environ.get('ENV_CREATION_LOG'):
  with open(os.environ['ENV_CREATION_LOG'],'a') as log:print(sample_id,file=log)
 if sample_id=='broken':raise RuntimeError('synthetic session failure')
 return Session()
''')
            (base/'fake_backend.py').write_text('''
class Backend:
 def reset_sample(self):
  import os
  with open(os.environ['RESET_LOG'],'a') as log:log.write('reset\\n')
 def transcribe(self,path,remaining):return 'Invented memo request'
 def decide(self,messages,tools,remaining):
  import os
  if os.environ.get('SIMULATE_FAILURE')=='1':raise RuntimeError('invented model failure')
  assert 'REFERENCE_MUST_NOT_LEAK' not in str(messages)
  assert 'LABEL_MUST_NOT_LEAK' not in str(messages)
  if any(m['role']=='tool' for m in messages):
   return []
  return [{'name':'memo.create','arguments':{'text':'invented memo'}}]
def create():return Backend()
''')
            output=base/'submission.jsonl'
            environment=os.environ.copy()
            environment['PYTHONPATH']=str(base)
            environment['RESET_LOG']=str(base/'resets.txt')
            result=subprocess.run([sys.executable,'-B',str(root/'scripts/infer.py'),
                '--split-dir',str(split),'--backend','fake_backend:create','--budget-seconds','60','--output',str(output),
                '--step-log',str(base/'steps.jsonl'),'--verbose-steps'],
                env=environment,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len((base/'resets.txt').read_text().splitlines()),2)
            row=json.loads(output.read_text())
            self.assertEqual(row['tool_calls'],[{'name':'memo.create','arguments':{'text':'invented memo'}}])
            self.assertEqual(row['state_hash'],'0'*64)
            trace=json.loads(output.with_suffix('.trace.jsonl').read_text())
            self.assertEqual(trace['termination'],'finished')
            self.assertEqual(trace['budget_seconds'],60)
            raw_steps=(base/'steps.jsonl').read_text(encoding='utf-8')
            self.assertNotIn('REFERENCE_MUST_NOT_LEAK',raw_steps)
            self.assertNotIn('LABEL_MUST_NOT_LEAK',raw_steps)
            events=[json.loads(line) for line in raw_steps.splitlines()]
            stages=[event['stage'] for event in events]
            self.assertEqual(stages[0],'sample.input')
            self.assertEqual(stages[-1],'sample.output')
            self.assertEqual(events[-1]['output'],row)
            self.assertEqual([event['step'] for event in events],list(range(1,len(events)+1)))
            self.assertTrue(all(event['id']=='invented_1' for event in events))
            for stage in ('asr.input','asr.output','decision.input','decision.output','tool.validation','tool.input','tool.output'):
                self.assertIn(stage,stages)
            self.assertIn('[invented_1]',result.stdout)
            self.assertIn('Invented memo request',result.stdout)
            self.assertIn('"input":',result.stdout)
            self.assertIn('"output":',result.stdout)
            self.assertNotIn('REFERENCE_MUST_NOT_LEAK',result.stdout)
            self.assertNotIn('LABEL_MUST_NOT_LEAK',result.stdout)
            failed_output=base/'failed.jsonl'
            failed_steps=base/'failed.steps.jsonl'
            failed=subprocess.run([sys.executable,'-B',str(root/'scripts/infer.py'),
                '--split-dir',str(split),'--backend','fake_backend:create','--output',str(failed_output),
                '--step-log',str(failed_steps)],env={**environment,'SIMULATE_FAILURE':'1'},
                capture_output=True,text=True,timeout=20)
            self.assertEqual(failed.returncode,0,failed.stderr)
            self.assertEqual(len((base/'resets.txt').read_text().splitlines()),4)
            failure_events=[json.loads(line) for line in failed_steps.read_text().splitlines()]
            self.assertIn('sample.error',[event['stage'] for event in failure_events])
            self.assertEqual(failure_events[-1]['termination'],'error')
            self.assertEqual(failure_events[-1]['output']['tool_calls'],[])
            original=json.loads((split/'train.jsonl').read_text())
            (split/'train.jsonl').write_text('\n'.join(json.dumps({**original,'id':sample_id})
                for sample_id in ('broken','good')))
            partial_output=base/'partial.jsonl'
            creations=base/'creations.txt'
            partial=subprocess.run([sys.executable,'-B',str(root/'scripts/infer.py'),
                '--split-dir',str(split),'--backend','fake_backend:create','--output',str(partial_output)],
                env={**environment,'ENV_CREATION_LOG':str(creations)},capture_output=True,text=True,timeout=20)
            self.assertNotEqual(partial.returncode,0)
            self.assertEqual(creations.read_text().splitlines(),['broken','good'])
            self.assertEqual([json.loads(line)['id'] for line in partial_output.read_text().splitlines()],['good'])
            partial_trace=[json.loads(line) for line in partial_output.with_suffix('.trace.jsonl').read_text().splitlines()]
            self.assertTrue(partial_trace[0]['missing_submission_row'])
            self.assertEqual(partial_trace[0]['error_type'],'RuntimeError')
            self.assertEqual(partial_trace[1]['termination'],'finished')
            self.assertEqual(len((base/'resets.txt').read_text().splitlines()),8)


if __name__=='__main__':unittest.main()

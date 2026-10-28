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
  self.calls.append({'name':name,'arguments':args});return {'ok':True}
 def get_tool_calls(self):return self.calls.copy()
 def get_state_hash(self):return '0'*64
def ToolEnv(split,sample_id,root):
 if sample_id in created:raise RuntimeError('Session created twice')
 created.add(sample_id)
 return Session()
''')
            (base/'fake_backend.py').write_text('''
class Backend:
 def transcribe(self,path,remaining):return 'Invented memo request'
 def decide(self,messages,tools,remaining):
  assert 'REFERENCE_MUST_NOT_LEAK' not in str(messages)
  assert 'LABEL_MUST_NOT_LEAK' not in str(messages)
  if any(m['role']=='tool' for m in messages):return []
  return [{'name':'memo.create','arguments':{'text':'invented memo'}}]
def create():return Backend()
''')
            output=base/'submission.jsonl'
            environment=os.environ.copy()
            environment['PYTHONPATH']=str(base)
            result=subprocess.run([sys.executable,'-B',str(root/'scripts/infer.py'),
                '--split-dir',str(split),'--backend','fake_backend:create','--budget-seconds','60','--output',str(output)],
                env=environment,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            row=json.loads(output.read_text())
            self.assertEqual(row['tool_calls'],[{'name':'memo.create','arguments':{'text':'invented memo'}}])
            self.assertEqual(row['state_hash'],'0'*64)
            trace=json.loads(output.with_suffix('.trace.jsonl').read_text())
            self.assertEqual(trace['termination'],'finished')
            self.assertEqual(trace['budget_seconds'],60)


if __name__=='__main__':unittest.main()

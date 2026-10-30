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
  if os.environ.get('LIFEMATE_PROMPT_VARIANT','').startswith('memory_'):
   from lifemate.planning import Decision
   if any(m['role']=='tool' for m in messages):return Decision(memory_update='No remaining work')
   return Decision(tool_calls=[{'name':'memo.create','arguments':{'text':'invented memo'}}],memory_update='Create memo')
  if os.environ.get('LIFEMATE_PROMPT_VARIANT')=='planning':
   from lifemate.planning import Decision
   if any(m['role']=='tool' for m in messages):return Decision(plan_update='No remaining work')
   return Decision([{'name':'memo.create','arguments':{'text':'invented memo'}}], 'Create the requested memo')
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
                '--step-log',str(base/'steps.jsonl'),'--verbose-steps','--prompt-variant','direct'],
                env=environment,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len((base/'resets.txt').read_text().splitlines()),2)
            row=json.loads(output.read_text())
            self.assertEqual(row['tool_calls'],[{'name':'memo.create','arguments':{'text':'invented memo'}}])
            self.assertEqual(row['state_hash'],'0'*64)
            trace=json.loads(output.with_suffix('.trace.jsonl').read_text())
            self.assertEqual(trace['termination'],'finished')
            self.assertEqual(trace['controller'], 'direct_v2')
            self.assertFalse(trace['task_success_verified'])
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
            planning_output = base/'planning.jsonl'
            planning_run = subprocess.run([sys.executable, '-B', str(root/'scripts/infer.py'),
                '--split-dir', str(split), '--backend', 'fake_backend:create', '--output', str(planning_output),
                '--prompt-variant', 'planning', '--max-turns', '16', '--budget-seconds', '90',
                '--step-log', str(base/'planning.steps.jsonl')],
                env=environment, capture_output=True, text=True, timeout=20)
            self.assertEqual(planning_run.returncode, 0, planning_run.stderr)
            planning_row = json.loads(planning_output.read_text())
            self.assertEqual(planning_row['tool_calls'], row['tool_calls'])
            self.assertNotIn('plan', planning_row)
            planning_trace = json.loads(planning_output.with_suffix('.trace.jsonl').read_text())
            self.assertTrue(planning_trace['planning'])
            self.assertEqual(planning_trace['budget_seconds'], 90)
            self.assertEqual(planning_trace['max_turns'], 16)
            self.assertEqual(len(planning_trace['plan_revisions']), 2)
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
            self.assertEqual(len((base/'resets.txt').read_text().splitlines()),10)
            # Exercise both new modes through the actual CLI, including fresh per-sample environments.
            for variant in ('memory_full', 'memory_compact'):
                target = base/(variant + '.jsonl')
                creations = base/(variant + '.creations.txt')
                run = subprocess.run([sys.executable, '-B', str(root/'scripts/infer.py'),
                    '--split-dir', str(split), '--backend', 'fake_backend:create',
                    '--output', str(target), '--prompt-variant', variant,
                    '--step-log', str(base/(variant + '.steps.jsonl'))],
                    env={**environment, 'ENV_CREATION_LOG': str(creations)},
                    capture_output=True, text=True, timeout=20)
                self.assertNotEqual(run.returncode, 0)  # synthetic broken sample remains isolated
                self.assertEqual(creations.read_text().splitlines(), ['broken', 'good'])
                exported = json.loads(target.read_text())
                self.assertEqual(exported['tool_calls'], row['tool_calls'])
                self.assertEqual(set(exported), {'id', 'tool_calls', 'state_hash'})
                traces = [json.loads(line) for line in target.with_suffix('.trace.jsonl').read_text().splitlines()]
                self.assertEqual(traces[1]['memory_mode'], variant)
                self.assertEqual(len(traces[1]['memory_revisions']), 2)
                self.assertEqual(traces[1]['termination'], 'finished')


if __name__=='__main__':unittest.main()

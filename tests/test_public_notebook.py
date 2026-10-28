"""Execute the public notebook's export gate on synthetic, metadata-only files."""
from collections import Counter
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicNotebookTests(unittest.TestCase):
    def test_progress_waits_for_complete_lines_without_double_counting(self):
        notebook=json.loads((ROOT/'notebooks/kaggle_public_test.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in notebook['cells']
            if c['cell_type']=='code' and 'def progress()' in ''.join(c['source'])))
        function=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='progress')
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary:
            trace=Path(temporary)/'trace.jsonl'
            first=json.dumps({'id':'p1','termination':'finished','elapsed_seconds':0.1})
            trace.write_text(first[:-1])
            environment={'completed':0,'trace_offset':0,'log_offset':0,'trace_path':trace,
                'log_path':Path(temporary)/'log','LIVE_STEPS':False,'expected_count':2,
                'terminations':Counter(),'json':json}
            exec(compile(ast.Module(body=[function],type_ignores=[]),'notebook-progress','exec'),environment)
            with contextlib.redirect_stdout(io.StringIO()):
                environment['progress']()
                self.assertEqual(environment['completed'],0)
                with trace.open('a') as target:target.write(first[-1:]+'\n')
                environment['progress']();environment['progress']()
                self.assertEqual(environment['completed'],1)
                with trace.open('a') as target:target.write(json.dumps({'id':'p2','termination':'deadline','elapsed_seconds':27})+'\n')
                environment['progress']()
            self.assertEqual(environment['completed'],2)
            self.assertEqual(environment['terminations'],{'finished':1,'deadline':1})

    def test_export_requires_complete_validation_and_preserves_existing_file(self):
        notebook=json.loads((ROOT/'notebooks/kaggle_public_test.ipynb').read_text(encoding='utf-8'))
        export=''.join(next(c['source'] for c in notebook['cells']
            if c['cell_type']=='code' and 'final_submission =' in ''.join(c['source'])))
        for scenario in ('complete','subset','existing'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary:
                base=Path(temporary);split=base/'public_test';split.mkdir()
                (split/'public_test.jsonl').write_text('\n'.join(json.dumps({
                    'id':sample_id,'context':{'installed_apps':{}}}) for sample_id in ('p1','p2')))
                run_dir=base/'run';run_dir.mkdir()
                submission=run_dir/'submission.jsonl'
                ids=('p1',) if scenario=='subset' else ('p1','p2')
                submission.write_text('\n'.join(json.dumps({'id':sample_id,'tool_calls':[],
                    'state_hash':'a'*64}) for sample_id in ids))
                trace=submission.with_suffix('.trace.jsonl')
                trace.write_text('\n'.join(json.dumps({'id':sample_id,'termination':'finished',
                    'elapsed_seconds':0.1}) for sample_id in ids))
                final=base/'final/submission.jsonl'
                if scenario=='existing':
                    final.parent.mkdir();final.write_text('previous result')
                def redirect_path(value):
                    return final if str(value)=='/kaggle/working/result/submission.jsonl' else Path(value)
                environment={'Path':redirect_path,'run_dir':run_dir,'submission':submission,
                    'trace_path':trace,'repo':ROOT,'split_dir':split,'sys':sys,'json':json,
                    'math':math,'Counter':Counter,
                    'subprocess':SimpleNamespace(run=lambda *a,**kw:subprocess.run(*a,capture_output=True,text=True,**kw))}
                with contextlib.redirect_stdout(io.StringIO()):
                    if scenario=='subset':
                        with self.assertRaises(RuntimeError):exec(compile(export,'notebook-export','exec'),environment)
                        self.assertFalse(final.exists())
                    elif scenario=='existing':
                        with self.assertRaises(FileExistsError):exec(compile(export,'notebook-export','exec'),environment)
                        self.assertEqual(final.read_text(),'previous result')
                    else:
                        exec(compile(export,'notebook-export','exec'),environment)
                        self.assertEqual(final.read_bytes(),submission.read_bytes())


if __name__=='__main__':unittest.main()

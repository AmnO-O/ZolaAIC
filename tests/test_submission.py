import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('submission_validator', ROOT / 'scripts/validate_submission.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class SubmissionTests(unittest.TestCase):
    def check(self, rows, raw=None):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temporary:
            directory = Path(temporary)
            (directory/'public_test.jsonl').write_text('\n'.join(json.dumps({
                'id':sample_id, 'context':{'installed_apps':{'notes':{'tools':['notes.create']}}}})
                for sample_id in ('public_1','public_2')), encoding='utf-8')
            candidate = directory/'submission.jsonl'
            candidate.write_text(raw if raw is not None else '\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            # There is no catalog, runtime, audio, or environment fixture: only metadata.
            return validator.validate_submission(directory,candidate)

    def row(self, sample_id='public_1', calls=None):
        return {'id':sample_id, 'tool_calls':calls or [], 'state_hash':'a'*64}

    def test_complete_file_without_environment_access(self):
        report = self.check([self.row(),self.row('public_2',[{'name':'notes.create','arguments':{'content':'x'}}])])
        self.assertTrue(report['valid'])
        self.assertEqual(report['expected_samples'],2)
        self.assertEqual(report['tool_calls'],1)
        self.assertEqual(len(report['sha256']),64)

    def test_subset_cannot_be_exported_as_complete_submission(self):
        report = self.check([self.row()])
        self.assertFalse(report['valid'])
        self.assertEqual(report['missing_ids'],['public_2'])

    def test_duplicate_and_unknown_ids_are_rejected(self):
        report = self.check([self.row(),self.row(),self.row('other')])
        self.assertFalse(report['valid'])
        self.assertEqual(report['duplicate_ids'],['public_1'])
        self.assertEqual(report['unknown_ids'],['other'])

    def test_malformed_hash_and_unauthorized_controls_are_rejected(self):
        for row in (dict(self.row(),state_hash='fake'),self.row(calls=[{'name':'agent.finish','arguments':{}}])):
            with self.subTest(row=row):
                self.assertFalse(self.check([row,self.row('public_2')])['valid'])

    def test_non_finite_values_and_duplicate_json_keys_are_rejected(self):
        raw='{"id":"public_1","id":"public_2","tool_calls":[],"state_hash":"'+('a'*64)+'"}'
        self.assertFalse(self.check([],raw=raw)['valid'])
        row=self.row(calls=[{'name':'notes.create','arguments':{'content':float('nan')}}])
        self.assertFalse(self.check([row,self.row('public_2')])['valid'])
        overflow=json.dumps(self.row(calls=[{'name':'notes.create','arguments':{'content':1.0}}])).replace('1.0','1e999')
        self.assertFalse(self.check([],raw=overflow)['valid'])


if __name__=='__main__':unittest.main()

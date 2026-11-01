import copy
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/score_train_results.py'
spec=importlib.util.spec_from_file_location('state_evaluation',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StateEvaluationTests(unittest.TestCase):
    def test_complete_changed_app_replaces_and_unchanged_app_survives(self):
        initial={'calendar':{'events':[1,2]},'contacts':{'items':[3]}}
        label={'calendar':{'events':[]}}
        initial_before=copy.deepcopy(initial);label_before=copy.deepcopy(label)
        class Env:
            def get_state(self):return initial
            def get_state_hash(self):return str(initial)
        result=module.expected_hash(Env(),label,str)
        self.assertEqual(result,str({'calendar':{'events':[]},'contacts':{'items':[3]}}))
        self.assertEqual(initial,initial_before)
        self.assertEqual(label,label_before)

    def test_inconsistent_runtime_hash_fails(self):
        class Env:
            def get_state(self):return {}
            def get_state_hash(self):return 'wrong'
        with self.assertRaises(ValueError):module.expected_hash(Env(),{},str)

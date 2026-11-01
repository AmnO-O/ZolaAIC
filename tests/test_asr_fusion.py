import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'experiments/asr_ensemble_cpu.py'
spec=importlib.util.spec_from_file_location('cpu_asr_fusion',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class FusionTests(unittest.TestCase):
    def test_majority_preserves_negation(self):
        self.assertEqual(module.fuse(['chuyển tiền','không chuyển tiền','không chuyển tiền']),
                         'không chuyển tiền')

    def test_majority_can_delete_an_extra_word(self):
        self.assertEqual(module.fuse(['không chuyển tiền','chuyển tiền','chuyển tiền']),'chuyển tiền')

    def test_three_way_tie_prefers_anchor(self):
        self.assertEqual(module.fuse(['đặt lúc sáu giờ','đặt lúc bảy giờ','đặt lúc tám giờ']),
                         'đặt lúc sáu giờ')

    def test_candidate_selection_is_reference_free(self):
        self.assertEqual(module.medoid(['gửi tiền cho mẹ','gửi tin cho mẹ','gửi tin cho mẹ']),
                         'gửi tin cho mẹ')

    def test_all_empty_and_normalization(self):
        self.assertEqual(module.fuse(['','','']),'')
        self.assertEqual(module.tokens('Không, CHUYỂN 50.000!'),['không','chuyển','50','000'])
        self.assertEqual(module.distance(['a'],['a','b']),1)

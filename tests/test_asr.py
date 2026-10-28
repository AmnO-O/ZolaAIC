import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.asr import GipformerASR


class ASRTests(unittest.TestCase):
    def test_local_assets_and_explicit_deterministic_cpu_settings(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'artifacts',prefix='asr-test-') as tmp:
            path=Path(tmp)
            self.assertTrue(path.resolve().is_relative_to((root/'artifacts').resolve()))
            (path/'config.json').write_text(json.dumps({'model_type':'zipformer-transducer','sample_rate':16000,'num_parameters':68625511}))
            for name in ('encoder.int8.onnx','decoder.int8.onnx','joiner.int8.onnx','tokens.txt'):
                (path/name).write_text('fixture')
            captured={}
            def factory(**kwargs):captured.update(kwargs);return object()
            asr=GipformerASR(path,recognizer_factory=factory)
            self.assertEqual(asr.parameters,68625511)
            self.assertEqual(captured['provider'],'cpu')
            self.assertEqual(captured['decoding_method'],'greedy_search')
            self.assertEqual(captured['num_threads'],4)
            with self.assertRaises(TimeoutError):asr.transcribe('not-opened.wav',0)
            (path/'encoder.int8.onnx').unlink()
            with self.assertRaises(FileNotFoundError):GipformerASR(path,recognizer_factory=factory)


if __name__=='__main__':unittest.main()

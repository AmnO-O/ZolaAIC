"""Local-only Gipformer ONNX recognizer, independent of the LLM runtime."""
import json
from pathlib import Path
import time


class GipformerASR:
    def __init__(self, model_dir, precision='int8', num_threads=4, recognizer_factory=None):
        if precision not in ('int8','fp32') or num_threads < 1:
            raise ValueError('Use int8/fp32 and a positive CPU thread count')
        root=Path(model_dir)
        config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        if config.get('model_type')!='zipformer-transducer' or config.get('sample_rate')!=16000:
            raise ValueError('Expected a local 16 kHz Gipformer transducer export')
        self.parameters=int(config['num_parameters'])
        if not 0 < self.parameters < 6_000_000_000:
            raise ValueError('Invalid ASR parameter metadata')
        suffix='.int8.onnx' if precision=='int8' else '.onnx'
        files={key:root/(key+suffix) for key in ('encoder','decoder','joiner')}
        files['tokens']=root/'tokens.txt'
        for path in files.values():
            if not path.is_file():raise FileNotFoundError(f'Missing local ASR asset: {path}')
        if recognizer_factory is None:
            import sherpa_onnx
            recognizer_factory=sherpa_onnx.OfflineRecognizer.from_transducer
        self.recognizer=recognizer_factory(**{key:str(path) for key,path in files.items()},
            sample_rate=16000,feature_dim=80,num_threads=num_threads,
            decoding_method='greedy_search',provider='cpu')
        self.precision=precision

    def transcribe(self,path,remaining_seconds):
        if remaining_seconds<=0:raise TimeoutError('No ASR budget')
        end=time.monotonic()+remaining_seconds
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        samples,rate=sf.read(path,dtype='float32')
        if samples.ndim==2:samples=samples.mean(axis=1)
        if not len(samples) or not np.isfinite(samples).all():raise ValueError('Invalid audio')
        if rate!=16000:
            divisor=gcd(rate,16000)
            samples=resample_poly(samples,16000//divisor,rate//divisor).astype(np.float32)
        if not np.any(samples):return ''
        if time.monotonic()>=end:raise TimeoutError('ASR budget exhausted before decoding')
        stream=self.recognizer.create_stream()
        stream.accept_waveform(16000,samples)
        # Native decoding is blocking: detect an overrun after it returns.
        # Do not claim a hard interrupt guarantee for this CPU call.
        self.recognizer.decode_streams([stream])
        if time.monotonic()>=end:raise TimeoutError('ASR decoding deadline reached')
        return stream.result.text.strip()

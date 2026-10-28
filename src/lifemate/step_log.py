"""Optional development trace of observed inputs/outputs, separate from submission."""
import json
import time


class StepLog:
    def __init__(self, path, verbose=False):
        self.file = path.open('x', encoding='utf-8')
        self.verbose = verbose
        self.sample_id = None
        self.started = time.monotonic()
        self.sequence = 0

    def begin(self, sample_id, started):
        self.sample_id, self.started, self.sequence = sample_id, started, 0

    def emit(self, stage, **payload):
        self.sequence += 1
        event = {'id': self.sample_id, 'step': self.sequence, 'stage': stage,
                 'elapsed_seconds': time.monotonic() - self.started, **payload}
        self.file.write(json.dumps(event, ensure_ascii=False) + '\n')
        self.file.flush()
        if self.verbose:
            print(f"[{self.sample_id}] {self.sequence:02d} {stage} ({event['elapsed_seconds']:.3f}s)", flush=True)
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)

    def close(self):
        self.file.close()

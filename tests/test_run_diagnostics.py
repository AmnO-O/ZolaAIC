"""Development budgets must not turn every slow/error sample into a timeout."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DiagnosticsTests(unittest.TestCase):
    def test_rejection_turns_and_actual_timeouts(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as temporary:
            base = Path(temporary)
            split = base/'train'
            split.mkdir()
            (split/'train_tools_catalog.json').write_text(json.dumps({'apps': []}))
            (split/'train.jsonl').write_text('\n'.join(json.dumps({
                'id': sid, 'label': {'tool_calls': []}}) for sid in ('slow', 'timed', 'crashed')))
            events = [
                {'id': 'slow', 'stage': 'decision.output', 'output': {'tool_calls': []}},
                {'id': 'slow', 'stage': 'tool.validation', 'step': 3, 'output': {'ok': False}},
                {'id': 'slow', 'stage': 'tool.validation', 'step': 4, 'output': {'ok': False}},
            ]
            for sid, elapsed, termination, error in (
                    ('slow', 60, 'finished', None),
                    ('timed', 90, 'error', 'TimeoutError'),
                    ('crashed', 1, 'error', 'RuntimeError')):
                events.append({'id': sid, 'stage': 'sample.output',
                    'output': {'tool_calls': []}, 'elapsed_seconds': elapsed,
                    'termination': termination, 'error_type': error})
            steps = base/'steps.jsonl'
            steps.write_text('\n'.join(json.dumps(event) for event in events))
            result = subprocess.run([sys.executable, '-B', str(root/'scripts/evaluate_run.py'),
                str(steps), '--train-dir', str(split)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((base/'evaluation_summary_steps.json').read_text())
            self.assertEqual(summary['timeouts'], 1)
            self.assertEqual(summary['error_samples'], 2)
            self.assertEqual(summary['samples_ge_50s'], 2)
            self.assertEqual(summary['blocked_turns_count'], 1)
            self.assertEqual(summary['blocked_calls_count'], 2)


if __name__ == '__main__':
    unittest.main()

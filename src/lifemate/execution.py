"""Execution facts and loop guards, independent of model-authored plans."""
import json

from .values import successful


def fingerprint(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def target(call):
    identifiers = tuple(sorted(
        (key, str(value)) for key, value in call['arguments'].items()
        if key.endswith('_id') or key in ('phone', 'phone_number')))
    return call['name'], identifiers or fingerprint(call['arguments'])


def outcome(result):
    if successful(result):
        return 'succeeded'
    if isinstance(result, dict) and result.get('ok') is False:
        return 'failed'
    return 'uncertain'


class ExecutionLedger:
    def __init__(self):
        self.completed_writes = {}
        self.failed_targets = set()
        self.records = []
        self.seen_batches = set()
        self.stagnant_rounds = 0

    def observe(self, call, result, kind):
        status = outcome(result)
        self.records.append({'call': call, 'status': status})
        if status == 'succeeded':
            self.failed_targets.discard(target(call))
            if kind not in ('read', 'meta'):
                key=fingerprint(call)
                self.completed_writes[key]=self.completed_writes.get(key,0)+1
        else:
            self.failed_targets.add(target(call))
        return status

    def stagnant(self, batch):
        """Plan changes do not count as new execution evidence."""
        signature = fingerprint(batch)
        if signature in self.seen_batches:
            self.stagnant_rounds += 1
        else:
            self.stagnant_rounds = 0
            self.seen_batches.add(signature)
        return self.stagnant_rounds >= 2

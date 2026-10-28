"""Stable grouped train-only split; never opens environment directories."""
import hashlib
import json
from pathlib import Path
import re


def main():
    groups = {}
    for line in Path('train/train.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        # Conservative template proxy; not proof of complete scenario isolation.
        text = re.sub(r'\d+', '#', row['audio_text'].lower())
        key = re.sub(r'\s+', ' ', text).strip()
        groups.setdefault(key, []).append(row['id'])
    train, validation = [], []
    for key, ids in sorted(groups.items()):
        digest = hashlib.sha256(('lifemate-42:'+key).encode()).digest()
        (validation if int.from_bytes(digest[:4], 'big') % 100 < 16 else train).extend(ids)
    probe_ids = sorted(validation)[:12]
    validation = sorted(set(validation)-set(probe_ids))
    result = {'seed': 42, 'grouping': 'lowercase transcript with digits masked',
              'limitation': 'Related paraphrases may cross groups; no claim of full template isolation',
              'train': sorted(train), 'development_probe': probe_ids, 'validation': validation}
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/train_split.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Train: {len(train)}; development probe: {len(probe_ids)}; untouched validation: {len(validation)}; groups: {len(groups)}')


if __name__ == '__main__':
    main()

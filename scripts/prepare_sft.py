"""Prepare private train-only trajectories; never import or open environment data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.input_adapter import SplitInput
from lifemate.training import trajectory_examples, split_group, validation_group
from lifemate.prompts import INSTRUCTION


def prepare(directory, output, development_ids=(), seed=42, validation_percent=15):
    split = SplitInput(directory)
    if split.split != 'train':
        raise ValueError('Only train metadata may be used for SFT')
    if not 1 <= validation_percent <= 50:
        raise ValueError('validation_percent must be 1..50')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = split.directory / 'train.jsonl'
    rows = [json.loads(l) for l in source.read_text(encoding='utf-8').splitlines() if l.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate train sample ID')
    # Previously inspected development samples (and exact text groups) cannot enter validation.
    dev_groups = {split_group(r['audio_text']) for r in rows if r['id'] in development_ids}
    manifest = {'seed': seed, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'prompt_sha256': hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
                'catalog_sha256': hashlib.sha256(json.dumps(split.catalog, sort_keys=True).encode()).hexdigest(),
                'grouping': 'casefolded transcript with digit runs masked; paraphrases may cross groups',
                'train': [], 'validation': [], 'rejected': [], 'examples': {'train': 0, 'validation': 0}}
    with (output/'train.jsonl').open('x', encoding='utf-8') as train, (output/'validation.jsonl').open('x', encoding='utf-8') as val:
        for row in rows:
            partition = ('validation' if split_group(row['audio_text']) not in dev_groups
                         and validation_group(row['audio_text'], seed, validation_percent) else 'train')
            try:
                examples = trajectory_examples(row, split.tools(row), split.catalog)
            except (ValueError, KeyError, TypeError) as error:
                manifest['rejected'].append({'id': row['id'], 'error': str(error)})
                continue
            manifest[partition].append(row['id'])
            for example in examples:
                (val if partition == 'validation' else train).write(json.dumps(example, ensure_ascii=False) + '\n')
            manifest['examples'][partition] += len(examples)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--development-ids', default='configs/development_ids.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    ids = json.loads(Path(args.development_ids).read_text())
    manifest = prepare(args.train_dir, args.output, ids, args.seed)
    print(json.dumps({'samples': {k: len(manifest[k]) for k in ('train','validation','rejected')},
                      'examples': manifest['examples']}, indent=2))


if __name__ == '__main__':
    main()

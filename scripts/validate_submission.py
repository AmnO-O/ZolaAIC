"""Check a complete submission using input metadata only; never instantiate ToolEnv."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def strict_json(text):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f'Duplicate JSON key: {key}')
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f'Non-finite JSON value: {value}')
    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f'Non-finite JSON number: {value}')
        return number
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def validate_submission(split_dir, predictions, split='public_test'):
    expected = {}
    for line in (Path(split_dir) / f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = strict_json(line)
        sample_id = item['id']
        if not isinstance(sample_id, str) or sample_id in expected:
            raise ValueError('Input metadata contains invalid or duplicate sample IDs')
        expected[sample_id] = {name for app in item['context']['installed_apps'].values()
                               for name in app['tools']} | {'sys.ask_missing_info'}
    if not expected:
        raise ValueError('Input split contains no samples')
    errors, seen, unknown, duplicates = [], set(), set(), set()
    rows = calls = 0
    path = Path(predictions)
    with path.open(encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = strict_json(line)
                if not isinstance(row, dict) or set(row) != {'id','tool_calls','state_hash'}:
                    raise ValueError('Expected exactly id, tool_calls, state_hash')
                sample_id = row['id']
                if not isinstance(sample_id, str):
                    raise ValueError('id must be a string')
                if sample_id in seen:
                    duplicates.add(sample_id)
                    errors.append(f'Line {number}: duplicate ID {sample_id}')
                seen.add(sample_id)
                if sample_id not in expected:
                    unknown.add(sample_id)
                    errors.append(f'Line {number}: unknown ID {sample_id}')
                if not isinstance(row['state_hash'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['state_hash']):
                    raise ValueError('state_hash must be a lowercase SHA-256 hexadecimal string')
                if not isinstance(row['tool_calls'], list):
                    raise ValueError('tool_calls must be a list')
                for call in row['tool_calls']:
                    if not isinstance(call, dict) or set(call) != {'name','arguments'}:
                        raise ValueError('Each tool call must contain exactly name and arguments')
                    if not isinstance(call['name'], str) or not isinstance(call['arguments'], dict):
                        raise ValueError('Tool name must be a string and arguments must be an object')
                    if sample_id in expected and call['name'] not in expected[sample_id]:
                        raise ValueError(f'Tool not permitted for {sample_id}: {call["name"]}')
                    calls += 1
            except (ValueError, TypeError, KeyError) as error:
                errors.append(f'Line {number}: {error}')
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f'Missing {len(missing)} input sample IDs')
    return {'valid':not errors, 'split':split, 'expected_samples':len(expected), 'prediction_rows':rows,
            'tool_calls':calls, 'missing_ids':missing, 'unknown_ids':sorted(unknown),
            'duplicate_ids':sorted(duplicates), 'errors':errors,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'Structure, complete ID coverage, tool permissions and hash format only; '
                    'not semantic correctness, hash recomputation, official score or runtime compliance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--split', choices=['train','public_test','private_test'], default='public_test')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = validate_submission(args.split_dir, args.predictions, args.split)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {'valid':False, 'errors':[str(error)]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

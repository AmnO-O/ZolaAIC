import sys
import json
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.input_adapter import SplitInput
from lifemate.evaluation import call_metrics


def compute_tool_name_f1(pred_calls, gold_calls):
    if not pred_calls and not gold_calls:
        return 1.0, 0, 0, 0
    if not pred_calls or not gold_calls:
        return 0.0, 0, len(pred_calls), len(gold_calls)
    matched = 0
    unmatched_gold = list(gold_calls)
    for p in pred_calls:
        p_name = p.get('name') if isinstance(p, dict) else None
        for i, g in enumerate(unmatched_gold):
            g_name = g.get('name') if isinstance(g, dict) else None
            if g_name == p_name:
                matched += 1
                unmatched_gold.pop(i)
                break
    p = matched / len(pred_calls)
    r = matched / len(gold_calls)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, matched, len(pred_calls), len(gold_calls)


def evaluate_target(target_path_str, train_dir='../train'):
    target_path = Path(target_path_str)
    split = SplitInput(train_dir)
    specs = {t['name']: t for app in split.catalog['apps'] for t in app['tools']}
    train_data_path = split.directory / 'train.jsonl'

    # Load gold
    gold_map = {}
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gold_map[row['id']] = row

    preds = {}
    trace_map = {}
    blocked_samples = set()
    blocked_turns_set = set()
    blocked_calls_count = 0
    sample_turns = {}

    if target_path.is_file() and target_path.name.endswith('.jsonl'):
        run_name = target_path.name
        summary_out_dir = target_path.parent
        with open(target_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = row.get('id')
                stage = row.get('stage')
                if stage == 'sample.output':
                    preds[sid] = row.get('output', {}).get('tool_calls', [])
                    trace_map[sid] = {
                        'elapsed_seconds': row.get('elapsed_seconds', 0.0),
                        'termination': row.get('termination', 'finished')
                    }
                elif stage == 'decision.output':
                    sample_turns[sid] = sample_turns.get(sid, 0) + 1
                elif stage == 'tool.validation':
                    out = row.get('output', {})
                    if out.get('ok') is False:
                        blocked_samples.add(sid)
                        blocked_turns_set.add((sid, row.get('step')))
                        blocked_calls_count += 1
        blocked_turns_count = len(blocked_turns_set)
    else:
        run_name = target_path.name
        summary_out_dir = target_path
        submission_path = target_path / 'submission.jsonl'
        trace_candidates = [
            target_path / 'trace.jsonl',
            target_path / 'submission.trace.jsonl',
            target_path / 'submission.jsonl.trace.jsonl',
            target_path / 'text_submission.trace.jsonl',
            target_path / 'audio_submission.trace.jsonl',
        ]
        trace_path = next((p for p in trace_candidates if p.exists()), None)
        steps_path = target_path / 'steps.jsonl'

        if not submission_path.exists():
            raise FileNotFoundError(f"Missing submission.jsonl in {target_path}")

        with open(submission_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    preds[row['id']] = row.get('tool_calls', [])

        if trace_path is not None and trace_path.exists():
            with open(trace_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        trace_map[row['id']] = row

        if steps_path.exists():
            with open(steps_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sid = row.get('id')
                    stage = row.get('stage')
                    if stage == 'sample.output' and sid not in trace_map:
                        trace_map[sid] = {
                            'elapsed_seconds': row.get('elapsed_seconds', 0.0),
                            'termination': row.get('termination', 'finished')
                        }
                    elif stage == 'decision.output':
                        sample_turns[sid] = sample_turns.get(sid, 0) + 1
                    elif stage == 'tool.validation':
                        out = row.get('output', {})
                        if out.get('ok') is False:
                            blocked_samples.add(sid)
                            blocked_turns_set.add((sid, row.get('step')))
                            blocked_calls_count += 1
            blocked_turns_count = len(blocked_turns_set)
        else:
            blocked_turns_count = 0

    # Compute metrics
    diag_f1_list = []
    name_f1_list = []
    latencies = []
    timeouts_50 = 0
    over_30 = 0
    detailed_rows = []

    print("=" * 105)
    print(f"{'ID':<12} | {'DiagF1':<6} | {'NmF1':<5} | {'P':<3} | {'G':<3} | {'M':<3} | {'Secs':<6} | {'Turns':<5} | {'Term':<10} | {'Status'}")
    print("=" * 105)

    for sid, pred_calls in sorted(preds.items()):
        gold_data = gold_map.get(sid, {})
        gold_calls = gold_data.get('label', {}).get('tool_calls', [])
        
        # Diagnostic call_metrics (with argument matching)
        diag_metric = call_metrics(pred_calls, gold_calls, specs)
        diag_f1 = diag_metric['f1']
        diag_f1_list.append(diag_f1)

        # Name-only F1
        nm_f1, nm_m, p_cnt, g_cnt = compute_tool_name_f1(pred_calls, gold_calls)
        name_f1_list.append(nm_f1)

        trace_info = trace_map.get(sid, {})
        elapsed = trace_info.get('elapsed_seconds', 0.0)
        latencies.append(elapsed)
        termination = trace_info.get('termination', 'unknown')
        if termination == 'error' or elapsed >= 50.0:
            timeouts_50 += 1
        if elapsed >= 30.0:
            over_30 += 1

        turns = sample_turns.get(sid, 1)

        status = "F1=1.0" if diag_f1 == 1.0 else ("PARTIAL" if diag_f1 > 0 else "ZERO")
        if termination == 'error':
            status += " (TIMEOUT)"
        if sid in blocked_samples:
            status += " [BLOCKED]"

        print(f"{sid:<12} | {diag_f1:<6.3f} | {nm_f1:<5.3f} | {diag_metric['predicted']:<3} | {diag_metric['gold']:<3} | {diag_metric['matched']:<3} | {elapsed:<6.2f} | {turns:<5} | {termination:<10} | {status}")

        detailed_rows.append({
            'id': sid,
            'diagnostic_f1': diag_f1,
            'official_f1': diag_f1,  # alias for backwards compatibility
            'name_f1': nm_f1,
            'pred_count': diag_metric['predicted'],
            'gold_count': diag_metric['gold'],
            'matched': diag_metric['matched'],
            'elapsed_seconds': elapsed,
            'termination': termination,
            'turns': turns,
            'blocked': sid in blocked_samples,
            'pred_calls': pred_calls,
            'gold_calls': gold_calls
        })

    macro_diag_f1 = sum(diag_f1_list) / len(diag_f1_list) if diag_f1_list else 0.0
    macro_name_f1 = sum(name_f1_list) / len(name_f1_list) if name_f1_list else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    mean_turns = sum(sample_turns.values()) / len(sample_turns) if sample_turns else 0.0

    print("=" * 105)
    print(f"Summary for: {run_name}")
    print(f"Total samples:                 {len(diag_f1_list)}")
    print(f"Diagnostic Macro Tool F1:      {macro_diag_f1:.4f}")
    print(f"Name-Only Macro Tool F1:       {macro_name_f1:.4f}")
    print(f"Mean Latency:                  {mean_latency:.2f}s (max: {max_latency:.2f}s)")
    print(f"Samples >= 30s:                {over_30}")
    print(f"Samples Timeout (>= 50s):      {timeouts_50}")
    print(f"Grounding Rejections:          {len(blocked_samples)} samples, {blocked_turns_count} turns, {blocked_calls_count} calls")
    print(f"Mean Turns / Sample:           {mean_turns:.2f}")
    print("=" * 105)

    summary_out = {
        'target': str(target_path),
        'samples': len(diag_f1_list),
        'diagnostic_macro_tool_f1': macro_diag_f1,
        'macro_official_f1': macro_diag_f1,  # alias for backwards compatibility
        'macro_name_f1': macro_name_f1,
        'mean_latency': mean_latency,
        'max_latency': max_latency,
        'samples_ge_30s': over_30,
        'timeouts_50s': timeouts_50,
        'blocked_samples_count': len(blocked_samples),
        'blocked_turns_count': blocked_turns_count,
        'blocked_calls_count': blocked_calls_count,
        'mean_turns': mean_turns,
        'details': detailed_rows
    }

    out_json = summary_out_dir / f"evaluation_summary_{Path(run_name).stem}.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"Wrote full evaluation summary to {out_json}")
    return summary_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate run predictions against train labels')
    parser.add_argument('target', type=str, help='Path to run directory or steps.jsonl file')
    parser.add_argument('--train-dir', type=str, default='../train', help='Path to train directory')
    args = parser.parse_args()
    evaluate_target(args.target, args.train_dir)

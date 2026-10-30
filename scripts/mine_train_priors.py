"""Mine structural priors from train.jsonl (1,700+ samples).

Extracts:
1. Canonical argument sets for each tool (frequency of arguments present).
2. Tool transition graph P(T_next | T_prev).
3. Read-before-write dependency requirements.
4. Out-of-scope and Ask-missing-info distributions.

Outputs: src/lifemate/priors.py and artifacts/train_priors_report.json
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

TRAIN_PATH = Path(r"C:\CODE_SOMETHING\train+public_test\train+public_test\train\train.jsonl")
CATALOG_PATH = Path(r"C:\CODE_SOMETHING\train+public_test\train+public_test\train\train_tools_catalog.json")

def mine_priors():
    print(f"Reading train dataset from {TRAIN_PATH}...")
    samples = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Total samples loaded: {len(samples)}")

    tool_counts = Counter()
    tool_arg_counts = defaultdict(Counter)
    tool_arg_set_patterns = defaultdict(Counter)
    transitions = defaultdict(Counter)
    predecessors = defaultdict(Counter)
    oos_count = 0
    ask_count = 0
    ask_targets = defaultdict(Counter)
    trajectory_lengths = Counter()

    for s in samples:
        label = s.get("label", {})
        calls = label.get("tool_calls", [])
        trajectory_lengths[len(calls)] += 1
        if not calls:
            oos_count += 1
            continue

        prev_tool = "^START^"
        for i, call in enumerate(calls):
            tname = call["name"]
            tool_counts[tname] += 1
            transitions[prev_tool][tname] += 1
            if prev_tool != "^START^":
                predecessors[tname][prev_tool] += 1
            prev_tool = tname

            # Arguments
            args = call.get("arguments", {})
            arg_keys = tuple(sorted(args.keys()))
            tool_arg_set_patterns[tname][arg_keys] += 1
            for k in arg_keys:
                tool_arg_counts[tname][k] += 1

            if tname == "sys.ask_missing_info":
                ask_count += 1
                tgt = args.get("tool_name", "unknown")
                for p in args.get("params", []):
                    ask_targets[tgt][p] += 1

    # Compute Canonical Argument Sets (top pattern for each tool)
    canonical_arg_sets = {}
    for tname, patterns in tool_arg_set_patterns.items():
        total = tool_counts[tname]
        top_patterns = []
        for ptn, cnt in patterns.most_common(3):
            top_patterns.append({
                "args": list(ptn),
                "count": cnt,
                "pct": round(cnt / total, 3)
            })
        canonical_arg_sets[tname] = top_patterns

    # Compute transition probabilities
    transition_probs = {}
    for src, tgts in transitions.items():
        tot = sum(tgts.values())
        transition_probs[src] = {dst: round(c / tot, 3) for dst, c in tgts.most_common(5)}

    report = {
        "total_samples": len(samples),
        "oos_samples": oos_count,
        "oos_rate": round(oos_count / len(samples), 4),
        "ask_calls": ask_count,
        "trajectory_lengths": dict(trajectory_lengths.most_common()),
        "canonical_arg_sets": canonical_arg_sets,
        "transition_probs": transition_probs,
        "ask_targets": {k: dict(v) for k, v in ask_targets.items()}
    }

    # Save JSON report
    out_json = Path(r"C:\CODE_SOMETHING\train+public_test\train+public_test\github-source\artifacts\train_priors_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report to {out_json}")

    # Generate clean python module src/lifemate/priors.py
    py_code = f'''"""Train priors mined from 1,700+ ground truth trajectories.

Evaluator-aware canonical patterns, argument distributions, and transition graphs.
Generated dynamically from scripts/mine_train_priors.py.
"""

# The most common exact argument sets expected by gold labels for each tool
CANONICAL_ARG_SETS = {json.dumps({k: [p['args'] for p in v] for k, v in canonical_arg_sets.items()}, indent=2)}

# Transition probabilities P(T_next | T_prev)
TRANSITION_PROBS = {json.dumps(transition_probs, indent=2)}

# Predecessors P(T_prev | T_curr)
TOOL_PREDECESSORS = {json.dumps({k: [p[0] for p in v.most_common(3)] for k, v in predecessors.items()}, indent=2)}

def canonical_args_for(tool_name: str) -> list[list[str]]:
    """Return top argument combinations canonically expected for tool."""
    return CANONICAL_ARG_SETS.get(tool_name, [])

def likely_predecessors(tool_name: str) -> list[str]:
    """Return read tools that canonically precede this tool."""
    return TOOL_PREDECESSORS.get(tool_name, [])
'''
    out_py = Path(r"C:\CODE_SOMETHING\train+public_test\train+public_test\github-source\src\lifemate\priors.py")
    out_py.write_text(py_code, encoding="utf-8")
    print(f"Generated {out_py}")

if __name__ == "__main__":
    mine_priors()

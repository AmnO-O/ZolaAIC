"""Evaluator Probe (Experiment 0): Reverse-engineering the exact matching rules.

Tests all boundary cases against src/lifemate/evaluation.py to construct
the formal EVALUATOR_CONTRACT.md.
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.evaluation import call_matches, call_metrics

# Load catalog specs to have real properties
CATALOG_PATH = Path(r"C:\CODE_SOMETHING\train+public_test\train+public_test\train\train_tools_catalog.json")
specs = {}
if CATALOG_PATH.is_file():
    cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    specs = {t["name"]: t for app in cat.get("apps", []) for t in app.get("tools", [])}

def run_probe():
    print("=== STARTING EVALUATOR PROBE (EXPERIMENT 0) ===")
    results = {}

    # 1. Permute independent calls: [A, B] vs [B, A]
    gold_1 = [
        {"name": "weather.get_forecast", "arguments": {"date": "2026-11-29"}},
        {"name": "calendar.list_events", "arguments": {}}
    ]
    pred_1 = [
        {"name": "calendar.list_events", "arguments": {}},
        {"name": "weather.get_forecast", "arguments": {"date": "2026-11-29"}}
    ]
    m_1 = call_metrics(pred_1, gold_1, specs)
    results["1_permute_order"] = m_1
    print(f"1. Order Permutation: F1={m_1['f1']}, matched={m_1['matched']}/{m_1['gold']}")

    # 2. Extra argument (e.g. description=None or location added)
    gold_2 = [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-29"}}]
    pred_2_extra_key = [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-29", "location": "Đà Nẵng"}}]
    pred_2_null_key = [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-29", "location": None}}]
    m_2_extra = call_metrics(pred_2_extra_key, gold_2, specs)
    m_2_null = call_metrics(pred_2_null_key, gold_2, specs)
    results["2_extra_key"] = m_2_extra
    results["2_null_key"] = m_2_null
    print(f"2a. Extra Key: F1={m_2_extra['f1']}, matched={m_2_extra['matched']}")
    print(f"2b. Null Key: F1={m_2_null['f1']}, matched={m_2_null['matched']}")

    # 3. Missing optional argument
    gold_3 = [{"name": "calendar.create_event", "arguments": {"title": "Họp", "start": "2026-11-29T09:00:00+07:00", "end": "2026-11-29T10:00:00+07:00"}}]
    pred_3_missing = [{"name": "calendar.create_event", "arguments": {"title": "Họp", "start": "2026-11-29T09:00:00+07:00"}}]
    m_3 = call_metrics(pred_3_missing, gold_3, specs)
    results["3_missing_optional"] = m_3
    print(f"3. Missing Optional Arg: F1={m_3['f1']}, matched={m_3['matched']}")

    # 4. Type sensitivity: int vs str ("500000" vs 500000)
    gold_4 = [{"name": "banking.transfer", "arguments": {"recipient_id": "c_01", "amount": 500000}}]
    pred_4_str = [{"name": "banking.transfer", "arguments": {"recipient_id": "c_01", "amount": "500000"}}]
    pred_4_float = [{"name": "banking.transfer", "arguments": {"recipient_id": "c_01", "amount": 500000.0}}]
    m_4_str = call_metrics(pred_4_str, gold_4, specs)
    m_4_float = call_metrics(pred_4_float, gold_4, specs)
    results["4_str_vs_int"] = m_4_str
    results["4_float_vs_int"] = m_4_float
    print(f"4a. String vs Int amount: F1={m_4_str['f1']}, matched={m_4_str['matched']}")
    print(f"4b. Float vs Int amount: F1={m_4_float['f1']}, matched={m_4_float['matched']}")

    # 5. Extra unneeded read call penalty
    gold_5 = [{"name": "alarm.create", "arguments": {"time": "07:00"}}]
    pred_5_extra_read = [
        {"name": "alarm.list", "arguments": {}},
        {"name": "alarm.create", "arguments": {"time": "07:00"}}
    ]
    m_5 = call_metrics(pred_5_extra_read, gold_5, specs)
    results["5_extra_read"] = m_5
    print(f"5. Extra Read Call: F1={m_5['f1']}, Precision={m_5['precision']}, Recall={m_5['recall']}")

    # 6. Empty calls behavior (Out-of-Scope)
    gold_6 = []
    pred_6_empty = []
    pred_6_hallucinated = [{"name": "weather.get_current", "arguments": {"location": "Hà Nội"}}]
    m_6_empty = call_metrics(pred_6_empty, gold_6, specs)
    m_6_hallucinated = call_metrics(pred_6_hallucinated, gold_6, specs)
    results["6_both_empty"] = m_6_empty
    results["6_hallucinated_oos"] = m_6_hallucinated
    print(f"6a. Gold [] vs Pred []: F1={m_6_empty['f1']}")
    print(f"6b. Gold [] vs Pred [Call]: F1={m_6_hallucinated['f1']}")

    # 7. Fuzzy match field: title in calendar.create_event (x-match: fuzzy)
    gold_7 = [{"name": "calendar.create_event", "arguments": {"title": "Cuộc họp với đối tác VIP", "start": "2026-11-29T09:00:00+07:00"}}]
    pred_7_fuzzy_good = [{"name": "calendar.create_event", "arguments": {"title": "Họp đối tác VIP", "start": "2026-11-29T09:00:00+07:00"}}]
    pred_7_fuzzy_bad = [{"name": "calendar.create_event", "arguments": {"title": "Đi cà phê", "start": "2026-11-29T09:00:00+07:00"}}]
    m_7_good = call_metrics(pred_7_fuzzy_good, gold_7, specs)
    m_7_bad = call_metrics(pred_7_fuzzy_bad, gold_7, specs)
    results["7_fuzzy_good"] = m_7_good
    results["7_fuzzy_bad"] = m_7_bad
    print(f"7a. Fuzzy Match Good Token Overlap: F1={m_7_good['f1']}")
    print(f"7b. Fuzzy Match Bad Token Overlap: F1={m_7_bad['f1']}")

    # 8. Duplicate identical calls
    gold_8 = [{"name": "smart_home.light_set", "arguments": {"device_id": "light_1", "power": "on"}}]
    pred_8_dup = [
        {"name": "smart_home.light_set", "arguments": {"device_id": "light_1", "power": "on"}},
        {"name": "smart_home.light_set", "arguments": {"device_id": "light_1", "power": "on"}}
    ]
    m_8 = call_metrics(pred_8_dup, gold_8, specs)
    results["8_duplicate_call"] = m_8
    print(f"8. Duplicate Call: F1={m_8['f1']}, Precision={m_8['precision']}, Recall={m_8['recall']}")

    return results

if __name__ == "__main__":
    run_probe()

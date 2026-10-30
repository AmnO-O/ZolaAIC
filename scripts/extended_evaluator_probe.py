import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.evaluation import call_matches, call_metrics, token_f1

CATALOG_PATH = Path(r"C:\CODE_SOMETHING\train+public_test\train+public_test\train\train_tools_catalog.json")
specs = {}
if CATALOG_PATH.is_file():
    cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    specs = {t["name"]: t for app in cat.get("apps", []) for t in app.get("tools", [])}

def run_extended_probes():
    print("=== EXTENDED EVALUATOR PROBES ===")

    # Probe 1: Exact fuzzy boundary (> 0.6 vs >= 0.6)
    # token_f1: shared=3, a=5, b=5 -> 2*3 / 10 = 0.600000
    t1 = "a b c d e"
    t2 = "a b c x y"
    f1_exact_0_6 = token_f1(t1, t2)
    print(f"Token F1 of '{t1}' and '{t2}': {f1_exact_0_6}")
    
    spec_fuzzy = {"name": "test.fuzzy", "parameters": {"properties": {"title": {"x-match": "fuzzy"}}}}
    p_exact = {"name": "test.fuzzy", "arguments": {"title": t1}}
    g_exact = {"name": "test.fuzzy", "arguments": {"title": t2}}
    match_0_6 = call_matches(p_exact, g_exact, {"test.fuzzy": spec_fuzzy})
    print(f"Match when Token F1 == 0.6: {match_0_6} (Proves strict > 0.6, NOT >= 0.6!)")

    # Probe 2: True vs 1 (Python equality edge)
    spec_bool = {"name": "test.bool", "parameters": {"properties": {"flag": {"type": "boolean"}}}}
    p_int_for_bool = {"name": "test.bool", "arguments": {"flag": 1}}
    g_bool = {"name": "test.bool", "arguments": {"flag": True}}
    match_bool = call_matches(p_int_for_bool, g_bool, {"test.bool": spec_bool})
    print(f"Match when pred=1 and gold=True: {match_bool} (type(1) == type(True) fails: int vs bool!)")

    # Probe 3: Nested list order
    spec_list = {"name": "test.list", "parameters": {"properties": {"items": {"type": "array"}}}}
    p_list_rev = {"name": "test.list", "arguments": {"items": ["b", "a"]}}
    g_list = {"name": "test.list", "arguments": {"items": ["a", "b"]}}
    match_list = call_matches(p_list_rev, g_list, {"test.list": spec_list})
    print(f"Match when list order is reversed: {match_list} (List order is strict/order-sensitive!)")

    # Probe 4: Nested dict key order
    spec_dict = {"name": "test.dict", "parameters": {"properties": {"meta": {"type": "object"}}}}
    p_dict = {"name": "test.dict", "arguments": {"meta": {"k2": 2, "k1": 1}}}
    g_dict = {"name": "test.dict", "arguments": {"meta": {"k1": 1, "k2": 2}}}
    match_dict = call_matches(p_dict, g_dict, {"test.dict": spec_dict})
    print(f"Match when dict key order differs: {match_dict} (Dict key order is invariant!)")

if __name__ == "__main__":
    run_extended_probes()

"""Train-only diagnostic metrics, not an official scoring implementation."""
from collections import Counter
import unicodedata


def token_f1(a, b):
    a = Counter(unicodedata.normalize('NFC', str(a)).lower().split())
    b = Counter(unicodedata.normalize('NFC', str(b)).lower().split())
    shared = sum((a & b).values())
    return 2*shared/(sum(a.values())+sum(b.values())) if a or b else 1.0


def call_matches(pred, gold, specs):
    if pred['name'] != gold['name']:
        return False
    a, b = pred['arguments'], gold['arguments']
    if set(a) != set(b):
        return False
    props = specs.get(pred['name'], {}).get('parameters', {}).get('properties', {})
    return all(token_f1(a[k], b[k]) > 0.6 if props.get(k, {}).get('x-match') == 'fuzzy'
               else type(a[k]) == type(b[k]) and a[k] == b[k] for k in b)


def call_metrics(pred, gold, specs):
    # Maximum bipartite matching: each gold/predicted call counts at most once.
    assigned = {}

    def augment(i, seen):
        for j, target in enumerate(gold):
            if j not in seen and call_matches(pred[i], target, specs):
                seen.add(j)
                if j not in assigned or augment(assigned[j], seen):
                    assigned[j] = i
                    return True
        return False

    matched = sum(augment(i, set()) for i in range(len(pred)))
    precision = matched/len(pred) if pred else float(not gold)
    recall = matched/len(gold) if gold else float(not pred)
    return {'matched': matched, 'predicted': len(pred), 'gold': len(gold),
            'precision': precision, 'recall': recall,
            'f1': 2*matched/(len(pred)+len(gold)) if pred or gold else 1.0}

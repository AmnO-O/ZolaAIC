"""Allowlisted lookup hints from explicit catalog descriptions only."""

def lookup_for(ref, parameter, available):
    """Resolve explicit read mentions; ambiguous references yield no hint."""
    import re
    app = ref.split('.')[0]
    names = set(re.findall(r'(?<![\w.])([a-zA-Z_][\w]*\.[a-zA-Z_][\w]*)(?![\w.])',
                           parameter.get('description', '')))
    matches = sorted(name for name in names if name in available
                     and name.split('.')[0] == app
                     and available[name].get('kind') == 'read')
    if len(matches) == 1:
        return matches[0]
    return None

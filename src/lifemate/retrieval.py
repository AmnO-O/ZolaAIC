"""Experimental catalog-only BM25 routing; never uses labels or app state."""
from collections import Counter
import math
import re
import unicodedata


def terms(text):
    text = unicodedata.normalize('NFKD', text.casefold()).replace('đ', 'd')
    text = ''.join(c for c in text if not unicodedata.combining(c))
    words = re.findall(r'[a-z0-9]+', text)
    result = Counter('w:'+w for w in words)
    for word in words:
        result.update('c:'+word[i:i+3] for i in range(len(word)-2))
    return result


def descriptions(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'description' and isinstance(child, str):
                yield child
            elif key == 'properties':
                yield ' '.join(child)
                yield from descriptions(child)
            elif isinstance(child, (dict, list)):
                yield from descriptions(child)
    elif isinstance(value, list):
        for child in value:
            yield from descriptions(child)


class CatalogSearch:
    def __init__(self, tools, catalog):
        self.tools = list(tools)
        self.specs = {t['name']:t for t in tools}
        apps = {a['id']:a.get('description', '') for a in catalog['apps']}
        docs = [' '.join([t['name'], apps.get(t['name'].split('.')[0], ''),
                         *descriptions(t)]) for t in tools]
        self.docs = [terms(doc) for doc in docs]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average = sum(self.lengths)/max(1,len(self.docs)) or 1
        frequency = Counter(term for doc in self.docs for term in doc)
        self.idf = {term:math.log(1+(len(tools)-df+0.5)/(df+0.5))
                    for term,df in frequency.items()}

    def rank(self, queries):
        scores = [0.0]*len(self.tools)
        for query in queries:
            query_terms = terms(query)
            query_weight = sum(1 if term.startswith('w:') else .2 for term in query_terms) or 1
            for i, doc in enumerate(self.docs):
                score = 0.0
                for term in query_terms:
                    tf = doc.get(term,0)
                    if tf:
                        weight = 1 if term.startswith('w:') else .2
                        score += weight*self.idf[term]*tf*2.2/(tf+1.2*(.25+.75*self.lengths[i]/self.average))
                scores[i] = max(scores[i], score/query_weight)
        return sorted(zip(self.tools,scores), key=lambda pair:(-pair[1],pair[0]['name']))

    def dependencies(self, names):
        names = set(names)
        while True:
            expanded = set(names)
            for name in names:
                tool = self.specs[name]
                # Exact tool mentions in catalog descriptions, not handwritten app mappings.
                for text in descriptions(tool):
                    expanded.update(n for n in re.findall(r'\b[a-zA-Z_][\w]*\.[a-zA-Z_][\w]*\b',text)
                                    if n in self.specs and self.specs[n].get('kind') == 'read')
                def refs(value):
                    if isinstance(value,dict):
                        if isinstance(value.get('x-ref'),str): yield value['x-ref']
                        for child in value.values(): yield from refs(child)
                    elif isinstance(value,list):
                        for child in value: yield from refs(child)
                for ref in refs(tool.get('parameters',{})):
                    app = ref.split('.')[0]
                    expanded.update(n for n,t in self.specs.items()
                                    if n.split('.')[0] == app and t.get('kind') == 'read')
            if expanded == names: return names
            names = expanded

    def select(self, queries, top_k=8):
        if top_k < 1: raise ValueError('top_k must be positive')
        ranking = self.rank(queries)
        seeds = [t['name'] for t,score in ranking if score > 0][:top_k]
        fallback = not seeds
        selected = self.dependencies(seeds) if seeds else set(self.specs)
        if 'sys.ask_missing_info' in self.specs:
            selected.add('sys.ask_missing_info')
        return [t for t in self.tools if t['name'] in selected], {
            'seed_names':seeds, 'selected_names':sorted(selected), 'fallback_full':fallback}

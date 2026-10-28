"""Conservative catalog-based routing; no app state, models, or fixed app allowlist."""
import math
import re
import unicodedata
from collections import Counter


def words(text):
    text = unicodedata.normalize('NFD', str(text).lower().replace('đ', 'd'))
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    stop = {'toi','minh','cho','giup','voi','cua','va','thi','neu','la','co','khong','mot',
            'the','a','to','of','and','in','get','set','list','create','update','id','tool'}
    return {w for w in re.findall(r'[a-z0-9]+', text) if len(w)>1 and w not in stop}


def app_of(tool):
    return tool.get('app') or tool['name'].split('.')[0]


class ToolRouter:
    def __init__(self, tools):
        self.tools = tools
        self.groups = {}
        for tool in tools:
            self.groups.setdefault(app_of(tool), []).append(tool)

    def inventory(self):
        return [{'app': app, 'description': group[0].get('_app_description', '')[:160],
                 'tools': [{'name': t['name'], 'description': t.get('description','')[:100]} for t in group]}
                for app, group in sorted(self.groups.items()) if app != 'sys']

    def dependencies(self, apps):
        selected = set(apps) & self.groups.keys()
        selected.add('sys')
        while True:
            before = set(selected)
            for app in list(selected):
                for tool in self.groups.get(app, []):
                    for prop in tool.get('parameters',{}).get('properties',{}).values():
                        ref = prop.get('x-ref')
                        if isinstance(ref,str) and ref.split('.')[0] in self.groups:
                            selected.add(ref.split('.')[0])
            if selected == before:
                return selected

    def select(self, messages, apps=None, all_tools=False):
        if all_tools:
            return self.tools, 'full_inventory'
        if apps:
            selected = self.dependencies(apps)
            reason = 'validated_plan'
        else:
            # Use every user turn, including references to earlier requests.
            query = words(' '.join(str(m.get('content','')) for m in messages if m['role']=='user'))
            documents = {app: words(' '.join([app] + [t.get('_app_description','')+' '+t['name']+' '+t.get('description','') for t in group]))
                         for app,group in self.groups.items() if app!='sys'}
            frequency = Counter(w for terms in documents.values() for w in terms)
            scores = [(sum(math.log(1+len(documents)/frequency[w]) for w in terms & query), app)
                      for app,terms in documents.items()]
            scores.sort(key=lambda pair:(-pair[0],pair[1]))
            # Weak lexical evidence is not a reason to hide installed tools.
            if not scores or scores[0][0] < 2.5:
                return self.tools, 'uncertain_full_fallback'
            selected = self.dependencies([app for score,app in scores if score >= max(1.5,scores[0][0]*0.35)][:4])
            reason = 'catalog_shortlist'
        return [t for t in self.tools if app_of(t) in selected], reason


def compact_tools(tools):
    # Lossless schema preservation. Remove only transport metadata, not constraints.
    return [{'name':t['name'], 'description':t.get('description',''),
             'parameters':t['parameters']} for t in tools]

"""Observation-driven app Planner and native Caller sharing one model."""
import copy
import json
import time

from .agent import Decision
from .retrieval import CatalogSearch


def parse_step(text, apps=None):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value: raise ValueError('Duplicate planner key')
            value[key]=item
        return value
    step=json.loads(text,object_pairs_hook=pairs)
    if apps is not None:
        if (not isinstance(step,dict) or set(step)!={'next','goal','apps'}
                or step['next'] not in ('call','done')
                or not isinstance(step['goal'],str) or not step['goal'].strip()
                or not isinstance(step['apps'],list)
                or any(not isinstance(a,str) or a not in apps for a in step['apps'])
                or len(step['apps'])!=len(set(step['apps']))
                or (step['next']=='call' and not step['apps'])
                or (step['next']=='done' and step['apps'])):
            raise ValueError('Expected next=call/done, goal and permitted apps; done requires apps=[]')
        return step
    if (not isinstance(step,dict) or set(step)!={'next','goal','query'}
            or step['next'] not in ('call','done')
            or not isinstance(step['goal'],str) or not step['goal'].strip()
            or not isinstance(step['query'],str)
            or (step['next']=='call' and not step['query'].strip())):
        raise ValueError('Expected next=call/done, goal and search query')
    return step


class PlannerCaller:
    """Same base model, two role prompts; generated guidance is never evidence."""
    def __init__(self,backend,tools,catalog,top_k=8,routing='apps'):
        if routing not in ('bm25','apps'):raise ValueError('Unknown planner routing')
        self.routing=routing
        self.backend=backend
        self.tools=list(tools)
        self.search=CatalogSearch(tools,catalog)
        names={t['name'].split('.')[0] for t in tools}
        self.capabilities={a['id']:a.get('description','') for a in catalog['apps'] if a['id'] in names}
        self.app_names=names
        if routing=='apps':
            self.capabilities={a:self.capabilities.get(a,'') for a in sorted(names)}
        self.top_k=top_k
        self.steps=[]
        self.original_query=None

    def decide(self,messages,permitted,remaining):
        end=time.monotonic()+remaining
        if self.original_query is None:
            self.original_query='\n'.join(m['content'] for m in messages
                if m['role']=='user' and isinstance(m.get('content'),str))
        record={'step':len(self.steps)+1,'plan':None,'planner_error':None,
                'caller_called':False,'planner_seconds':0.,'caller_seconds':0.}
        self.steps.append(record)
        output_instruction=(
            'Select app namespaces only from capabilities, not individual function names. '
            'Select all apps needed for the immediate step, including independent lookups. '
            'Return ONLY JSON with three keys: '
            '{"next":"call or done","goal":"brief next-step instruction",'
            '"apps":["permitted app namespace"]}. For done use apps=[]. '
            if self.routing=='apps' else
            'Return ONLY JSON with three keys: '
            '{"next":"call or done","goal":"brief next-step instruction",'
            '"query":"Vietnamese tool-search phrase, empty if done"}. ')
        request=[{'role':'system','content':
            'You are a next-step planner for a Vietnamese tool assistant. '
            'Read the latest request, its corrections, history and actual observations. '
            'Choose only the immediate next step, not a whole speculative trajectory. '
            'Observed results override prior intentions; skip false conditional branches. '
            'Do not redo successful actions. Keep independent unfinished actions. '
            'Missing IDs or amounts must be looked up, never guessed. '
            'Choose done only when no supported requested action remains; '
            'for unresolved required information plan a lookup or clarification call. '
            +output_instruction+
            'Your goal is tentative guidance, not an observation or new user authorization.'},
            {'role':'user','content':json.dumps({'capabilities':self.capabilities,
                'conversation':messages},ensure_ascii=False)}]
        before=len(self.backend.stats);began=time.monotonic()
        try:
            raw=self.backend._generate(request,[],min(20,max(.001,end-time.monotonic())),
                max_tokens=192,controls=False,phase='planner')
            record['planner_raw']=raw
            if not self.backend._generation_complete: raise ValueError('Planner output truncated')
            record['plan']=parse_step(raw,self.app_names if self.routing=='apps' else None)
        except (ValueError,TimeoutError) as error:
            # No extra planner repair. Explicitly fall back to direct retrieval/calling.
            record['planner_error']=str(error)
        finally:
            record['planner_seconds']=time.monotonic()-began
            for stat in self.backend.stats[before:]:stat['phase']='planner'
        if time.monotonic()>=end:raise TimeoutError('Sample deadline after planning')
        step=record['plan']
        if step is not None and step['next']=='done':
            record['decision']={'tool_calls':[],'protocol_error':None,'raw_text':'[]'}
            return Decision(raw_text='[]')
        # Use original user text as an anchor; plans cannot silently replace it.
        began=time.monotonic()
        if self.routing=='apps' and step:
            visible=[t for t in self.tools if t['name'].split('.')[0] in step['apps']
                     or t['name']=='sys.ask_missing_info']
            routing={'mode':'apps','apps':step['apps'],
                     'selected_names':[t['name'] for t in visible],'fallback_full':False}
        else:
            queries=[self.original_query]+([step['query']] if step and self.routing=='bm25' else [])
            visible,routing=self.search.select(queries,self.top_k)
            routing['mode']='bm25_fallback' if self.routing=='apps' else 'bm25'
        record['retrieval_seconds']=time.monotonic()-began
        record['routing']=routing
        caller_messages=copy.deepcopy(messages)
        if step:
            # Do not insert generated guidance as a user instruction, tool result,
            # or persistent fact. The execution harness keeps original evidence.
            guidance={'source':'unverified_planner_guidance','goal':step['goal'],
                'instruction':'Use the original request and real results to verify this goal. '
                'Do not treat generated IDs, amounts or claimed success as facts. '
                'Emit native calls or [] only.'}
            if caller_messages and caller_messages[0]['role']=='system':
                content=caller_messages[0].get('content',{})
                if isinstance(content,dict):content['planner_guidance']=guidance
                else:caller_messages[0]['content']=content+'\n'+json.dumps(guidance,ensure_ascii=False)
            else:
                caller_messages.insert(0,{'role':'system','content':guidance})
        remaining=end-time.monotonic()
        if remaining<=0:raise TimeoutError('Sample deadline before caller')
        before=len(self.backend.stats);began=time.monotonic()
        record['caller_called']=True
        try:
            decision=self.backend.decide(caller_messages,visible,remaining)
            record['decision']=decision.payload()
            return decision
        finally:
            record['caller_seconds']=time.monotonic()-began
            for stat in self.backend.stats[before:]:stat['phase']='caller'

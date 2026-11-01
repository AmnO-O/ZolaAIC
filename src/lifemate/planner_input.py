"""Lossless event projection for Planner; immutable request boundary per sample."""
import copy
import json


def content(value):
    if isinstance(value,str):
        try:
            return json.loads(value)
        except (ValueError,TypeError):
            return value
    return copy.deepcopy(value)


def execution_events(messages):
    events=[]
    pending=[]
    for message in messages:
        role=message['role']
        if role=='assistant':
            pending.extend(copy.deepcopy(message.get('tool_calls',[])))
            text=message.get('text',message.get('content',''))
            if text:events.append({'type':'assistant_text','text':text})
        elif role=='tool':
            observation=content(message.get('content'))
            name=message.get('name')
            index=next((i for i,c in enumerate(pending)
                        if c.get('function',c).get('name')==name),None)
            call=pending.pop(index) if index is not None else None
            event={'type':'tool_observation','call':call}
            if isinstance(observation,dict) and 'source_type' in observation:
                event.update(observation)
            else:
                event.update(source_type='unspecified_observation',result=observation)
            events.append(event)
        elif role=='user':
            payload=content(message.get('content',''))
            if isinstance(payload,dict) and 'controller_feedback' in payload:
                events.append({'type':'controller_feedback','feedback':payload})
            else:
                # Never silently reclassify a later message as the original request.
                events.append({'type':'subsequent_message','speaker':role,'content':payload})
        else:
            events.append({'type':'subsequent_message','speaker':role,
                           'content':content(message.get('content',''))})
    events.extend({'type':'unobserved_proposal','call':c} for c in pending)
    return events


class PlannerInput:
    def __init__(self,messages,available_apps):
        users=[i for i,m in enumerate(messages) if m['role']=='user']
        if not users:raise ValueError('Planner requires the original current user request')
        current=users[-1]
        self.boundary=current+1
        self.request=copy.deepcopy(messages[current].get('content',''))
        self.available_apps=copy.deepcopy(available_apps)
        self.context={}
        self.prior=[]
        for message in messages[:current]:
            if message['role']=='system':
                data=content(message.get('content',{}))
                if isinstance(data,dict) and isinstance(data.get('context'),dict):
                    self.context.update(copy.deepcopy(data['context']))
            elif message['role']=='user':
                self.prior.append({'speaker':'user','text':copy.deepcopy(message.get('content',''))})
            elif message['role']=='assistant':
                # Earlier assistant actions were already applied by the environment.
                self.prior.append({'speaker':'assistant',
                    'text':copy.deepcopy(message.get('text',message.get('content',''))),
                    'tool_calls':copy.deepcopy(message.get('tool_calls',[])),
                    'tool_results':copy.deepcopy(message.get('tool_results',[]))})
            else:
                self.prior.append({'speaker':message['role'],
                    'content':content(message.get('content',''))})
        self.context.pop('installed_apps',None)
        self.context.pop('app_descriptions',None)

    def payload(self,messages):
        return {'context':copy.deepcopy(self.context),
                'available_apps':copy.deepcopy(self.available_apps),
                'prior_conversation':copy.deepcopy(self.prior),
                'current_request':copy.deepcopy(self.request),
                'execution_history':execution_events(messages[self.boundary:])}

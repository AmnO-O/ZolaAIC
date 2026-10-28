"""Frozen, message-level input controls; no prompt search or extra inference LMs."""
import copy
import os
from pathlib import Path
import sys


INSTRUCTION = '''You are an assistant that fulfils Vietnamese user requests using the provided tools.
Use only currently available tools and the exact parameter names in their schemas.
- Demonstration and schema examples are not current user data. Never copy their IDs, phone numbers, or amounts.
- App facts and identifiers must come from tool results, observed history, or explicit user input. Look up unknown data first; never invent IDs or guess balances, bills, recipients, or amounts.
- Interpret the latest request in the full conversation. Historical actions have already happened; do not repeat them. Follow the latest corrections and respect negation.
- Resolve relative dates and times using the sample's current_time and timezone, not the machine clock.
- Batching: you may return multiple tool calls in a single response when they are independent — for example, two read calls whose results do not depend on each other, or two write calls that act on different resources. Do NOT batch a write call with any call whose result you need to determine the write arguments. Wait for read results before calling writes that use those results.
- For conditional requests, obtain the relevant facts, choose the applicable branch, and complete every action in that branch. Continue calling tools while requested actions remain unfinished.
- Include required arguments and only optional arguments needed by the request. Do not add unrequested label, title, repeat, location, or default values.
- Look up missing information when tools can provide it. Complete independent actions with sufficient information before asking. If required information remains unavailable, call sys.ask_missing_info with the exact tool_name and missing params, then stop. Never invent values.
- If no installed capability supports the request, finish without unrelated calls. A read capability does not imply support for the requested operation.
- Inspect done, message, and returned data. ok=true alone does not establish that the operation succeeded. Do not claim completion without supporting execution results.
- Use native tool-call format. Do not write Python, lengthy explanations, or unnecessary reads/confirmations.
Before finishing, check that every requested part has been handled. Do not output a lengthy reasoning narrative.'''

# Authored synthetic examples, not extracted from inspected/public/private labels.
# Filter by supported tool names, and keep order fixed across model decisions.
# Each example shows ONLY the first tool call response; the conversation continues
# with results before any dependent calls are made.
EXAMPLES = [
    # Multi-read batch: weather + calendar list are independent reads — return both at once.
    # Then writes (update_event, sms) happen in the NEXT turn after seeing results.
    ({'weather.get_forecast', 'calendar.list_events', 'calendar.update_event',
      'contacts.search', 'sms.send'}, {
      'request': 'Nếu mai mưa thì dời lịch chạy bộ muộn 1 tiếng và nhắn sếp Dũng giờ mới.',
      'behavior': '<tool_call>{"name":"weather.get_forecast","arguments":{"date":"2030-06-03"}}</tool_call>'
                  '<tool_call>{"name":"calendar.list_events","arguments":{}}</tool_call>'}),
    # Missing info: no content supplied — ask before doing anything.
    ({'notes.create', 'sys.ask_missing_info'}, {
      'request': 'Tạo một ghi chú giúp tôi.',
      'behavior': '<tool_call>{"name":"sys.ask_missing_info","arguments":{"tool_name":"notes.create","params":["content"]}}</tool_call>'}),
    # Lookup before action: must find contact_id from contacts.search first.
    ({'contacts.search', 'sms.send'}, {
      'request': 'Nhắn cho cô Lan là tôi đến muộn.',
      'behavior': '<tool_call>{"name":"contacts.search","arguments":{"query":"cô Lan"}}</tool_call>'}),
]


class PromptControls:
    def __init__(self, mode='steerability'):
        if mode not in ('baseline','system','steerability'):
            raise ValueError('Unknown prompt mode')
        self.mode=mode
        if mode=='baseline':return
        source=os.environ.get('LIFEMATE_STEERABILITY_PATH')
        if source is None:
            source=str(Path(__file__).resolve().parents[2]/'vendor'/'steerability')
        if Path(source,'steerability','__init__.py').is_file():
            sys.path.insert(0,str(Path(source).resolve()))
        try:
            from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
            from steerability.algorithms.input_control.few_shot.control import FewShot
        except ImportError as error:
            raise RuntimeError('Prepare the real steerability source with scripts/prepare_steerability.py; no fallback prompt implementation is used') from error
        self.system=SystemPrompt(text=INSTRUCTION,mode='prepend')
        self.system.steer()
        self.fewshot=FewShot(system_mode='append')
        self.fewshot.steer()

    def apply(self, messages, tools):
        batch=[copy.deepcopy(messages)]
        if self.mode=='baseline':return batch[0]
        batch=self.system.adapt_messages(batch)
        if self.mode=='steerability':
            allowed={t['name'] for t in tools}
            examples=[example for required,example in EXAMPLES if required<=allowed][:2]
            if examples:
                batch=self.fewshot.adapt_messages(batch,runtime_kwargs={'positive_examples':examples})
        return batch[0]

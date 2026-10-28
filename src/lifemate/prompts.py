"""Frozen, message-level input controls; no prompt search or extra inference LMs."""
import copy
import os
from pathlib import Path
import sys


INSTRUCTION = '''You are an assistant that fulfils Vietnamese user requests by calling the provided tools.
Use only currently available tools and the exact parameter names in their schemas.

CRITICAL RULES:
1. NEVER INVENT OR GUESS ENTITY IDs:
   - Any parameter with an ID constraint (e.g. contact_id, event_id, device_id, bill_id, lock_id, scene_id) MUST be an actual ID observed from a previous tool result.
   - NEVER fabricate placeholder IDs such as "contact_123", "evt_01", "device_1"!
   - You CANNOT call calendar.update_event or calendar.cancel_event until you have FIRST called calendar.list_events and observed the real event_id and original time! When calling calendar.list_events, pass NO arguments (use {}) unless the user explicitly asked to filter the calendar view by date.
   - You CANNOT control a smart device until you have FIRST called smart_home.get_status and observed the real device_id! When calling smart_home.get_status, pass NO arguments (use {}) so all devices in the home are returned.
   - If the request involves paying or transferring an amount equal to a bill (e.g. "tiền điện", "tiền nước", "tiền mạng", "tiền internet", "hóa đơn") without stating an exact numeric amount, you MUST call bill_payment.list_bills with {} first to discover the exact bill amount before transferring!
   - You CANNOT call sms.send or chat_ott.send until you have FIRST called contacts.search and observed the real contact_id!
   - If a condition (e.g. weather forecast, wallet balance) is satisfied, your immediate next step is to issue all lookup tools together to discover the real IDs before writing.

2. STRICT MINIMAL ARGUMENTS:
   - Pass ONLY arguments that are either marked `required` or EXPLICITLY requested by the user.
   - In weather tools: omit `location` if the user is asking about the weather locally or does not name a different city (as the schema states: 'bỏ trống = thành phố của người dùng').
   - In alarm tools: DO NOT add unrequested `label` or `repeat`.
   - In smart home tools: DO NOT add unrequested `floor` or `room`.
   - In ewallet tools: DO NOT add unrequested `note`.

3. VIETNAMESE SPOKEN AMOUNTS AND RELATIVE DATES:
   - 'năm mươi nghìn' / 'năm mươi ngàn' / '50k' = 50000.
   - 'một trăm nghìn' / 'một trăm ngàn' / '100k' = 100000.
   - 'hai mươi nghìn' / 'hai mươi ngàn' / '20k' = 20000.
   - 'hai trăm nghìn' / 'hai trăm ngàn' / '200k' = 200000.
   - Compute dates strictly relative to `current_time` in context: 'hôm nay' = current date, 'mai' = current date + 1 day. Never copy dates from few-shot examples!

4. EXECUTE ALL REQUESTED ACTIONS:
   - The user request often contains multiple operations (e.g. "dời lịch chạy bộ và nhắn sếp"). You MUST execute EVERY part!
   - For "dời lịch" (postpone/reschedule an event): you MUST update the calendar event (calendar.update_event), AND if the user asked to notify or message someone (e.g. "nhắn sếp Dũng giờ mới"), you MUST also search the contact and send the message. Do NOT stop after updating the calendar!
   - For conditional requests (e.g. "Nếu mai mưa thì... không mưa thì thôi"): check the condition first (e.g. weather.get_forecast). If the condition is met, execute ALL requested actions in that branch!

5. MESSAGE CONTENT FIDELITY:
   - When sending an SMS or message, write a short, direct message conveying the user's intent (e.g. "mai chạy bộ dời sang 09:00 nhé").
   - DO NOT write conversational chatbot pleasantries like "Chào bạn, hôm nay có mưa...".

6. BATCHING INDEPENDENT READS:
   - You may return multiple tool calls in a single response when they are independent (e.g. calling weather.get_forecast, calendar.list_events, or contacts.search together).
   - Wait for tool results before issuing dependent write calls that require observed IDs.

7. RIDE HAILING & CALENDAR EVENT DESTINATIONS:
   - When the user asks to book or quote a ride to an event or appointment (e.g. "đi họp...", "tới chỗ hẹn...", "đi khám..."), you MUST call calendar.list_events with {} first to discover the event's location and time!
   - Resolve relative locations from user_profile in context:
     - "công ty", "văn phòng", "chỗ làm" -> user_profile.work_address
     - "nhà", "về nhà" -> user_profile.home_address
   - When the user says "đặt xe", "đặt ngay bây giờ", "gọi ô tô", call ride_hailing.book directly! Do NOT call ride_hailing.get_quotes unless the user explicitly asks for quotes or prices.
   - For vehicle: "bốn chỗ" / "4 chỗ" = "car4", "bảy chỗ" / "7 chỗ" = "car7", "xe máy" = "bike".

Before finishing, check that every requested action has been executed. Use native tool-call format. Do not output conversational explanations.'''

# Authored synthetic examples, not extracted from inspected/public/private labels.
# Filter by supported tool names, and keep order fixed across model decisions.
# Each example shows ONLY the first tool call response; the conversation continues
# with results before any dependent calls are made.
EXAMPLES = [
    # Multi-action lookups: both calendar.list_events and contacts.search are independent lookups to discover IDs before writing.
    ({'calendar.list_events', 'calendar.update_event', 'contacts.search', 'sms.send'}, {
      'request': 'Dời lịch chạy bộ muộn 1 tiếng và nhắn sếp Dũng giờ mới.',
      'behavior': '<tool_call>{"name":"calendar.list_events","arguments":{}}</tool_call>'
                  '<tool_call>{"name":"contacts.search","arguments":{"query":"Sếp Dũng"}}</tool_call>'}),
    # Conditional check: check condition first before taking action.
    ({'weather.get_forecast', 'calendar.cancel_event'}, {
      'request': 'Nếu mai mưa thì hủy lịch đá bóng, không mưa thì thôi.',
      'behavior': '<tool_call>{"name":"weather.get_forecast","arguments":{"date":"2035-12-01"}}</tool_call>'}),
    # Bill payment + contact lookups before money transfer/topup.
    ({'bill_payment.list_bills', 'contacts.search', 'ewallet.topup_mobile', 'ewallet.transfer'}, {
      'request': 'Nạp 50k điện thoại cho bác Ba và chuyển tiền điện tháng này cho cô Mai.',
      'behavior': '<tool_call>{"name":"bill_payment.list_bills","arguments":{}}</tool_call>'
                  '<tool_call>{"name":"contacts.search","arguments":{"query":"Bác Ba"}}</tool_call>'
                  '<tool_call>{"name":"contacts.search","arguments":{"query":"Cô Mai"}}</tool_call>'}),
    # Smart home device status check before control:
    ({'smart_home.get_status', 'smart_home.light_set'}, {
      'request': 'Giảm độ sáng đèn phòng khách xuống còn một nửa.',
      'behavior': '<tool_call>{"name":"smart_home.get_status","arguments":{}}</tool_call>'}),
    # Ride hailing to a calendar event: discover event location and time from calendar first.
    ({'calendar.list_events', 'ride_hailing.book'}, {
      'request': 'Đặt giúp tôi xe bốn chỗ từ công ty đi họp phụ huynh chiều nay nhé, đặt ngay bây giờ luôn.',
      'behavior': '<tool_call>{"name":"calendar.list_events","arguments":{}}</tool_call>'}),
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
            user_text=' '.join(str(m.get('content','')) for m in messages if m.get('role')=='user').lower()
            candidates=[example for required,example in EXAMPLES if required<=allowed]
            if candidates:
                def score(cand):
                    req_words=cand['request'].lower().split()
                    return sum(1 for w in req_words if len(w)>2 and w in user_text)
                ranked=sorted(candidates,key=score,reverse=True)
                examples=ranked[:2]
                batch=self.fewshot.adapt_messages(batch,runtime_kwargs={'positive_examples':examples})
        return batch[0]

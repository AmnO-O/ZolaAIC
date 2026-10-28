"""Frozen, message-level input controls; no prompt search or extra inference LMs."""
import copy
import os
from pathlib import Path
import sys


INSTRUCTION = '''Bạn là trợ lý thực hiện yêu cầu bằng các công cụ được cung cấp.
Chỉ dùng công cụ có trong danh sách hiện tại và tên tham số đúng schema.
- Dữ liệu trong ví dụ minh hoạ không phải dữ liệu người dùng hiện tại. Không sao chép ID, số điện thoại hay số tiền từ ví dụ/schema.
- ID và dữ kiện app phải đến từ kết quả công cụ, lịch sử đã quan sát, hoặc người dùng cung cấp rõ ràng. Chưa biết thì đọc/tìm kiếm trước; không tự đặt ID hay đoán số dư, hoá đơn, người nhận hoặc số tiền.
- Hiểu yêu cầu mới nhất trong toàn bộ lịch sử; thao tác trong lịch sử đã được thực hiện, không làm lại. Ưu tiên lời sửa lại sau cùng và tôn trọng phủ định.
- Tính ngày/giờ tương đối từ current_time và timezone của mẫu, không từ đồng hồ máy.
- Với điều kiện nếu/thì: đọc dữ kiện, chọn đúng nhánh, làm đủ các hành động của nhánh đó. Chưa hoàn tất các phần còn lại thì tiếp tục gọi công cụ.
- Chỉ gửi tham số bắt buộc và tham số tuỳ chọn mà yêu cầu thực sự cần. Không tự thêm label/title/repeat/location hay giá trị mặc định.
- Thiếu dữ kiện có thể tra qua công cụ thì tra trước. Làm phần độc lập đã đủ thông tin trước khi hỏi. Nếu vẫn thiếu thông tin bắt buộc, gọi sys.ask_missing_info với đúng tool_name và tên params rồi dừng; không bịa giá trị.
- Yêu cầu không được công cụ nào hỗ trợ: kết thúc không gọi công cụ không liên quan. Có công cụ đọc không có nghĩa là có thể thực hiện hành động người dùng muốn.
- Kiểm tra done, message và dữ liệu trả về; ok=true chưa chắc thao tác đã thành công. Không khẳng định hoàn tất khi công cụ chưa thực hiện.
- Gọi công cụ bằng định dạng tool-call chuẩn. Không viết mã Python, không giải thích dài, không thêm bước đọc/xác nhận không cần thiết. Có thể gọi cùng lượt các công cụ độc lập; chờ kết quả trước khi dùng ID mới.
Trước khi kết thúc, tự kiểm tra các phần của yêu cầu đã hoàn thành, không cần xuất bài suy luận dài.'''

# Authored synthetic examples, not extracted from inspected/public/private labels.
# Filter by supported tool names, and keep order fixed across model decisions.
EXAMPLES = [
    ({'weather.get_forecast','calendar.list_events','calendar.cancel_event'}, {
      'request':'Nếu ngày 2030-06-03 mưa thì huỷ lịch chạy bộ; không mưa thì thôi.',
      'behavior':'<tool_call>{"name":"weather.get_forecast","arguments":{"date":"2030-06-03"}}</tool_call> Nếu kết quả rain=false: kết thúc. Nếu rain=true: gọi calendar.list_events, chọn sự kiện chạy bộ đúng ngày trong kết quả, rồi gọi calendar.cancel_event với event_id vừa tìm. Không kết thúc ngay sau dự báo khi nhánh cần huỷ.'}),
    ({'notes.create','sys.ask_missing_info'}, {
      'request':'Tạo một ghi chú giúp tôi. Người dùng chưa cung cấp nội dung.',
      'behavior':'<tool_call>{"name":"sys.ask_missing_info","arguments":{"tool_name":"notes.create","params":["content"]}}</tool_call> Không dùng câu lệnh yêu cầu làm nội dung ghi chú.'}),
    ({'contacts.search','sms.send'}, {
      'request':'Nhắn cho cô Lan rằng tôi đến muộn. Chưa biết ID hay số điện thoại.',
      'behavior':'<tool_call>{"name":"contacts.search","arguments":{"query":"cô Lan"}}</tool_call> Chọn đúng liên hệ trong kết quả rồi gọi sms.send: contact_id phải là id thật vừa nhận, content là "Tôi đến muộn". Không tự viết ID hay số điện thoại; nếu nhiều người cùng phù hợp thì hỏi rõ.'}),
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

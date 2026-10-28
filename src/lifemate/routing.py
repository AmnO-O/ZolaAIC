"""Experimental catalog routing; callers retain full schemas for recovery."""
import re
import unicodedata

DOMAINS = {
    'weather': ['thời tiết', 'mưa', 'nắng', 'bão', 'dự báo', 'nhiệt độ', 'độ c', 'không khí', 'gió'],
    'calendar': ['lịch', 'họp', 'hẹn', 'sự kiện', 'dời', 'hoãn', 'lùi', 'chạy bộ', 'đá bóng', 'sinh nhật', 'lịch biểu'],
    'smart_home': ['điều hòa', 'máy lạnh', 'đèn', 'quạt', 'công tắc', 'rèm', 'ổ cắm', 'phòng khách', 'phòng ngủ', 'nhiệt độ', 'bật', 'tắt', 'độ'],
    'sms': ['nhắn', 'gửi tin', 'sms', 'soạn tin'],
    'chat_ott': ['chat', 'zalo', 'nhắn', 'gửi tin', 'gọi nhóm', 'vị trí'],
    'contacts': ['danh bạ', 'số điện thoại', 'sđt', 'liên hệ', 'tìm số', 'tìm người'],
    'phone': ['gọi điện', 'gọi thoại', 'gọi cho', 'quay số', 'alo'],
    'banking': ['ngân hàng', 'chuyển khoản', 'chuyển tiền', 'chuyển', 'số dư', 'tài khoản', 'tiết kiệm', 'sao kê', 'giao dịch'],
    'ewallet': ['ví', 'nạp tiền', 'nạp điện thoại', 'thanh toán qr', 'momo', 'zalopay', 'nạp'],
    'bill_payment': ['hóa đơn', 'tiền điện', 'tiền nước', 'tiền mạng', 'internet', 'học phí', 'trả điện', 'đóng tiền'],
    'ride_hailing': ['đặt xe', 'đón', 'gọi xe', 'xe máy', 'ô tô', 'chở', 'grab', 'be', 'xe bốn chỗ', 'xe 4 chỗ', 'xe 7 chỗ'],
    'food_delivery': ['đặt đồ ăn', 'giao tận nơi', 'đặt món', 'cơm', 'phở', 'bún', 'trà sữa', 'nhà hàng', 'quán', 'đặt hộ', 'món'],
    'alarm': ['báo thức', 'đánh thức', 'hẹn giờ', 'chuông'],
    'reminders': ['nhắc', 'nhắc việc', 'nhắc tôi', 'đừng quên', 'ghi nhớ'],
    'medication': ['thuốc', 'uống thuốc', 'liều', 'viên', 'uống'],
    'notes': ['ghi chú', 'note', 'viết lại', 'lưu lại'],
    'tasks': ['việc cần làm', 'công việc', 'task', 'hạn chót', 'deadline'],
    'email': ['email', 'thư điện tử', 'gửi mail', 'hòm thư', 'mail'],
    'maps': ['bản đồ', 'chỉ đường', 'đường đi', 'khoảng cách', 'địa điểm'],
    'music': ['nhạc', 'bài hát', 'phát nhạc', 'ca sĩ', 'playlist', 'bài'],
    'tv': ['tivi', 'truyền hình', 'kênh', 'chiếu phim', 'ghi hình'],
    'robot_vacuum': ['hút bụi', 'robot', 'dọn nhà', 'lau nhà'],
    'door_lock': ['khóa cửa', 'mở cửa', 'mã khóa', 'mật khẩu cửa'],
    'flights': ['vé máy bay', 'chuyến bay', 'bay', 'hàng không'],
    'gold_price': ['giá vàng', 'vàng sjc', 'vàng nhẫn', 'vàng'],
    'fuel_price': ['giá xăng', 'giá dầu', 'xăng ron', 'xăng'],
    'lottery': ['xổ số', 'vé số', 'kết quả sổ xố', 'dò vé']
}


def vi_words(text):
    text = unicodedata.normalize('NFC', str(text).lower())
    return set(re.findall(r'[a-z0-9àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+', text))


def clean_context_for_prompt(context):
    """Strip redundant installed_apps JSON to save ~1,500 prompt characters."""
    if not isinstance(context, dict):
        return context
    cleaned = dict(context)
    if 'installed_apps' in cleaned:
        # Keep app names only, remove deep schemas and tool name arrays
        cleaned['installed_apps'] = list(cleaned['installed_apps'].keys())
    return cleaned


class SafeToolRouter:
    def __init__(self, tools):
        self.tools = tools
        self.groups = {}
        for tool in tools:
            app = tool.get('app') or tool['name'].split('.')[0]
            self.groups.setdefault(app, []).append(tool)

    def route(self, query_text, enabled=False):
        """Experimental shortlist; the caller must retain the full catalog for recovery."""
        if not enabled or len(self.groups) <= 4:
            return self.tools

        query = str(query_text).lower()
        q_words = vi_words(query)

        matched = set()
        for app, kws in DOMAINS.items():
            if app in self.groups:
                for kw in kws:
                    if ' ' in kw:
                        if kw in query:
                            matched.add(app)
                            break
                    elif kw in q_words:
                        matched.add(app)
                        break

        # If matching is empty or query is complex/uncertain, keep all installed tools
        if not matched:
            return self.tools

        # Dependency expansion
        if any(a in matched for a in ('sms', 'chat_ott', 'phone')) and 'contacts' in self.groups:
            matched.add('contacts')
        if any(w in query for w in ('mưa', 'nắng', 'thời tiết', 'bão')) and 'weather' in self.groups:
            matched.add('weather')
        if any(w in q_words for w in ('sếp', 'anh', 'chị', 'em', 'mẹ', 'bố', 'ba', 'cô', 'chú', 'bác', 'ông', 'bà', 'bạn')) and 'contacts' in self.groups:
            matched.add('contacts')

        # Unrecognized apps must not disappear because a known domain matched.
        matched.update(app for app in self.groups if app not in DOMAINS)
        if 'bill_payment' in matched and any(word in q_words for word in ('chuyển', 'gửi')):
            matched.update(app for app, group in self.groups.items()
                           if any(t['name'].endswith('.transfer') for t in group))
        # Follow declared references instead of relying only on app-name rules.
        while True:
            previous = set(matched)
            for app in list(matched):
                for tool in self.groups.get(app, []):
                    for prop in tool.get('parameters', {}).get('properties', {}).values():
                        ref = prop.get('x-ref')
                        if isinstance(ref, str) and ref.split('.')[0] in self.groups:
                            matched.add(ref.split('.')[0])
            if matched == previous:
                break

        # Always include sys.ask_missing_info
        matched.add('sys')

        selected = [t for t in self.tools if (t.get('app') or t['name'].split('.')[0]) in matched]
        return selected if selected else self.tools

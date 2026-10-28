import sys
import json
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.agent import run_agent
from lifemate.input_adapter import SplitInput


class FakeEnv:
    def __init__(self):
        self.calls = []

    def validate(self, name, args):
        return {'ok': name != 'bad', 'error': 'invalid'}

    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': args})
        return {'ok': True}

    def get_tool_calls(self):
        return self.calls.copy()

    def get_state_hash(self):
        return 'fake-hash'


TOOLS = [{'name': 'weather.get_forecast', 'kind': 'read'},
         {'name': 'calendar.list_events', 'kind': 'read'},
         {'name': 'calendar.update_event', 'kind': 'write'},
         {'name': 'sms.send', 'kind': 'write'},
         {'name': 'sys.ask_missing_info', 'kind': 'meta'}]


def read():
    return [{'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-29'}}]


def multi_read():
    return [{'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-29'}},
            {'name': 'calendar.list_events', 'arguments': {}}]


def write():
    return [{'name': 'calendar.update_event', 'arguments': {'event_id': 'evt_01'}}]


def multi_write():
    return [{'name': 'calendar.update_event', 'arguments': {'event_id': 'evt_01'}},
            {'name': 'sms.send', 'arguments': {'contact_id': 'c_08', 'content': 'ok'}}]


class PipelineTests(unittest.TestCase):
    def test_validation_repair_does_not_execute_invalid_call(self):
        env = FakeEnv()
        decisions = iter([[{'name': 'bad', 'arguments': {}}],
                          [{'name': 'valid', 'arguments': {}}], []])
        row, log = run_agent(env, lambda *_: next(decisions), [], [], time.monotonic()+5)
        self.assertEqual([c['name'] for c in row['tool_calls']], ['valid'])
        self.assertEqual(log['termination'], 'finished')

    def test_deadline_executes_nothing(self):
        row, log = run_agent(FakeEnv(), lambda *_: self.fail('backend called'), [], [], 0)
        self.assertEqual(row['tool_calls'], [])
        self.assertEqual(log['termination'], 'deadline')

    def test_continue_after_read_reminds_when_empty_after_read(self):
        env = FakeEnv()
        decisions = iter([read(), [], write(), []])
        row, log = run_agent(env, lambda *_: next(decisions), [], TOOLS, time.monotonic() + 5, continue_after_read=True)
        self.assertEqual([c['name'] for c in row['tool_calls']], ['weather.get_forecast', 'calendar.update_event'])
        self.assertTrue(log['continued_after_read'])
        self.assertEqual(log['termination'], 'finished')

    def test_batched_reads_execute_and_count_as_read_executed(self):
        """Both reads in a multi-read batch should execute, and write follows on next turn."""
        env = FakeEnv()
        decisions = iter([multi_read(), multi_write(), []])
        row, log = run_agent(env, lambda *_: next(decisions), [], TOOLS, time.monotonic() + 5)
        names = [c['name'] for c in row['tool_calls']]
        self.assertIn('weather.get_forecast', names)
        self.assertIn('calendar.list_events', names)
        self.assertIn('calendar.update_event', names)
        self.assertIn('sms.send', names)
        self.assertEqual(log['termination'], 'finished')
        # No spurious reminder when write did execute
        self.assertFalse(log['continued_after_read'])

    def test_batched_reads_trigger_remind_if_no_write_follows(self):
        """Multi-read batch with no subsequent write should trigger the continue reminder."""
        env = FakeEnv()
        decisions = iter([multi_read(), [], write(), []])
        row, log = run_agent(env, lambda *_: next(decisions), [], TOOLS, time.monotonic() + 5, continue_after_read=True)
        self.assertTrue(log['continued_after_read'])
        self.assertIn('calendar.update_event', [c['name'] for c in row['tool_calls']])

    def test_clarification_stops_remaining_batch(self):
        env=FakeEnv()
        row,log=run_agent(env,lambda *_:[{'name':'sys.ask_missing_info','arguments':{'tool_name':'a.b','params':['id']}},
                                       {'name':'must_not_execute','arguments':{}}],[],[],time.monotonic()+5)
        self.assertEqual([c['name'] for c in row['tool_calls']],['sys.ask_missing_info'])
        self.assertEqual(log['termination'],'clarification')

    def test_ungrounded_xref_rejected_with_validation_error(self):
        env = FakeEnv()
        tools_with_xref = [
            {'name': 'sms.send', 'kind': 'write', 'parameters': {
                'properties': {'contact_id': {'type': 'string', 'x-ref': 'contacts.contacts'}}}}
        ]
        # Attempt to call sms.send with fake ungrounded contact_id
        decisions = iter([[{'name': 'sms.send', 'arguments': {'contact_id': 'contact_123'}}], []])
        messages = [{'role': 'user', 'content': 'Nhắn tin cho mẹ'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools_with_xref, time.monotonic() + 5)
        # Fake call should NOT be executed on env
        self.assertEqual(row['tool_calls'], [])
        # Tool result in trace should record validation error
        self.assertIn('validation_error', log['trace'][0]['result'])

    def test_weather_location_is_not_silently_removed(self):
        env = FakeEnv()
        decisions = iter([[{'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-29', 'location': 'Đà Nẵng'}}], []])
        # User message did NOT mention Đà Nẵng
        messages = [{'role': 'user', 'content': 'Nếu mai mưa thì thôi'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, TOOLS, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(len(row['tool_calls']), 1)
        self.assertEqual(row['tool_calls'][0]['arguments'], {'date': '2026-11-29', 'location': 'Đà Nẵng'})

    def test_correct_model_relative_date_is_preserved(self):
        env = FakeEnv()
        # The model resolves tomorrow from the supplied context; controller preserves it.
        decisions = iter([[{'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-21'}}], []])
        messages = [
            {'role': 'system', 'content': {'context': {'current_time': '2026-11-20T15:30:00+07:00'}}},
            {'role': 'user', 'content': 'Xem giúp mai có mưa không'}
        ]
        row, log = run_agent(env, lambda *_: next(decisions), messages, TOOLS, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(row['tool_calls'][0]['arguments']['date'], '2026-11-21')

    def test_calendar_lookup_filters_are_preserved(self):
        env = FakeEnv()
        decisions = iter([[{'name': 'calendar.list_events', 'arguments': {'date_from': '2026-11-29', 'date_to': '2026-11-29'}}], []])
        messages = [{'role': 'user', 'content': 'Nếu mai mưa thì dời chạy bộ muộn 1 tiếng'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, TOOLS, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(len(row['tool_calls']), 1)
        self.assertEqual(row['tool_calls'][0]['arguments'], {'date_from': '2026-11-29', 'date_to': '2026-11-29'})

    def test_requested_calendar_dates_are_preserved(self):
        env = FakeEnv()
        decisions = iter([[{'name': 'calendar.list_events', 'arguments': {'date_from': '2026-11-29', 'date_to': '2026-11-29'}}], []])
        messages = [{'role': 'user', 'content': 'Xem lịch hôm nay giúp tôi'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, TOOLS, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(len(row['tool_calls']), 1)
        self.assertEqual(row['tool_calls'][0]['arguments'], {'date_from': '2026-11-29', 'date_to': '2026-11-29'})

    def test_contacts_search_preserves_person_names_and_query(self):
        env = FakeEnv()
        tools = [{'name': 'contacts.search', 'kind': 'read'}]
        decisions = iter([
            [{'name': 'contacts.search', 'arguments': {'query': 'Xếp Dũng'}}],
            [{'name': 'contacts.search', 'arguments': {'query': 'xếp dựng'}}],
            [{'name': 'contacts.search', 'arguments': {'query': 'Sếp dung'}}],
            [{'name': 'contacts.search', 'arguments': {'query': 'cô mai'}}],
            [{'name': 'contacts.search', 'arguments': {'query': 'bác ba'}}],
            []
        ])
        messages = [{'role': 'user', 'content': 'Nhắn xếp Dũng giờ mới'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(row['tool_calls'][0]['arguments']['query'], 'Xếp Dũng')
        self.assertEqual(row['tool_calls'][1]['arguments']['query'], 'xếp dựng')
        self.assertEqual(row['tool_calls'][2]['arguments']['query'], 'Sếp dung')
        self.assertEqual(row['tool_calls'][3]['arguments']['query'], 'cô mai')
        self.assertEqual(row['tool_calls'][4]['arguments']['query'], 'bác ba')

    def test_smarthome_get_status_preserves_requested_filters(self):
        env = FakeEnv()
        tools = [{'name': 'smart_home.get_status', 'kind': 'read'}]
        decisions = iter([[{'name': 'smart_home.get_status', 'arguments': {'room': 'phòng ngủ', 'type': 'light'}}], []])
        messages = [{'role': 'user', 'content': 'Giảm độ sáng đèn phòng ngủ nhỏ xuống còn 50'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(row['tool_calls'][0]['arguments'], {'room': 'phòng ngủ', 'type': 'light'})

    def test_model_spoken_amount_is_preserved(self):
        env = FakeEnv()
        tools = [{'name': 'ewallet.topup_mobile', 'kind': 'write'}]
        decisions = iter([[{'name': 'ewallet.topup_mobile', 'arguments': {'phone': '0912345678', 'amount': 50000}}], []])
        messages = [{'role': 'user', 'content': 'Nạp giúp mình năm mươi ngàn tiền điện thoại 0912345678'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(row['tool_calls'][0]['arguments']['amount'], 50000)

    def test_bill_transfer_amount_from_observation_is_preserved(self):
        class BillEnv(FakeEnv):
            def call(self, name, args):
                super().call(name, args)
                if name == 'bill_payment.list_bills':
                    return {'ok': True, 'bills': [{'type': 'electricity', 'amount': 845000}]}
                return {'ok': True}

        env = BillEnv()
        tools = [{'name': 'bill_payment.list_bills', 'kind': 'read'}, {'name': 'ewallet.transfer', 'kind': 'write'}]
        decisions = iter([
            [{'name': 'bill_payment.list_bills', 'arguments': {}}],
            [{'name': 'ewallet.transfer', 'arguments': {'phone': '0912345678', 'amount': 845000}}],
            []
        ])
        messages = [{'role': 'user', 'content': 'Chuyển tiền điện tháng này tới số 0912345678'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        transfer_call = [c for c in row['tool_calls'] if c['name'] == 'ewallet.transfer'][0]
        self.assertEqual(transfer_call['arguments']['amount'], 845000)
        self.assertNotIn('note', transfer_call['arguments'])

    def test_bill_transfer_without_bill_lookup_is_intercepted(self):
        env = FakeEnv()
        tools = [{'name': 'bill_payment.list_bills', 'kind': 'read'}, {'name': 'ewallet.transfer', 'kind': 'write'}]
        decisions = iter([
            [{'name': 'ewallet.transfer', 'arguments': {'phone': '0912345678', 'amount': 50000}}],
            []
        ])
        messages = [{'role': 'user', 'content': 'Chuyển tiền điện tháng này tới số 0912345678'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(row['tool_calls'], [])
        self.assertIn('validation_error', log['trace'][0]['result'])

    def test_calendar_update_event_computes_end_time_from_duration(self):
        class CalendarEnv(FakeEnv):
            def call(self, name, args):
                super().call(name, args)
                if name == 'calendar.list_events':
                    return {'ok': True, 'events': [{'id': 'evt_01', 'start': '2026-11-29T08:00:00+07:00', 'end': '2026-11-29T09:00:00+07:00'}]}
                return {'ok': True}

        env = CalendarEnv()
        decisions = iter([
            [{'name': 'calendar.list_events', 'arguments': {}}],
            [{'name': 'calendar.update_event', 'arguments': {'event_id': 'evt_01', 'start': '2026-11-29T09:00:00+07:00'}}],
            []
        ])
        messages = [{'role': 'user', 'content': 'Dời chạy bộ muộn 1 tiếng'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, TOOLS, time.monotonic() + 5, continue_after_read=False)
        update_call = [c for c in row['tool_calls'] if c['name'] == 'calendar.update_event'][0]
        self.assertEqual(update_call['arguments']['end'], '2026-11-29T10:00:00+07:00')

    def test_fabricated_phone_number_is_rejected(self):
        env = FakeEnv()
        tools = [{'name': 'ewallet.topup_mobile', 'kind': 'write'}]
        decisions = iter([[{'name': 'ewallet.topup_mobile', 'arguments': {'phone': '0901234567', 'amount': 50000}}], []])
        messages = [{'role': 'user', 'content': 'Nạp 50k cho bác Ba'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual(row['tool_calls'], [])
        self.assertIn('validation_error', log['trace'][0]['result'])

    def test_multi_action_incomplete_triggers_reminder(self):
        env = FakeEnv()
        # Turn 1: write calendar.update_event
        # Turn 2: empty (model thinks it's done, but hasn't sent sms)
        # Turn 3: after reminder, calls sms.send
        # Turn 4: empty (finished)
        def mock_decide(msgs, tools, remaining):
            last_msg = msgs[-1]
            if last_msg.get('role') == 'user' and 'requested message/notification' in str(last_msg.get('content', '')):
                return [{'name': 'sms.send', 'arguments': {'contact_id': 'c_08', 'content': 'done'}}]
            if len(env.get_tool_calls()) == 0:
                return [{'name': 'calendar.update_event', 'arguments': {'event_id': 'evt_01'}}]
            return []

        messages = [{'role': 'user', 'content': 'Dời lịch chạy bộ và nhắn tin cho Dũng nhé'}]
        row, log = run_agent(env, mock_decide, messages, TOOLS, time.monotonic() + 5)
        called_names = [c['name'] for c in row['tool_calls']]
        self.assertEqual(called_names, ['calendar.update_event', 'sms.send'])

    def test_ride_hailing_resolves_work_and_event_addresses(self):
        class RideEnv(FakeEnv):
            def call(self, name, args):
                super().call(name, args)
                if name == 'calendar.list_events':
                    return {'ok': True, 'events': [{'id': 'evt_01', 'title': 'họp phụ huynh', 'location': 'Trường TH Nguyễn Bỉnh Khiêm, Quận 1'}]}
                return {'ok': True}

        env = RideEnv()
        tools = [{'name': 'calendar.list_events', 'kind': 'read'}, {'name': 'ride_hailing.book', 'kind': 'write'}]
        decisions = iter([
            [{'name': 'calendar.list_events', 'arguments': {}}],
            [{'name': 'ride_hailing.book', 'arguments': {'pickup': 'công ty', 'dropoff': 'họp phụ huynh', 'vehicle': 'car4'}}],
            []
        ])
        messages = [
            {'role': 'system', 'content': {'context': {'user_profile': {'work_address': 'Khu Công nghệ cao, TP.HCM'}}}},
            {'role': 'user', 'content': 'Đặt xe bốn chỗ từ công ty đi họp phụ huynh'}
        ]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        book_call = [c for c in row['tool_calls'] if c['name'] == 'ride_hailing.book'][0]
        self.assertEqual(book_call['arguments']['pickup'], 'Khu Công nghệ cao, TP.HCM')
        self.assertEqual(book_call['arguments']['dropoff'], 'Trường TH Nguyễn Bỉnh Khiêm, Quận 1')
        self.assertNotIn('payment', book_call['arguments'])

    def test_ride_hailing_preserves_requested_quote_and_booking(self):
        env = FakeEnv()
        tools = [{'name': 'ride_hailing.get_quotes', 'kind': 'read'}, {'name': 'ride_hailing.book', 'kind': 'write'}]
        decisions = iter([
            [{'name': 'ride_hailing.get_quotes', 'arguments': {'pickup': 'A', 'dropoff': 'B', 'vehicle': 'car4'}},
             {'name': 'ride_hailing.book', 'arguments': {'pickup': 'A', 'dropoff': 'B', 'vehicle': 'car4'}}],
            []
        ])
        messages = [{'role': 'user', 'content': 'Lấy báo giá và đặt xe giúp tôi'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        self.assertEqual([c['name'] for c in row['tool_calls']], ['ride_hailing.get_quotes', 'ride_hailing.book'])

    def test_transfer_observed_contact_id_is_preserved(self):
        class ContactEnv(FakeEnv):
            def call(self, name, args):
                super().call(name, args)
                if name == 'contacts.search':
                    return {'ok': True, 'contacts': [{'id': 'c_02', 'name': 'Bác Ba'}, {'id': 'c_06', 'name': 'Cô giáo Mai'}]}
                return {'ok': True}

        env = ContactEnv()
        tools = [{'name': 'contacts.search', 'kind': 'read'}, {'name': 'ewallet.transfer', 'kind': 'write'}]
        decisions = iter([
            [{'name': 'contacts.search', 'arguments': {'query': 'Cô Mai'}}],
            [{'name': 'ewallet.transfer', 'arguments': {'contact_id': 'c_06', 'amount': 100000}}],
            []
        ])
        messages = [{'role': 'user', 'content': 'Chuyển ví cho cô Mai tiền điện tháng này nhé'}]
        row, log = run_agent(env, lambda *_: next(decisions), messages, tools, time.monotonic() + 5, continue_after_read=False)
        transfer_call = [c for c in row['tool_calls'] if c['name'] == 'ewallet.transfer'][0]
        self.assertEqual(transfer_call['arguments']['contact_id'], 'c_06')

    def test_train_labels_and_transcripts_are_removed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as temporary:
            root = Path(temporary)
            (root / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': [
                {'name': 'sys.ask_missing_info'}, {'name': 'uninstalled.action'}]}]}), encoding='utf-8')
            (root / 'train.jsonl').write_text(json.dumps({
                'id': 'synthetic_1', 'audio': 'audio/example.wav',
                'context': {'installed_apps': {}}, 'label': {'synthetic': True},
                'audio_text': 'REFERENCE_ONLY',
                'history': [{'role': 'user', 'audio': 'audio/history.wav', 'audio_text': 'HISTORY_REFERENCE'}]
            }), encoding='utf-8')
            split = SplitInput(root)
            item = next(split.records())
            self.assertNotIn('label', item)
            self.assertNotIn('audio_text', item)
            self.assertTrue(all('audio_text' not in turn for turn in item['history']))
            self.assertEqual([t['name'] for t in split.tools(item)], ['sys.ask_missing_info'])


if __name__ == '__main__':
    unittest.main()

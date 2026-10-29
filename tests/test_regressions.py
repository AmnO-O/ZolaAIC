"""Adversarial cases from review: preserve valid arguments and isolate failures."""
import copy
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.agent import run_agent


def call(name, **args):return {'name':name,'arguments':args}
def spec(name,kind='write',props=None):
    return {'name':name,'kind':kind,'parameters':{'properties':props or {}}}
def user(text):return {'role':'user','content':text}
def result(name, data):return {'role':'tool','name':name,'content':{'ok':True,**data}}


class Env:
    def __init__(self, responses=None):self.calls=[];self.responses=responses or {}
    def validate(self,name,args):return {'ok':True}
    def call(self,name,args):
        self.calls.append(call(name,**copy.deepcopy(args)))
        return self.responses.get(name,{'ok':True})
    def get_tool_calls(self):return self.calls.copy()
    def get_state_hash(self):return 'synthetic-hash'


class RegressionTests(unittest.TestCase):
    def execute(self,messages,batch,tools=None,responses=None):
        env=Env(responses)
        decisions=iter([batch,[],[],[]])
        row,log=run_agent(env,lambda *_:next(decisions),messages,
            tools or [spec(c['name']) for c in batch],time.monotonic()+5)
        return row['tool_calls'],log

    def test_150k_is_not_treated_as_50k(self):
        batch=[call('ewallet.topup_mobile',phone='0901111111',amount=150000)]
        actual,_=self.execute([user('Nạp 150k cho số 0901111111')],batch)
        self.assertEqual(actual,batch)

    def test_two_topups_preserve_separate_amounts(self):
        batch=[call('ewallet.topup_mobile',phone='0901111111',amount=50000),
               call('ewallet.topup_mobile',phone='0902222222',amount=100000)]
        actual,_=self.execute([user('Nạp 50k cho 0901111111 và 100k cho 0902222222')],batch)
        self.assertEqual(actual,batch)

    def test_latest_amount_not_overwritten_from_history(self):
        batch=[call('ewallet.topup_mobile',phone='0901111111',amount=100000)]
        actual,_=self.execute([user('Nạp 50k cho 0901111111'),user('Sửa lại, nạp 100k cho 0901111111')],batch)
        self.assertEqual(actual,batch)

    def test_explicit_partial_bill_amount_is_not_replaced_by_full_bill(self):
        batch=[call('ewallet.transfer',phone='0901111111',amount=100000)]
        messages=[user('Chuyển 100k tiền điện cho số 0901111111'),
                  result('bill_payment.list_bills',{'bills':[{'type':'electricity','amount':845000}]})]
        actual,_=self.execute(messages,batch)
        self.assertEqual(actual,batch)

    def test_phone_bill_not_replaced_by_electricity_amount(self):
        batch=[call('ewallet.transfer',phone='0901111111',amount=120000)]
        messages=[user('Chuyển tiền điện thoại cho 0901111111'),result('bill_payment.list_bills',
            {'bills':[{'type':'electricity','amount':845000},{'type':'phone','amount':120000}]})]
        actual,_=self.execute(messages,batch)
        self.assertEqual(actual,batch)

    def test_explicit_bill_amount_does_not_force_an_extra_lookup(self):
        batch=[call('ewallet.transfer',phone='0901111111',amount=100000)]
        actual,_=self.execute([user('Chuyển tiền điện 100000 cho số 0901111111')],batch,
            [spec('ewallet.transfer'),spec('bill_payment.list_bills','read')])
        self.assertEqual(actual,batch)

    def test_same_honorific_does_not_change_recipient(self):
        batch=[call('ewallet.transfer',contact_id='c_lan',amount=50000)]
        messages=[user('Chuyển cho chị Lan số tiền 50000'),result('contacts.search',
            {'contacts':[{'id':'c_mai','name':'Chị Mai'},{'id':'c_lan','name':'Chị Lan'}]})]
        actual,_=self.execute(messages,batch,[spec('ewallet.transfer',props={'contact_id':{'x-ref':'contacts.contacts'}})])
        self.assertEqual(actual,batch)

    def test_person_names_are_not_phonetically_rewritten(self):
        batch=[call('contacts.search',query='Phương Dung')]
        actual,_=self.execute([user('Tìm cả Dũng và Phương Dung')],batch)
        self.assertEqual(actual,batch)

    def test_restaurant_is_not_home_alias(self):
        batch=[call('ride_hailing.book',pickup='Nhà hàng Sen, 5 Trần Phú',dropoff='12 Lê Lợi',vehicle='car4')]
        messages=[{'role':'system','content':{'context':{'user_profile':{'home_address':'HOME'}}}},
                  user('Đặt xe từ Nhà hàng Sen, 5 Trần Phú tới 12 Lê Lợi')]
        actual,_=self.execute(messages,batch)
        self.assertEqual(actual,batch)

    def test_explicit_destination_not_changed_by_event_mention(self):
        batch=[call('ride_hailing.book',pickup='A',dropoff='12 Lê Lợi',vehicle='car4')]
        messages=[user('Đặt xe tới 12 Lê Lợi sau cuộc họp'),result('calendar.list_events',
            {'events':[{'id':'e1','title':'cuộc họp','location':'88 Nguyễn Trãi'}]})]
        actual,_=self.execute(messages,batch)
        self.assertEqual(actual,batch)

    def test_controller_does_not_resolve_ambiguous_event_destination(self):
        batch=[call('ride_hailing.book',pickup='A',dropoff='họp',vehicle='car4')]
        messages=[user('Đặt xe đi họp'),result('calendar.list_events',{'events':[
            {'id':'a','title':'họp','location':'A'},{'id':'b','title':'họp','location':'B'}]})]
        actual,log=self.execute(messages,batch)
        self.assertEqual(actual,batch)

    def test_explicit_date_survives_person_named_mai_and_history(self):
        batch=[call('weather.get_forecast',date='2026-12-05')]
        messages=[{'role':'system','content':{'context':{'current_time':'2026-11-28T07:10:00+07:00'}}},
                  user('Mai có mưa không?'),user('Mai Anh sửa lại: xem ngày 2026-12-05, không phải mai')]
        actual,_=self.execute(messages,batch)
        self.assertEqual(actual,batch)

    def test_historical_tool_results_ground_ids_without_replaying(self):
        batch=[call('calendar.cancel_event',event_id='event_real')]
        messages=[user('Tìm lịch chạy bộ'),{'role':'assistant',
            'tool_calls':[call('calendar.list_events')],
            'tool_results':[{'ok':True,'events':[{'id':'event_real'}]}]},user('Hủy lịch đó')]
        actual,_=self.execute(messages,batch,[spec('calendar.cancel_event',props={'event_id':{'x-ref':'calendar.events'}})])
        self.assertEqual(actual,batch)

    def test_historical_duration_does_not_inject_argument(self):
        messages=[{'role':'assistant','tool_calls':[call('calendar.list_events')],
            'tool_results':[{'ok':True,'events':[{'id':'e1','start':'2026-11-29T08:00:00+07:00','end':'2026-11-29T09:00:00+07:00'}]}]},user('Dời lịch muộn một giờ')]
        actual,_=self.execute(messages,[call('calendar.update_event',event_id='e1',start='2026-11-29T09:00:00+07:00')])
        self.assertNotIn('end', actual[0]['arguments'])

    def test_do_not_change_end_when_user_explicitly_keeps_it(self):
        batch=[call('calendar.update_event',event_id='e1',start='2026-11-29T08:30:00+07:00')]
        messages=[user('Dời giờ bắt đầu, giữ nguyên giờ kết thúc'),result('calendar.list_events',
            {'events':[{'id':'e1','start':'2026-11-29T08:00:00+07:00','end':'2026-11-29T09:00:00+07:00'}]})]
        actual,_=self.execute(messages,batch)
        self.assertEqual(actual,batch)

    def test_created_event_duration_does_not_inject_argument(self):
        messages=[{'role':'assistant','tool_calls':[call('calendar.create_event')],
            'tool_results':[{'ok':True,'item':{'id':'e1','start':'2026-11-29T08:00:00+07:00','end':'2026-11-29T09:00:00+07:00'}}]},
            user('Dời lịch vừa tạo muộn một giờ')]
        actual,_=self.execute(messages,[call('calendar.update_event',event_id='e1',start='2026-11-29T09:00:00+07:00')],
            [spec('calendar.update_event',props={'event_id':{'x-ref':'calendar.events'}})])
        self.assertNotIn('end', actual[0]['arguments'])

    def test_wrong_app_result_does_not_ground_id(self):
        actual,_=self.execute([user('Nhắn cho mẹ'),result('banking.get_balance',{'accounts':[{'id':'same_id'}]})],
            [call('sms.send',contact_id='same_id',content='hello')],
            [spec('sms.send',props={'contact_id':{'x-ref':'contacts.contacts'}})])
        self.assertEqual(actual,[])

    def test_id_substring_is_not_explicit_user_identifier(self):
        actual,_=self.execute([user('Hủy event_e10')],[call('calendar.cancel_event',event_id='event_e1')],
            [spec('calendar.cancel_event',props={'event_id':{'x-ref':'calendar.events'}})])
        self.assertEqual(actual,[])

    def test_failed_observation_does_not_ground_ids(self):
        messages=[user('Hủy lịch đó'),{'role':'tool','name':'calendar.list_events',
            'content':{'ok':False,'events':[{'id':'fake'}]}}]
        actual,_=self.execute(messages,[call('calendar.cancel_event',event_id='fake')],
            [spec('calendar.cancel_event',props={'event_id':{'x-ref':'calendar.events'}})])
        self.assertEqual(actual,[])

    def test_all_phone_recipient_fields_reject_invented_number(self):
        for name,key in (('sms.send','phone_number'),('ewallet.transfer','phone'),('phone.call','phone')):
            with self.subTest(name=name):
                actual,_=self.execute([user('Gửi cho mẹ')],[call(name,**{key:'0999999999'})])
                self.assertEqual(actual,[])

    def test_phone_formatting_and_country_prefix_can_match_history(self):
        batch=[call('sms.send',phone_number='0912345678',content='hello')]
        messages=[{'role':'assistant','tool_calls':[call('contacts.search')],
            'tool_results':[{'ok':True,'contacts':[{'id':'c1','phone':'+84 912 345 678'}]}]},user('Nhắn cho người đó')]
        actual,_=self.execute(messages,batch)
        self.assertEqual(actual,batch)

    def test_logging_time_is_deducted_before_decide(self):
        clock=[0.0];received=[]
        def emit(stage,**payload):
            if stage=='decision.input':clock[0]+=5
        def decide(*args):received.append(args[-1]);clock[0]+=6;return []
        with patch('lifemate.agent.time.monotonic',side_effect=lambda:clock[0]):
            row,log=run_agent(Env(),decide,[],[],10,emit=emit)
        self.assertEqual(received,[5])
        self.assertEqual(log['termination'],'deadline')

    def test_logging_before_execution_cannot_allow_late_write(self):
        clock=[0.0];env=Env()
        def emit(stage,**payload):
            if stage=='tool.input':clock[0]=11
        with patch('lifemate.agent.time.monotonic',side_effect=lambda:clock[0]):
            row,log=run_agent(env,lambda *_:[call('notes.create',content='x')],[],[spec('notes.create')],10,emit=emit)
        self.assertEqual(row['tool_calls'],[])
        self.assertEqual(log['termination'],'deadline')

    def test_failed_write_is_attempted_not_successful(self):
        actual,log=self.execute([user('Tạo ghi chú')],[call('notes.create',content='x')],
            responses={'notes.create':{'ok':False,'error':'failed'}})
        self.assertEqual(len(actual),1)
        self.assertFalse(log['write_succeeded'])
        self.assertEqual(log['attempted_writes'],1)
        self.assertEqual(log['termination'],'execution_failed')

    def test_no_failure_cleared_by_unrelated_success(self):
        actual,log=self.execute([user('Tạo ghi chú và báo thức')],
            [call('notes.create',content='x'),call('alarm.create',time='t')],
            responses={'notes.create':{'ok':False,'error':'failed'}})
        self.assertTrue(log['write_succeeded'])
        self.assertEqual(log['termination'],'execution_failed')

    def test_full_catalog_is_provided_without_lexical_pruning(self):
        env=Env();seen=[]
        tools=[spec('notes.create'),spec('alarm.create'),spec('unknown_app.act')]
        def decide(messages, offered, remaining):
            seen.append(offered)
            return []
        row,log=run_agent(env,decide,[],tools,time.monotonic()+5)
        self.assertEqual(seen,[tools])
        self.assertEqual(row['tool_calls'],[])

    def test_full_catalog_schema_grounds_every_call(self):
        env=Env();decisions=iter([[call('sms.send',contact_id='invented')],[]])
        tools=[spec('notes.create'),spec('sms.send',props={'contact_id':{'x-ref':'contacts.contacts'}})]
        row,log=run_agent(env,lambda *_:next(decisions),[user('Nhắn cho mẹ')],tools,time.monotonic()+5)
        self.assertEqual(row['tool_calls'],[])
        self.assertIn('Ungrounded',log['trace'][0]['result']['validation_error'])

    def test_spoken_phone_number_is_grounded_and_accepted(self):
        batch=[call('ewallet.topup_mobile',phone='09505360367',amount=50000)]
        actual,_=self.execute([user('Nạp dùm 50 ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha.')],batch)
        self.assertEqual(actual,batch)

    def test_event_title_containment_does_not_rewrite_dropoff(self):
        batch=[call('ride_hailing.book',pickup='A',dropoff='hội họp phụ huynh',vehicle='car4')]
        messages=[user('Đặt xe đi hội họp phụ huynh'),result('calendar.list_events',
            {'events':[{'id':'e1','title':'họp phụ huynh','location':'Trường Nguyễn Bỉnh Khiêm'}]})]
        actual,_=self.execute(messages,batch)
        self.assertEqual(actual, batch)

    def test_controller_does_not_classify_event_destinations_by_keywords(self):
        batch=[call('ride_hailing.book',pickup='A',dropoff='hội họp phụ huynh',vehicle='car4')]
        actual,log=self.execute([user('Đặt xe đi hội họp phụ huynh')],batch)
        self.assertEqual(actual,batch)


if __name__=='__main__':unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.routing import SafeToolRouter, clean_context_for_prompt


class RoutingTests(unittest.TestCase):
    def test_clean_context_for_prompt_replaces_dict_with_names(self):
        context = {
            'current_time': '2026-11-28T07:10:00+07:00',
            'installed_apps': {
                'weather': {'tools': ['weather.get_current']},
                'calendar': {'tools': ['calendar.list_events']}
            }
        }
        cleaned = clean_context_for_prompt(context)
        self.assertEqual(cleaned['installed_apps'], ['weather', 'calendar'])
        self.assertEqual(cleaned['current_time'], context['current_time'])

    def test_few_apps_not_pruned(self):
        tools = [
            {'name': 'weather.get_forecast'},
            {'name': 'calendar.update_event'},
            {'name': 'sys.ask_missing_info'}
        ]
        router = SafeToolRouter(tools)
        # 3 apps (weather, calendar, sys) <= 4 -> no pruning
        self.assertEqual(router.route('Bật đèn phòng khách', enabled=True), tools)

    def test_pruning_routes_and_expands_dependencies(self):
        tools = [
            {'name': 'weather.get_forecast'},
            {'name': 'calendar.update_event'},
            {'name': 'contacts.search'},
            {'name': 'sms.send'},
            {'name': 'banking.transfer'},
            {'name': 'ewallet.pay_qr'},
            {'name': 'food_delivery.place_order'},
            {'name': 'sys.ask_missing_info'}
        ]
        router = SafeToolRouter(tools)
        # Query about sending SMS to a person -> should include sms, contacts, and sys
        selected = router.route('Nhắn tin cho mẹ là con sắp về', enabled=True)
        selected_names = {t['name'] for t in selected}
        self.assertIn('sms.send', selected_names)
        self.assertIn('contacts.search', selected_names)
        self.assertIn('sys.ask_missing_info', selected_names)
        self.assertNotIn('banking.transfer', selected_names)
        self.assertNotIn('food_delivery.place_order', selected_names)

    def test_uncertain_fallback_preserves_all_tools(self):
        tools = [
            {'name': 'weather.get_forecast'},
            {'name': 'calendar.update_event'},
            {'name': 'contacts.search'},
            {'name': 'sms.send'},
            {'name': 'banking.transfer'},
            {'name': 'sys.ask_missing_info'}
        ]
        router = SafeToolRouter(tools)
        # Unmatched query -> safely falls back to full list
        selected = router.route('xyz 123 không rõ ràng gì cả', enabled=True)
        self.assertEqual(selected, tools)

    def test_default_is_full_catalog_and_unknown_apps_are_not_pruned(self):
        tools=[{'name':name} for name in ('sms.send','contacts.search','weather.get_forecast',
            'calendar.list_events','new_service.act','sys.ask_missing_info')]
        router=SafeToolRouter(tools)
        self.assertEqual(router.route('Nhắn tin cho mẹ'),tools)
        self.assertIn('new_service.act',{t['name'] for t in router.route('Nhắn tin cho mẹ',enabled=True)})

    def test_bill_transfer_keeps_transfer_capabilities(self):
        tools=[{'name':name} for name in ('ewallet.transfer','bill_payment.list_bills','contacts.search',
            'weather.get_forecast','notes.create','sys.ask_missing_info')]
        selected=SafeToolRouter(tools).route('Chuyển tiền điện cho anh Nam',enabled=True)
        self.assertIn('ewallet.transfer',{t['name'] for t in selected})

    def test_declared_dependencies_are_followed_transitively(self):
        tools=[{'name':'sms.send','parameters':{'properties':{'target':{'x-ref':'email.messages'}}}},
               {'name':'email.lookup','parameters':{'properties':{'place':{'x-ref':'maps.places'}}}},
               {'name':'maps.get'},{'name':'weather.get_forecast'},
               {'name':'calendar.list_events'},{'name':'sys.ask_missing_info'}]
        selected=SafeToolRouter(tools).route('Nhắn tin',enabled=True)
        self.assertIn('maps.get',{t['name'] for t in selected})


if __name__ == '__main__':
    unittest.main()

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


if __name__ == '__main__':
    unittest.main()

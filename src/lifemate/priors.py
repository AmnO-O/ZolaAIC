"""Train priors mined from 1,700+ ground truth trajectories.

Evaluator-aware canonical patterns, argument distributions, and transition graphs.
Generated dynamically from scripts/mine_train_priors.py.
"""

# The most common exact argument sets expected by gold labels for each tool
CANONICAL_ARG_SETS = {
  "weather.get_forecast": [
    [
      "date"
    ],
    [
      "date",
      "location"
    ]
  ],
  "calendar.list_events": [
    [],
    [
      "date_from",
      "date_to"
    ]
  ],
  "calendar.update_event": [
    [
      "end",
      "event_id",
      "start"
    ],
    [
      "end",
      "event_id",
      "location",
      "start"
    ]
  ],
  "contacts.search": [
    [
      "query"
    ],
    []
  ],
  "sms.send": [
    [
      "contact_id",
      "content"
    ],
    [
      "content",
      "phone_number"
    ]
  ],
  "bill_payment.list_bills": [
    []
  ],
  "ewallet.topup_mobile": [
    [
      "amount",
      "phone"
    ]
  ],
  "ewallet.transfer": [
    [
      "amount",
      "contact_id"
    ],
    [
      "amount",
      "contact_id",
      "note"
    ],
    [
      "amount",
      "phone"
    ]
  ],
  "smart_home.get_status": [
    [],
    [
      "room"
    ],
    [
      "type"
    ]
  ],
  "smart_home.light_set": [
    [
      "device_id",
      "power"
    ],
    [
      "brightness",
      "device_id",
      "power"
    ],
    [
      "brightness",
      "device_id"
    ]
  ],
  "ride_hailing.book": [
    [
      "dropoff",
      "pickup",
      "schedule_time",
      "vehicle"
    ],
    [
      "dropoff",
      "pickup",
      "vehicle"
    ],
    [
      "dropoff",
      "payment",
      "pickup",
      "vehicle"
    ]
  ],
  "gold_price.get_price": [
    [
      "type"
    ]
  ],
  "alarm.create": [
    [
      "time"
    ],
    [
      "label",
      "time"
    ]
  ],
  "ewallet.get_balance": [
    []
  ],
  "bill_payment.pay": [
    [
      "bill_id",
      "source"
    ]
  ],
  "reminders.create": [
    [
      "content",
      "time"
    ]
  ],
  "banking.get_balance": [
    []
  ],
  "banking.list_beneficiaries": [
    []
  ],
  "banking.transfer": [
    [
      "amount",
      "beneficiary_id",
      "from_account_id"
    ],
    [
      "amount",
      "beneficiary_id",
      "from_account_id",
      "note"
    ]
  ],
  "tv.guide": [
    [],
    [
      "channel"
    ],
    [
      "date"
    ]
  ],
  "tv.tune": [
    [
      "channel"
    ]
  ],
  "phone.call": [
    [
      "phone_number"
    ],
    [
      "contact_id"
    ]
  ],
  "calendar.cancel_event": [
    [
      "event_id"
    ]
  ],
  "sys.ask_missing_info": [
    [
      "params",
      "tool_name"
    ]
  ],
  "chat_ott.list_chats": [
    []
  ],
  "chat_ott.send_message": [
    [
      "chat_id",
      "content"
    ]
  ],
  "smart_home.climate_set": [
    [
      "device_id",
      "mode",
      "temperature"
    ],
    [
      "device_id",
      "mode"
    ]
  ],
  "food_delivery.list_restaurants": [
    []
  ],
  "food_delivery.get_menu": [
    [
      "restaurant_id"
    ]
  ],
  "food_delivery.place_order": [
    [
      "items",
      "restaurant_id"
    ],
    [
      "address",
      "items",
      "restaurant_id"
    ],
    [
      "items",
      "payment",
      "restaurant_id"
    ]
  ],
  "alarm.list": [
    []
  ],
  "alarm.delete": [
    [
      "alarm_id"
    ]
  ],
  "alarm.update": [
    [
      "alarm_id",
      "time"
    ]
  ],
  "medication.add": [
    [
      "name",
      "times"
    ],
    [
      "days",
      "name",
      "times"
    ]
  ],
  "weather.get_current": [
    [],
    [
      "location"
    ]
  ],
  "calculator.convert_unit": [
    [
      "from_unit",
      "to_unit",
      "value"
    ]
  ],
  "shopping_list.add": [
    [
      "name",
      "quantity"
    ],
    [
      "name"
    ]
  ],
  "news.search": [
    [
      "query"
    ]
  ],
  "maps.search_place": [
    [
      "query"
    ],
    [
      "near",
      "query"
    ]
  ],
  "maps.get_route": [
    [
      "destination",
      "origin"
    ],
    [
      "destination",
      "mode",
      "origin"
    ]
  ],
  "bike_rental.reserve": [
    [
      "bike_id"
    ]
  ],
  "learning.enroll": [
    [
      "course_id"
    ]
  ],
  "library.reserve": [
    [
      "book_id"
    ]
  ],
  "subscriptions.list": [
    []
  ],
  "subscriptions.cancel": [
    [
      "subscription_id"
    ]
  ],
  "email.create_draft": [
    [
      "body",
      "subject",
      "to"
    ]
  ],
  "home_security.list_alerts": [
    [],
    [
      "since"
    ]
  ],
  "door_lock.list_locks": [
    []
  ],
  "door_lock.set_lock": [
    [
      "lock_id",
      "locked"
    ]
  ],
  "appliances.list": [
    [],
    [
      "type"
    ]
  ],
  "appliances.start_cycle": [
    [
      "appliance_id",
      "cycle"
    ],
    [
      "appliance_id",
      "cycle",
      "delay_minutes"
    ]
  ],
  "household_chores.list_chores": [
    []
  ],
  "household_chores.assign": [
    [
      "assignee",
      "chore_id"
    ]
  ],
  "package_tracking.schedule_pickup": [
    [
      "address",
      "date"
    ]
  ],
  "flights.book": [
    [
      "flight_id",
      "passengers"
    ],
    [
      "flight_id",
      "passengers",
      "seat_class"
    ]
  ],
  "chat_ott.read_chat": [
    [
      "chat_id"
    ]
  ],
  "tasks.add": [
    [
      "due",
      "title"
    ],
    [
      "title"
    ],
    [
      "due",
      "priority",
      "title"
    ]
  ],
  "notes.create": [
    [
      "content",
      "title"
    ],
    [
      "content"
    ]
  ],
  "calendar.create_event": [
    [
      "end",
      "start",
      "title"
    ],
    [
      "start",
      "title"
    ],
    [
      "location",
      "start",
      "title"
    ]
  ],
  "video_meeting.create": [
    [
      "start",
      "title"
    ],
    [
      "attendees",
      "start",
      "title"
    ]
  ],
  "contacts.update": [
    [
      "contact_id",
      "phone"
    ],
    [
      "contact_id",
      "email"
    ]
  ],
  "home_energy.get_usage": [
    [
      "period"
    ]
  ],
  "home_automation.create_rule": [
    [
      "action",
      "device_id",
      "time",
      "title"
    ],
    [
      "action",
      "device_id",
      "time"
    ]
  ],
  "fitness.log_workout": [
    [
      "minutes",
      "type"
    ],
    [
      "minutes",
      "time",
      "type"
    ]
  ],
  "robot_vacuum.start": [
    [
      "rooms"
    ],
    [
      "mode",
      "rooms"
    ],
    [
      "mode"
    ]
  ],
  "library.search": [
    [
      "query"
    ],
    []
  ],
  "music.play": [
    [
      "track_id"
    ],
    [
      "playlist_id"
    ]
  ],
  "podcast_radio.search": [
    [
      "query"
    ]
  ],
  "podcast_radio.queue": [
    [
      "episode_id"
    ]
  ],
  "robot_vacuum.get_status": [
    []
  ],
  "email.list": [
    []
  ],
  "learning.list_courses": [
    []
  ],
  "music.list_tracks": [
    []
  ],
  "flights.search": [
    [
      "date",
      "destination",
      "origin"
    ]
  ],
  "bike_rental.find_bikes": [
    [
      "near"
    ],
    []
  ],
  "home_security.acknowledge_alert": [
    [
      "alert_id"
    ]
  ],
  "calculator.calculate": [
    [
      "expression"
    ]
  ],
  "home_security.get_camera": [
    [
      "location"
    ]
  ],
  "food_delivery.order_history": [
    []
  ],
  "notes.list": [
    []
  ],
  "notes.append": [
    [
      "content",
      "note_id"
    ]
  ],
  "smart_home.fan_set": [
    [
      "device_id",
      "power"
    ]
  ],
  "chat_ott.call": [
    [
      "chat_id",
      "video"
    ],
    [
      "chat_id"
    ]
  ],
  "fitness.get_activity": [
    [
      "date"
    ],
    []
  ],
  "fitness.set_goal": [
    [
      "steps"
    ]
  ],
  "loyalty.get_points": [
    [
      "program"
    ],
    []
  ],
  "loyalty.redeem": [
    [
      "program",
      "reward_id"
    ]
  ],
  "fuel_price.get_price": [
    [
      "type"
    ]
  ],
  "music.add_to_playlist": [
    [
      "playlist_id",
      "track_id"
    ]
  ],
  "medication.list": [
    []
  ],
  "tv.record": [
    [
      "program_id"
    ]
  ],
  "ride_hailing.get_quotes": [
    [
      "dropoff",
      "pickup"
    ]
  ],
  "loyalty.list_vouchers": [
    [],
    [
      "merchant"
    ]
  ],
  "video_meeting.list_meetings": [
    []
  ],
  "exchange_rate.get_rate": [
    [
      "from_currency",
      "to_currency"
    ]
  ],
  "home_automation.list_rules": [
    []
  ],
  "home_automation.delete_rule": [
    [
      "rule_id"
    ]
  ],
  "medication.log_taken": [
    [
      "schedule_id",
      "time"
    ],
    [
      "schedule_id"
    ]
  ],
  "sms.read_recent": [
    [
      "contact_id"
    ],
    []
  ],
  "learning.schedule_lesson": [
    [
      "course_id",
      "time"
    ]
  ],
  "bill_payment.schedule_autopay": [
    [
      "enabled",
      "type"
    ]
  ],
  "tasks.list": [
    []
  ],
  "tv.control": [
    [
      "action"
    ],
    [
      "action",
      "volume"
    ]
  ],
  "learning.log_progress": [
    [
      "course_id",
      "minutes"
    ]
  ],
  "robot_vacuum.stop": [
    []
  ],
  "reminders.list": [
    [],
    [
      "date"
    ]
  ],
  "reminders.delete": [
    [
      "reminder_id"
    ]
  ],
  "home_energy.set_alert": [
    [
      "threshold_kwh"
    ]
  ],
  "package_tracking.track": [
    []
  ],
  "podcast_radio.play": [
    [
      "episode_id"
    ],
    [
      "station"
    ]
  ],
  "phone.recent_calls": [
    [],
    [
      "missed_only"
    ]
  ],
  "smart_home.scene_run": [
    [
      "scene_id"
    ]
  ],
  "shopping_list.get": [
    []
  ],
  "shopping_list.check": [
    [
      "item_id"
    ]
  ],
  "podcast_radio.subscribe": [
    [
      "show_id"
    ]
  ],
  "medication.remove": [
    [
      "schedule_id"
    ]
  ],
  "door_lock.grant_guest_code": [
    [
      "lock_id",
      "valid_until"
    ],
    [
      "label",
      "lock_id",
      "valid_until"
    ]
  ],
  "ewallet.pay_qr": [
    [
      "amount",
      "merchant_id"
    ]
  ],
  "smart_home.switch_set": [
    [
      "device_id",
      "power"
    ]
  ],
  "lottery.get_results": [
    [
      "province"
    ],
    [
      "date",
      "province"
    ]
  ],
  "news.save": [
    [
      "article_id"
    ]
  ],
  "flights.my_bookings": [
    []
  ],
  "flights.check_in": [
    [
      "booking_id"
    ]
  ],
  "shopping_list.remove": [
    [
      "item_id"
    ]
  ],
  "ride_hailing.get_ride": [
    []
  ],
  "ride_hailing.cancel": [
    [
      "ride_id"
    ]
  ],
  "lottery.check_ticket": [
    [
      "date",
      "number",
      "province"
    ]
  ],
  "appliances.stop": [
    [
      "appliance_id"
    ]
  ],
  "email.archive": [
    [
      "action",
      "email_id"
    ]
  ],
  "video_meeting.join": [
    [
      "meeting_id"
    ]
  ],
  "maps.save_place": [
    [
      "label",
      "place_id"
    ]
  ],
  "tasks.complete": [
    [
      "task_id"
    ]
  ],
  "reminders.complete": [
    [
      "reminder_id"
    ]
  ],
  "home_security.arm": [
    [
      "mode"
    ]
  ],
  "music.control": [
    [
      "action",
      "volume"
    ],
    [
      "action"
    ]
  ],
  "package_tracking.reschedule_delivery": [
    [
      "time",
      "tracking_id"
    ]
  ],
  "exchange_rate.convert": [
    [
      "amount",
      "from_currency",
      "to_currency"
    ]
  ],
  "flights.cancel": [
    [
      "booking_id"
    ]
  ],
  "news.headlines": [
    [
      "topic"
    ]
  ],
  "email.read": [
    [
      "email_id"
    ]
  ],
  "chat_ott.send_location": [
    [
      "chat_id"
    ]
  ],
  "email.reply": [
    [
      "body",
      "email_id"
    ]
  ],
  "banking.list_transactions": [
    [],
    [
      "account_id"
    ]
  ],
  "food_delivery.cancel_order": [
    [
      "order_id"
    ]
  ],
  "banking.open_savings": [
    [
      "amount",
      "from_account_id",
      "term_months"
    ]
  ],
  "email.send": [
    [
      "body",
      "cc",
      "subject",
      "to"
    ],
    [
      "body",
      "subject",
      "to"
    ]
  ]
}

# Transition probabilities P(T_next | T_prev)
TRANSITION_PROBS = {
  "^START^": {
    "bill_payment.list_bills": 0.076,
    "smart_home.get_status": 0.061,
    "contacts.search": 0.053,
    "calendar.list_events": 0.05,
    "sys.ask_missing_info": 0.042
  },
  "weather.get_forecast": {
    "calendar.list_events": 0.622,
    "contacts.search": 0.044,
    "alarm.update": 0.022,
    "calendar.cancel_event": 0.022,
    "bike_rental.reserve": 0.022
  },
  "calendar.list_events": {
    "calendar.update_event": 0.314,
    "ride_hailing.book": 0.294,
    "calendar.cancel_event": 0.167,
    "calendar.create_event": 0.039,
    "alarm.create": 0.029
  },
  "calendar.update_event": {
    "alarm.create": 0.615,
    "contacts.search": 0.269,
    "video_meeting.create": 0.077,
    "chat_ott.list_chats": 0.038
  },
  "contacts.search": {
    "sys.ask_missing_info": 0.32,
    "sms.send": 0.286,
    "ewallet.transfer": 0.117,
    "alarm.create": 0.074,
    "ewallet.topup_mobile": 0.039
  },
  "bill_payment.list_bills": {
    "ewallet.get_balance": 0.333,
    "banking.get_balance": 0.288,
    "bill_payment.pay": 0.279,
    "contacts.search": 0.063,
    "bill_payment.schedule_autopay": 0.027
  },
  "ewallet.topup_mobile": {
    "contacts.search": 1.0
  },
  "smart_home.get_status": {
    "smart_home.light_set": 0.743,
    "home_automation.create_rule": 0.152,
    "smart_home.climate_set": 0.029,
    "smart_home.fan_set": 0.019,
    "smart_home.switch_set": 0.019
  },
  "ewallet.get_balance": {
    "bill_payment.pay": 0.881,
    "reminders.list": 0.024,
    "subscriptions.cancel": 0.024,
    "contacts.search": 0.024,
    "home_security.arm": 0.024
  },
  "bill_payment.pay": {
    "bill_payment.pay": 0.456,
    "contacts.search": 0.368,
    "reminders.create": 0.105,
    "notes.create": 0.035,
    "home_energy.set_alert": 0.018
  },
  "banking.get_balance": {
    "banking.list_beneficiaries": 0.917,
    "banking.open_savings": 0.033,
    "bill_payment.pay": 0.017,
    "banking.transfer": 0.017,
    "banking.list_transactions": 0.017
  },
  "banking.list_beneficiaries": {
    "banking.transfer": 0.982,
    "sys.ask_missing_info": 0.018
  },
  "banking.transfer": {
    "banking.transfer": 0.852,
    "contacts.search": 0.037,
    "package_tracking.track": 0.037,
    "chat_ott.call": 0.037,
    "chat_ott.list_chats": 0.037
  },
  "smart_home.light_set": {
    "smart_home.light_set": 0.37,
    "smart_home.climate_set": 0.359,
    "contacts.search": 0.196,
    "notes.create": 0.011,
    "smart_home.fan_set": 0.011
  },
  "tv.guide": {
    "tv.record": 0.5,
    "tv.tune": 0.333,
    "tasks.add": 0.167
  },
  "calendar.cancel_event": {
    "contacts.search": 0.929,
    "email.list": 0.071
  },
  "chat_ott.list_chats": {
    "chat_ott.send_message": 0.538,
    "food_delivery.list_restaurants": 0.194,
    "chat_ott.read_chat": 0.14,
    "sys.ask_missing_info": 0.065,
    "chat_ott.send_location": 0.032
  },
  "alarm.create": {
    "contacts.search": 0.618,
    "sys.ask_missing_info": 0.309,
    "alarm.create": 0.018,
    "subscriptions.list": 0.018,
    "lottery.get_results": 0.018
  },
  "ride_hailing.book": {
    "reminders.create": 0.944,
    "lottery.check_ticket": 0.056
  },
  "gold_price.get_price": {
    "banking.get_balance": 0.583,
    "reminders.create": 0.25,
    "sys.ask_missing_info": 0.042,
    "banking.list_beneficiaries": 0.042,
    "maps.save_place": 0.042
  },
  "food_delivery.list_restaurants": {
    "food_delivery.get_menu": 0.92,
    "sys.ask_missing_info": 0.04,
    "food_delivery.order_history": 0.04
  },
  "food_delivery.get_menu": {
    "food_delivery.place_order": 1.0
  },
  "reminders.create": {
    "contacts.search": 0.444,
    "reminders.create": 0.167,
    "fitness.log_workout": 0.111,
    "phone.call": 0.056,
    "notes.create": 0.056
  },
  "phone.call": {
    "notes.create": 0.286,
    "alarm.create": 0.143,
    "contacts.search": 0.143,
    "email.create_draft": 0.143,
    "package_tracking.schedule_pickup": 0.143
  },
  "alarm.list": {
    "alarm.delete": 0.25,
    "alarm.update": 0.25,
    "appliances.start_cycle": 0.125,
    "door_lock.grant_guest_code": 0.125,
    "medication.list": 0.125
  },
  "smart_home.climate_set": {
    "reminders.create": 0.2,
    "alarm.update": 0.2,
    "smart_home.fan_set": 0.2,
    "smart_home.switch_set": 0.2,
    "sys.ask_missing_info": 0.2
  },
  "medication.add": {
    "reminders.create": 1.0
  },
  "weather.get_current": {
    "reminders.create": 0.167,
    "smart_home.get_status": 0.167,
    "contacts.search": 0.167,
    "notes.list": 0.167,
    "tasks.add": 0.083
  },
  "calculator.convert_unit": {
    "shopping_list.add": 0.857,
    "notes.create": 0.143
  },
  "news.search": {
    "news.save": 0.364,
    "reminders.create": 0.136,
    "tasks.add": 0.136,
    "sys.ask_missing_info": 0.091,
    "calendar.create_event": 0.045
  },
  "maps.search_place": {
    "maps.get_route": 0.4,
    "maps.save_place": 0.4,
    "ride_hailing.book": 0.067,
    "maps.search_place": 0.067,
    "appliances.stop": 0.067
  },
  "subscriptions.list": {
    "subscriptions.cancel": 0.875,
    "sys.ask_missing_info": 0.125
  },
  "subscriptions.cancel": {
    "email.create_draft": 0.286,
    "contacts.search": 0.286,
    "tasks.add": 0.143,
    "reminders.create": 0.143,
    "calendar.create_event": 0.071
  },
  "home_security.list_alerts": {
    "door_lock.list_locks": 0.435,
    "home_security.acknowledge_alert": 0.174,
    "notes.create": 0.13,
    "chat_ott.list_chats": 0.087,
    "phone.call": 0.043
  },
  "door_lock.list_locks": {
    "door_lock.set_lock": 0.778,
    "door_lock.grant_guest_code": 0.111,
    "chat_ott.list_chats": 0.037,
    "alarm.list": 0.037,
    "sys.ask_missing_info": 0.037
  },
  "appliances.list": {
    "appliances.start_cycle": 0.667,
    "appliances.stop": 0.222,
    "sys.ask_missing_info": 0.111
  },
  "appliances.start_cycle": {
    "household_chores.list_chores": 0.5,
    "reminders.create": 0.3,
    "notes.create": 0.1,
    "appliances.start_cycle": 0.1
  },
  "household_chores.list_chores": {
    "household_chores.assign": 0.81,
    "sys.ask_missing_info": 0.095,
    "robot_vacuum.stop": 0.048,
    "reminders.create": 0.048
  },
  "maps.get_route": {
    "reminders.create": 0.294,
    "maps.get_route": 0.118,
    "calendar.create_event": 0.059,
    "calculator.calculate": 0.059,
    "ride_hailing.book": 0.059
  },
  "shopping_list.add": {
    "tasks.add": 0.75,
    "notes.append": 0.25
  },
  "calendar.create_event": {
    "reminders.create": 0.3,
    "chat_ott.list_chats": 0.2,
    "tasks.add": 0.1,
    "package_tracking.schedule_pickup": 0.1,
    "calendar.create_event": 0.1
  },
  "contacts.update": {
    "chat_ott.list_chats": 0.6,
    "phone.call": 0.2,
    "email.create_draft": 0.2
  },
  "chat_ott.read_chat": {
    "music.list_tracks": 0.167,
    "calendar.create_event": 0.083,
    "phone.call": 0.083,
    "contacts.search": 0.083,
    "sys.ask_missing_info": 0.083
  },
  "home_energy.get_usage": {
    "smart_home.get_status": 0.455,
    "home_energy.set_alert": 0.273,
    "tasks.add": 0.182,
    "appliances.list": 0.091
  },
  "home_automation.create_rule": {
    "smart_home.light_set": 0.5,
    "robot_vacuum.get_status": 0.167,
    "notes.create": 0.167,
    "home_security.list_alerts": 0.167
  },
  "library.search": {
    "library.reserve": 0.733,
    "tasks.add": 0.067,
    "chat_ott.send_message": 0.067,
    "library.search": 0.067,
    "sys.ask_missing_info": 0.067
  },
  "podcast_radio.search": {
    "podcast_radio.queue": 0.438,
    "sys.ask_missing_info": 0.188,
    "podcast_radio.play": 0.188,
    "podcast_radio.subscribe": 0.188
  },
  "chat_ott.send_message": {
    "chat_ott.send_message": 0.3,
    "email.create_draft": 0.2,
    "calendar.create_event": 0.1,
    "flights.my_bookings": 0.1,
    "alarm.create": 0.1
  },
  "door_lock.set_lock": {
    "chat_ott.list_chats": 0.368,
    "phone.call": 0.105,
    "door_lock.set_lock": 0.105,
    "notes.create": 0.053,
    "home_security.list_alerts": 0.053
  },
  "ewallet.transfer": {
    "notes.create": 0.4,
    "chat_ott.list_chats": 0.1,
    "chat_ott.send_message": 0.1,
    "calendar.create_event": 0.1,
    "ewallet.transfer": 0.1
  },
  "robot_vacuum.get_status": {
    "robot_vacuum.start": 1.0
  },
  "email.list": {
    "email.archive": 0.375,
    "email.read": 0.25,
    "email.create_draft": 0.125,
    "email.reply": 0.125,
    "loyalty.list_vouchers": 0.125
  },
  "fitness.log_workout": {
    "reminders.create": 0.4,
    "music.list_tracks": 0.2,
    "fitness.get_activity": 0.2,
    "alarm.create": 0.2
  },
  "learning.list_courses": {
    "learning.log_progress": 0.476,
    "learning.enroll": 0.238,
    "sys.ask_missing_info": 0.143,
    "learning.schedule_lesson": 0.143
  },
  "music.list_tracks": {
    "music.play": 0.667,
    "music.add_to_playlist": 0.25,
    "sys.ask_missing_info": 0.083
  },
  "robot_vacuum.start": {
    "tasks.add": 0.333,
    "smart_home.get_status": 0.167,
    "chat_ott.list_chats": 0.167,
    "home_energy.get_usage": 0.167,
    "reminders.create": 0.167
  },
  "flights.search": {
    "flights.book": 1.0
  },
  "email.create_draft": {
    "tasks.add": 0.5,
    "chat_ott.send_message": 0.25,
    "video_meeting.create": 0.25
  },
  "library.reserve": {
    "reminders.create": 0.222,
    "tasks.add": 0.222,
    "sms.send": 0.111,
    "sys.ask_missing_info": 0.111,
    "contacts.search": 0.111
  },
  "learning.enroll": {
    "calendar.create_event": 0.429,
    "notes.create": 0.143,
    "tasks.add": 0.143,
    "learning.log_progress": 0.143,
    "contacts.search": 0.143
  },
  "bike_rental.find_bikes": {
    "bike_rental.reserve": 0.867,
    "sys.ask_missing_info": 0.133
  },
  "bike_rental.reserve": {
    "maps.get_route": 0.294,
    "bike_rental.reserve": 0.294,
    "reminders.create": 0.118,
    "calendar.update_event": 0.059,
    "sms.send": 0.059
  },
  "video_meeting.create": {
    "calendar.create_event": 0.333,
    "calendar.list_events": 0.333,
    "alarm.create": 0.333
  },
  "flights.book": {
    "ride_hailing.book": 0.25,
    "package_tracking.schedule_pickup": 0.125,
    "calendar.create_event": 0.125,
    "tasks.add": 0.125,
    "chat_ott.send_message": 0.125
  },
  "podcast_radio.queue": {
    "tasks.add": 0.4,
    "calendar.create_event": 0.2,
    "podcast_radio.queue": 0.2,
    "podcast_radio.subscribe": 0.2
  },
  "music.play": {
    "smart_home.get_status": 0.333,
    "fitness.log_workout": 0.167,
    "podcast_radio.search": 0.167,
    "music.control": 0.167,
    "alarm.delete": 0.167
  },
  "food_delivery.place_order": {
    "bill_payment.list_bills": 0.5,
    "food_delivery.place_order": 0.5
  },
  "household_chores.assign": {
    "household_chores.assign": 0.6,
    "music.list_tracks": 0.1,
    "smart_home.get_status": 0.1,
    "gold_price.get_price": 0.1,
    "calendar.create_event": 0.1
  },
  "tasks.add": {
    "calendar.create_event": 0.5,
    "tasks.add": 0.25,
    "notes.create": 0.25
  },
  "home_security.acknowledge_alert": {
    "home_security.acknowledge_alert": 0.333,
    "contacts.search": 0.222,
    "calendar.list_events": 0.111,
    "calendar.create_event": 0.111,
    "alarm.create": 0.111
  },
  "package_tracking.schedule_pickup": {
    "calendar.create_event": 0.4,
    "reminders.create": 0.2,
    "chat_ott.list_chats": 0.2,
    "email.reply": 0.2
  },
  "calculator.calculate": {
    "tasks.add": 0.2,
    "chat_ott.list_chats": 0.2,
    "reminders.create": 0.2,
    "bill_payment.schedule_autopay": 0.2,
    "contacts.search": 0.2
  },
  "home_security.get_camera": {
    "robot_vacuum.get_status": 0.25,
    "smart_home.get_status": 0.25,
    "home_security.arm": 0.25,
    "home_security.list_alerts": 0.25
  },
  "food_delivery.order_history": {
    "food_delivery.cancel_order": 0.167,
    "sys.ask_missing_info": 0.167,
    "food_delivery.place_order": 0.083,
    "reminders.list": 0.083,
    "robot_vacuum.stop": 0.083
  },
  "notes.list": {
    "sys.ask_missing_info": 0.455,
    "notes.append": 0.273,
    "appliances.list": 0.091,
    "shopping_list.add": 0.091,
    "chat_ott.send_message": 0.091
  },
  "smart_home.fan_set": {
    "reminders.create": 0.2,
    "tasks.list": 0.2,
    "smart_home.fan_set": 0.2,
    "gold_price.get_price": 0.2,
    "sys.ask_missing_info": 0.2
  },
  "fitness.get_activity": {
    "fitness.get_activity": 0.385,
    "fitness.set_goal": 0.308,
    "shopping_list.remove": 0.077,
    "tasks.list": 0.077,
    "calendar.create_event": 0.077
  },
  "loyalty.get_points": {
    "loyalty.redeem": 0.812,
    "sys.ask_missing_info": 0.188
  },
  "loyalty.redeem": {
    "loyalty.redeem": 0.5,
    "alarm.create": 0.125,
    "loyalty.list_vouchers": 0.125,
    "email.reply": 0.125,
    "weather.get_forecast": 0.125
  },
  "medication.list": {
    "medication.log_taken": 0.333,
    "medication.remove": 0.267,
    "reminders.create": 0.133,
    "alarm.create": 0.067,
    "chat_ott.list_chats": 0.067
  },
  "ride_hailing.get_quotes": {
    "ride_hailing.book": 1.0
  },
  "loyalty.list_vouchers": {
    "chat_ott.send_message": 0.167,
    "reminders.create": 0.167,
    "loyalty.get_points": 0.167,
    "calendar.create_event": 0.167,
    "chat_ott.list_chats": 0.167
  },
  "video_meeting.list_meetings": {
    "video_meeting.join": 0.385,
    "sys.ask_missing_info": 0.154,
    "alarm.create": 0.154,
    "reminders.create": 0.077,
    "smart_home.fan_set": 0.077
  },
  "fuel_price.get_price": {
    "contacts.search": 0.3,
    "bike_rental.reserve": 0.1,
    "alarm.create": 0.1,
    "podcast_radio.search": 0.1,
    "loyalty.get_points": 0.1
  },
  "home_automation.list_rules": {
    "home_automation.delete_rule": 0.818,
    "appliances.list": 0.091,
    "home_automation.create_rule": 0.091
  },
  "sms.read_recent": {
    "library.search": 0.2,
    "ride_hailing.get_ride": 0.2,
    "exchange_rate.convert": 0.2,
    "bike_rental.find_bikes": 0.2,
    "fitness.log_workout": 0.2
  },
  "learning.schedule_lesson": {
    "smart_home.light_set": 0.5,
    "learning.schedule_lesson": 0.5
  },
  "tasks.list": {
    "tasks.complete": 0.462,
    "sys.ask_missing_info": 0.077,
    "music.list_tracks": 0.077,
    "contacts.search": 0.077,
    "weather.get_forecast": 0.077
  },
  "reminders.list": {
    "reminders.complete": 0.222,
    "sys.ask_missing_info": 0.222,
    "reminders.delete": 0.111,
    "smart_home.get_status": 0.111,
    "door_lock.list_locks": 0.111
  },
  "package_tracking.track": {
    "sys.ask_missing_info": 0.429,
    "package_tracking.reschedule_delivery": 0.429,
    "reminders.create": 0.143
  },
  "home_automation.delete_rule": {
    "home_automation.delete_rule": 0.25,
    "reminders.create": 0.167,
    "chat_ott.send_message": 0.167,
    "phone.recent_calls": 0.083,
    "robot_vacuum.start": 0.083
  },
  "phone.recent_calls": {
    "phone.call": 0.6,
    "sms.send": 0.2,
    "contacts.search": 0.2
  },
  "exchange_rate.get_rate": {
    "chat_ott.list_chats": 0.333,
    "contacts.search": 0.222,
    "ride_hailing.book": 0.111,
    "notes.append": 0.111,
    "alarm.create": 0.111
  },
  "shopping_list.get": {
    "shopping_list.remove": 0.429,
    "shopping_list.check": 0.286,
    "shopping_list.add": 0.143,
    "sms.send": 0.143
  },
  "shopping_list.check": {
    "shopping_list.check": 0.5,
    "calendar.list_events": 0.167,
    "phone.call": 0.167,
    "chat_ott.list_chats": 0.167
  },
  "robot_vacuum.stop": {
    "smart_home.get_status": 0.286,
    "sys.ask_missing_info": 0.143,
    "banking.get_balance": 0.143,
    "bike_rental.reserve": 0.143,
    "robot_vacuum.start": 0.143
  },
  "learning.log_progress": {
    "learning.log_progress": 0.143,
    "bill_payment.list_bills": 0.143,
    "reminders.delete": 0.143,
    "reminders.complete": 0.143,
    "learning.schedule_lesson": 0.143
  },
  "smart_home.switch_set": {
    "contacts.search": 1.0
  },
  "lottery.get_results": {
    "sms.send": 0.2,
    "lottery.get_results": 0.2,
    "ewallet.transfer": 0.1,
    "reminders.create": 0.1,
    "contacts.search": 0.1
  },
  "news.save": {
    "news.save": 0.333,
    "alarm.create": 0.222,
    "chat_ott.list_chats": 0.222,
    "contacts.search": 0.111,
    "news.search": 0.111
  },
  "flights.my_bookings": {
    "flights.check_in": 0.571,
    "sys.ask_missing_info": 0.286,
    "flights.cancel": 0.143
  },
  "flights.check_in": {
    "alarm.create": 0.5,
    "banking.list_beneficiaries": 0.5
  },
  "ride_hailing.get_ride": {
    "ride_hailing.cancel": 0.5,
    "food_delivery.order_history": 0.25,
    "reminders.create": 0.25
  },
  "lottery.check_ticket": {
    "lottery.check_ticket": 0.4,
    "calendar.list_events": 0.1,
    "maps.search_place": 0.1,
    "shopping_list.add": 0.1,
    "tasks.add": 0.1
  },
  "ride_hailing.cancel": {
    "fitness.get_activity": 0.2,
    "sms.send": 0.2,
    "phone.call": 0.2,
    "ride_hailing.book": 0.2,
    "bill_payment.list_bills": 0.2
  },
  "email.archive": {
    "email.archive": 0.667,
    "email.reply": 0.333
  },
  "music.add_to_playlist": {
    "music.add_to_playlist": 0.5,
    "music.control": 0.5
  },
  "video_meeting.join": {
    "contacts.search": 0.333,
    "sms.send": 0.333,
    "chat_ott.list_chats": 0.167,
    "medication.remove": 0.167
  },
  "sms.send": {
    "sms.send": 1.0
  },
  "maps.save_place": {
    "contacts.search": 0.5,
    "home_automation.list_rules": 0.25,
    "maps.save_place": 0.25
  },
  "tasks.complete": {
    "tasks.complete": 0.6,
    "sms.send": 0.2,
    "notes.list": 0.2
  },
  "podcast_radio.subscribe": {
    "notes.list": 0.5,
    "lottery.get_results": 0.5
  },
  "notes.append": {
    "maps.search_place": 1.0
  },
  "appliances.stop": {
    "reminders.create": 0.25,
    "appliances.start_cycle": 0.25,
    "tasks.add": 0.25,
    "contacts.search": 0.25
  },
  "home_energy.set_alert": {
    "ride_hailing.book": 0.25,
    "smart_home.get_status": 0.25,
    "fitness.get_activity": 0.25,
    "loyalty.get_points": 0.25
  },
  "exchange_rate.convert": {
    "banking.get_balance": 0.333,
    "exchange_rate.convert": 0.222,
    "reminders.create": 0.111,
    "calendar.create_event": 0.111,
    "chat_ott.list_chats": 0.111
  },
  "alarm.update": {
    "sys.ask_missing_info": 0.5,
    "chat_ott.list_chats": 0.5
  },
  "tv.control": {
    "smart_home.get_status": 0.333,
    "lottery.get_results": 0.333,
    "sys.ask_missing_info": 0.333
  },
  "news.headlines": {
    "contacts.search": 0.5,
    "calendar.create_event": 0.5
  },
  "email.read": {
    "smart_home.climate_set": 0.25,
    "library.search": 0.25,
    "home_security.get_camera": 0.25,
    "lottery.check_ticket": 0.25
  },
  "chat_ott.send_location": {
    "ewallet.get_balance": 0.5,
    "chat_ott.send_location": 0.5
  },
  "banking.list_transactions": {
    "home_automation.create_rule": 0.333,
    "sys.ask_missing_info": 0.333,
    "calculator.calculate": 0.333
  },
  "reminders.complete": {
    "reminders.complete": 1.0
  },
  "flights.cancel": {
    "flights.cancel": 0.5,
    "chat_ott.send_message": 0.5
  },
  "food_delivery.cancel_order": {
    "alarm.list": 0.5,
    "food_delivery.cancel_order": 0.5
  },
  "shopping_list.remove": {
    "shopping_list.add": 0.5,
    "smart_home.climate_set": 0.5
  },
  "bill_payment.schedule_autopay": {
    "bill_payment.pay": 0.5,
    "tasks.add": 0.5
  },
  "medication.remove": {
    "medication.add": 0.5,
    "contacts.search": 0.25,
    "medication.remove": 0.25
  },
  "email.send": {
    "alarm.create": 1.0
  },
  "banking.open_savings": {
    "contacts.search": 0.5,
    "chat_ott.list_chats": 0.5
  },
  "door_lock.grant_guest_code": {
    "door_lock.grant_guest_code": 0.5,
    "chat_ott.list_chats": 0.5
  },
  "fitness.set_goal": {
    "contacts.search": 0.333,
    "phone.call": 0.333,
    "reminders.create": 0.333
  },
  "ewallet.pay_qr": {
    "ride_hailing.book": 1.0
  },
  "medication.log_taken": {
    "medication.log_taken": 0.667,
    "reminders.create": 0.333
  },
  "home_security.arm": {
    "reminders.create": 1.0
  }
}

# Predecessors P(T_prev | T_curr)
TOOL_PREDECESSORS = {
  "calendar.list_events": [
    "weather.get_forecast",
    "video_meeting.create",
    "news.search"
  ],
  "calendar.update_event": [
    "calendar.list_events",
    "bike_rental.reserve"
  ],
  "contacts.search": [
    "alarm.create",
    "bill_payment.pay",
    "smart_home.light_set"
  ],
  "sms.send": [
    "contacts.search",
    "lottery.get_results",
    "video_meeting.join"
  ],
  "ewallet.topup_mobile": [
    "contacts.search"
  ],
  "ewallet.transfer": [
    "contacts.search",
    "lottery.get_results",
    "ewallet.transfer"
  ],
  "smart_home.light_set": [
    "smart_home.get_status",
    "smart_home.light_set",
    "home_automation.create_rule"
  ],
  "ride_hailing.book": [
    "calendar.list_events",
    "flights.book",
    "ride_hailing.get_quotes"
  ],
  "ewallet.get_balance": [
    "bill_payment.list_bills",
    "chat_ott.send_location"
  ],
  "bill_payment.pay": [
    "ewallet.get_balance",
    "bill_payment.list_bills",
    "bill_payment.pay"
  ],
  "reminders.create": [
    "ride_hailing.book",
    "bill_payment.pay",
    "gold_price.get_price"
  ],
  "banking.list_beneficiaries": [
    "banking.get_balance",
    "flights.check_in",
    "gold_price.get_price"
  ],
  "banking.transfer": [
    "banking.list_beneficiaries",
    "banking.transfer",
    "banking.get_balance"
  ],
  "banking.get_balance": [
    "bill_payment.list_bills",
    "gold_price.get_price",
    "exchange_rate.convert"
  ],
  "tv.tune": [
    "tv.guide"
  ],
  "phone.call": [
    "contacts.search",
    "phone.recent_calls",
    "door_lock.set_lock"
  ],
  "calendar.cancel_event": [
    "calendar.list_events",
    "weather.get_forecast",
    "lottery.check_ticket"
  ],
  "sys.ask_missing_info": [
    "contacts.search",
    "alarm.create",
    "chat_ott.list_chats"
  ],
  "alarm.create": [
    "contacts.search",
    "calendar.update_event",
    "calendar.list_events"
  ],
  "chat_ott.send_message": [
    "chat_ott.list_chats",
    "chat_ott.send_message",
    "home_automation.delete_rule"
  ],
  "smart_home.climate_set": [
    "smart_home.light_set",
    "smart_home.get_status",
    "email.read"
  ],
  "food_delivery.list_restaurants": [
    "chat_ott.list_chats",
    "medication.list",
    "food_delivery.order_history"
  ],
  "food_delivery.get_menu": [
    "food_delivery.list_restaurants"
  ],
  "food_delivery.place_order": [
    "food_delivery.get_menu",
    "food_delivery.order_history",
    "food_delivery.place_order"
  ],
  "alarm.delete": [
    "alarm.list",
    "music.play"
  ],
  "alarm.update": [
    "alarm.list",
    "weather.get_forecast",
    "smart_home.climate_set"
  ],
  "shopping_list.add": [
    "calculator.convert_unit",
    "lottery.check_ticket",
    "shopping_list.get"
  ],
  "maps.get_route": [
    "maps.search_place",
    "bike_rental.reserve",
    "maps.get_route"
  ],
  "subscriptions.cancel": [
    "subscriptions.list",
    "ewallet.get_balance",
    "subscriptions.cancel"
  ],
  "email.create_draft": [
    "contacts.search",
    "subscriptions.cancel",
    "chat_ott.send_message"
  ],
  "door_lock.list_locks": [
    "home_security.list_alerts",
    "reminders.list"
  ],
  "door_lock.set_lock": [
    "door_lock.list_locks",
    "door_lock.set_lock"
  ],
  "appliances.start_cycle": [
    "appliances.list",
    "alarm.list",
    "appliances.start_cycle"
  ],
  "household_chores.list_chores": [
    "appliances.start_cycle",
    "home_security.list_alerts",
    "door_lock.set_lock"
  ],
  "household_chores.assign": [
    "household_chores.list_chores",
    "household_chores.assign",
    "tasks.list"
  ],
  "chat_ott.read_chat": [
    "chat_ott.list_chats"
  ],
  "tasks.add": [
    "shopping_list.add",
    "news.search",
    "email.create_draft"
  ],
  "notes.create": [
    "ewallet.transfer",
    "home_security.list_alerts",
    "phone.call"
  ],
  "video_meeting.create": [
    "calendar.update_event",
    "contacts.search",
    "calendar.list_events"
  ],
  "contacts.update": [
    "contacts.search"
  ],
  "chat_ott.list_chats": [
    "door_lock.set_lock",
    "contacts.update",
    "contacts.search"
  ],
  "calendar.create_event": [
    "calendar.list_events",
    "learning.enroll",
    "tasks.add"
  ],
  "smart_home.get_status": [
    "home_energy.get_usage",
    "music.play",
    "weather.get_current"
  ],
  "home_automation.create_rule": [
    "smart_home.get_status",
    "banking.list_transactions",
    "home_automation.list_rules"
  ],
  "fitness.log_workout": [
    "reminders.create",
    "music.play",
    "weather.get_current"
  ],
  "library.reserve": [
    "library.search",
    "weather.get_forecast",
    "library.reserve"
  ],
  "podcast_radio.queue": [
    "podcast_radio.search",
    "podcast_radio.queue",
    "fuel_price.get_price"
  ],
  "robot_vacuum.get_status": [
    "home_automation.create_rule",
    "calendar.list_events",
    "home_security.get_camera"
  ],
  "robot_vacuum.start": [
    "robot_vacuum.get_status",
    "home_automation.delete_rule",
    "robot_vacuum.stop"
  ],
  "email.list": [
    "calendar.cancel_event"
  ],
  "learning.enroll": [
    "learning.list_courses"
  ],
  "music.play": [
    "music.list_tracks"
  ],
  "flights.book": [
    "flights.search",
    "exchange_rate.get_rate"
  ],
  "bike_rental.reserve": [
    "bike_rental.find_bikes",
    "bike_rental.reserve",
    "weather.get_forecast"
  ],
  "music.list_tracks": [
    "chat_ott.read_chat",
    "fitness.log_workout",
    "household_chores.assign"
  ],
  "bill_payment.list_bills": [
    "food_delivery.place_order",
    "learning.log_progress",
    "ride_hailing.cancel"
  ],
  "appliances.list": [
    "home_energy.get_usage",
    "weather.get_current",
    "home_automation.list_rules"
  ],
  "flights.search": [
    "calendar.list_events"
  ],
  "package_tracking.schedule_pickup": [
    "flights.book",
    "calendar.create_event",
    "phone.call"
  ],
  "home_security.list_alerts": [
    "door_lock.set_lock",
    "home_automation.create_rule",
    "home_security.get_camera"
  ],
  "home_security.acknowledge_alert": [
    "home_security.list_alerts",
    "home_security.acknowledge_alert",
    "door_lock.set_lock"
  ],
  "podcast_radio.search": [
    "news.search",
    "music.play",
    "calendar.list_events"
  ],
  "calculator.calculate": [
    "maps.get_route",
    "banking.list_transactions"
  ],
  "home_energy.get_usage": [
    "robot_vacuum.start",
    "smart_home.get_status"
  ],
  "notes.append": [
    "notes.list",
    "exchange_rate.get_rate",
    "shopping_list.add"
  ],
  "weather.get_forecast": [
    "calendar.list_events",
    "tasks.list",
    "loyalty.redeem"
  ],
  "smart_home.fan_set": [
    "smart_home.get_status",
    "smart_home.light_set",
    "video_meeting.list_meetings"
  ],
  "chat_ott.call": [
    "chat_ott.list_chats",
    "banking.transfer"
  ],
  "fitness.get_activity": [
    "fitness.get_activity",
    "ride_hailing.cancel",
    "home_energy.set_alert"
  ],
  "fitness.set_goal": [
    "fitness.get_activity",
    "maps.get_route",
    "weather.get_forecast"
  ],
  "loyalty.redeem": [
    "loyalty.get_points",
    "loyalty.redeem",
    "news.search"
  ],
  "tv.guide": [
    "maps.get_route"
  ],
  "tv.record": [
    "tv.guide",
    "calendar.list_events"
  ],
  "ride_hailing.get_quotes": [
    "calendar.list_events",
    "weather.get_forecast"
  ],
  "subscriptions.list": [
    "alarm.create"
  ],
  "home_automation.delete_rule": [
    "home_automation.list_rules",
    "home_automation.delete_rule",
    "chat_ott.read_chat"
  ],
  "medication.log_taken": [
    "medication.list",
    "medication.log_taken"
  ],
  "sms.read_recent": [
    "contacts.search"
  ],
  "library.search": [
    "sms.read_recent",
    "flights.book",
    "library.search"
  ],
  "bill_payment.schedule_autopay": [
    "bill_payment.list_bills",
    "calculator.calculate",
    "bill_payment.pay"
  ],
  "music.add_to_playlist": [
    "music.list_tracks",
    "music.add_to_playlist"
  ],
  "tasks.list": [
    "smart_home.fan_set",
    "food_delivery.order_history",
    "fitness.get_activity"
  ],
  "reminders.list": [
    "food_delivery.order_history",
    "ewallet.get_balance"
  ],
  "reminders.delete": [
    "reminders.list",
    "learning.log_progress"
  ],
  "podcast_radio.play": [
    "podcast_radio.search"
  ],
  "phone.recent_calls": [
    "home_automation.delete_rule"
  ],
  "shopping_list.check": [
    "shopping_list.check",
    "shopping_list.get"
  ],
  "medication.remove": [
    "medication.list",
    "chat_ott.read_chat",
    "video_meeting.join"
  ],
  "learning.log_progress": [
    "learning.list_courses",
    "learning.log_progress",
    "food_delivery.order_history"
  ],
  "learning.schedule_lesson": [
    "learning.list_courses",
    "calendar.list_events",
    "learning.schedule_lesson"
  ],
  "door_lock.grant_guest_code": [
    "door_lock.list_locks",
    "alarm.list",
    "door_lock.set_lock"
  ],
  "smart_home.switch_set": [
    "smart_home.get_status",
    "smart_home.climate_set",
    "home_automation.delete_rule"
  ],
  "robot_vacuum.stop": [
    "food_delivery.order_history",
    "household_chores.list_chores"
  ],
  "news.save": [
    "news.search",
    "news.save"
  ],
  "video_meeting.list_meetings": [
    "contacts.search",
    "chat_ott.read_chat"
  ],
  "flights.check_in": [
    "flights.my_bookings",
    "calendar.list_events"
  ],
  "loyalty.get_points": [
    "calendar.list_events",
    "loyalty.list_vouchers",
    "calendar.create_event"
  ],
  "tv.control": [
    "food_delivery.order_history"
  ],
  "shopping_list.remove": [
    "shopping_list.get",
    "fitness.get_activity"
  ],
  "lottery.get_results": [
    "lottery.get_results",
    "podcast_radio.subscribe",
    "tv.control"
  ],
  "ride_hailing.get_ride": [
    "sms.read_recent",
    "smart_home.light_set"
  ],
  "ride_hailing.cancel": [
    "ride_hailing.get_ride",
    "weather.get_forecast",
    "fuel_price.get_price"
  ],
  "appliances.stop": [
    "appliances.list",
    "maps.search_place"
  ],
  "email.archive": [
    "email.list",
    "email.archive",
    "maps.get_route"
  ],
  "home_automation.list_rules": [
    "maps.save_place",
    "smart_home.light_set"
  ],
  "package_tracking.track": [
    "banking.transfer"
  ],
  "food_delivery.order_history": [
    "ride_hailing.get_ride",
    "food_delivery.list_restaurants"
  ],
  "tasks.complete": [
    "tasks.list",
    "tasks.complete",
    "ewallet.transfer"
  ],
  "home_energy.set_alert": [
    "home_energy.get_usage",
    "bill_payment.pay"
  ],
  "podcast_radio.subscribe": [
    "podcast_radio.search",
    "podcast_radio.queue"
  ],
  "notes.list": [
    "weather.get_current",
    "podcast_radio.subscribe",
    "tasks.complete"
  ],
  "maps.search_place": [
    "notes.append",
    "lottery.check_ticket",
    "loyalty.list_vouchers"
  ],
  "maps.save_place": [
    "maps.search_place",
    "maps.get_route",
    "maps.save_place"
  ],
  "video_meeting.join": [
    "video_meeting.list_meetings"
  ],
  "reminders.complete": [
    "reminders.list",
    "learning.log_progress",
    "reminders.complete"
  ],
  "home_security.get_camera": [
    "home_security.list_alerts",
    "email.read"
  ],
  "home_security.arm": [
    "home_security.get_camera",
    "ewallet.get_balance"
  ],
  "music.control": [
    "music.add_to_playlist",
    "music.play",
    "door_lock.set_lock"
  ],
  "alarm.list": [
    "door_lock.list_locks",
    "food_delivery.cancel_order",
    "weather.get_forecast"
  ],
  "flights.my_bookings": [
    "chat_ott.send_message"
  ],
  "package_tracking.reschedule_delivery": [
    "package_tracking.track"
  ],
  "medication.list": [
    "alarm.list",
    "weather.get_forecast",
    "calendar.list_events"
  ],
  "exchange_rate.convert": [
    "exchange_rate.convert",
    "sms.read_recent"
  ],
  "gold_price.get_price": [
    "smart_home.fan_set",
    "household_chores.assign"
  ],
  "flights.cancel": [
    "flights.my_bookings",
    "flights.cancel"
  ],
  "chat_ott.send_location": [
    "chat_ott.list_chats",
    "weather.get_current",
    "chat_ott.send_location"
  ],
  "news.search": [
    "news.save",
    "news.search"
  ],
  "bike_rental.find_bikes": [
    "reminders.create",
    "sms.read_recent"
  ],
  "learning.list_courses": [
    "maps.get_route",
    "calendar.list_events"
  ],
  "loyalty.list_vouchers": [
    "loyalty.redeem",
    "email.list"
  ],
  "email.reply": [
    "loyalty.redeem",
    "email.list",
    "package_tracking.schedule_pickup"
  ],
  "lottery.check_ticket": [
    "lottery.check_ticket",
    "ride_hailing.book",
    "email.read"
  ],
  "food_delivery.cancel_order": [
    "food_delivery.order_history",
    "food_delivery.cancel_order"
  ],
  "exchange_rate.get_rate": [
    "flights.book",
    "tasks.list"
  ],
  "fuel_price.get_price": [
    "smart_home.light_set"
  ],
  "banking.open_savings": [
    "banking.get_balance"
  ],
  "email.read": [
    "email.list"
  ],
  "email.send": [
    "contacts.search"
  ],
  "weather.get_current": [
    "chat_ott.read_chat"
  ],
  "medication.add": [
    "medication.remove"
  ],
  "calculator.convert_unit": [
    "maps.get_route"
  ],
  "banking.list_transactions": [
    "banking.get_balance"
  ],
  "news.headlines": [
    "alarm.create"
  ]
}

def canonical_args_for(tool_name: str) -> list[list[str]]:
    """Return top argument combinations canonically expected for tool."""
    return CANONICAL_ARG_SETS.get(tool_name, [])

def likely_predecessors(tool_name: str) -> list[str]:
    """Return read tools that canonically precede this tool."""
    return TOOL_PREDECESSORS.get(tool_name, [])

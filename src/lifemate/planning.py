"""Brief action plans alongside native calls, never an execution authority."""
from dataclasses import asdict, dataclass, field
import re

from .messages import parse_calls

MAX_PLAN_CHARACTERS = 600


@dataclass
class Decision:
    tool_calls: list = field(default_factory=list)
    plan_update: str | None = None
    protocol_error: str | None = None
    raw_text: str = ''

    def payload(self):
        return asdict(self)


def parse_planning_response(text):
    """Plan-only and malformed responses cannot silently become successful stops."""
    try:
        plans = re.findall(r'<plan>\s*(.*?)\s*</plan>', text, re.S)
        if text.count('<plan>') != text.count('</plan>') or len(plans) > 1:
            raise ValueError('Use at most one complete <plan> block')
        plan = plans[0].strip() if plans else None
        if plans and (not plan or len(plan) > MAX_PLAN_CHARACTERS):
            raise ValueError('Plan must contain 1 to 600 characters')
        body = re.sub(r'<plan>.*?</plan>', '', text, flags=re.S)
        calls = parse_calls(body)
        remainder = re.sub(r'<tool_call>.*?</tool_call>', '', body, flags=re.S).strip()
        if remainder not in ('', '[]') or (calls and remainder == '[]'):
            raise ValueError('Emit native calls or [], with optional <plan>; no other prose')
        if not calls and remainder != '[]':
            raise ValueError('A plan alone is not completion; emit native calls or []')
        return Decision(calls, plan, raw_text=text)
    except (ValueError, TypeError) as error:
        return Decision(protocol_error=str(error), raw_text=text)

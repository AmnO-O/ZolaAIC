"""Experimental working memory protocol; never an evidence source."""
import re
from .planning import Decision, parse_planning_response

MAX_MEMORY_CHARACTERS = 1600

def parse_memory_response(text):
    blocks = re.findall(r'<memory>(.*?)</memory>', text, re.S)
    if (len(blocks) != 1 or text.count('<memory>') != 1
            or text.count('</memory>') != 1):
        return Decision(protocol_error='Emit exactly one complete <memory> block', raw_text=text)
    memory = blocks[0].strip()
    if not memory or len(memory) > MAX_MEMORY_CHARACTERS:
        return Decision(protocol_error='Memory must contain 1 to 1600 characters', raw_text=text)
    body = re.sub(r'<memory>.*?</memory>', '', text, flags=re.S)
    # Plans are not part of this protocol. Reuse the strict native-call parser.
    if '<plan>' in body or '</plan>' in body:
        return Decision(protocol_error='Use memory, not a separate plan', raw_text=text)
    decision = parse_planning_response(body)
    if decision.protocol_error == 'A plan alone is not completion; emit native calls or []':
        decision.protocol_error = 'Memory alone is not a termination signal; emit native calls or literal []'
    decision.raw_text = text
    if not decision.protocol_error:
        decision.memory_update = memory
    return decision

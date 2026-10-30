"""Direct baseline: native actions, immutable evidence, no generated memory."""
import copy
import json
import re
import time
from dataclasses import dataclass, field

from .messages import parse_calls
from .grounding import Evidence
from .execution import ExecutionLedger


@dataclass
class DirectDecision:
    tool_calls: list = field(default_factory=list)
    protocol_error: str | None = None
    raw_text: str = ''

    def payload(self):
        return {'tool_calls': self.tool_calls, 'protocol_error': self.protocol_error,
                'raw_text': self.raw_text}


def parse_direct_response(text):
    """Only complete native calls or explicit [] may control execution."""
    try:
        calls = parse_calls(text)
        remainder = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S).strip()
        if not calls and remainder == '[]':
            return DirectDecision(raw_text=text)
        if not calls or remainder:
            raise ValueError('Emit only complete native tool calls, or literal [] to stop')
        # Duplicate keys and non-finite JSON numbers cannot be silently accepted.
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate JSON key: ' + key)
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError('Non-finite JSON value: ' + value)
        for block in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S):
            obj = json.loads(block, object_pairs_hook=pairs, parse_constant=invalid_constant)
            if set(obj) != {'name', 'arguments'}:
                raise ValueError('Each call must contain exactly name and arguments')
            if not isinstance(obj['arguments'], dict):
                raise ValueError('arguments must be a JSON object')
        json.dumps(calls, allow_nan=False)
        return DirectDecision(calls, raw_text=text)
    except (ValueError, TypeError) as error:
        return DirectDecision(protocol_error=str(error), raw_text=text)


def run_direct_agent(env, decide, messages, tools, deadline, max_turns=8, emit=None):
    """One environment. Preserve actual arguments/results and all attempted calls."""
    messages = copy.deepcopy(messages)
    tools = list(tools)
    specs = {t['name']: t for t in tools}
    evidence = Evidence(messages)
    ledger = ExecutionLedger()
    trace = []
    reason = 'turn_limit'
    decisions = protocol_errors = validation_rejections = 0
    attempted_reads = attempted_writes = 0
    last_rejected = False

    def event(stage, **data):
        if emit:
            emit(stage, **data)

    for _ in range(max_turns):
        if time.monotonic() >= deadline:
            reason = 'deadline'
            break
        event('decision.input', input={'messages': messages, 'tools': tools},
              remaining_seconds=max(0, deadline-time.monotonic()))
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            reason = 'deadline'
            break
        decisions += 1
        try:
            response = decide(messages, tools, remaining)
        except TimeoutError:
            reason = 'deadline'
            break
        if isinstance(response, list):  # Backend plugin compatibility.
            response = DirectDecision(response)
        if not isinstance(response, DirectDecision):
            raise ValueError('Direct backend must return DirectDecision or a list of calls')
        event('decision.output', output=response.payload())
        if time.monotonic() >= deadline:
            reason = 'deadline'
            break
        calls = response.tool_calls
        if not isinstance(calls, list) or any(
            not isinstance(c, dict) or not isinstance(c.get('name'), str)
            or not isinstance(c.get('arguments'), dict) for c in calls):
            raise ValueError('Malformed backend calls')
        if response.protocol_error:
            protocol_errors += 1
            # Raw rejected output stays in logs, never in executable assistant history.
            feedback = {'controller_feedback': 'output_rejected', 'executed': False,
                        'error': response.protocol_error,
                        'instruction': 'No calls in that response ran. Emit native calls or literal []. Do not claim completion of rejected actions.'}
            messages.append({'role': 'user', 'content': json.dumps(feedback, ensure_ascii=False)})
            event('controller.feedback', output=feedback)
            if protocol_errors >= 2:
                reason = 'protocol_error'
                break
            continue
        if not calls:
            reason = ('execution_failed' if ledger.failed_targets else
                      'validation_failed' if last_rejected else 'finished')
            break
        # Keep each proposal paired with a result, including explicit nonexecution feedback.
        assistant = {'role': 'assistant', 'tool_calls': []}
        messages.append(assistant)
        batch = []
        last_rejected = False
        for original in calls:
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            call = copy.deepcopy(original)
            name, args = call['name'], call['arguments']
            kind = specs.get(name, {}).get('kind')
            error = 'Tool is not permitted for this sample.' if name not in specs else None
            if not error and ledger.duplicate_write(call, kind):
                error = 'Identical write already succeeded; do not repeat it.'
            if not error:
                error = evidence.check(specs[name], args, specs)
            check = {'ok': False, 'error': error} if error else env.validate(name, args)
            event('tool.validation', input=call, output=check)
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            executed = False
            status = 'rejected'
            if not check['ok']:
                last_rejected = True
                validation_rejections += 1
                result = {'validation_error': check.get('error'), 'executed': False}
            else:
                event('tool.input', input=call)
                if time.monotonic() >= deadline:
                    reason = 'deadline'
                    break
                executed = True
                try:
                    result = env.call(name, args)
                except Exception as error:
                    result = {'execution_error': str(error), 'execution_attempted': True}
                event('tool.output', input=call, output=result)
                attempted_reads += kind == 'read'
                attempted_writes += kind == 'write'
                status = ledger.observe(call, result, kind)
                if status == 'succeeded':
                    evidence.observe(name, result)
            assistant['tool_calls'].append(call)
            wrapped = {'source_id': 'observation_' + str(len(trace)+1),
                       'source_type': 'environment_result' if executed else 'controller_feedback',
                       'executed': executed, 'execution_status': status, 'result': result}
            messages.append({'role': 'tool', 'name': name, 'content': wrapped})
            entry = {'call': call, 'result': result, 'executed': executed, 'status': status}
            trace.append(entry)
            batch.append(entry)
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            if executed and name == 'sys.ask_missing_info':
                reason = 'clarification'
                break
            if executed and status == 'uncertain' and kind not in ('read', 'meta'):
                reason = 'uncertain_outcome'
                break
        if reason in ('deadline', 'clarification', 'uncertain_outcome'):
            break
        if ledger.stagnant(batch):
            reason = 'no_progress'
            break
    return {'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}, {
        'termination': reason, 'task_success_verified': False, 'controller': 'direct_v2',
        'trace': trace, 'execution_ledger': ledger.records, 'decisions': decisions,
        'protocol_errors': protocol_errors, 'validation_rejections': validation_rejections,
        'attempted_reads': attempted_reads, 'attempted_writes': attempted_writes}

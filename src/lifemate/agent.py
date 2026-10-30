"""Direct baseline: native actions, immutable evidence, no generated memory."""
import copy
import json
import re
import time
from dataclasses import dataclass, field

from .grounding import Evidence
from .execution import ExecutionLedger


@dataclass
class Decision:
    tool_calls: list = field(default_factory=list)
    protocol_error: str | None = None
    raw_text: str = ''

    def payload(self):
        return {'tool_calls': self.tool_calls, 'protocol_error': self.protocol_error,
                'raw_text': self.raw_text}


def _parameter_value(text, schema):
    """Decode Qwen3.5 XML wire values using their declared JSON types."""
    kinds = schema.get('type', 'string')
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if text in ('null', 'None') and 'null' in kinds:
        return None
    if 'string' in kinds:
        return text
    if 'boolean' in kinds and text.lower() in ('true', 'false'):
        return text.lower() == 'true'
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key in structured parameter')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite parameter')
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    valid = (('integer' in kinds and type(value) is int)
             or ('number' in kinds and type(value) in (int, float))
             or ('boolean' in kinds and type(value) is bool)
             or ('object' in kinds and isinstance(value, dict))
             or ('array' in kinds and isinstance(value, list)))
    if not valid:
        raise ValueError('Parameter does not match its declared wire type')
    json.dumps(value, allow_nan=False)
    return value


def parse_decision(text, tools=(), protocol="xml"):
    """Parse the selected native wire format; never accept truncated calls."""
    if protocol not in ("xml", "json"):
        raise ValueError("Unknown native protocol")
    try:
        body = text.strip()
        if body == '[]':
            return Decision(raw_text=text)
        blocks = re.findall(r'<tool_call>(.*?)</tool_call>', body, re.S)
        if (not blocks or body.count('<tool_call>') != len(blocks)
                or body.count('</tool_call>') != len(blocks)
                or re.sub(r'<tool_call>.*?</tool_call>', '', body, flags=re.S).strip()):
            raise ValueError('Emit only complete native tool-call blocks or literal []; no other text')
        if protocol == 'json':
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('Duplicate JSON key')
                    result[key] = value
                return result
            def invalid(value):
                raise ValueError('Non-finite JSON value')
            calls = []
            for block in blocks:
                call = json.loads(block, object_pairs_hook=pairs, parse_constant=invalid)
                if (not isinstance(call, dict) or set(call) != {'name', 'arguments'}
                        or not isinstance(call['name'], str) or not call['name']
                        or not isinstance(call['arguments'], dict)):
                    raise ValueError('Expected native name/arguments JSON object')
                json.dumps(call, allow_nan=False)
                calls.append(call)
            return Decision(calls, raw_text=text)
        specs = {t['name']: t for t in tools}
        calls = []
        for block in blocks:
            function = re.fullmatch(r'\s*<function=([^<>\s]+)>(.*?)</function>\s*', block, re.S)
            if not function:
                raise ValueError('Expected a native <function=...> block')
            name, parameters = function.groups()
            matches = list(re.finditer(r'<parameter=([^<>\s]+)>(.*?)</parameter>', parameters, re.S))
            if re.sub(r'<parameter=[^<>\s]+>.*?</parameter>', '', parameters, flags=re.S).strip():
                raise ValueError('Malformed parameter block')
            props = specs.get(name, {}).get('parameters', {}).get('properties', {})
            arguments = {}
            for match in matches:
                key, value = match.groups()
                if key in arguments:
                    raise ValueError('Duplicate parameter: ' + key)
                # Remove template framing newlines, preserve meaningful string whitespace.
                if value.startswith('\n'):
                    value = value[1:]
                if value.endswith('\n'):
                    value = value[:-1]
                arguments[key] = _parameter_value(value, props.get(key, {}))
            calls.append({'name': name, 'arguments': arguments})
        return Decision(calls, raw_text=text)
    except (ValueError, TypeError) as error:
        return Decision(protocol_error=str(error), raw_text=text)


def run_agent(env, decide, messages, tools, deadline, max_turns=8, emit=None):
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
            response = Decision(response)
        if not isinstance(response, Decision):
            raise ValueError('Direct backend must return Decision or a list of calls')
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
        'termination': reason, 'task_success_verified': False, 'controller': 'single',
        'trace': trace, 'execution_ledger': ledger.records, 'decisions': decisions,
        'protocol_errors': protocol_errors, 'validation_rejections': validation_rejections,
        'attempted_reads': attempted_reads, 'attempted_writes': attempted_writes}

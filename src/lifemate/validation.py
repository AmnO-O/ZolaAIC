"""Evidence guards and clarification validation; never executes tools."""
import re
from .catalog import lookup_for
from .values import phone_key, explicit_phone

def validate_evidence(evidence, tool, arguments, available=()):
    available_names = set(available.keys()) if isinstance(available, dict) else {
        t['name'] if isinstance(t, dict) else t for t in available}
    available_specs = available if isinstance(available, dict) else {
        t['name']: t for t in available if isinstance(t, dict)}

    if tool.get('name') == 'sys.ask_missing_info':
        target_name = arguments.get('tool_name')
        if not target_name or not isinstance(target_name, str):
            return "Missing or invalid 'tool_name' in sys.ask_missing_info arguments."
        if target_name == 'sys.ask_missing_info':
            return "Cannot request clarification for sys.ask_missing_info."
        if target_name not in available_names:
            return f"Tool {target_name!r} is not permitted for this sample."
        target_spec = available_specs.get(target_name, {})
        params = arguments.get('params')
        if (not isinstance(params, list) or not params
                or any(not isinstance(p, str) for p in params)):
            return "Missing or empty 'params' list in sys.ask_missing_info arguments."
        target_props = target_spec.get('parameters', {}).get('properties', {})
        if 'properties' in target_spec.get('parameters', {}):
            invalid_params = [p for p in params if p not in target_props]
            if invalid_params:
                return f"Parameter(s) {invalid_params!r} do not exist in schema for tool {target_name!r}."
        for p in params:
            ref = target_props.get(p, {}).get('x-ref') if target_props else None
            if ref:
                lookup_tool = lookup_for(ref, target_props.get(p, {}), available_specs)
                if (lookup_tool and not available_specs[lookup_tool].get('parameters', {}).get('required')
                        and lookup_tool not in evidence.successful_tools):
                    return (f"Cannot ask missing info for parameter {p!r}; lookup tool {lookup_tool!r} "
                            f"is available and has not been executed yet. Call '{lookup_tool}' first.")
        return None

    for key, schema in tool.get('parameters', {}).get('properties', {}).items():
        ref = schema.get('x-ref')
        if not ref or key not in arguments:
            continue
        values = arguments[key] if isinstance(arguments[key], list) else [arguments[key]]
        for value in values:
            if type(value) not in (str, int):
                continue  # Type/shape errors belong to the supplied schema validator.
            known = str(value) in evidence.ids.get(ref.split('.')[0], set())
            explicit = re.search(r'(?<!\w)' + re.escape(str(value)) + r'(?!\w)', evidence.user_text) is not None
            if not known and not explicit:
                lookup_tool = lookup_for(ref, schema, available_specs)
                if lookup_tool and lookup_tool in available_names:
                    return (f"Ungrounded ID {value!r} for parameter '{key}'. "
                            f"Call '{lookup_tool}' first to look up valid IDs, or ask for missing info if no entity exists.")
                return f"Ungrounded ID {value!r} for parameter '{key}'; obtain it from explicit user input or relevant context."
    for key in ('phone', 'phone_number'):
        if key in arguments:
            value = arguments[key]
            if phone_key(value) not in evidence.phones and not explicit_phone(value, evidence.user_text):
                return f"Ungrounded phone number {value!r} for parameter '{key}'; use explicit user data or an observed contact result."
    return None

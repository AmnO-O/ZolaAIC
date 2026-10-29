"""Compact redundant catalog metadata without changing user facts."""

def clean_context_for_prompt(context):
    """Strip redundant installed_apps JSON to save ~1,500 prompt characters."""
    if not isinstance(context, dict):
        return context
    cleaned = dict(context)
    if 'installed_apps' in cleaned:
        # Keep app names only, remove deep schemas and tool name arrays
        cleaned['installed_apps'] = list(cleaned['installed_apps'].keys())
    return cleaned


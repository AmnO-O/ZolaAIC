"""Render installed app descriptions once, without duplicating tool schemas."""

def clean_context_for_prompt(context, catalog):
    """Map installed app IDs to catalog descriptions; preserve other context."""
    if not isinstance(context, dict):
        return context
    cleaned = dict(context)
    cleaned.pop('app_descriptions', None)
    if 'installed_apps' in cleaned:
        descriptions = {app.get('id'): app.get('description') or app.get('name_vi', '')
                        for app in catalog.get('apps', [])}
        # An unknown app remains visible, without inventing its capabilities.
        cleaned['installed_apps'] = {name: descriptions.get(name, '')
                                     for name in context['installed_apps']}
    return cleaned


"""Single-sample KV ownership with exact token-prefix checks."""


class PromptCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cache = None
        self.token_ids = []

    def take(self, token_ids):
        # Transfer ownership: any failed forward/generation leaves no stale cache.
        cache, previous = self.cache, self.token_ids
        self.reset()
        if cache is None:
            return None, 0
        limit = min(len(previous), max(0, len(token_ids) - 1))
        shared = 0
        while shared < limit and previous[shared] == token_ids[shared]:
            shared += 1
        if not shared or not hasattr(cache, 'crop') or cache.get_seq_length() < shared:
            return None, 0
        if cache.get_seq_length() > shared:
            cache.crop(shared - cache.get_seq_length())
        if cache.get_seq_length() != shared:
            return None, 0
        return cache, shared

    def store(self, cache, token_ids):
        self.reset()
        if cache is None or not token_ids or not hasattr(cache, 'crop'):
            return
        # Save only known prompt tokens; generated text is reserialized by the
        # controller/template and may have different whitespace or token IDs.
        if cache.get_seq_length() < len(token_ids):
            return
        if cache.get_seq_length() > len(token_ids):
            # Negative removal counts work with pinned 5.16 and newer APIs.
            cache.crop(len(token_ids) - cache.get_seq_length())
        if cache.get_seq_length() == len(token_ids):
            self.cache, self.token_ids = cache, list(token_ids)

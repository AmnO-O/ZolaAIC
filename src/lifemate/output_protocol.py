"""Early rejection of text outside the native tool-call envelope."""


def valid_output_prefix(text):
    """Allow partial tags, multiple calls, or literal []; not a body grammar."""
    body = text.lstrip()
    if body.startswith('['):
        return '[]'.startswith(body) or body.rstrip() == '[]'
    while body:
        if '<tool_call>'.startswith(body):
            return True
        if not body.startswith('<tool_call>'):
            return False
        close = body.find('</tool_call>', len('<tool_call>'))
        if close < 0:
            return True  # Complete bodies are checked by the strict parser.
        body = body[close + len('</tool_call>'):].lstrip()
    return True


class NativeOutputGuard:
    """Single-sample Transformers stopping criterion; never declares success."""
    def __init__(self, tokenizer, prompt_tokens, stop_ids):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.stop_ids = set(stop_ids)
        self.error = None

    def __call__(self, input_ids, scores, **kwargs):
        output = input_ids[0, self.prompt_tokens:]
        if len(output) and int(output[-1]) in self.stop_ids:
            output = output[:-1]
        # Preserve unexpected special tokens so they cannot disappear into a stop.
        text = self.tokenizer.decode(output, skip_special_tokens=False)
        if not valid_output_prefix(text):
            self.error = 'Output outside native tool-call blocks or literal []; generation stopped early'
        return self.error is not None

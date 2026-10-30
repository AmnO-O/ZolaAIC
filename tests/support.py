import copy

def call(name, **arguments):
    return {'name': name, 'arguments': arguments}

class Env:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}
    def validate(self, name, arguments):
        return {'ok': True}
    def call(self, name, arguments):
        self.calls.append(call(name, **copy.deepcopy(arguments)))
        result = self.results.get(name, {'ok': True})
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)
    def get_tool_calls(self):
        return copy.deepcopy(self.calls)
    def get_state_hash(self):
        return 'synthetic'

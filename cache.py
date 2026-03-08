class SemanticCache:
    def __init__(self):
        self.cache = {}

    def get(self, query):
        return self.cache.get(query)

    def set(self, query, results):
        self.cache[query] = results
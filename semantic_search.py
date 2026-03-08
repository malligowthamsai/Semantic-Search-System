import pickle
from sklearn.metrics.pairwise import cosine_similarity
from app.cache import SemanticCache

class SemanticSearch:
    def __init__(self):
        # Load precomputed embeddings (example)
        try:
            with open("data/embeddings.pkl", "rb") as f:
                self.documents, self.embeddings = pickle.load(f)
        except FileNotFoundError:
            self.documents = []
            self.embeddings = []
        self.cache = SemanticCache()

    def search(self, query: str, top_k: int = 5):
        # Check cache first
        cached = self.cache.get(query)
        if cached:
            return cached

        # Simple dummy embedding for query (replace with real model)
        query_vector = [len(query)] * len(self.embeddings[0]) if self.embeddings else []

        if not self.embeddings:
            return []

        scores = cosine_similarity([query_vector], self.embeddings)[0]
        results = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)[:top_k]

        # Cache results
        self.cache.set(query, results)
        return results
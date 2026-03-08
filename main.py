from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Example documents
# (replace with your dataset)
# -----------------------------
documents = [
    "Space exploration and NASA missions",
    "Religion and atheism debates",
    "Computer graphics and GPU technology",
    "Baseball and sports news",
    "Political discussions and policies"
]

# -----------------------------
# Generate embeddings
# -----------------------------
doc_embeddings = model.encode(documents)

# -----------------------------
# FAISS index
# -----------------------------
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype("float32"))

# -----------------------------
# Example cluster labels
# (use your fuzzy clustering result)
# -----------------------------
cluster_labels = [0,1,2,3,4]

# -----------------------------
# Semantic Cache
# -----------------------------
semantic_cache = {}

cache_stats = {
    "hit_count": 0,
    "miss_count": 0
}

SIMILARITY_THRESHOLD = 0.85


# -----------------------------
# Request Model
# -----------------------------
class QueryRequest(BaseModel):
    query: str


# -----------------------------
# Helper: cosine similarity
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# POST /query
# -----------------------------
@app.post("/query")
def query_service(req: QueryRequest):

    query = req.query

    query_embedding = model.encode([query])[0]

    # -------- check cache --------
    for cached_query, cached_data in semantic_cache.items():

        score = cosine_similarity(query_embedding, cached_data["embedding"])

        if score >= SIMILARITY_THRESHOLD:

            cache_stats["hit_count"] += 1

            return {
                "query": query,
                "cache_hit": True,
                "matched_query": cached_query,
                "similarity_score": float(score),
                "result": cached_data["result"],
                "dominant_cluster": cached_data["cluster"]
            }

    # -------- cache miss --------
    cache_stats["miss_count"] += 1

    D, I = index.search(np.array([query_embedding]).astype("float32"), k=1)

    result_doc = documents[I[0][0]]

    cluster = cluster_labels[I[0][0]]

    # store in cache
    semantic_cache[query] = {
        "embedding": query_embedding,
        "result": result_doc,
        "cluster": cluster
    }

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result_doc,
        "dominant_cluster": cluster
    }


# -----------------------------
# GET /cache/stats
# -----------------------------
@app.get("/cache/stats")
def cache_stats_endpoint():

    total = len(semantic_cache)

    hits = cache_stats["hit_count"]
    misses = cache_stats["miss_count"]

    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0

    return {
        "total_entries": total,
        "hit_count": hits,
        "miss_count": misses,
        "hit_rate": hit_rate
    }


# -----------------------------
# DELETE /cache
# -----------------------------
@app.delete("/cache")
def clear_cache():

    semantic_cache.clear()

    cache_stats["hit_count"] = 0
    cache_stats["miss_count"] = 0

    return {"message": "Cache cleared"}


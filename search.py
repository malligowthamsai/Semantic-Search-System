
import faiss
import pickle
from sentence_transformers import SentenceTransformer


def search(query):

    index = faiss.read_index("vector_db/news_index.faiss")

    with open("embeddings/document_embeddings.pkl", "rb") as f:
        embeddings, docs = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_vector = model.encode([query])

    D, I = index.search(query_vector, 5)

    for idx in I[0]:
        print(docs[idx][:200])
        print("------")
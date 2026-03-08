
import faiss
import pickle
import numpy as np


def build_index():

    with open("embeddings/document_embeddings.pkl", "rb") as f:
        embeddings, docs = pickle.load(f)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(index, "vector_db/news_index.faiss")

    print("Index built with", index.ntotal, "vectors")
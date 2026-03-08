import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

dataset_path = "/content/dataset/20_newsgroups"

documents = []
labels = []
categories = sorted(os.listdir(dataset_path))

for label, category in enumerate(categories):

    category_path = os.path.join(dataset_path, category)

    if os.path.isdir(category_path):

        for file in os.listdir(category_path):

            file_path = os.path.join(category_path, file)

            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()

                    documents.append(text)
                    labels.append(label)

            except:
                pass

print("Total documents:", len(documents))
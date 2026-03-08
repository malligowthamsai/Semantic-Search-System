embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

for idx in I[0]:
    print("Category:", categories[labels[idx]])
    print(documents[idx][:300])
    print("------")
    
    
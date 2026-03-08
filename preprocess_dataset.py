cleaned_docs = [clean_text(doc) for doc in documents]

print("Example cleaned text:\n")
print(cleaned_docs[0][:500])
model = SentenceTransformer("all-MiniLM-L6-v2")
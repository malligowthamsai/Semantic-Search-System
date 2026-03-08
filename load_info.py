lengths = [len(doc) for doc in documents]

print("Average length:", sum(lengths)/len(lengths))
print("Longest document:", max(lengths))
print("Shortest document:", min(lengths))
embeddings = model.encode(
    cleaned_docs,
    batch_size=128,
    show_progress_bar=True
    
# Semantic-Search-System
Semantic Search API using NLP with SentenceTransformers, FAISS, and Fuzzy C-Means clustering. It converts documents to embeddings, performs fast semantic retrieval, and uses a semantic cache to speed up repeated queries. Built with FastAPI and includes query and cache management endpoints.

# Semantic Search System

A FastAPI-based semantic search system with caching and precomputed embeddings.

## Features
- Semantic search using embeddings
- Query caching for faster responses
- Easy to extend with custom models

## Installation
```bash
git clone <repo_url>
cd semantic-search-system
pip install -r requirements.txt

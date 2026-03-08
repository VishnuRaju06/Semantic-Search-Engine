# Semantic Search Engine with Semantic Cache

This project implements a semantic search system using embeddings, FAISS vector search, clustering, and a semantic cache.

## Features

• SentenceTransformer embeddings
• FAISS vector database
• Fuzzy clustering
• Semantic cache
• FastAPI backend
• Simple frontend UI

## Architecture

User Query
↓
Embedding Model
↓
Semantic Cache
↓
FAISS Search
↓
Results

## API Endpoints

POST /query

GET /cache/stats

DELETE /cache

## Run Backend

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

## Run Frontend

cd frontend
python -m http.server 8080
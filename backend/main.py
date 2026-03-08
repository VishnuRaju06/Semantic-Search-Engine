from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# -------------------------
# Load Data
# -------------------------

df = pd.read_csv("data/news_clustered.csv")
embeddings = np.load("data/embeddings.npy")
index = faiss.read_index("data/faiss_index.index")

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Semantic Cache
# -------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SemanticCache:

    def __init__(self, threshold=0.85):
        self.cache = []
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):

        best_score = 0
        best_item = None

        for item in self.cache:

            score = cosine_similarity(query_embedding, item["embedding"])

            if score > best_score:
                best_score = score
                best_item = item

        if best_score >= self.threshold:
            self.hit_count += 1
            return True, best_item, best_score

        self.miss_count += 1
        return False, None, best_score

    def add(self, query, embedding, result, cluster):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = len(self.cache)

        hit_rate = self.hit_count / (self.hit_count + self.miss_count + 1e-9)

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0


cache = SemanticCache()

# -------------------------
# Search System
# -------------------------

def search_system(query):

    query_embedding = model.encode([query])[0]

    hit, item, score = cache.lookup(query_embedding)

    if hit:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": item["query"],
            "similarity_score": float(score),
            "result": item["result"],
            "dominant_cluster": item["cluster"]
        }

    D, I = index.search(query_embedding.reshape(1,-1), k=5)

    results = df.iloc[I[0]]["clean_text"].tolist()

    cluster = int(df.iloc[I[0][0]]["dominant_cluster"])

    cache.add(query, query_embedding, results, cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0.0,
        "result": results,
        "dominant_cluster": cluster
    }

# -------------------------
# FastAPI
# -------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_api(request: QueryRequest):

    return search_system(request.query)

@app.get("/cache/stats")
def cache_stats():

    return cache.stats()

@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}
from __future__ import annotations

import json
from functools import lru_cache

import faiss
import numpy as np
from fastembed import TextEmbedding

from app.settings import get_settings


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()

        self.embedding_model = TextEmbedding(
            model_name=settings.embedding_model
        )

        with open(settings.embedding_documents_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.texts = [doc["text"] for doc in self.documents]
        self.ids = [str(doc["id"]) for doc in self.documents]

        self.index_path = settings.data_dir / "faiss.index"

        if self.index_path.exists():
            print("Loading FAISS index from disk...")
            self.index = faiss.read_index(str(self.index_path))
        else:
            print("Building FAISS index...")

            embeddings = list(self.embedding_model.embed(self.texts))
            embeddings = np.array(embeddings).astype("float32")

            faiss.normalize_L2(embeddings)

            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)

            faiss.write_index(self.index, str(self.index_path))

        print(f"Loaded {len(self.texts)} assessment embeddings.")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not query.strip():
            return []

        embedding = list(self.embedding_model.embed([query]))[0]
        embedding = np.array([embedding]).astype("float32")

        faiss.normalize_L2(embedding)

        scores, indices = self.index.search(embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                {
                    "id": self.ids[idx],
                    "score": float(score),
                }
            )

        return results


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()
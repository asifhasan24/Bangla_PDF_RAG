import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.texts = [item["text"] for item in self.metadata]
        self.embed_model = SentenceTransformer(model_name)

    def get_top_k(self, query: str, k: int = 3):
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_emb, k)
        return [self.texts[i] for i in indices[0]]

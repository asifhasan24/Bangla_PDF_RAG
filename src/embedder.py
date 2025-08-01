import json
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        # returns numpy array of shape (n_texts, embedding_dim)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def build_index(self, chunks, index_path, metadata_path):
        texts = [c["text"] for c in chunks]
        embeddings = self.embed(texts).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, index_path)

        # save chunk metadata (ids + text)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=str,
        default="/home/asif/Documents/rag_project/vector_store/chunks.json",
        help="Path to chunk metadata JSON"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="/home/asif/Documents/rag_project/vector_store/index.faiss",
        help="Where to write FAISS index"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/home/asif/Documents/rag_project/vector_store/metadata.json",
        help="Where to write FAISS metadata"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/LaBSE",
        help="Sentence-Transformer model"
    )
    args = parser.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = Embedder(args.model)
    embedder.build_index(chunks, args.index, args.metadata)

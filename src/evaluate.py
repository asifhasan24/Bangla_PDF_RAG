import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Imports
from retriever import Retriever
from generator import GeminiGenerator


def load_testcases(path: str) -> list[dict]:
    """
    Load test cases from a JSON file.
    Expected format:
    [
      {
        "query": "...",
        "gold_texts": ["exact chunk text1", "exact chunk text2", ...]
      },
      ...
    ]
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_relevance(
    testcases: list[dict],
    retriever: Retriever,
    k: int = 3
) -> float:
    """
    Computes average Hit@k: fraction of gold_texts appearing in top-k retrieved.
    """
    hits = []
    for tc in testcases:
        query = tc.get('query', '')
        gold = set(tc.get('gold_texts', []))
        if not query or not gold:
            continue
        retrieved = retriever.get_top_k(query=query, k=k)
        hit_count = len(set(retrieved) & gold)
        hits.append(hit_count / min(len(gold), k))
    return float(np.mean(hits)) if hits else 0.0


def evaluate_groundedness(
    testcases: list[dict],
    retriever: Retriever,
    generator: GeminiGenerator,
    model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
) -> float:
    """
    Computes average max cosine similarity between generated answer and retrieved docs.
    Higher means the answer is more grounded in the docs.
    """
    embed_model = SentenceTransformer(model_name)
    scores = []
    for tc in testcases:
        query = tc.get('query', '')
        if not query:
            continue
        docs = retriever.get_top_k(query=query, k=3)
        answer = generator.generate(question=query, context="", documents=docs)
        ans_emb = embed_model.encode([answer], convert_to_numpy=True)
        docs_emb = embed_model.encode(docs, convert_to_numpy=True)
        sims = cosine_similarity(ans_emb, docs_emb)[0]
        scores.append(float(np.max(sims)))
    return float(np.mean(scores)) if scores else 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system for relevance and groundedness.")
    parser.add_argument(
        '--testcases',
        type=str,
        default='evaluation/test_queries.json',
        help='Path to JSON with test queries and gold_texts'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of docs to retrieve for evaluation'
    )
    args = parser.parse_args()

    # Initialize components
    retriever = Retriever(
        index_path='vector_store/index.faiss',
        metadata_path='vector_store/metadata.json'
    )
    # Instantiate GeminiGenerator without api_key arg
    model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-flash')
    generator = GeminiGenerator(model_name=model_name)

    # Load test cases
    testcases = load_testcases(args.testcases)

    # Compute metrics
    relevance_score = evaluate_relevance(testcases, retriever, k=args.k)
    groundedness_score = evaluate_groundedness(testcases, retriever, generator)

    # Output results
    print(f"Relevance (Hit@{args.k}): {relevance_score:.4f}")
    print(f"Groundedness (Max Cosine Sim): {groundedness_score:.4f}")

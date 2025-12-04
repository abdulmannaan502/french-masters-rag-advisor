from pathlib import Path
import json
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INDEX_DIR = Path("index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def load_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"{FAISS_INDEX_PATH} not found. Run build_index.py first.")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    return index


def load_metadata() -> List[Dict]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} not found. Run build_index.py first.")

    metadata = []
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))
    return metadata


def embed_query(query: str, model) -> np.ndarray:
    emb = model.encode([query], convert_to_numpy=True)
    return emb.astype("float32")


def search(query: str, k: int = 5):
    print(f"\n=== Query: {query}")
    print("=" * 60)

    # Load model, index, metadata
    model = SentenceTransformer(MODEL_NAME)
    index = load_index()
    metadata = load_metadata()

    # Embed query
    query_emb = embed_query(query, model)  

    # Search
    distances, indices = index.search(query_emb, k)

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        chunk = metadata[idx]
        print(f"\n--- Result #{rank} ---")
        print(f"Source: {chunk['source']}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Distance: {dist:.4f}")
        print("Text:")
        print(chunk["text"][:500], "...")  


if __name__ == "__main__":
    test_queries = [
        "What are the steps to apply for a master's in France?",
        "English language requirements for Master in Artificial Intelligence",
        "How to apply to Université Paris-Saclay AI master's?",
        "Application procedure through Campus France",
    ]

    for q in test_queries:
        search(q, k=3)

from pathlib import Path
import json
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"


def load_chunks() -> List[Dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run preprocess.py first.")

    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_embeddings(chunks: List[Dict], model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> np.ndarray:
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks...")

    embeddings_list = []
    batch_size = 32

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings_list.append(batch_embeddings)

    embeddings = np.vstack(embeddings_list).astype("float32")
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    print(f"Creating FAISS index with dimension {dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index now contains {index.ntotal} vectors")
    return index


def save_index(index: faiss.IndexFlatL2, chunks: List[Dict]):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"Saving metadata to {METADATA_PATH}")
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("Index and metadata saved.")


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    embeddings = build_embeddings(chunks)
    index = build_faiss_index(embeddings)
    save_index(index, chunks)


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import List, Dict
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


def load_text_files() -> List[Path]:
    return list(PROCESSED_DIR.glob("*.txt"))


def chunk_text(text: str, source_name: str) -> List[Dict]:
    """
    Split long text into overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # characters per chunk
        chunk_overlap=200,    # overlap between chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    chunks = []

    for i, chunk in enumerate(raw_chunks):
        chunks.append({
            "id": f"{source_name}_{i}",
            "source": source_name,
            "chunk_index": i,
            "text": chunk.strip()
        })

    return chunks


def build_chunks():
    text_files = load_text_files()
    if not text_files:
        print(f"No .txt files found in {PROCESSED_DIR.resolve()}")
        return

    all_chunks: List[Dict] = []

    for txt_file in text_files:
        print(f"Chunking {txt_file.name} ...")
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        source_name = txt_file.stem
        file_chunks = chunk_text(text, source_name)
        print(f"  -> {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    # Save as JSONL (one JSON object per line)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    build_chunks()

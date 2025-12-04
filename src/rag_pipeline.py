from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ---------------- PATHS ----------------
BASE_DIR = Path(".")
INDEX_PATH = BASE_DIR / "index/faiss_index.bin"
META_PATH = BASE_DIR / "index/metadata.jsonl"
MODEL_PATH = BASE_DIR / "models/phi3.gguf"

# ---------------- MODELS ----------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


# ---------------- LOAD ----------------
def load_faiss():
    index = faiss.read_index(str(INDEX_PATH))

    documents = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    return index, documents


def load_llm():
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_threads=6
    )


# ---------------- RETRIEVAL ----------------
def retrieve(query, embedder, index, docs, k=4):

    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    scores, ids = index.search(q_emb, k)

    contexts = []
    sources = set()

    for idx in ids[0]:
        contexts.append(docs[idx]["text"])
        sources.add(docs[idx]["source"])

    return contexts, list(sources)


# ---------------- PROMPT ----------------
def build_prompt(q, ctx):

    return f"""
You are an academic advisor assistant for Masters applications in France.
Use ONLY the context provided.
Cite facts from context directly.

CONTEXT:
{ "\n\n".join(ctx) }

QUESTION:
{q}

ANSWER:
"""


# ---------------- QA ----------------
def answer_question(q):

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    index, docs = load_faiss()

    llm = load_llm()

    contexts, sources = retrieve(q, embedder, index, docs)

    prompt = build_prompt(q, contexts)

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.2
    )

    raw_text = output["choices"][0]["text"]
    answer = raw_text.split("QUESTION:")[0].strip()

    return answer, sources


# ---------------- CHAT ----------------
if __name__ == "__main__":

    print("\n===== AI Masters Advisor (France - FAST Mode) =====\n")

    while True:
        q = input("Ask your question (or type 'exit'): ")

        if q.lower() in ["exit", "quit"]:
            break

        answer, srcs = answer_question(q)

        print("\n--- ANSWER ---")
        print(answer)

        print("\n--- SOURCES ---")
        for s in srcs:
            print("-", s)

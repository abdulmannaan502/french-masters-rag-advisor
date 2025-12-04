import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ================= PATHS =================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "index" / "faiss_index.bin"
META_PATH = BASE_DIR / "index" / "metadata.jsonl"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL_ID = "microsoft/phi-3-mini-4k-instruct"


# ================= LOADERS =================

@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index(str(INDEX_PATH))

    docs: List[Dict] = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    return index, docs


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model=LLM_MODEL_ID,
        tokenizer=LLM_MODEL_ID,
    )


# ================= RAG CORE =================

def retrieve(query: str, embedder, index, docs, k=4):
    embedding = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, ids = index.search(embedding, k)

    contexts = []
    sources = set()

    for i in ids[0]:
        doc = docs[i]
        contexts.append(doc["text"])
        sources.add(doc["source"])

    return contexts, list(sources)


def build_prompt(question: str, contexts: List[str], target_lang: str) -> str:
    ctx_text = "\n\n".join(contexts)

    if target_lang == "Français":
        lang_instruction = (
            "Réponds en français clair et simple. "
            "Utilise uniquement les informations factuelles contenues dans le CONTEXTE."
        )
    else:
        lang_instruction = (
            "Answer in clear and simple English. "
            "Use only the factual information from the CONTEXT."
        )

    return f"""
You are an academic advisor assistant for international students applying to Master's programs in France.

{lang_instruction}
Respond in concise bullet points (1 sentence per bullet).
Do not repeat marketing or motivational content.
Stop when the question has been fully answered.

CONTEXT:
{ctx_text}

QUESTION:
{question}

ANSWER:
""".strip()


def generate_answer(question, language):
    index, docs = load_index_and_metadata()
    embedder = load_embedder()
    llm = load_llm()

    contexts, sources = retrieve(question, embedder, index, docs)

    if not contexts:
        return "No relevant information was found in the documents.", []

    prompt = build_prompt(question, contexts, language)

    output = llm(
        prompt,
        max_new_tokens=220,
        temperature=0.2,
        do_sample=True,
    )

    text = output[0]["generated_text"]

    if "ANSWER:" in text:
        text = text.split("ANSWER:", 1)[1]

    answer = text.split("QUESTION:")[0].strip()

    return answer, sources


# ================= STREAMLIT UI =================

def main():

    st.set_page_config(
        page_title="🎓 AI Academic Advisor – France",
        page_icon="🎓",
        layout="wide"
    )

    st.markdown(
        """
        # 🎓 AI Academic Advisor – France
        ### Retrieval-Augmented LLM for French Master's Admissions
        """
    )

    st.info(
        "Ask questions grounded strictly in official university documents "
        "(Campus France, Paris-Saclay, Grenoble, HEC)."
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## 🔎 System Overview")

        st.markdown(
            """
            **AI Stack**
            - Phi-3 Mini (Transformers, CPU)
            - FAISS document retrieval
            - Multilingual sentence embeddings
            
            **Question types**
            - Application steps  
            - Language requirements  
            - Deadlines & procedures  
            - Program structure  
            """
        )

        st.markdown("## 🌐 Answer language")

        lang = st.radio(
            "Select response language:",
            ["English", "Français"],
            index=0
        )

        st.session_state["target_lang"] = lang


    # ---- Message History ----
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                if "sources" in msg:
                    st.markdown("##### 📚 Sources")
                    for s in msg["sources"]:
                        st.markdown(f"- `{s}`")


    # ---- User input ----
    user_input = st.chat_input("Ask about French Master's admissions...")


    if user_input:

        st.session_state["messages"].append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                answer, sources = generate_answer(
                    user_input,
                    st.session_state["target_lang"]
                )

                st.markdown(answer)

                if sources:
                    st.markdown("##### 📚 Sources")
                    for s in sources:
                        st.markdown(f"- `{s}`")

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


if __name__ == "__main__":
    main()

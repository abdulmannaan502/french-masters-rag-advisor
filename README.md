# 🎓 AI Masters Advisor – France

This Space hosts a **Retrieval-Augmented Generation (RAG) system** designed to answer student questions about applying to **Master’s programs in France**.

The assistant supports **English and French queries** and is grounded strictly on official academic documents, including:

- Campus France "Études en France" guide  
- Campus France Master's admission documentation  
- Université Paris-Saclay AI Master's catalogue  
- Grenoble AI4OneHealth Master's guide  
- HEC MSc Data Science program brochure

All answers are generated using a retrieval + grounding pipeline to minimize hallucinations and ensure factual accuracy.

---

## 🔍 System Features

✅ Multilingual question answering (English & French)  
✅ FAISS vector retrieval over official university PDFs  
✅ Phi-3 LLM response synthesis  
✅ Faithfulness + Recall@k benchmarking  
✅ Fully reproducible evaluation pipeline  

---

## 📊 Evaluation Results

Benchmarking was conducted on bilingual admissions QA sets:

| Language | Faithfulness | Recall@1 | Recall@3 | Recall@5 |
|----------|----------------|-----------|-----------|-----------|
| English  | **90%** | **50%** | **90%** | **90%** |
| French   | **90%** | **70%** | **90%** | **90%** |

The results demonstrate strong cross-lingual grounding performance with higher top-rank retrieval precision observed for French queries.

---

## 🔗 Project Links

- ✅ Source code & experiments:  
  https://github.com/abdulmannaan502/french-masters-rag-advisor  

- ✅ Reproducible evaluation notebook (Kaggle):  
  *(Link will be added after notebook publication)*

---

## ⚙️ Architecture Overview

**Pipeline Flow:**

PDF documents → Chunking → Embedding → FAISS Vector Search →  
Top-K Grounded Context → Phi-3 Generation → Verified Answer

All outputs are constrained to retrieved document context to maintain factual reliability.

---

## 🎯 Use Case

This assistant is intended as:

- A proof-of-concept admissions advisor chatbot  
- A research artifact for multilingual NLP evaluation  
- A portfolio system for graduate AI program applications

---

## 📄 Research

A technical evaluation of this system is documented in a short research preprint:

**_“Multilingual Evaluation of a Retrieval-Augmented Generation System for Admissions Question Answering”_**

*(arXiv submission pending)*

---

## 🛠 Maintenance

This project is actively maintained and expanded for:

- Larger bilingual datasets  
- Retrieval reranking experiments  
- Response faithfulness auditing


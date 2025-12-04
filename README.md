# Multilingual RAG Advisor for French Master's Admissions

This repository contains a **Retrieval-Augmented Generation (RAG)** system that answers questions about applying to **Master's programs in France**, in both **English and French**.  

The system is evaluated as a small research project with public code, data, and a live demo.

---

## 🧠 Overview

**Main features**

- Multilingual questions: **English 🇬🇧 / French 🇫🇷**
- RAG pipeline:
  - Sentence embeddings: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - Vector DB: **FAISS**
  - LLM: **Phi-3 Mini**
- Grounded on real documents:
  - Campus France guides
  - Université Paris-Saclay AI Master's catalogue
  - Grenoble AI4OneHealth guide
  - HEC MSc Data Science for Business brochure
- Evaluation:
  - Faithfulness Accuracy
  - Recall@1 / Recall@3 / Recall@5
  - English vs French comparison

---

## 🔗 Live Demo and Artifacts

- 🚀 **Hugging Face Space (Streamlit app)**  
  https://huggingface.co/spaces/abdulmannaan1/ai-masters-advisor-france  

- 💾 **Evaluation Dataset (questions + results)**  
  https://www.kaggle.com/datasets/abdulmannaan12/french-masters-rag-eval  

- 📊 **Evaluation Notebook (metrics + plots)**  
  https://www.kaggle.com/code/abdulmannaan12/multilingual-rag-evaluation-for-french-admissions  

---

## 📂 Project Structure

```
french-masters-rag-advisor/
├─ data/
├─ index/
│  ├─ faiss_index.bin
│  └─ metadata.jsonl
├─ src/
│  ├─ preprocess.py
│  ├─ build_index.py
│  ├─ rag_pipeline.py
│  └─ app_streamlit.py
├─ eval/
│  ├─ questions.jsonl
│  ├─ questions_fr.jsonl
│  ├─ run_faithfulness_eval.py
│  └─ results_*.jsonl
├─ requirements.txt
├─ Dockerfile
└─ paper.tex
```

---

## 🛠 Local Setup

```bash
git clone https://github.com/abdulmannaan502/french-masters-rag-advisor.git
cd french-masters-rag-advisor

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🔧 Building the Index

```bash
python src/preprocess.py
python src/build_index.py
```

---

## 💬 CLI Question Answering

```bash
python src/rag_pipeline.py
```

---

## 🌐 Streamlit App

```bash
streamlit run src/app_streamlit.py
```

---

## 📊 Evaluation

```bash
python eval/run_faithfulness_eval.py
```

**Key metrics:**

| Language | Faithfulness | Recall@1 | Recall@3 | Recall@5 |
|----------|--------------:|----------:|----------:|----------:|
| English  | 90%           | 50%       | 90%       | 90%       |
| French   | 90%           | 70%       | 90%       | 90%       |

---

## 🧪 Research Paper

See `paper.tex` for the submission-ready LaTeX paper.

---

## ⚖️ License

MIT License.

---

## 🙌 Acknowledgements

Campus France, French universities, Sentence-Transformers, FAISS, Hugging Face, Streamlit, and Phi-3 Mini.

# Medical RAG Chatbot — Differential Diagnosis Assistant

**Multilingual Medical RAG System for Differential Diagnosis Support**

> Decision-support tool for trained medical professionals. This system assists physicians — it never diagnoses autonomously. Clinical responsibility remains with the physician at all times.

---

## Project Overview

| Field | Details |
|---|---|
| **Student** | Mountassir BENLAKBIRA — CI3, SUPMTI Rabat |
| **Supervisor** | Pr. Soufiane HAMIDA — SUPMTI |
| **Period** | 23/03/2026 → 22/06/2026 (16 weeks) |
| **Languages** | French · English · Spanish |

## What It Does

A doctor types a clinical question in French, English, or Spanish. The system searches through a structured medical corpus and returns the most likely diagnoses ranked by confidence, with the exact source passages that support each one. The goal is to save time during consultation, not to replace clinical judgment.

---

## Architecture

```
User query (FR / EN / ES)
        │
        ▼
 Language detection (langdetect, threshold 0.90)
        │
        ▼
  Multilingual embedding (BGE-M3)
        │
        ▼
  Vector retrieval (ChromaDB / FAISS)
        │
        ▼
  LangChain orchestration
    ├── Medical prompt library
    ├── ConversationBufferMemory (multi-turn)
    └── Post-generation verification (double LLM call)
        │
        ▼
  Structured output (Pydantic)
    ├── Ranked differential diagnoses
    ├── Confidence score
    └── Cited source passages
        │
        ▼
  Response in user's language
```

**Frameworks:** LangChain (orchestration) + LlamaIndex (indexing & chunking)  
**LLM:** Llama 3.1 via Groq API  
**Vector DB (dev):** ChromaDB · **Vector DB (prod):** FAISS (self-hosted, GDPR-compliant)  
**Embeddings (dev):** all-MiniLM-L6-v2 · **Embeddings (prod):** PubMedBERT + BGE-M3

---

## Repository Structure

```
medical-rag-chatbot/
├── corpus/
│   ├── collect_pubmed.py        # PubMed E-utilities API collection
│   ├── parse_mesh.py            # MeSH term parsing
│   ├── parse_icd10.py           # ICD-10 code parsing
│   ├── merge_datasets.py        # Semantic chunking (NLTK)
│   └── build_index.py           # ChromaDB vector indexing
├── rag_pipeline/
│   └── rag_chain.py             # Core RAG chain (LangChain + Groq)
├── poc/
│   ├── poc_dataset.py           # POC subset builder
│   ├── poc_langchain.py         # LangChain-only POC
│   ├── poc_Hybrid.py            # Hybrid LlamaIndex+LangChain POC
│   └── poc_compare.py           # Comparative results & verdict
├── embeddings/                  # (Phase 3) PubMedBERT index
├── prompts/                     # (Phase 3) Medical prompt library
├── evaluation/                  # (Phase 5) RAGAS evaluation scripts
├── data/
│   ├── raw/pubmed/              # 1,665 PubMed abstracts
│   ├── raw/mesh/                # 30,946 MeSH terms
│   ├── raw/icd10/               # 46,498 ICD-10 codes
│   ├── processed/               # unified_corpus.json (81,838 chunks)
│   └── chromadb/                # 81,838 vectors indexed
├── .env.example                 # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Corpus

| Source | Raw documents | Chunks indexed |
|---|---|---|
| PubMed abstracts | 1,665 | 3,950 |
| MeSH terms | 30,946 | 31,390 |
| ICD-10 codes | 46,498 | 46,498 |
| **Total** | **79,109** | **81,838** |

Chunking strategy: NLTK sentence tokenizer, max 5 sentences per chunk, minimum 100 characters.

---

## POC Results

500-document subset · 20 medical questions · 5 specialties
Evaluation method: LLM-as-judge quality scoring (0–10)

| Metric | LangChain only | Hybrid (LlamaIndex + LangChain) |
|---|---|---|
| Avg quality score (0–10) | 5.3 | **6.4** |
| Avg total time | 5.753s | 6.684s |
| Index build time | ~14s | ~17s |

**Verdict:** Hybrid architecture retained — quality difference (1.1 points) is clinically significant, speed difference (0.93s) is not.

---

## Setup

### Prerequisites
- Python 3.13
- Windows (VS Code recommended)
- Groq API key (free tier)

### Installation

```powershell
git clone https://github.com/montassir-benlakbira/medical-rag-chatbot.git
cd medical-rag-chatbot

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Environment variables

```powershell
copy .env.example .env
# Then edit .env and add your GROQ_API_KEY
```

`.env.example`:
```
GROQ_API_KEY=gsk_your_key_here
```

### Run the RAG chatbot (terminal)

```powershell
python rag_pipeline/rag_chain.py
```

---

## Evaluation Framework (Phase 5)

**RAGAS metrics:** Faithfulness · Answer Relevancy · Context Precision · Context Recall · Answer Semantic Similarity

**Differential diagnosis-specific:** MRR (Mean Reciprocal Rank) · NDCG@k (k=3, k=5)

**Clinical validation:** 200 annotated test cases · 20 expert physicians · 5 specialties

---

## Ethical & Legal Framework

- System positioned as a **decision-support tool**, never an autonomous diagnostic agent
- No real patient data used in development (synthetic cases and anonymised published cases only)
- GDPR / Loi 09-08 (Morocco) compliant architecture — all data self-hosted
- Clinical responsibility remains with the physician user at all times
- Reference: Article 73, Code de déontologie médicale marocain

---

## Roadmap

- [x] Phase 1 — Corpus collection & vector indexing (81,838 chunks)
- [x] Phase 2 — POC comparison (LangChain vs Hybrid) — verdict documented
- [ ] Phase 3 — RAG engine (PubMedBERT, memory, medical prompts, hallucination check)
- [ ] Phase 4 — Interface & multilingual (Streamlit, FastAPI, BGE-M3, PDF export)
- [ ] Phase 5 — Evaluation & deployment (RAGAS, clinical validation, Docker)

---

## License

Academic project — SUPMTI 2025/2026. All rights reserved.

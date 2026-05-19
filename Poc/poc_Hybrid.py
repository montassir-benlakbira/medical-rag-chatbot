import json
import os
import time
import sys
sys.path.append("..")

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaGroq
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─── Configuration ───────────────────────────────────────────
POC_CORPUS    = "../data/poc/poc_corpus.json"
POC_QUESTIONS = "../data/poc/poc_questions.json"
OUTPUT_FILE   = "../data/poc/results_hybrid.json"
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.1-8b-instant"

# ─── Load POC corpus ─────────────────────────────────────────
print("Loading POC corpus...")
with open(POC_CORPUS, encoding="utf-8") as f:
    docs = json.load(f)

with open(POC_QUESTIONS, encoding="utf-8") as f:
    questions = json.load(f)

print(f"  {len(docs)} documents, {len(questions)} questions")

# ─── HYBRID: LlamaIndex handles indexing ─────────────────────
print("\n[HYBRID] LlamaIndex building the index...")
Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)
Settings.llm = LlamaGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
Settings.chunk_size = 512

llama_docs = [
    Document(
        text=d["text"][:1000],
        metadata={"source": d["source"], "specialty": d["specialty"]}
    )
    for d in docs
]

t_index_start = time.time()
index = VectorStoreIndex.from_documents(llama_docs, show_progress=True)
index_time = round(time.time() - t_index_start, 3)
print(f"✅ LlamaIndex index built in {index_time}s")

# LlamaIndex retriever only — no generation
retriever = index.as_retriever(similarity_top_k=5)

# ─── HYBRID: LangChain handles generation ────────────────────
print("[HYBRID] LangChain handling generation...")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.1
)

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a medical assistant. Answer based ONLY on the documents below.
Cite your sources. Never invent information.

DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""
)

# ─── Run 20 questions ─────────────────────────────────────────
print("\nRunning 20 questions with Hybrid (LlamaIndex + LangChain)...\n")
results = []

for i, question in enumerate(questions):
    print(f"Q{i+1}: {question[:60]}...")

    # LlamaIndex retrieves
    t0 = time.time()
    retrieved_nodes = retriever.retrieve(question)
    retrieval_time = round(time.time() - t0, 3)

    # Build context from LlamaIndex results
    context = "\n\n".join([
        f"[{node.metadata.get('source', 'Unknown')}] {node.text}"
        for node in retrieved_nodes
    ])

    # LangChain generates
    t1 = time.time()
    prompt = PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    generation_time = round(time.time() - t1, 3)

    results.append({
        "question_id": i + 1,
        "question": question,
        "answer": response.content,
        "retrieval_time_sec": retrieval_time,
        "generation_time_sec": generation_time,
        "total_time_sec": round(retrieval_time + generation_time, 3),
        "sources": list(set([
            node.metadata.get("source", "Unknown")
            for node in retrieved_nodes
        ]))
    })

    print(f"   ✅ Retrieval: {retrieval_time}s | Generation: {generation_time}s")
    time.sleep(0.5)

# ─── Save results ─────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

avg_retrieval  = round(sum(r["retrieval_time_sec"] for r in results) / len(results), 3)
avg_generation = round(sum(r["generation_time_sec"] for r in results) / len(results), 3)
avg_total      = round(sum(r["total_time_sec"] for r in results) / len(results), 3)

print(f"\n✅ Hybrid POC complete")
print(f"📊 Avg retrieval:   {avg_retrieval}s")
print(f"📊 Avg generation:  {avg_generation}s")
print(f"📊 Avg total:       {avg_total}s")
print(f"📁 Results saved → {OUTPUT_FILE}")
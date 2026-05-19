import json
import os
import time
import sys
sys.path.append("..")

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuration
POC_CORPUS    = "../data/poc/poc_corpus.json"
POC_QUESTIONS = "../data/poc/poc_questions.json"
OUTPUT_FILE   = "../data/poc/results_langchain.json"
MODEL_NAME    = "all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.1-8b-instant"
CHROMA_DIR    = "../data/poc/chroma_poc"

# Load POC corpus
print("Loading POC corpus...")
with open(POC_CORPUS, encoding="utf-8") as f:
    docs = json.load(f)

with open(POC_QUESTIONS, encoding="utf-8") as f:
    questions = json.load(f)

print(f"  {len(docs)} documents, {len(questions)} questions")

# Build POC ChromaDB index
print("\nBuilding POC ChromaDB index (500 docs)...")
embedder = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection("poc_langchain")
except:
    pass

collection = client.create_collection(
    name="poc_langchain",
    metadata={"hnsw:space": "cosine"}
)

# Index in one batch (500 docs is small)
texts      = [d["text"][:1000] for d in docs]
ids        = [d["id"] for d in docs]
metadatas  = [{"source": d["source"], "specialty": d["specialty"]} for d in docs]
embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)
print(f"✅ Indexed {collection.count()} documents")

# Setup LLM 
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

# Run 20 questions 
print("\nRunning 20 questions with LangChain...\n")
results = []

for i, question in enumerate(questions):
    print(f"Q{i+1}: {question[:60]}...")

    # Measure retrieval time
    t0 = time.time()
    query_vec = embedder.encode([question]).tolist()
    retrieved = collection.query(
        query_embeddings=query_vec,
        n_results=5
    )
    retrieval_time = round(time.time() - t0, 3)

    # Build context
    context = "\n\n".join([
        f"[{retrieved['metadatas'][0][j]['source']}] {retrieved['documents'][0][j]}"
        for j in range(len(retrieved['documents'][0]))
    ])

    # Measure generation time
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
        "sources": [m["source"] for m in retrieved["metadatas"][0]]
    })

    print(f"   ✅ Retrieval: {retrieval_time}s | Generation: {generation_time}s")
    time.sleep(0.5)  # avoid Groq rate limit

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

avg_retrieval   = round(sum(r["retrieval_time_sec"] for r in results) / len(results), 3)
avg_generation  = round(sum(r["generation_time_sec"] for r in results) / len(results), 3)
avg_total       = round(sum(r["total_time_sec"] for r in results) / len(results), 3)

print(f"\n✅ LangChain POC complete")
print(f"📊 Avg retrieval:   {avg_retrieval}s")
print(f"📊 Avg generation:  {avg_generation}s")
print(f"📊 Avg total:       {avg_total}s")
print(f"📁 Results saved → {OUTPUT_FILE}")
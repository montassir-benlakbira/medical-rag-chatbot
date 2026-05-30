import os
import sys
import torch
torch.set_num_threads(2)

from dotenv import load_dotenv
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in .env")
    sys.exit(1)

# Paths (always relative to project root)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chromadb_pubmedbert")
COLLECTION = "medical_rag_pubmedbert"
PUBMEDBERT = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
GROQ_MODEL = "llama-3.3-70b-versatile"
N_RESULTS  = 5

# Load components 
print("Loading PubMedBERT embedder...")
embedder = HuggingFaceEmbedding(model_name=PUBMEDBERT)
print("✅ Embedder ready")

print("Connecting to ChromaDB...")
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION)
print(f"✅ ChromaDB — {collection.count()} nodes")

print("Loading Groq LLM...")
llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.1)
print("✅ LLM ready\n")

# Memory 
# Simple manual memory
# Stores last 5 turns so doctors can ask follow-up questions
class SimpleMemory:
    def __init__(self, max_turns: int = 5):
        self.history = []
        self.max_turns = max_turns

    def add(self, question: str, answer: str):
        self.history.append((question, answer))
        if len(self.history) > self.max_turns:
            self.history.pop(0)  # drop oldest turn

    def get(self) -> str:
        if not self.history:
            return "No prior conversation."
        parts = []
        for q, a in self.history:
            parts.append(f"Physician: {q}\nAssistant: {a}")
        return "\n\n".join(parts)

    def clear(self):
        self.history = []

memory = SimpleMemory(max_turns=10)

# Medical prompt 
MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are a senior clinical decision-support assistant for trained physicians.
You speak like a knowledgeable colleague — clear, direct, and medically precise.

STRICT RULES:
- Base every factual claim ONLY on the retrieved documents below
- Cite sources using [Source X] after every claim
- For PubMed sources: quote the exact citable sentence so the physician can verify it
- Never invent medical information not present in the documents
- If documents are insufficient for a claim, say so explicitly
- The physician makes the final decision — you only support

FOLLOW-UP RULE (critical):
If there is prior conversation, do NOT repeat what was already covered.
Acknowledge what changed in the new question, then focus ONLY on what is
different or new. Be direct: "Given the age and diabetes, here is what changes..."

PRIOR CONVERSATION:
{chat_history}

RETRIEVED MEDICAL DOCUMENTS:
{context}

PHYSICIAN QUESTION:
{question}

YOUR RESPONSE:

**Assessment:**
[2-3 sentence direct clinical summary acknowledging any new patient details]

**Differential Diagnosis:**
1. [Diagnosis] — Confidence: [High/Medium/Low]
   Evidence: "[exact citable sentence]" [Source X — PubMed/MeSH/ICD-10]

2. [Diagnosis] — Confidence: [High/Medium/Low]
   Evidence: "[exact citable sentence]" [Source Y — PubMed/MeSH/ICD-10]

**What the Documents Support for Verification:**
[List the exact sentences the physician can search on PubMed or clinical databases to verify]

**Recommended Next Steps:**
- [Specific tests or referrals cited from documents]

**Limitations of this response:**
[What the documents don't cover — be honest]
"""
)

# Retrieval 
def retrieve(question: str, n: int = N_RESULTS) -> str:
    query_vector = embedder.get_text_embedding(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        source     = meta.get("source",    "Unknown")
        specialty  = meta.get("specialty", "general")
        sentence   = meta.get("sentence",  doc)
        window     = meta.get("window",    doc)
        similarity = round(1 - dist, 3)

        # Build a source label the LLM can reference clearly
        source_label = f"Source {i+1} | {source} | {specialty} | confidence {similarity}"

        parts.append(
            f"[{source_label}]\n"
            f"Context window (for reasoning):\n{window}\n\n"
            f"Citable sentence (exact — physician can search this):\n\"{sentence}\""
        )

    return "\n\n" + ("─" * 50 + "\n\n").join(parts)

# RAG pipeline 
def ask(question: str) -> str:
    """Retrieve → Augment → Generate → Remember."""
    print("🔍 Retrieving...")
    context = retrieve(question)

    chat_history = memory.get()

    prompt_text = MEDICAL_PROMPT.format(
        context=context,
        question=question,
        chat_history=chat_history
    )

    print("🤖 Generating...")
    response = llm.invoke(prompt_text)
    answer   = response.content

    memory.add(question, answer)
    return answer

# Interactive terminal loop 
if __name__ == "__main__":
    print("=" * 60)
    print("  MEDICAL RAG CHATBOT — PubMedBERT + SentenceWindow + Llama 3")
    print("=" * 60)
    print("Commands: 'quit' to exit | 'reset' to clear memory\n")

    while True:
        question = input("👨‍⚕️ Doc: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break

        if question.lower() == "reset":
            memory.clear()
            print("✅ Conversation cleared.\n")
            continue

        if not question:
            continue

        answer = ask(question)
        print(f"\n🏥 Assistant:\n{answer}\n")
        print("-" * 60)
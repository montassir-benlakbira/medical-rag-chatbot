import os
import sys
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables 
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY not found in .env file")
    sys.exit(1)

# Configuration
CHROMA_DIR  = "../data/chromadb"
MODEL_NAME  = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"  # free, fast, open source

# Load embedding model 
print("Loading embedding model...")
embedder = SentenceTransformer(MODEL_NAME)
print("✅ Embedding model loaded")

# Connect to ChromaDB 
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("medical_rag")
print(f"✅ ChromaDB connected — {collection.count()} documents available")

# Load LLM 
print("Loading Groq LLM...")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.1  # low temperature = more factual, less creative
)
print("✅ LLM loaded")

# Medical prompt template 
MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a medical assistant helping physicians with differential diagnosis.
You must base your answer ONLY on the provided medical documents.
Always cite the source of each piece of information.
If the documents do not contain enough information, say so clearly.
Never invent medical information.

MEDICAL DOCUMENTS:
{context}

PHYSICIAN'S QUESTION:
{question}

STRUCTURED RESPONSE:
Provide a clear, structured answer with:
1. Most likely diagnoses based on the documents
2. Supporting evidence from the documents
3. Sources cited

Answer:"""
)

# Retrieval function
def retrieve_documents(question, n_results=5):
    """Embed the question and retrieve most relevant documents."""
    query_vector = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    # Format context with sources
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        source = meta.get("source", "Unknown")
        context_parts.append(f"[Document {i+1} — {source}]\n{doc}")
    
    return "\n\n".join(context_parts)

# RAG Chain 
def ask(question):
    """Full RAG pipeline: retrieve → augment → generate."""
    print(f"\n🔍 Retrieving documents for: '{question}'")
    context = retrieve_documents(question)
    
    print("🤖 Generating answer...")
    prompt = MEDICAL_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    
    return response.content

# Interactive loop 
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   MEDICAL RAG CHATBOT — Powered by Llama 3 + ChromaDB")
    print("="*60)
    print("Type your medical question. Type 'quit' to exit.\n")
    
    while True:
        question = input("👨‍⚕️ Physician: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
            
        answer = ask(question)
        print(f"\n🏥 Assistant:\n{answer}\n")
        print("-"*60)
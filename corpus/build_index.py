import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
CORPUS_FILE = "../data/processed/unified_corpus.json"
CHROMA_DIR  = "../data/chromadb"
MODEL_NAME  = "all-MiniLM-L6-v2"  # general model, biomedical model comparison in POC phase
BATCH_SIZE  = 100
TEST_MODE   = False   # True = PubMed only, False = full 79k

os.makedirs(CHROMA_DIR, exist_ok=True)

# Load corpus
print("Loading corpus...")
with open(CORPUS_FILE, encoding="utf-8") as f:
    all_docs = json.load(f)

if TEST_MODE:
    docs = [d for d in all_docs if d["source"] == "PubMed"]
    print(f"TEST MODE: using {len(docs)} PubMed documents only")
else:
    docs = all_docs
    print(f"FULL MODE: indexing {len(docs)} documents")

# Load embedding model 
print(f"\nLoading embedding model: {MODEL_NAME}")
print("(First run downloads ~80MB — please wait...)")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# Setup ChromaDB 
print("\nSetting up ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete existing collection if re-running
try:
    client.delete_collection("medical_rag")
except:
    pass

collection = client.create_collection(
    name="medical_rag",
    metadata={"hnsw:space": "cosine"}
)

# Index in batches
print(f"\nIndexing {len(docs)} documents in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, len(docs), BATCH_SIZE)):
    batch = docs[i:i + BATCH_SIZE]

    texts = [d["text"][:1000] for d in batch]  # cap at 1000 chars
    ids   = [d["id"] for d in batch]
    metas = [{"source": d["source"], "specialty": d["specialty"]} for d in batch]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metas
    )

print(f"\n✅ Indexing complete! {collection.count()} documents in ChromaDB")

# Quick test search
print("\n── Test query: 'chest pain and shortness of breath' ──")
query = "chest pain and shortness of breath"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

for i, doc in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i]["source"]
    print(f"\nResult {i+1} [{source}]:\n{doc[:200]}...")
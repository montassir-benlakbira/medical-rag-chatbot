import json
import os
import nltk

# Configuration 
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_CHUNK_SENTENCES = 5   # max sentences per chunk
MIN_CHUNK_CHARS     = 100 # skip chunks too short to be useful

all_documents = []
chunk_id = 0

# Semantic chunking function 
def chunk_text(text, source, specialty, id_prefix):
    """Split text into semantic chunks at sentence boundaries."""
    global chunk_id
    chunks = []
    
    sentences = nltk.sent_tokenize(text)
    
    current_chunk = []
    current_len   = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        current_len += len(sentence)
        
        # Commit chunk when we hit max sentences
        if len(current_chunk) >= MAX_CHUNK_SENTENCES:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "id":       f"{id_prefix}_{chunk_id}",
                    "text":     chunk_text,
                    "source":   source,
                    "specialty": specialty
                })
                chunk_id += 1
            current_chunk = []
            current_len   = 0
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append({
                "id":       f"{id_prefix}_{chunk_id}",
                "text":     chunk_text,
                "source":   source,
                "specialty": specialty
            })
            chunk_id += 1
    
    return chunks

# Load PubMed 
print("Loading PubMed abstracts...")
with open("../data/raw/pubmed/all_abstracts.json", encoding="utf-8") as f:
    pubmed = json.load(f)

for doc in pubmed:
    text = f"Title: {doc['title']}\nAbstract: {doc['abstract']}"
    chunks = chunk_text(text, "PubMed", doc["specialty"], f"pubmed_{doc['pmid']}")
    all_documents.extend(chunks)

print(f"  ✅ PubMed → {len([d for d in all_documents if d['source'] == 'PubMed'])} chunks")

# Load MeSH 
print("Loading MeSH terms...")
mesh_start = len(all_documents)

with open("../data/raw/mesh/mesh_terms.json", encoding="utf-8") as f:
    mesh = json.load(f)

for i, doc in enumerate(mesh):
    chunks = chunk_text(doc["full_text"], "MeSH", "general", f"mesh_{i}")
    all_documents.extend(chunks)

print(f"  ✅ MeSH → {len(all_documents) - mesh_start} chunks")

# Load ICD-10 
print("Loading ICD-10 codes...")
icd_start = len(all_documents)

with open("../data/raw/icd10/icd10_codes.json", encoding="utf-8") as f:
    icd10 = json.load(f)

for i, doc in enumerate(icd10):
    chunks = chunk_text(doc["full_text"], "ICD-10", "general", f"icd10_{i}")
    all_documents.extend(chunks)

print(f"  ✅ ICD-10 → {len(all_documents) - icd_start} chunks")

# Save unified dataset 
output_path = os.path.join(OUTPUT_DIR, "unified_corpus.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_documents, f, ensure_ascii=False, indent=2)

print(f"\n✅ Merge complete!")
print(f"📊 Total chunks: {len(all_documents)}")
print(f"📁 Saved in: {output_path}")

# Preview
print("\n── Preview of first PubMed chunk ──")
pubmed_chunks = [d for d in all_documents if d['source'] == 'PubMed']
if pubmed_chunks:
    print(json.dumps(pubmed_chunks[0], indent=2, ensure_ascii=False))
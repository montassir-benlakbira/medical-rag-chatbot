import json
import random
import os

# ─── Configuration ───────────────────────────────────────────
CORPUS_FILE = "../data/processed/unified_corpus.json"
OUTPUT_DIR  = "../data/poc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)  # reproducible results

# ─── Load full corpus ─────────────────────────────────────────
print("Loading corpus...")
with open(CORPUS_FILE, encoding="utf-8") as f:
    all_docs = json.load(f)

# ─── Select 500 balanced docs ────────────────────────────────
# 100 PubMed + 200 MeSH + 200 ICD-10
pubmed = [d for d in all_docs if d["source"] == "PubMed"]
mesh   = [d for d in all_docs if d["source"] == "MeSH"]
icd10  = [d for d in all_docs if d["source"] == "ICD-10"]

subset = (
    random.sample(pubmed, min(100, len(pubmed))) +
    random.sample(mesh,   min(200, len(mesh)))   +
    random.sample(icd10,  min(200, len(icd10)))
)

print(f"  PubMed: {len([d for d in subset if d['source'] == 'PubMed'])}")
print(f"  MeSH:   {len([d for d in subset if d['source'] == 'MeSH'])}")
print(f"  ICD-10: {len([d for d in subset if d['source'] == 'ICD-10'])}")
print(f"  Total:  {len(subset)}")

# ─── Save 500 doc subset ──────────────────────────────────────
subset_path = os.path.join(OUTPUT_DIR, "poc_corpus.json")
with open(subset_path, "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

# ─── 20 medical test questions ────────────────────────────────
# Covering all 5 specialties + multilevel complexity
questions = [
    # Cardiology (4)
    "What are the main causes of chest pain in adult patients?",
    "How is myocardial infarction diagnosed and what are its symptoms?",
    "What is the difference between stable and unstable angina?",
    "What are the treatment options for heart failure?",
    # Pneumology (4)
    "What are the symptoms and causes of pulmonary embolism?",
    "How is pneumonia diagnosed and treated?",
    "What distinguishes asthma from COPD?",
    "What are the risk factors for lung cancer?",
    # Neurology (4)
    "What are the early warning signs of a stroke?",
    "How is epilepsy classified and treated?",
    "What are the symptoms of multiple sclerosis?",
    "What causes severe headache with neck stiffness?",
    # Gastroenterology (4)
    "What are the symptoms of Crohn disease?",
    "How is hepatitis B different from hepatitis C?",
    "What are the signs of liver cirrhosis?",
    "What causes acute abdominal pain in adults?",
    # Internal medicine (4)
    "What are the diagnostic criteria for diabetes mellitus type 2?",
    "How is hypertension classified and managed?",
    "What are the causes of unexplained weight loss?",
    "What conditions can cause high fever with skin rash?"
]

questions_path = os.path.join(OUTPUT_DIR, "poc_questions.json")
with open(questions_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"\n✅ POC dataset ready")
print(f"📁 500 docs → {subset_path}")
print(f"📁 20 questions → {questions_path}")
import requests
import json
import time
import os

# Configuration
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OUTPUT_DIR = "../data/raw/pubmed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5 medical specialties
SPECIALTIES = {
    "cardiology": "chest pain OR myocardial infarction OR heart failure OR arrhythmia",
    "pneumology": "pneumonia OR pulmonary embolism OR asthma OR COPD OR dyspnea",
    "neurology": "stroke OR epilepsy OR multiple sclerosis OR headache OR neuropathy",
    "gastroenterology": "abdominal pain OR Crohn disease OR hepatitis OR cirrhosis OR colitis",
    "internal_medicine": "fever OR fatigue OR anemia OR diabetes OR hypertension"
}

MAX_PER_SPECIALTY = 600  # 5 x 600 = 3000 abstracts total

# Search PubMed → get article IDs 
def search_pubmed(query, max_results):
    print(f"  Searching: {query[:50]}...")
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }
    response = requests.get(BASE_URL + "esearch.fcgi", params=params)
    data = response.json()
    ids = data["esearchresult"]["idlist"]
    print(f"  Found {len(ids)} articles")
    return ids

# Fetch abstracts for a list of IDs
def fetch_abstracts(pmids):
    ids_str = ",".join(pmids)
    params = {
        "db": "pubmed",
        "id": ids_str,
        "retmode": "xml",
        "rettype": "abstract"
    }
    response = requests.get(BASE_URL + "efetch.fcgi", params=params)
    return response.text

# Parse and clean each article
def parse_articles(xml_text, specialty):
    from lxml import etree
    articles = []

    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except Exception as e:
        print(f"  Warning: XML parsing failed — {e}")
        return articles

    for article in root.findall(".//PubmedArticle"):
        try:
            # PMID
            pmid = article.findtext(".//PMID", default="")

            # Title
            title = article.findtext(".//ArticleTitle", default="")

            # Abstract — can have multiple sections
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join([
                (a.text or "") for a in abstract_parts if a.text
            ])

            if not abstract or len(abstract) < 50:
                continue

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "specialty": specialty,
                "source": "PubMed"
            })
        except Exception:
            continue

    return articles

# Main pipeline
all_articles = []

for specialty, query in SPECIALTIES.items():
    print(f"\n── Collecting: {specialty.upper()} ──")

    # Search
    pmids = search_pubmed(query, MAX_PER_SPECIALTY)

    if not pmids:
        print("  No results, skipping.")
        continue

    # Fetch in batches of 100 (API limit)
    batch_size = 100
    specialty_articles = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        print(f"  Fetching batch {i//batch_size + 1}...")
        raw = fetch_abstracts(batch)
        parsed = parse_articles(raw, specialty)
        specialty_articles.extend(parsed)
        time.sleep(0.4)  # Respect API rate limit (3 req/s)

    print(f"  Saved {len(specialty_articles)} abstracts for {specialty}")
    all_articles.extend(specialty_articles)

    # Save per specialty
    out_path = os.path.join(OUTPUT_DIR, f"{specialty}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(specialty_articles, f, ensure_ascii=False, indent=2)

# Save everything in one file too
all_path = os.path.join(OUTPUT_DIR, "all_abstracts.json")
with open(all_path, "w", encoding="utf-8") as f:
    json.dump(all_articles, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Total articles collected: {len(all_articles)}")
print(f"📁 Saved in: {OUTPUT_DIR}")
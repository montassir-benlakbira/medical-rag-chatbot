import xml.etree.ElementTree as ET
import json
import os

# Configuration
MESH_FILE = "../../desc2026/desc2026.xml"
OUTPUT_DIR = "../data/raw/mesh"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading MeSH XML... (this may take 30-60 seconds)")
tree = ET.parse(MESH_FILE)
root = tree.getroot()
print("XML loaded. Parsing...")

# Parse each MeSH descriptor
mesh_terms = []

for descriptor in root.findall("DescriptorRecord"):
    try:
        # Main term name
        name = descriptor.findtext("DescriptorName/String", default="").strip()
        if not name:
            continue

        # Scope note = the definition of the term
        scope_note = descriptor.findtext(
            "ConceptList/Concept/ScopeNote", default=""
        ).strip()

        # Tree numbers = medical classification (e.g. C14.280 = cardiovascular)
        tree_numbers = [
            tn.text for tn in descriptor.findall("TreeNumberList/TreeNumber")
            if tn.text
        ]

        # Related synonyms/terms
        synonyms = [
            t.findtext("String", default="")
            for concept in descriptor.findall("ConceptList/Concept")
            for t in concept.findall("TermList/Term")
        ]
        synonyms = list(set([s.strip() for s in synonyms if s.strip() and s != name]))

        # Only keep terms that have a definition
        if not scope_note or len(scope_note) < 20:
            continue

        # Build the text we'll later embed
        full_text = f"Term: {name}\nDefinition: {scope_note}"
        if synonyms:
            full_text += f"\nSynonyms: {', '.join(synonyms[:5])}"
        if tree_numbers:
            full_text += f"\nClassification: {', '.join(tree_numbers)}"

        mesh_terms.append({
            "term": name,
            "definition": scope_note,
            "synonyms": synonyms[:10],
            "tree_numbers": tree_numbers,
            "full_text": full_text,
            "source": "MeSH 2026"
        })

    except Exception:
        continue

# &Save results
output_path = os.path.join(OUTPUT_DIR, "mesh_terms.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(mesh_terms, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Total MeSH terms extracted: {len(mesh_terms)}")
print(f"📁 Saved in: {output_path}")

# Quick preview of first term
if mesh_terms:
    print("\n── Preview of first term ──")
    print(json.dumps(mesh_terms[0], indent=2))
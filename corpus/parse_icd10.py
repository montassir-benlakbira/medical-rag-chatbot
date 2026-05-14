import xml.etree.ElementTree as ET
import json
import os

# Configuration
ICD10_FILE = "../../icd10cm-table-index-2025/icd-10-cm-tabular-2025.xml"
OUTPUT_DIR = "../data/raw/icd10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading ICD-10 XML...")
tree = ET.parse(ICD10_FILE)
root = tree.getroot()
print("XML loaded. Parsing...")

# Parse ICD-10 codes
icd_entries = []

def extract_codes(element, parent_description=""):
    """Recursively extract all diagnostic codes and descriptions."""

    # Chapter level
    for chapter in element.findall("chapter"):
        chapter_desc = chapter.findtext("desc", default="").strip()
        extract_from_section(chapter, chapter_desc)

def extract_from_section(element, chapter_desc):
    """Extract codes from sections and diags recursively."""

    for section in element.findall("section"):
        section_desc = section.findtext("desc", default="").strip()

        for diag in section.findall(".//diag"):
            try:
                code = diag.findtext("name", default="").strip()
                description = diag.findtext("desc", default="").strip()

                if not code or not description:
                    continue

                # Build full text for embedding
                full_text = f"ICD-10 Code: {code}\nDiagnosis: {description}"
                if chapter_desc:
                    full_text += f"\nCategory: {chapter_desc}"
                if section_desc:
                    full_text += f"\nSection: {section_desc}"

                icd_entries.append({
                    "code": code,
                    "description": description,
                    "chapter": chapter_desc,
                    "section": section_desc,
                    "full_text": full_text,
                    "source": "ICD-10-CM 2025"
                })

            except Exception:
                continue

# Run extraction
extract_codes(root, "")

# Save results
output_path = os.path.join(OUTPUT_DIR, "icd10_codes.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(icd_entries, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Total ICD-10 codes extracted: {len(icd_entries)}")
print(f"📁 Saved in: {output_path}")

# Quick preview
if icd_entries:
    print("\n── Preview of first entry ──")
    print(json.dumps(icd_entries[0], indent=2))
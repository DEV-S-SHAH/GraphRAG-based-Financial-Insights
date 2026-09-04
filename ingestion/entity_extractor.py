import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "normalized"
    / "LTM_FY2025-26_normalized.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "entities"
    / "LTM_FY2025-26_entities.json"
)


ENTITY_PATTERNS = {
    "strategy": [
        "AI",
        "digital transformation",
        "cost optimization",
        "vendor consolidation",
        "pyramid optimization",
        "capital allocation",
    ],

    "program": [
        "Fit4Future",
        "New Horizons",
    ],

    "business_theme": [
        "operational efficiency",
        "productivity",
        "recurring revenues",
        "operating leverage",
        "client demand",
        "large deal wins",
        "transformation initiatives",
    ],

    "risk": [
        "currency risks",
        "currency risk",
        "labor codes",
        "margin moderation",
        "discretionary spending",
    ],

    "capability": [
        "AI platforms",
        "capability building",
        "ecosystem partnerships",
        "digital transformation",
    ],
}


def load_document():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Normalized document not found:\n{INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def find_entities(text):

    entities = []

    for entity_type, patterns in ENTITY_PATTERNS.items():

        for pattern in patterns:

            # Case-insensitive search
            matches = re.finditer(
                re.escape(pattern),
                text,
                flags=re.IGNORECASE,
            )

            for match in matches:

                entities.append(
                    {
                        "name": match.group(0),
                        "canonical_name": pattern,
                        "entity_type": entity_type,
                    }
                )

    return entities


def deduplicate_entities(entities):

    seen = set()
    result = []

    for entity in entities:

        key = (
            entity["canonical_name"].lower(),
            entity["entity_type"],
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(entity)

    return result


def extract_entities():

    print("=" * 80)
    print("LTM ENTITY EXTRACTOR")
    print("=" * 80)

    data = load_document()

    pages = data["pages"]

    print(f"Pages: {len(pages)}")

    extracted = []

    for page in pages:

        page_number = page["page"]
        text = page.get("text", "")

        entities = find_entities(text)

        entities = deduplicate_entities(entities)

        if not entities:
            continue

        for entity in entities:

            extracted.append(
                {
                    "entity_id": (
                        f"LTM_FY26_AR_001_"
                        f"{entity['entity_type']}_"
                        f"{entity['canonical_name'].lower().replace(' ', '_')}"
                    ),

                    "name": entity["canonical_name"],

                    "entity_type": entity["entity_type"],

                    "source": {
                        "document_id": "LTM_FY26_AR_001",
                        "page": page_number,
                    },

                    "confidence": 1.0,
                }
            )

    # Global deduplication
    unique = {}

    for entity in extracted:

        key = entity["entity_id"]

        if key not in unique:
            unique[key] = entity

    entities = list(unique.values())

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "document_id": "LTM_FY26_AR_001",
        "company": "LTM Limited",
        "ticker": "LTM",
        "reporting_period": "FY2025-26",
        "dataset_type": "narrative_entities",
        "entity_count": len(entities),
        "entities": entities,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Entities found: {len(entities)}")
    print()
    print("Entities:")

    for entity in entities:

        print(
            f"{entity['entity_type']:<20} "
            f"{entity['name']:<35} "
            f"page={entity['source']['page']}"
        )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    extract_entities()
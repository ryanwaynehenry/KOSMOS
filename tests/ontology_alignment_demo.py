"""
Manual demo script to align mentions from an interim JSON to ontologies.

Usage:
    python tests/ontology_alignment_demo.py data/interim/altered_session_2348_1.json

The script will:
  - Load the JSON (expects a top-level "mentions" list).
  - Treat each mention as an entity candidate (canonical_name = mention["text"], entity_type = mention["type"]).
  - Call align_entities_with_ontology from clinical_kg.umls.lookup.
  - Print a simple table of canonical_name -> ontology mapping.

Requires DB access and env vars for UMLS connection.
"""

import json
import sys
from pathlib import Path

from clinical_kg.umls.lookup import align_entities_with_ontology


def main(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # In the interim files, "mentions" is actually the grouped entity list
    entities = data.get("mentions", [])
    aligned = align_entities_with_ontology(entities)

    print("Aligned entities:")
    for ent in aligned:
        ont = ent.get("ontology")
        print(
            f"- {ent.get('canonical_name')!r} ({ent.get('entity_type')}): "
            f"{ont.get('source')}:{ont.get('source_code')} [{ont.get('preferred_term')}]"
            if ont
            else f"- {ent.get('canonical_name')!r} ({ent.get('entity_type')}): no match"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/ontology_alignment_demo.py <interim_json_path>")
        sys.exit(1)
    main(sys.argv[1])

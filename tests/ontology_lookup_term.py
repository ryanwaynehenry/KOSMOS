"""
Lookup a single term in UMLS using the alignment helper and print the result.

Usage:
    python tests/ontology_lookup_term.py "<term>" "<entity_type>"

Examples:
    python tests/ontology_lookup_term.py "depression" PROBLEM
    python tests/ontology_lookup_term.py "lisinopril" MEDICATION
    python tests/ontology_lookup_term.py "blood pressure" LAB_TEST
"""

import json
import sys

from clinical_kg.umls.lookup import align_entities_with_ontology


def main(term: str, entity_type: str) -> None:
    # Build a single-entity payload matching what align_entities_with_ontology expects
    entities = [
        {
            "canonical_name": term,
            "entity_type": entity_type,
            "turn_ids": [],
            "mentions": [{"mention_id": "m0001", "turn_id": "", "text": term, "type": entity_type}],
        }
    ]
    aligned = align_entities_with_ontology(entities)

    print("Aligned ontology result:")
    for ent in aligned:
        ont = ent.get("ontology")
        if not ont:
            print(f"- {ent.get('canonical_name')!r} ({ent.get('entity_type')}): no match")
        else:
            print(
                f"- {ent.get('canonical_name')!r} ({ent.get('entity_type')}): "
                f"{ont.get('source')}:{ont.get('source_code')} "
                f"[preferred_term={ont.get('preferred_term')}, cui={ont.get('cui')}, score={ont.get('score')}, searched_term={ont.get('searched_term')}]"
            )

    # Also dump the full JSON in case you need the raw structure
    print("\nFull JSON:")
    print(json.dumps(aligned, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tests/ontology_lookup_term.py <term> <entity_type>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

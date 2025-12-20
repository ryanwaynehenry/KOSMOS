"""
Quick test helper to load an interim transcript JSON and populate KG nodes.

Usage:
  python scripts/build_nodes_from_interim.py data/interim/altered_session_2348_1.json
  python scripts/build_nodes_from_interim.py data/interim/altered_session_2348_1.json --output data/interim/file_with_nodes.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from clinical_kg.kg.builder import build_nodes
from clinical_kg.kg.schema import schema_for_entity_type, shacl_turtle


def _enrich_nodes(nodes: List[Dict[str, Any]], encounter_id: str) -> List[Dict[str, Any]]:
    """
    Ensure nodes have schema-backed attributes/options and stable ids.
    """
    enriched: List[Dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        schema = schema_for_entity_type(node.get("entity_type"))
        attributes = node.get("attributes") or {}
        filtered_attrs = {k: v for k, v in attributes.items() if k in schema.attribute_options}

        canonical = node.get("canonical_name") or node.get("name") or node.get("text")
        if schema.class_name == "Person" and canonical:
            filtered_attrs.setdefault("name", canonical)

        enriched.append(
            {
                **node,
                "id": node.get("id") or f"{encounter_id}_n{idx + 1:04d}",
                "class": node.get("class") or schema.class_name,
                "attribute_options": schema.attribute_options,
                "attributes": filtered_attrs,
            }
        )
    return enriched


def main():
    parser = argparse.ArgumentParser(description="Build KG nodes from an interim transcript JSON.")
    parser.add_argument("interim_path", help="Path to data/interim/<file>.json")
    parser.add_argument(
        "--output",
        help="Optional path to write the enriched JSON (defaults to stdout only).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore any existing nodes in the file and rebuild from mentions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="How many concepts to send to the LLM at once when building nodes.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM enrichment and fall back to deterministic node building.",
    )
    args = parser.parse_args()

    interim_path = Path(args.interim_path)
    with open(interim_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    encounter_id = payload.get("encounter_id") or interim_path.stem
    turns = payload.get("turns") or []
    mentions_or_entities = payload.get("mentions") or payload.get("nodes") or []
    existing_nodes = payload.get("nodes") or []

    if existing_nodes and not args.rebuild and args.no_llm:
        nodes = _enrich_nodes(existing_nodes, encounter_id=encounter_id)
    else:
        nodes = build_nodes(
            mentions_or_entities,
            encounter_id=encounter_id,
            turns=turns,
            use_llm=not args.no_llm,
            batch_size=max(1, args.batch_size),
        )

    payload["nodes"] = nodes
    payload["shacl_shapes_ttl"] = shacl_turtle()

    print(f"Loaded {len(mentions_or_entities)} grouped entities; produced {len(nodes)} nodes.")
    for node in nodes[:3]:
        print(json.dumps(node, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote enriched JSON to {out_path}")

        ttl_path = out_path.with_suffix(".shapes.ttl")
        with open(ttl_path, "w", encoding="utf-8") as f:
            f.write(shacl_turtle())
        print(f"Wrote SHACL shapes to {ttl_path}")


if __name__ == "__main__":
    main()


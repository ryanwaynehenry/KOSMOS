"""
Quick test helper to load an interim transcript JSON and populate KG nodes.

Usage:
  python scripts/build_nodes_from_interim.py data/interim/altered_session_2348_1.json
  python scripts/build_nodes_from_interim.py data/interim/altered_session_2348_1.json --output data/interim/file_with_nodes.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from clinical_kg.config import load_config
from clinical_kg.kg.builder import build_nodes
from clinical_kg.kg.schema import schema_for_entity_type, shacl_turtle
from clinical_kg.nlp.llm_client import call_llm_for_extraction

LLM_NODE_SYSTEM_PROMPT = """You are a clinical knowledge-graph node builder.

You will receive a JSON array of concepts. Each concept has:
- concept_id: unique string that must be echoed back exactly.
- canonical_name and entity_type: starting labels for the concept.
- attribute_options: the only attribute keys you may populate.
- mentions: span-level mentions (id, turn_id, text, type) for this concept.
- mention_context: per-mention transcript snippets that include the turn with the mention plus the immediately previous and next turns (if they exist).
- context_turns: deduplicated list of turns relevant to this concept for quick reference.
- ontology: (optional) code metadata; keep untouched.

Task:
For each concept, use ONLY the provided context (mention_context and context_turns) to infer node attributes.
- Fill attributes that are explicitly supported by attribute_options, guided by the provided attribute definitions/examples. Use short string values drawn from the transcript (age, severity, dose, frequency, status, onset, role, occupation, etc.).
- If information is absent, leave the attribute out (do not invent or guess).
- You may refine canonical_name if the transcript clearly suggests a better clinical name; otherwise keep it.
- Keep entity_type unless the transcript clearly indicates a better choice.
- Never merge or split concepts; return one output object per input concept in the SAME ORDER.
- Do not carry information across concepts; treat each concept independently.

Output STRICT JSON ONLY:
[
  {
    "concept_id": "<from input>",
    "canonical_name": "...",
    "entity_type": "...",
    "attributes": {
      "<attribute_option>": "<value from transcript>"
    }
  },
  ...
]

Return only the JSON array."""


def _attribute_definitions_for_schema(schema) -> List[Dict[str, Any]]:
    """Return ordered attribute definitions/examples for the schema's options."""
    defs: List[Dict[str, Any]] = []
    for opt in schema.attribute_options:
        spec = schema.attribute_definitions.get(opt)
        if spec:
            defs.append({"attribute": spec.name, "definition": spec.definition, "examples": spec.examples})
    return defs


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


def _turn_lookup(turns: List[Dict[str, Any]]) -> Dict[str, int]:
    return {t.get("turn_id"): idx for idx, t in enumerate(turns)}


def _turn_at(turns: List[Dict[str, Any]], idx: Optional[int]) -> Optional[Dict[str, Any]]:
    if idx is None or idx < 0 or idx >= len(turns):
        return None
    turn = turns[idx]
    return {
        "turn_id": turn.get("turn_id"),
        "speaker": turn.get("speaker"),
        "text": turn.get("text"),
    }


def _concept_context(
    concept: Dict[str, Any],
    turns: List[Dict[str, Any]],
    turn_index: Dict[str, int],
) -> Dict[str, Any]:
    """
    Build mention-level and aggregated context for a concept.

    For each mention, include the prior and following turn in case nearby context
    changes the meaning.
    """
    mention_contexts: List[Dict[str, Any]] = []
    context_turn_ids: List[Tuple[int, str]] = []

    for mention in concept.get("mentions") or []:
        tid = mention.get("turn_id")
        idx = turn_index.get(tid)
        turn = _turn_at(turns, idx)
        prev_turn = _turn_at(turns, idx - 1 if idx is not None else None)
        next_turn = _turn_at(turns, idx + 1 if idx is not None else None)

        mention_contexts.append(
            {
                "mention_id": mention.get("mention_id"),
                "turn_id": tid,
                "text": mention.get("text"),
                "turn": turn,
                "previous_turn": prev_turn,
                "next_turn": next_turn,
            }
        )

        for ctx_turn in (turn, prev_turn, next_turn):
            if ctx_turn and ctx_turn.get("turn_id") and ctx_turn.get("turn_id") in turn_index:
                context_turn_ids.append((turn_index[ctx_turn["turn_id"]], ctx_turn["turn_id"]))

    deduped_ids = []
    seen = set()
    for idx, tid in sorted(context_turn_ids, key=lambda pair: pair[0]):
        if tid in seen:
            continue
        seen.add(tid)
        deduped_ids.append(tid)

    context_turns = [_turn_at(turns, turn_index[tid]) for tid in deduped_ids if tid in turn_index]

    return {
        "mention_context": mention_contexts,
        "context_turns": context_turns,
    }


def _chunked(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _merge_llm_updates(
    concepts: List[Dict[str, Any]],
    updates: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for idx, concept in enumerate(concepts):
        concept_id = f"c{idx:04d}"
        update = updates.get(concept_id, {})
        merged_concept = {**concept}

        if isinstance(update, dict):
            if update.get("canonical_name"):
                merged_concept["canonical_name"] = update["canonical_name"]
            if update.get("entity_type"):
                merged_concept["entity_type"] = update["entity_type"]
            if isinstance(update.get("attributes"), dict):
                merged_concept["attributes"] = update["attributes"]

        merged.append(merged_concept)
    return merged


def build_nodes_with_llm(
    concepts: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    encounter_id: str,
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Use an LLM to build/enrich nodes in small batches with per-concept context.
    """
    cfg = load_config()
    turn_index = _turn_lookup(turns)
    llm_updates: Dict[str, Dict[str, Any]] = {}

    for batch_num, batch in enumerate(_chunked(concepts, batch_size), start=1):
        payload_for_llm = []
        for offset, concept in enumerate(batch):
            global_idx = (batch_num - 1) * batch_size + offset
            concept_id = f"c{global_idx:04d}"
            schema = schema_for_entity_type(concept.get("entity_type"))
            context = _concept_context(concept, turns, turn_index)
            attr_defs = _attribute_definitions_for_schema(schema)

            payload_for_llm.append(
                {
                    "concept_id": concept_id,
                    "canonical_name": concept.get("canonical_name") or concept.get("text"),
                    "entity_type": concept.get("entity_type"),
                    "attribute_options": schema.attribute_options,
                    "attribute_definitions": attr_defs,
                    "mentions": concept.get("mentions") or [],
                    "turn_ids": concept.get("turn_ids") or [],
                    "ontology": concept.get("ontology"),
                    **context,
                }
            )

        messages = [
            {"role": "system", "content": LLM_NODE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Each concept includes attribute_definitions (use only those listed for that concept) and attribute_options.\nConcept batch JSON:\n"
                + json.dumps(payload_for_llm, ensure_ascii=False, indent=2)
                + "\nReturn only JSON as specified.",
            },
        ]

        try:
            llm_response = call_llm_for_extraction(messages, cfg)
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            print(f"[llm] Batch {batch_num} failed, keeping originals: {exc}")
            continue

        if not isinstance(llm_response, list):
            print(f"[llm] Batch {batch_num} returned non-list; skipping.")
            continue

        for item in llm_response:
            if not isinstance(item, dict):
                continue
            cid = item.get("concept_id")
            if not cid:
                continue
            llm_updates[str(cid)] = item

    merged_concepts = _merge_llm_updates(concepts, llm_updates)
    return build_nodes(merged_concepts, encounter_id=encounter_id)


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
    elif args.no_llm:
        nodes = build_nodes(mentions_or_entities, encounter_id=encounter_id)
    else:
        nodes = build_nodes_with_llm(
            mentions_or_entities,
            turns,
            encounter_id=encounter_id,
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

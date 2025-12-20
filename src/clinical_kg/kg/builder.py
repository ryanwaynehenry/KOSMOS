"""
Utilities to turn grouped entities into knowledge-graph nodes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from clinical_kg.data_models import Mention
from clinical_kg.kg.schema import NodeSchema, schema_for_entity_type


LLM_NODE_SYSTEM_PROMPT = """You are a clinical knowledge-graph node builder.

You will receive a JSON array of concepts. Each concept has:
- concept_id: unique string that must be echoed back exactly.
- canonical_name and entity_type: starting labels for the concept.
- attribute_options: the only attribute keys you may populate.
- attribute_definitions: definitions/examples for attribute_options.
- mentions: span-level mentions for this concept.
- mention_context: per-mention transcript snippets that include the turn with the mention plus the immediately previous and next turns (if they exist).
- context_turns: deduplicated list of turns relevant to this concept for quick reference.
- ontology: (optional) code metadata; keep untouched.

Task:
For each concept, use ONLY the provided context (mention_context and context_turns) to infer node attributes.
- Fill attributes that are explicitly supported by attribute_options, guided by attribute_definitions.
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


def _as_dict(entity: Any) -> Dict[str, Any]:
    if isinstance(entity, Mention):
        return entity.__dict__
    return entity if isinstance(entity, dict) else {}


def _filter_attributes(entity: Dict[str, Any], schema: NodeSchema) -> Dict[str, Any]:
    attrs = entity.get("attributes") or {}
    if not schema.attribute_options or not isinstance(attrs, dict):
        return {}
    return {k: v for k, v in attrs.items() if k in schema.attribute_options}

def _entity_type(entity: Dict[str, Any]) -> Optional[str]:
    return entity.get("entity_type") or entity.get("type")


def _canonical_name(entity: Dict[str, Any]) -> Optional[str]:
    return entity.get("canonical_name") or entity.get("name") or entity.get("text")


def _turn_ids_from_mentions(mentions: Sequence[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    seen = set()
    for m in mentions:
        tid = m.get("turn_id")
        if tid and tid not in seen:
            ids.append(str(tid))
            seen.add(tid)
    return ids


def _normalize_turns(turns: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    if not turns:
        return []
    normalized: List[Dict[str, Any]] = []
    for t in turns:
        if isinstance(t, dict):
            turn_id = t.get("turn_id")
            speaker = t.get("speaker")
            text = t.get("text")
        else:
            turn_id = getattr(t, "turn_id", None)
            speaker = getattr(t, "speaker", None)
            text = getattr(t, "text", None)
        normalized.append({"turn_id": turn_id, "speaker": speaker, "text": text})
    return normalized


def _turn_lookup(turns: List[Dict[str, Any]]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, t in enumerate(turns):
        tid = t.get("turn_id")
        if tid:
            lookup[str(tid)] = idx
    return lookup


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

    mentions = concept.get("mentions") or []
    if isinstance(mentions, list):
        mention_list = [m if isinstance(m, dict) else {} for m in mentions]
    else:
        mention_list = []

    for mention in mention_list:
        tid = mention.get("turn_id")
        idx = turn_index.get(str(tid)) if tid is not None else None
        turn = _turn_at(turns, idx)
        prev_turn = _turn_at(turns, idx - 1 if idx is not None else None)
        next_turn = _turn_at(turns, idx + 1 if idx is not None else None)

        mention_contexts.append(
            {
                "mention_id": mention.get("mention_id"),
                "turn_id": tid,
                "text": mention.get("text"),
                "type": mention.get("type"),
                "turn": turn,
                "previous_turn": prev_turn,
                "next_turn": next_turn,
            }
        )

        for ctx_turn in (turn, prev_turn, next_turn):
            if ctx_turn and ctx_turn.get("turn_id"):
                ctx_tid = str(ctx_turn["turn_id"])
                if ctx_tid in turn_index:
                    context_turn_ids.append((turn_index[ctx_tid], ctx_tid))

    deduped_ids: List[str] = []
    seen = set()
    for _, tid in sorted(context_turn_ids, key=lambda pair: pair[0]):
        if tid in seen:
            continue
        seen.add(tid)
        deduped_ids.append(tid)

    context_turns = [_turn_at(turns, turn_index[tid]) for tid in deduped_ids if tid in turn_index]

    return {
        "mention_context": mention_contexts,
        "context_turns": context_turns,
    }


def _chunked(seq: Sequence[Any], size: int) -> Iterable[Tuple[int, List[Any]]]:
    for i in range(0, len(seq), size):
        yield i, list(seq[i : i + size])


def _attribute_definitions_for_schema(schema: NodeSchema) -> List[Dict[str, Any]]:
    defs: List[Dict[str, Any]] = []
    for opt in schema.attribute_options:
        spec = schema.attribute_definitions.get(opt)
        if spec:
            defs.append({"attribute": spec.name, "definition": spec.definition, "examples": spec.examples})
    return defs


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


def _enrich_concepts_with_llm(
    concepts: List[Dict[str, Any]],
    turns: Sequence[Any],
    batch_size: int = 5,
    llm_system_prompt: str = LLM_NODE_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """
    Use an LLM to fill schema-backed attributes for concepts in small batches.
    """
    normalized_turns = _normalize_turns(turns)
    if not normalized_turns:
        return concepts

    # Lazy import so deterministic builds don't require LLM deps.
    from clinical_kg.nlp.llm_client import call_llm_for_extraction

    turn_index = _turn_lookup(normalized_turns)
    llm_updates: Dict[str, Dict[str, Any]] = {}

    for start_idx, batch in _chunked(concepts, max(1, batch_size)):
        payload_for_llm: List[Dict[str, Any]] = []
        for offset, concept in enumerate(batch):
            global_idx = start_idx + offset
            concept_id = f"c{global_idx:04d}"

            etype = _entity_type(concept)
            schema = schema_for_entity_type(etype)
            context = _concept_context(concept, normalized_turns, turn_index)

            mentions = concept.get("mentions") or []
            mention_list = [m if isinstance(m, dict) else {} for m in mentions] if isinstance(mentions, list) else []

            turn_ids = concept.get("turn_ids") or _turn_ids_from_mentions(mention_list)

            payload_for_llm.append(
                {
                    "concept_id": concept_id,
                    "canonical_name": _canonical_name(concept),
                    "entity_type": etype,
                    "attribute_options": schema.attribute_options,
                    "attribute_definitions": _attribute_definitions_for_schema(schema),
                    "mentions": mention_list,
                    "turn_ids": turn_ids,
                    "ontology": concept.get("ontology"),
                    **context,
                }
            )

        messages = [
            {"role": "system", "content": llm_system_prompt},
            {
                "role": "user",
                "content": "Each concept includes attribute_definitions and attribute_options. Use only those attributes for that concept.\nConcept batch JSON:\n"
                + json.dumps(payload_for_llm, ensure_ascii=False, indent=2)
                + "\nReturn only JSON as specified.",
            },
        ]

        try:
            llm_response = call_llm_for_extraction(messages, cfg=None, label="node_build_llm_batch")
        except Exception as exc:
            print(f"[llm] Node enrichment batch starting at {start_idx} failed; keeping originals: {exc}")
            continue

        if not isinstance(llm_response, list):
            print(f"[llm] Node enrichment batch starting at {start_idx} returned non-list; skipping.")
            continue

        for item in llm_response:
            if not isinstance(item, dict):
                continue
            cid = item.get("concept_id")
            if cid:
                llm_updates[str(cid)] = item

    return _merge_llm_updates(concepts, llm_updates)


def _build_nodes_deterministic(entities: List[Any], encounter_id: str) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for idx, entity in enumerate(entities):
        data = _as_dict(entity)
        etype = _entity_type(data)
        schema = schema_for_entity_type(etype)

        canonical = _canonical_name(data)
        node_id = data.get("entity_id") or data.get("id") or f"{encounter_id}_n{idx + 1:04d}"

        attributes = _filter_attributes(data, schema)
        if schema.class_name == "Person" and canonical:
            attributes.setdefault("name", canonical)

        nodes.append(
            {
                "id": node_id,
                "encounter_id": encounter_id,
                "class": schema.class_name,
                "entity_type": etype,
                "shacl_shape_id": schema.shacl_shape.shape_id,
                "canonical_name": canonical,
                "ontology": data.get("ontology"),
                "ontology_strategy": data.get("ontology_strategy"),
                "turn_ids": data.get("turn_ids") or [],
                "mentions": data.get("mentions") or [],
                "attribute_options": schema.attribute_options,
                "attributes": attributes,
            }
        )

    return nodes


def build_nodes(
    entities: List[Any],
    encounter_id: str,
    turns: Optional[Sequence[Any]] = None,
    *,
    use_llm: bool = True,
    batch_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convert grouped entities (with ontology annotations) into KG node payloads.

    When use_llm=True and turns are provided, an LLM is used in batches to fill
    schema-backed node attributes using per-concept transcript context (mention turn
    plus previous/next turns). The final node payloads are still normalized by the
    deterministic builder.
    """
    if use_llm and turns:
        concept_dicts = [_as_dict(e) for e in entities]
        enriched = _enrich_concepts_with_llm(concept_dicts, turns, batch_size=batch_size)
        return _build_nodes_deterministic(enriched, encounter_id=encounter_id)
    return _build_nodes_deterministic(entities, encounter_id=encounter_id)

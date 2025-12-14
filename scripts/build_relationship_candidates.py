"""
Build relationship candidates from an interim JSON by:
1) computing an entity co-occurrence matrix using adjacent turns (neighbor overlap of 1),
2) batching entity pairs to an LLM with focused context to propose relationships.

Usage:
  python scripts/build_relationship_candidates.py data/interim/file_with_nodes2.json --output data/interim/file_with_relationships.json

Outputs:
  - Writes the updated payload with relationship_candidates to --output (or stdout if omitted).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from clinical_kg.config import load_config
from clinical_kg.kg.schema import schema_for_entity_type
from clinical_kg.nlp.llm_client import call_llm_for_extraction

RELATION_OPTIONS = [
    {"name": "has_condition", "description": "Person → Condition: Patient is documented as having the condition (active, chronic, resolved, etc.)."},
    {"name": "has_symptom", "description": "Person → Condition: Patient reports a symptom or complaint modeled as a Condition."},
    {"name": "has_diagnosis", "description": "Person → Condition: Clinician-assigned diagnosis for the patient (can be confirmed/provisional via attributes)."},
    {"name": "denies_condition", "description": "Person → Condition: Patient explicitly denies having the condition/symptom."},
    {"name": "has_observation", "description": "Person → Observation: Patient has a qualitative finding or non-lab observation (mood, general findings, vitals if modeled as Observation)."},
    {"name": "denies_observation", "description": "Person → Observation: Patient explicitly denies an observation (e.g., “no suicidal thoughts”)."},
    {"name": "has_lab_test", "description": "Person → LabTest: A lab or measurement is associated with the patient (may include results in node attributes)."},
    {"name": "has_medication", "description": "Person → Medication: Patient is taking/was prescribed the medication statement (dose, route, frequency in attributes)."},
    {"name": "has_activity", "description": "Person → Activity: Patient performs or has performed a behavior/activity (status current/former/never in attributes)."},
    {"name": "denies_activity", "description": "Person → Activity: Patient explicitly denies an activity/behavior."},
    {"name": "has_provider", "description": "Person(patient) → Person(clinician): Links the patient to their clinician/provider of care."},
    {"name": "evaluated_by", "description": "Person(patient) → Person(clinician): Clinician evaluated the patient in this context."},
    {"name": "documented_by", "description": "Condition/Medication/LabTest/Observation/Activity → Person(clinician): Clinician authored or documented the clinical fact."},
    {"name": "reported_by", "description": "Condition/Medication/LabTest/Observation/Activity → Person: Who stated the fact (patient or clinician)."},
    {"name": "diagnosed", "description": "Person(clinician) → Condition: Clinician diagnosed/assessed the condition."},
    {"name": "prescribed", "description": "Person(clinician) → Medication: Clinician prescribed or initiated the medication."},
    {"name": "ordered_test", "description": "Person(clinician) → LabTest: Clinician ordered the test/measurement."},
    {"name": "recommended", "description": "Person(clinician) → Activity or Medication: Clinician recommended an activity or medication change."},
    {"name": "indicated_for", "description": "Medication → Condition: Medication is intended to treat the condition (indication)."},
    {"name": "treats", "description": "Medication → Condition: Medication actively treats the condition."},
    {"name": "prevents", "description": "Medication → Condition: Medication is used to prevent the condition or complication."},
    {"name": "contraindicated_for", "description": "Medication → Condition: Medication should not be used with the condition."},
    {"name": "causes_adverse_effect", "description": "Medication → Condition or Observation: Medication caused a side effect or adverse event."},
    {"name": "worsens", "description": "Medication → Condition: Medication worsens the condition/symptoms."},
    {"name": "improves", "description": "Medication → Condition: Medication improves the condition/symptoms."},
    {"name": "interacts_with", "description": "Medication → Medication: Medication interaction is stated or clinically relevant."},
    {"name": "duplicate_therapy_with", "description": "Medication → Medication: Two meds are the same therapy class/redundant."},
    {"name": "replaces", "description": "Medication → Medication: One medication was substituted for another."},
    {"name": "supports_diagnosis_of", "description": "LabTest → Condition: Lab result supports the condition diagnosis."},
    {"name": "consistent_with", "description": "LabTest → Condition: Lab result is consistent with the condition."},
    {"name": "rules_out", "description": "LabTest → Condition: Lab result argues against/excludes the condition."},
    {"name": "monitors", "description": "LabTest → Condition: Lab test is used to monitor the condition over time."},
    {"name": "indicates_severity_of", "description": "LabTest → Condition: Lab result reflects severity of the condition."},
    {"name": "causes", "description": "Condition → Condition or Observation: One condition leads to another condition or a symptom/finding."},
    {"name": "risk_factor_for", "description": "Condition → Condition: One condition increases risk for another."},
    {"name": "complicates", "description": "Condition → Condition: One condition complicates the course/management of another."},
    {"name": "associated_with", "description": "Condition → Condition: Conditions are clinically associated without a clear causal claim."},
    {"name": "no_relation", "description": "No clinically meaningful relation in context."},
]

LLM_RELATION_SYSTEM = """You are a clinical relation extraction assistant.

You will receive:
- A list of entity pairs that co-occur in nearby turns (same or adjacent turn only).
- For each pair, the node metadata plus per-mention context (turn plus previous/next turns).
- A small set of allowed relation labels with definitions.

For EACH pair, decide if a clinically meaningful relation exists. If none, return "no_relation".

If a relation exists:
- Pick the single best relation label from the allowed list.
- Provide direction as "source -> target" (use the order given).
- Give a concise explanation grounded in the supplied context.
- List the turn_ids that support the relation (only turns provided).

Output STRICT JSON array, one item per pair in the SAME order:
[
  {
    "pair_id": "<pair_id from input>",
    "relation": "<one of the allowed relation names>",
    "direction": "source->target",
    "explanation": "...",
    "evidence_turn_ids": ["t0006", "t0007"]
  },
  ...
]

Return only JSON."""


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


def _node_turn_indices(node: Dict[str, Any], turn_index: Dict[str, int]) -> List[int]:
    indices = []
    for tid in node.get("turn_ids") or []:
        if tid in turn_index:
            indices.append(turn_index[tid])
    return sorted(set(indices))


def _adjacent_cooccurrence(indices_a: List[int], indices_b: List[int]) -> Tuple[int, int]:
    """
    Return (count, min_distance) for overlaps where abs(diff) <= 1.
    """
    b_set = set(indices_b)
    count = 0
    min_dist = None
    for ia in indices_a:
        for nb in (ia - 1, ia, ia + 1):
            if nb in b_set:
                count += 1
                dist = abs(ia - nb)
                min_dist = dist if min_dist is None else min(min_dist, dist)
    return count, (min_dist if min_dist is not None else 9999)


def _pair_context(node: Dict[str, Any], turns: List[Dict[str, Any]], turn_index: Dict[str, int]) -> Dict[str, Any]:
    schema = schema_for_entity_type(node.get("entity_type"))
    attr_defs = [
        {"attribute": spec.name, "definition": spec.definition, "examples": spec.examples}
        for spec in schema.attribute_definitions.values()
    ]
    mention_contexts: List[Dict[str, Any]] = []
    for mention in node.get("mentions") or []:
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
    return {
        "node_id": node.get("id"),
        "canonical_name": node.get("canonical_name"),
        "entity_type": node.get("entity_type"),
        "attributes": node.get("attributes") or {},
        "attribute_options": schema.attribute_options,
        "attribute_definitions": attr_defs,
        "mentions": mention_contexts,
    }


def _build_cooccurrence_pairs(nodes: List[Dict[str, Any]], turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turn_index = _turn_lookup(turns)
    node_turns = [(_node_turn_indices(n, turn_index)) for n in nodes]

    pairs: List[Dict[str, Any]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            count, min_dist = _adjacent_cooccurrence(node_turns[i], node_turns[j])
            if count > 0:
                pairs.append(
                    {
                        "pair_id": f"p{i:03d}_{j:03d}",
                        "source_node_id": nodes[i].get("id"),
                        "target_node_id": nodes[j].get("id"),
                        "cooccurrence_count": count,
                        "min_turn_distance": min_dist,
                    }
                )
    return pairs


def _batch(iterable: List[Any], size: int) -> List[List[Any]]:
    return [iterable[i : i + size] for i in range(0, len(iterable), size)]


def _llm_label_pairs(
    pairs: List[Dict[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    turns: List[Dict[str, Any]],
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    cfg = load_config()
    labeled: Dict[str, Dict[str, Any]] = {}
    turn_index = _turn_lookup(turns)

    for batch_pairs in _batch(pairs, batch_size):
        payload_pairs = []
        for pair in batch_pairs:
            src = nodes_by_id.get(pair["source_node_id"])
            tgt = nodes_by_id.get(pair["target_node_id"])
            if not src or not tgt:
                continue
            payload_pairs.append(
                {
                    "pair_id": pair["pair_id"],
                    "source": _pair_context(src, turns, turn_index),
                    "target": _pair_context(tgt, turns, turn_index),
                }
            )

        if not payload_pairs:
            continue

        messages = [
            {"role": "system", "content": LLM_RELATION_SYSTEM},
            {
                "role": "user",
                "content": "Allowed relations:\n"
                + json.dumps(RELATION_OPTIONS, ensure_ascii=False, indent=2)
                + "\n\nEntity pairs:\n"
                + json.dumps(payload_pairs, ensure_ascii=False, indent=2)
                + "\nReturn only JSON as specified.",
            },
        ]
        print(messages)

        try:
            resp = call_llm_for_extraction(messages, cfg)
        except Exception as exc:  # pragma: no cover
            print(f"[llm] relation batch failed: {exc}")
            continue

        if not isinstance(resp, list):
            continue
        for item in resp:
            if not isinstance(item, dict) or "pair_id" not in item:
                continue
            labeled[item["pair_id"]] = item

    results = []
    for pair in pairs:
        enriched = {**pair}
        if pair["pair_id"] in labeled:
            enriched["llm_relation"] = labeled[pair["pair_id"]]
        results.append(enriched)
    return results


def main():
    parser = argparse.ArgumentParser(description="Build relationship candidates from an interim transcript JSON.")
    parser.add_argument("interim_path", help="Path to data/interim/<file>.json")
    parser.add_argument(
        "--output",
        help="Optional path to write the enriched JSON (defaults to input path if omitted).",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="LLM batch size for relation labeling.")
    args = parser.parse_args()

    interim_path = Path(args.interim_path)
    with open(interim_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    turns = payload.get("turns") or []
    nodes = payload.get("nodes") or []
    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}

    pairs = _build_cooccurrence_pairs(nodes, turns)
    print(f"Identified {len(pairs)} co-occurring pairs (neighbor overlap <= 1 turn).")

    labeled_pairs = _llm_label_pairs(pairs, nodes_by_id, turns, batch_size=max(1, args.batch_size))
    payload["relationship_candidates"] = labeled_pairs

    out_path = Path(args.output) if args.output else interim_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote relationship candidates to {out_path}")


if __name__ == "__main__":
    main()

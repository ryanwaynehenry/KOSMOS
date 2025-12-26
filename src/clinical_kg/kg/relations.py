"""
LLM-assisted relationship candidate construction.

This module encapsulates the logic previously in scripts/build_relationship_candidates.py
so it can be reused from the pipeline and the CLI script.
"""

import json
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from clinical_kg.config import load_config
from clinical_kg.kg.schema import schema_for_entity_type
from clinical_kg.nlp.llm_client import call_llm_for_extraction

RELATION_OPTIONS = [
    {"name": "has_condition", "description": "Person(patient) -> Condition: Patient is documented as having the condition.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "has_symptom", "description": "Person(patient) -> Condition: Patient reports a symptom/complaint.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "has_diagnosis", "description": "Person(patient) -> Condition: Clinician-assigned diagnosis for the patient.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "denies_condition", "description": "Person(patient) -> Condition: Patient denies having the condition/symptom.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "has_observation", "description": "Person (patient or clinician) -> Observation: A qualitative finding or non-lab observation is associated with the patient.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT", "PERSON_CLINICIAN"], "target_classes": ["Observation"], "target_entity_types": ["OBS_VALUE", "OTHER", "UNIT"]},
    {"name": "denies_observation", "description": "Person (patient or clinician) -> Observation: Denial of an observation.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT", "PERSON_CLINICIAN"], "target_classes": ["Observation"], "target_entity_types": ["OBS_VALUE", "OTHER", "UNIT"]},
    {"name": "has_lab_test", "description": "Person(patient) -> LabTest: A lab or measurement is associated with the patient.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["LabTest"], "target_entity_types": ["LAB_TEST"]},
    {"name": "has_medication", "description": "Person(patient) -> Medication: Patient is taking/was prescribed the medication statement.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["MedicationStatement"], "target_entity_types": ["MEDICATION"]},
    {"name": "has_activity", "description": "Person(patient) -> Activity: Patient performs or has performed a behavior/activity.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Activity"], "target_entity_types": ["ACTIVITY"]},
    {"name": "denies_activity", "description": "Person(patient) -> Activity: Patient denies an activity/behavior.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Activity"], "target_entity_types": ["ACTIVITY"]},
    {"name": "has_procedure", "description": "Person(patient) -> Procedure: Patient underwent, is undergoing, or is scheduled for the procedure/intervention.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Procedure"], "target_entity_types": ["PROCEDURE"]},
    {"name": "has_provider", "description": "Person(patient) -> Person(clinician): Links the patient to their clinician/provider of care.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Person"], "target_entity_types": ["PERSON_CLINICIAN"]},
    {"name": "evaluated_by", "description": "Person(patient) -> Person(clinician): Clinician evaluated the patient in this context.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Person"], "target_entity_types": ["PERSON_CLINICIAN"]},
    {"name": "documented_by", "description": "Condition/Medication/LabTest/Observation/Activity/Procedure -> Person(clinician): Clinician authored or documented the clinical fact.", "source_classes": ["Condition", "MedicationStatement", "LabTest", "Observation", "Activity", "Procedure"], "target_classes": ["Person"], "target_entity_types": ["PERSON_CLINICIAN"]},
    {"name": "reported_by", "description": "Condition/Medication/LabTest/Observation/Activity/Procedure -> Person: Who stated the fact (patient or clinician).", "source_classes": ["Condition", "MedicationStatement", "LabTest", "Observation", "Activity", "Procedure"], "target_classes": ["Person"], "target_entity_types": ["PERSON_PATIENT", "PERSON_CLINICIAN"]},
    {"name": "diagnosed", "description": "Person(clinician) -> Condition: Clinician diagnosed/assessed the condition.", "source_classes": ["Person"], "source_entity_types": ["PERSON_CLINICIAN"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "prescribed", "description": "Person(clinician) -> Medication: Clinician prescribed or initiated the medication.", "source_classes": ["Person"], "source_entity_types": ["PERSON_CLINICIAN"], "target_classes": ["MedicationStatement"], "target_entity_types": ["MEDICATION"]},
    {"name": "ordered_test", "description": "Person(clinician) -> LabTest: Clinician ordered the test/measurement.", "source_classes": ["Person"], "source_entity_types": ["PERSON_CLINICIAN"], "target_classes": ["LabTest"], "target_entity_types": ["LAB_TEST"]},
    {"name": "recommended", "description": "Person(clinician) -> Activity, Medication, or Procedure: Clinician recommended an activity, procedure/therapy, or medication change.", "source_classes": ["Person"], "source_entity_types": ["PERSON_CLINICIAN"], "target_classes": ["Activity", "MedicationStatement", "Procedure"], "target_entity_types": ["ACTIVITY", "MEDICATION", "PROCEDURE"]},
    {"name": "indicated_for", "description": "Medication -> Condition: Medication is intended to treat the condition (indication).", "source_classes": ["MedicationStatement"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "treats", "description": "Medication -> Condition: Medication actively treats the condition.", "source_classes": ["MedicationStatement"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "prevents", "description": "Medication -> Condition: Medication is used to prevent the condition or complication.", "source_classes": ["MedicationStatement"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "contraindicated_for", "description": "Medication -> Condition: Medication should not be used with the condition.", "source_classes": ["MedicationStatement"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "causes_adverse_effect", "description": "Medication -> Condition or Observation: Medication caused a side effect or adverse event.", "source_classes": ["MedicationStatement"], "target_classes": ["Condition", "Observation"], "target_entity_types": ["PROBLEM", "OBS_VALUE", "OTHER", "UNIT"]},
    {"name": "worsens", "description": "Medication -> Condition: Medication worsens the condition/symptoms.", "source_classes": ["MedicationStatement"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "improves", "description": "Medication -> Condition: Medication improves the condition/symptoms.", "source_classes": ["MedicationStatement"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "interacts_with", "description": "Medication -> Medication: Medication interaction is stated or clinically relevant.", "source_classes": ["MedicationStatement"], "target_classes": ["MedicationStatement"]},
    {"name": "duplicate_therapy_with", "description": "Medication -> Medication: Two meds are the same therapy class/redundant.", "source_classes": ["MedicationStatement"], "target_classes": ["MedicationStatement"]},
    {"name": "replaces", "description": "Medication -> Medication: One medication was substituted for another.", "source_classes": ["MedicationStatement"], "target_classes": ["MedicationStatement"]},
    {"name": "supports_diagnosis_of", "description": "LabTest -> Condition: Lab result supports the condition diagnosis.", "source_classes": ["LabTest"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "consistent_with", "description": "LabTest -> Condition: Lab result is consistent with the condition.", "source_classes": ["LabTest"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "rules_out", "description": "LabTest -> Condition: Lab result argues against/excludes the condition.", "source_classes": ["LabTest"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "monitors", "description": "LabTest -> Condition: Lab test is used to monitor the condition over time.", "source_classes": ["LabTest"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "indicates_severity_of", "description": "LabTest -> Condition: Lab result reflects severity of the condition.", "source_classes": ["LabTest"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "causes", "description": "Condition -> Condition or Observation: One condition leads to another condition or a symptom/finding.", "source_classes": ["Condition"], "target_classes": ["Condition", "Observation"]},
    {"name": "risk_factor_for", "description": "Condition -> Condition: One condition increases risk for another.", "source_classes": ["Condition"], "target_classes": ["Condition"]},
    {"name": "complicates", "description": "Condition -> Condition: One condition complicates the course/management of another.", "source_classes": ["Condition"], "target_classes": ["Condition"]},
    {"name": "associated_with", "description": "Condition -> Condition: Conditions are clinically associated without a clear causal claim.", "source_classes": ["Condition"], "target_classes": ["Condition"]},
    {"name": "has_family_history_of", "description": "Person -> Condition: Patient reports a family history of the condition (does not imply the patient has it).", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "denies_family_history_of", "description": "Person -> Condition: Patient explicitly denies family history of the condition.", "source_classes": ["Person"], "source_entity_types": ["PERSON_PATIENT"], "target_classes": ["Condition"], "target_entity_types": ["PROBLEM"]},
    {"name": "no_relation", "description": "No clinically meaningful relation in context."},
]

LLM_RELATION_SYSTEM = """
You are an expert clinical relationship extractor. Given entity pairs and the surrounding conversation
turns, assign at most one relation per pair.

Use ONLY the relations provided in relation_options. Each option includes the allowed source/target
classes and entity_types. You MUST obey those constraints. If the entities do not match the allowed
source/target classes/entity_types for the direction you are asserting, respond with "no_relation".

Evidence standard
- Choose relations that are supported by the transcript, including information that is strongly implied
  by normal clinical conversation structure (for example, structured screening questions).
- Do not invent new facts. If the transcript does not support a link, output "no_relation".
- If there is ambiguity about who did something or whether something is present, prefer "no_relation".

Conversational inference rules (allowed)
1) Structured screening or checklist questions
- When a speaker asks a grouped, closed screening question that lists multiple symptoms or options,
  and the respondent answers in a way that clearly indicates a complete response (for example, they
  affirm only a subset, or they reject the set except for named positives), you may treat the
  non-endorsed options as negated.
- Apply this only when the response is clearly complete and not hedged.
- Do NOT apply this when the respondent appears unsure, gives a partial answer, changes the topic,
  or says they cannot recall.
- If your relation_options include a negation relation (for example denies_symptom, no_symptom,
  negative_for, absent), use it for the non-endorsed options.
- If your relation_options do not include a negation relation, do not force a positive link. Output
  "no_relation" for those non-endorsed pairs.

2) Direct yes/no answers
- If a question is clearly yes/no and the respondent clearly answers yes, you may treat the entity as
  present. If they clearly answer no, you may treat it as absent and use a negation relation when
  available.

3) Topic scoping
- If the clinician explicitly scopes a question to a specific timeframe (today, recently, since last visit),
  keep relations within that scope when possible. If timeframe is unclear, do not add one.

Clinician-attribution guardrails
- Only assign clinician -> condition relations like diagnosed, has_diagnosis, evaluated_by when the
  clinician is performing that assessment in this encounter (for example they state belief, assessment,
  diagnosis, or evaluation as their current conclusion).
- For historical conditions reported as prior diagnoses or long-standing history, link as patient -> condition
  using has_condition (or the closest allowed option) and optionally use reported_by or documented_by if
  your relation_options allow it. Do not attribute the diagnosis to the current clinician unless the
  transcript supports that.
- If it is unclear whether a clinician is diagnosing versus recounting prior history or chart history, avoid
  clinician -> condition diagnosis relations.

Clinician and observation pairs
- For clinician + observation pairs, consider has_observation with clinician as source when the clinician
  explicitly notes the observation during the encounter, or use documented_by with target->source
  direction when that is the correct option in relation_options.

Family history phrasing
- If a condition is mentioned in a family-history context, do not treat it as the patient having the
  condition unless the transcript supports that. Use a family-history relation if relation_options provides
  one. Otherwise prefer "no_relation" unless your schema explicitly encodes family-history conditions as
  patient conditions.

Speaker mapping
- speaker "D" or text starting with "D:" is the clinician speaking.
- speaker "P" or text starting with "P:" is the patient speaking.
Use the speaker to attribute clinician or patient actions even when their name is not repeated.

Direction
- Direction is relative to the provided source/target nodes ("source->target" or "target->source").
- Default to "source->target" unless the evidence clearly supports the reverse orientation.

Direction requirement (hard constraint)
- If a relation option matches only when direction is reversed (for example documented_by or reported_by),
  you MUST set direction to "target->source" rather than using "no_relation".

If there is no clinically meaningful relation supported by the evidence, output "no_relation".

Return a JSON array where each object has:
- pair_id
- relation (one of relation_options or "no_relation")
- direction ("source->target" or "target->source")
- explanation (brief rationale, note when you used a screening inference)
- evidence_turn_ids (list of turn_ids that support the choice)
"""


def _relation_lookup() -> Dict[str, Dict[str, Any]]:
    return {r["name"]: r for r in RELATION_OPTIONS}


def _matches_constraint(value: Optional[str], allowed: Optional[List[str]]) -> bool:
    if not allowed:
        return True
    return value in allowed


def _relation_allowed(
    relation: str,
    source_node: Dict[str, Any],
    target_node: Dict[str, Any],
    lookup: Dict[str, Dict[str, Any]],
    direction: str = "source->target",
) -> bool:
    if relation == "no_relation":
        return True
    rel_spec = lookup.get(relation)
    if not rel_spec:
        return False

    src = source_node
    tgt = target_node
    if direction == "target->source":
        src, tgt = tgt, src

    src_class_ok = _matches_constraint(src.get("class"), rel_spec.get("source_classes"))
    src_type_ok = _matches_constraint(src.get("entity_type"), rel_spec.get("source_entity_types"))
    tgt_class_ok = _matches_constraint(tgt.get("class"), rel_spec.get("target_classes"))
    tgt_type_ok = _matches_constraint(tgt.get("entity_type"), rel_spec.get("target_entity_types"))
    return src_class_ok and src_type_ok and tgt_class_ok and tgt_type_ok


def _turn_lookup(turns: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, Dict[str, Any]]]:
    by_id: Dict[str, int] = {}
    by_index: Dict[int, Dict[str, Any]] = {}
    for idx, turn in enumerate(turns):
        if hasattr(turn, "turn_id"):
            tid = getattr(turn, "turn_id", f"t{idx:04d}")
            turn_dict = {
                "turn_id": tid,
                "speaker": getattr(turn, "speaker", None),
                "text": getattr(turn, "text", None),
            }
        else:
            tid = turn.get("turn_id", f"t{idx:04d}")
            turn_dict = turn
        by_id[tid] = idx
        by_index[idx] = turn_dict
    return by_id, by_index


def _turn_at(turns_by_index: Dict[int, Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
    return turns_by_index.get(idx)


def _node_turn_indices(node: Dict[str, Any], turn_index_by_id: Dict[str, int]) -> List[int]:
    indices: List[int] = []
    for mention in node.get("mentions", []):
        tid = mention.get("turn_id")
        if tid is None:
            continue
        if tid not in turn_index_by_id:
            continue
        indices.append(turn_index_by_id[tid])
    return sorted(set(indices))


def _pair_id(src_id: str, tgt_id: str) -> str:
    src_suffix = src_id.split("_")[-1]
    tgt_suffix = tgt_id.split("_")[-1]
    return f"p{src_suffix}_{tgt_suffix}"


def _adjacent_cooccurrence(
    nodes_by_id: Dict[str, Dict[str, Any]],
    turn_index_by_id: Dict[str, int],
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    node_ids = sorted(nodes_by_id)
    turn_idx_cache = {nid: _node_turn_indices(nodes_by_id[nid], turn_index_by_id) for nid in node_ids}

    for i, src_id in enumerate(node_ids):
        for tgt_id in node_ids[i + 1 :]:
            src_turns = turn_idx_cache[src_id]
            tgt_turns = turn_idx_cache[tgt_id]
            if not src_turns or not tgt_turns:
                continue
            distances = [abs(s - t) for s in src_turns for t in tgt_turns]
            if not distances:
                continue
            min_dist = min(distances)
            if min_dist > 1:
                continue
            co_count = sum(1 for d in distances if d <= 1)
            pairs.append(
                {
                    "pair_id": _pair_id(src_id, tgt_id),
                    "source_node_id": src_id,
                    "target_node_id": tgt_id,
                    "cooccurrence_count": co_count,
                    "min_turn_distance": min_dist,
                }
            )
    return pairs


def _ensure_anchor_pairs(
    pairs: List[Dict[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    turn_index_by_id: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Guarantee candidate pairs from each patient and clinician to every other node.
    This ensures anchor nodes (patient/clinician) are considered with all entities,
    even if they are not adjacent in the transcript.
    """
    unordered = {frozenset({p["source_node_id"], p["target_node_id"]}) for p in pairs}
    anchor_ids = [
        nid
        for nid, n in nodes_by_id.items()
        if n.get("class") == "Person" and n.get("entity_type") in {"PERSON_PATIENT", "PERSON_CLINICIAN"}
    ]
    all_ids = list(nodes_by_id.keys())

    def _distances(a_ids: List[int], b_ids: List[int]) -> List[int]:
        return [abs(a - b) for a in a_ids for b in b_ids]

    for anchor_id in anchor_ids:
        for other_id in all_ids:
            if other_id == anchor_id:
                continue
            key = frozenset({anchor_id, other_id})
            if key in unordered:
                continue
            a_turns = _node_turn_indices(nodes_by_id[anchor_id], turn_index_by_id)
            o_turns = _node_turn_indices(nodes_by_id[other_id], turn_index_by_id)
            distances = _distances(a_turns, o_turns) if a_turns and o_turns else []
            min_dist = min(distances) if distances else None
            co_count = sum(1 for d in distances if d <= 1)
            new_pair = {
                "pair_id": _pair_id(anchor_id, other_id),
                "source_node_id": anchor_id,
                "target_node_id": other_id,
                "cooccurrence_count": co_count,
                "min_turn_distance": min_dist if min_dist is not None else None,
            }
            pairs.append(new_pair)
            unordered.add(key)
    return pairs


def _mention_turn_contexts(
    node: Dict[str, Any],
    turn_index_by_id: Dict[str, int],
    turns_by_index: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    turn_ids = [m.get("turn_id") for m in node.get("mentions", []) if m.get("turn_id") in turn_index_by_id]
    seen: set = set()
    contexts: List[Dict[str, Any]] = []
    for tid in sorted(turn_ids, key=lambda t: turn_index_by_id[t]):  # type: ignore[index]
        if tid in seen:
            continue
        seen.add(tid)
        idx = turn_index_by_id[tid]
        turn = _turn_at(turns_by_index, idx) or {}
        prev_turn = _turn_at(turns_by_index, idx - 1) or {}
        next_turn = _turn_at(turns_by_index, idx + 1) or {}
        contexts.append(
            {
                "turn_id": tid,
                "speaker": turn.get("speaker"),
                "text": turn.get("text"),
                "prev_text": prev_turn.get("text"),
                "next_text": next_turn.get("text"),
            }
        )
    return contexts


def _node_context(
    node: Dict[str, Any],
    turn_index_by_id: Dict[str, int],
    turns_by_index: Dict[int, Dict[str, Any]],
    include_turn_contexts: bool = True,
) -> Dict[str, Any]:
    schema = schema_for_entity_type(node.get("entity_type"))
    attr_defs = {
        name: {"definition": spec.definition, "examples": spec.examples}
        for name, spec in schema.attribute_definitions.items()
    }
    context = {
        "node_id": node.get("id"),
        "canonical_name": node.get("canonical_name"),
        "class": node.get("class"),
        "entity_type": node.get("entity_type"),
        "attributes": node.get("attributes", {}),
        "attribute_options": schema.attribute_options,
        "attribute_definitions": attr_defs,
        "mentions": node.get("mentions", []),
    }
    if include_turn_contexts:
        context["mention_turn_contexts"] = _mention_turn_contexts(
            node, turn_index_by_id, turns_by_index
        )
    return context


def _pair_context(
    pair: Dict[str, Any],
    nodes_by_id: Dict[str, Dict[str, Any]],
    turn_index_by_id: Dict[str, int],
    turns_by_index: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    src = nodes_by_id[pair["source_node_id"]]
    tgt = nodes_by_id[pair["target_node_id"]]
    return {
        "pair_id": pair["pair_id"],
        "cooccurrence_count": pair["cooccurrence_count"],
        "min_turn_distance": pair["min_turn_distance"],
        "source": _node_context(src, turn_index_by_id, turns_by_index, include_turn_contexts=False),
        "target": _node_context(tgt, turn_index_by_id, turns_by_index, include_turn_contexts=False),
    }


def _batch(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _batch_turn_contexts(
    batch_pairs: List[Dict[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    turn_index_by_id: Dict[str, int],
    turns_by_index: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    turn_indices: set = set()
    max_idx = len(turns_by_index) - 1
    for pair in batch_pairs:
        for node_id in (pair.get("source_node_id"), pair.get("target_node_id")):
            node = nodes_by_id.get(node_id or "")
            if not node:
                continue
            for idx in _node_turn_indices(node, turn_index_by_id):
                turn_indices.add(idx)
                if idx - 1 >= 0:
                    turn_indices.add(idx - 1)
                if idx + 1 <= max_idx:
                    turn_indices.add(idx + 1)
    contexts: List[Dict[str, Any]] = []
    for idx in sorted(turn_indices):
        turn = _turn_at(turns_by_index, idx) or {}
        contexts.append(
            {
                "turn_id": turn.get("turn_id"),
                "speaker": turn.get("speaker"),
                "text": turn.get("text"),
            }
        )
    return contexts


def _extract_results(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("pairs", "relations", "results", "relationship_candidates"):
            val = raw.get(key)
            if isinstance(val, list):
                return val
    raise ValueError(f"Unexpected LLM response format: {raw}")


def _llm_label_pairs(
    pairs: List[Dict[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    turns: List[Dict[str, Any]],
    cfg: Any,
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    if not pairs:
        return {}
    turn_index_by_id, turns_by_index = _turn_lookup(turns)
    rel_lookup = _relation_lookup()
    pair_map = {p["pair_id"]: p for p in pairs}
    labeled: Dict[str, Dict[str, Any]] = {}

    total_batches = math.ceil(len(pairs) / batch_size) if pairs else 0
    for batch_idx, batch in enumerate(_batch(pairs, batch_size), start=1):
        turn_contexts = _batch_turn_contexts(batch, nodes_by_id, turn_index_by_id, turns_by_index)
        batch_payload = {
            "instructions": (
                "Select at most one relation per pair from relation_options. "
                "Only choose a relation if the source/target classes and entity_types satisfy the allowed "
                "constraints in relation_options. "
                "Only assign clinician -> condition relations (diagnosed/has_diagnosis/evaluated_by) when the clinician diagnoses/assesses/suggests in this encounter; do not infer past diagnoses from patient history. "
                "Otherwise use 'no_relation'. "
                "Return a JSON array with pair_id, relation, direction, explanation, evidence_turn_ids."
            ),
            "relation_options": RELATION_OPTIONS,
            "turn_contexts": turn_contexts,
            "pairs": [
                _pair_context(p, nodes_by_id, turn_index_by_id, turns_by_index) for p in batch
            ],
        }
        messages = [
            {"role": "system", "content": LLM_RELATION_SYSTEM},
            {"role": "user", "content": json.dumps(batch_payload, ensure_ascii=False, indent=2)},
        ]

        start = time.time()
        raw = call_llm_for_extraction(messages, cfg, label="relationships_batch")
        elapsed = time.time() - start
        print(f"[relationships] LLM batch {batch_idx}/{total_batches} (size={len(batch)}) took {elapsed:.2f}s")
        try:
            rel_items = _extract_results(raw)
        except ValueError:
            continue

        for rel_obj in rel_items:
            pid = rel_obj.get("pair_id")
            if pid not in pair_map:
                continue
            relation = rel_obj.get("relation") or "no_relation"
            direction = rel_obj.get("direction", "source->target")
            if direction not in ("source->target", "target->source"):
                direction = "source->target"
            explanation = rel_obj.get("explanation", "")
            evidence_turn_ids = rel_obj.get("evidence_turn_ids", [])
            if not isinstance(evidence_turn_ids, list):
                evidence_turn_ids = []

            pair = pair_map[pid]
            src_node = nodes_by_id.get(pair["source_node_id"], {})
            tgt_node = nodes_by_id.get(pair["target_node_id"], {})
            if not _relation_allowed(relation, src_node, tgt_node, rel_lookup, direction=direction):
                relation = "no_relation"
                if not explanation:
                    explanation = (
                        "Relation not allowed for the provided source/target classes or entity types."
                    )

            labeled[pid] = {
                "pair_id": pid,
                "relation": relation,
                "direction": direction,
                "explanation": explanation,
                "evidence_turn_ids": evidence_turn_ids,
            }
    return labeled


def build_relationship_candidates(
    turns: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    cfg: Any = None,
    batch_size: int = 20,
) -> List[Dict[str, Any]]:
    """
    Construct relationship candidates for the provided turns and nodes.
    Returns a list of candidate dicts with llm_relation populated when available.
    """
    cfg = cfg or load_config()
    nodes_by_id = {n["id"]: n for n in nodes if "id" in n}
    turn_index_by_id, _ = _turn_lookup(turns)
    pairs = _adjacent_cooccurrence(nodes_by_id, turn_index_by_id)
    pairs = _ensure_anchor_pairs(pairs, nodes_by_id, turn_index_by_id)
    print(f"[relationships] Evaluating {len(pairs)} candidate pairs with batch size {batch_size}")

    llm_relations = _llm_label_pairs(pairs, nodes_by_id, turns, cfg, batch_size=batch_size)

    candidates: List[Dict[str, Any]] = []
    for pair in pairs:
        candidate = dict(pair)
        source_node = nodes_by_id.get(pair["source_node_id"], {})
        target_node = nodes_by_id.get(pair["target_node_id"], {})
        candidate["source_canonical_name"] = source_node.get("canonical_name")
        candidate["target_canonical_name"] = target_node.get("canonical_name")
        if pair["pair_id"] in llm_relations:
            candidate["llm_relation"] = llm_relations[pair["pair_id"]]
        candidates.append(candidate)
    return candidates

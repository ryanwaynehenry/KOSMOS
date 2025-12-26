"""
Grouping via LLM (replaces previous coref logic).
"""

import json
from typing import Dict, List, Optional, Tuple

from clinical_kg.config import PipelineConfig, load_config
from clinical_kg.data_models import Mention, Turn
from clinical_kg.nlp.llm_client import call_llm_for_extraction


GROUPING_PROMPT = """You are a clinical entity normalization assistant.

Inputs
1) A transcript of a clinician–patient conversation split into turns.
2) A JSON list of cleaned span-level mentions (already filtered to clinical content).

Each mention has:
- mention_id
- turn_id
- text
- type (coarse clinical label)

Task
Group mentions that refer to the same underlying entity concept and output:
- a list of entity objects
- a mention assignment log that covers every mention_id and explains what concept it was assigned to and why

Hard constraints
- Every input mention_id must appear exactly once in exactly one output entity’s mentions list.
- Do not invent new mentions. Use mention objects exactly as provided.
- Do not output commentary or markdown outside the required JSON object.

Temporal context rule (required, high priority)
- Do not create entities that are primarily time spans or temporal phrases.
  - No entities whose canonical_name is a time expression like "today", "yesterday", "last week", "last year", "in 2016", "for two months", "since Monday".
- Time spans are context, not standalone concepts.
- Any mention that is a time span, date, duration, timing phrase, or relative time must be grouped into the entity it modifies.
  - Example behaviors you must follow:
    - If a time span modifies a condition, symptom, procedure, test, or medication, attach the time span mention to that entity.
    - The entity canonical_name must be the clinical concept the time span is describing, not the time span.
- If a time span could apply to multiple entities, attach it to the closest clearly referenced concept in the same sentence or turn. If still ambiguous, attach it to the most clinically central concept in that turn.
- If a time span mention appears alone without a clear target, keep it attached to the nearest relevant concept mentioned adjacent in the transcript rather than making a time-only entity.

People rules
- Do not group people with non-people, except for the demographic and special family history bundle rule below.
- Demographics attachment rule
  - Do not create separate entities for patient demographics.
  - Any mention that is an age, sex, pregnancy or postpartum status, or race or ethnicity term that refers to a patient must be grouped into the patient entity.
- Family member handling
  - Mentions that are family member roles do not refer to the patient and must not be placed in the patient entity.
  - Treat the following as family member roles when present as standalone or role spans: father, mother, parent, sister, brother, sibling, child, son, daughter, spouse, husband, wife, partner, grandparent, grandmother, grandfather, aunt, uncle, cousin.

Diagnosis handling
- Always keep named conditions that function as diagnoses separate from symptom complaint entities, even if they are related.
  - Example: "pneumonia" must be its own entity and must not be merged into "cough" or "shortness of breath".
- Do not merge diagnosis entities with symptom complaint entities.

Medication regimen rule (required, high priority)
You must distinguish current medications from medications the clinician is prescribing or changing.

A) Medication regimen categories
Treat medication entities as regimen instances, not just drug names. Do not merge mentions across different regimen categories.
- current: medications the patient is currently taking or using as part of their existing regimen
- new: medications the clinician is starting or prescribing as a new regimen
- updated: medications where the clinician changes an existing regimen (dose change, frequency change, route change, formulation change)
- discontinued: medications the clinician stops or discontinues

B) Required separation behavior
- If a patient is taking a medication and the clinician changes it, you must output two separate medication entities:
  1) the current regimen entity
  2) the updated regimen entity
- If a clinician starts a medication the patient was not on, output a new regimen entity.
- If a clinician stops a medication, output a discontinued regimen entity.

C) Assigning medication-related mentions
- Group medication name, dose, frequency, route, and adherence mentions into the correct regimen entity.
- Do not leave dose or frequency as standalone entities when they clearly modify a medication, attach them to the medication entity for the correct regimen.
- If both an old dose and a new dose are present, keep old dose mentions with the current regimen entity and new dose mentions with the updated regimen entity.

D) Canonical naming for medication regimen entities
- canonical_name must include the regimen context so downstream steps can distinguish them, even when the drug name is the same.
Use this pattern:
  - "<medication name> (current)"
  - "<medication name> (new)"
  - "<medication name> (updated)"
  - "<medication name> (discontinued)"
- The medication name part should remain terminology-friendly (use the drug name if present).

E) Deciding regimen category from transcript structure
- current cues: patient states they take it, are on it, use it, have been taking it, continue it, or lists it as home medication
- new cues: clinician states they will start, prescribe, add, initiate, give, or provide it
- updated cues: clinician states increase, decrease, change dose, adjust, switch, titrate, or modify it
- discontinued cues: clinician states stop, discontinue, hold, or no longer take it
- If cues are unclear, prefer separate entities rather than merging, and explain the decision in the mention log.

Core grouping rules
- Group only when two mentions refer to the same entity concept.
- Prefer granular entities and avoid umbrella entities unless the umbrella term is explicitly stated.
- Keep qualifiers (severity, onset, course, negation, improving, worsening) attached to the entity they describe. Qualifiers alone are not a reason to merge different entities.
- Keep temporal phrases attached to the entity they describe and never let them determine canonical_name.

Tests and measurements rule (required)
- If a test or measurement name and its value refer to the same measurement, they must be grouped into a single test entity.
  - Example: "blood pressure" plus "135/85" belong to one entity.
- Canonical naming for test entities
  - canonical_name must be the test or measurement name, not the value.
- Entity typing override for test entities
  - If an entity contains any LAB_TEST mention, set entity_type = LAB_TEST even if OBS_VALUE mentions are included.

Physical exam grouping rule (system-level bundling)
When physical exam findings are present as multiple mentions, avoid over-fragmentation.
- If multiple OBS_VALUE mentions clearly belong to the same physical exam target in the same system statement, group them into a single entity representing that exam target.
  - Example: group "good air entry bilaterally" and "no wheezing" into one respiratory exam entity.
- Preferred anchor behavior
  - If a relevant exam anchor mention exists (often typed LAB_TEST in your pipeline, such as "lungs", "heart sounds", "abdomen", "pupils"), group the related exam findings into that same entity.
  - If no anchor mention exists, create one bundled entity using a system-level canonical_name such as:
    - "respiratory exam"
    - "cardiovascular exam"
    - "abdominal exam"
    - "neurological exam"
  - Do not create new mention spans. The bundling is done by grouping existing mentions only.

Anti-fragmentation rule for symptom and finding synonyms (narrow merge)
- Do not create separate entities for descriptive variants of the same symptom or finding concept only when both mentions explicitly share the same body region term in the mention text.
  - A merge is allowed only if both mentions include the same region token, such as "chest", "abdomen", "head", "knee".
  - Example: "chest pain" and "chest discomfort" must be merged (shared region token "chest").
  - Example: "tightness" and "chest pain" must NOT be merged unless both mention texts explicitly include the same region token.
- Do not merge symptoms across different anatomical sites.
- Do not merge distinct symptom families into one entity just because they co-occur.

Generic symptom rule (keep generic unless explicitly equated)
- Keep a generic symptom mention as its own entity unless the transcript clearly equates it to a specific complaint.

Family history bundle rule (required, special exception)
Family history must not be lost.
- When a family member role and a condition are stated together as family history, create a single family history bundle entity that groups:
  - the family member role mention(s)
  - the condition mention(s)
- Family history bundle canonical naming
  - canonical_name must follow: "family history of <condition>"
- Family history bundle typing override
  - Set entity_type = PROBLEM for family history bundle entities, even if they contain person-role mentions.
- Do not place family member role mentions into the patient entity when they are used in a family history bundle.
- Each mention_id still must appear exactly once overall.

Entity creation rule
- If a mention does not match any other mention, it must still form a single-mention entity by itself, except that time-only mentions must be attached to the entity they modify per the temporal context rule.

Final reconciliation step (required)
Before producing output, do a merge sweep:
- Consider merging two entities only if:
  - same entity_type
  - compatible meaning
  - and no anatomical site conflict
- Apply the narrow synonym merge rule for symptom and finding entities.
- Do not merge diagnosis entities with symptom entities.
- Do not merge family history bundle entities with patient condition entities.
- Do not merge medication entities across different regimen categories (current vs new vs updated vs discontinued), even if the drug name is the same.
- Do not merge lab test entities across different temporal categories (previous vs at the appointment vs future), even if the lab test name is the same.
  - Use the transcript context to determine whether a lab test/value refers to past results, current visit measurements, or future/ordered tests; keep those separate unless clearly the same event.

Entity fields
For each entity, output:
- canonical_name: concise name optimized for clinical terminologies when possible, but must follow the medication regimen naming rule and must not be a time expression
- entity_type: choose a coarse category using this rule order, with the overrides defined above applied first:
  - If entity is a family history bundle, entity_type = PROBLEM
  - Else if entity is a test entity containing any LAB_TEST mention, entity_type = LAB_TEST
  - Else if any mention type is PERSON_PATIENT, entity_type = PERSON_PATIENT
  - Else if any mention type is PERSON_CLINICIAN, entity_type = PERSON_CLINICIAN
  - Else if any mention type is MEDICATION, entity_type = MEDICATION
  - Else if any mention type is LAB_TEST, entity_type = LAB_TEST
  - Else if any mention type is PROCEDURE, entity_type = PROCEDURE
  - Else if any mention type is OBS_VALUE, entity_type = OBS_VALUE
  - Else if any mention type is PROBLEM, entity_type = PROBLEM
  - Else if any mention type is ACTIVITY, entity_type = ACTIVITY
  - Else entity_type = OTHER
- turn_ids: sorted unique list of turn_id values where this entity is mentioned
- mentions: the list of mention objects exactly as provided:
  - mention_id
  - turn_id
  - text
  - type

Mention assignment log (required)
You must also output a log that contains one entry for every input mention_id.
Each log entry must include:
- mention_id
- turn_id
- text
- type
- assigned_entity_index (0-based index into the entities list you output)
- assigned_entity_canonical_name
- assigned_entity_type
- rationale (1 to 2 sentences describing why this mention belongs in that entity, referencing the relevant rule)

Output format (strict)
Return strict JSON only with exactly two top-level keys:
- "entities": the list of entity objects
- "mention_log": the list of log entries, one per input mention_id

Return only the JSON object with keys "entities" and "mention_log".
"""

MERGE_PROMPT = """You are performing the FINAL cross-batch consolidation step for clinical entity normalization.

Context
- Entities were created in earlier batch runs. Those earlier steps already output full entities and ensured every mention was assigned to exactly one entity within each batch.
- This final step does NOT output rewritten entities or a mention log.
- This final step ONLY outputs which existing entity groups (by index) should be merged across batches.
- You will also receive the full preprocessed transcript (JSON turns). Use it as supporting context when reasoning about temporal attributes for lab tests and other time-sensitive mentions (past vs current vs future).

Input
- The full preprocessed transcript as a JSON list of turns (speaker/text).
- A numbered (0-based) list of current entities.
- Each entity includes: canonical_name, entity_type, mentions (with mention_id, turn_id, text, type), and temporal_context when available.

Task
- Identify entity groups that should be merged because they refer to the same underlying clinical concept.
- For each merge group, propose:
  1) the merged canonical_name
  2) the merged entity_type
  3) the list of entity_indices to combine
- Ensure that after merging there is at most one PERSON_PATIENT entity and at most one PERSON_CLINICIAN entity.

Hard constraints
- Suggest merges only. Do NOT rewrite, delete, or re-order entities in the input.
- Do NOT invent new mentions or remove mentions.
- Each input entity index may appear in at most one merge set (no overlapping merge sets).
- Do not output commentary or markdown outside the required JSON object.

Core merge rule
Merge entities only when they clearly represent the same underlying concept, not merely because they co-occur or are discussed together.

Do NOT merge rules (high priority)
- Do not merge diagnosis entities with symptom complaint entities.
- Do not merge entities that conflict on anatomical site.
- Do not merge distinct symptom families into one entity just because they co-occur.
- Do not merge family history bundle entities with the patient’s own condition entities.
- Do not merge medication entities across different regimen contexts (current vs new vs updated vs discontinued), even if the drug name is the same.
  - Regimen context may appear in canonical_name as "(current)", "(new)", "(updated)", "(discontinued)".
  - If regimen context is not present in canonical_name, infer it from mention texts and turn context (start, prescribe, increase, decrease, change, switch, stop, discontinue, continue, taking).
- Do not merge lab test entities across different temporal categories (previous vs at the appointment vs future), even if the lab test name is the same.
  - Use the transcript context and temporal_context values to determine whether a lab test/value refers to past results, current visit measurements, or future/ordered tests; keep those separate unless clearly the same event.
- Do not merge medication entities when temporal_context differs (for example, past home meds vs current visit meds vs planned/ordered); keep them separate unless the transcript explicitly equates them as the same regimen.

Allowed merge patterns (with typing rules)
A) Same-type same-concept merge (default)
- Allowed when entity_type matches and meaning matches with no site conflict.
- Output entity_type must stay the same as the inputs.

B) Person consolidation (required)
- Merge all PERSON_PATIENT entities into one patient entity.
- Merge all PERSON_CLINICIAN entities into one clinician entity.
- Output canonical_name should keep an existing person name when available; otherwise use "patient" or "clinician".

C) Narrow synonym merge for symptoms/findings (only when strict)
- You may merge descriptive variants of the same symptom or finding ONLY if BOTH sides explicitly contain the same body region token in the mention text (or clearly equivalent region token).
- Do not merge across different anatomical sites.

D) Test and measurement consolidation (cross-batch rescue)
- If one entity is the test name and another entity is the value for that same measurement, merge them.
- This merge is allowed even if the entity_types differ (commonly LAB_TEST vs OBS_VALUE).
- Output canonical_name must be the test or measurement name.
- Output entity_type must be LAB_TEST.

E) Physical exam bundling (cross-batch rescue)
- If multiple entities contain physical exam findings that clearly belong to the same exam target or system statement, merge to avoid over-fragmentation.
- If an exam anchor entity exists (often LAB_TEST like "lungs", "heart sounds", "abdomen"), merge related exam finding entities into it.
- Output canonical_name should be the anchor name if present, otherwise a system label like "respiratory exam", "cardiovascular exam", "abdominal exam", or "neurological exam".
- Output entity_type:
  - If the merged set includes any LAB_TEST mention, use LAB_TEST.
  - Otherwise use OBS_VALUE.

F) Patient demographics attachment (cross-batch rescue)
- If an entity contains only patient demographic descriptors and another entity is the patient person entity, merge them into the patient entity.
- Output canonical_name should remain the patient entity’s canonical_name.
- Output entity_type must be PERSON_PATIENT.

G) Family history bundle (cross-batch rescue, required)
- If a family member role entity and a condition entity are clearly part of the same family history statement, merge them into a single family history bundle.
- Output canonical_name must be: "family history of <condition>".
- Output entity_type must be PROBLEM.

H) Temporal-only entity rescue (cross-batch rescue, required)
- If an entity contains only time spans, dates, durations, or relative time phrases, it should not remain standalone.
- Merge a temporal-only entity into the single most clearly modified target entity when the linkage is obvious from nearby mention texts and turn context.
- Output canonical_name must be the target clinical concept, never the time phrase.
- If there is no clearly modified target, do not merge and leave it alone.

Canonical naming guidance
- Prefer concise terminology-appropriate names.
- Preserve specificity (include site when present).
- For merges into an existing anchor entity (patient entity, test entity, exam anchor entity), keep the anchor’s canonical_name unless a clearer equivalent is needed.
- For family history bundles, always use the required pattern.
- Never output a merged canonical_name that is primarily a time expression.

Output format (strict)
Return strict JSON only in this shape:
{
  "merge_sets": [
    {
      "canonical_name": "...",
      "entity_type": "...",
      "entity_indices": [0, 3, 5]
    }
  ]
}

- Use zero-based indices exactly as provided in the input list.
- Each merge set must contain 2 or more indices.
- If no merges are needed, output: {"merge_sets": []}
"""



def _mention_to_dict(mention: Mention) -> Dict[str, str]:
    if isinstance(mention, Mention):
        return {
            "mention_id": mention.mention_id,
            "turn_id": mention.turn_id,
            "text": mention.text,
            "type": mention.type,
        }
    return {
        "mention_id": mention.get("mention_id"),
        "turn_id": mention.get("turn_id"),
        "text": mention.get("text"),
        "type": mention.get("type"),
    }


def _turn_id_of(mention: Mention) -> Optional[str]:
    if isinstance(mention, Mention):
        return mention.turn_id
    return mention.get("turn_id")


def _batch_mentions_by_turn(mentions: List[Mention], target_size: int = 20) -> List[List[Mention]]:
    """
    Split mentions into batches aiming for target_size, but only split when turn_id changes.
    """
    batches: List[List[Mention]] = []
    current: List[Mention] = []
    for mention in mentions:
        if not current:
            current.append(mention)
            continue
        if len(current) >= target_size and _turn_id_of(mention) != _turn_id_of(current[-1]):
            batches.append(current)
            current = [mention]
        else:
            current.append(mention)
    if current:
        batches.append(current)
    return batches


def _annotate_temporal_context(
    entities: List[dict],
    turns: List[Turn],
    cfg: PipelineConfig,
) -> None:
    """
    Use an LLM to assign temporal_context ("past" | "current" | "future" | "uncertain")
    to lab test and medication entities based on the full preprocessed turns.
    """
    relevant = [
        {
            "index": idx,
            "canonical_name": ent.get("canonical_name"),
            "entity_type": ent.get("entity_type"),
            "mentions": ent.get("mentions", []),
        }
        for idx, ent in enumerate(entities)
        if (ent.get("entity_type") or "").upper() in {"LAB_TEST", "MEDICATION"}
    ]
    if not relevant:
        return

    turn_payload = [
        {"turn_id": t.turn_id, "speaker": t.speaker, "text": t.text}
        for t in turns
    ]

    prompt = (
        "You will receive:\n"
        "1) A list of entities (lab tests or medications) with indices.\n"
        "2) The full preprocessed transcript turns (speaker/text).\n\n"
        "For EACH entity, assign a temporal_context from this set:\n"
        "  - past: refers to prior history, previous results/meds, or earlier measurements.\n"
        "  - current: refers to this visit/encounter's measurement or the medication the patient is currently taking.\n"
        "  - future: refers to ordered/pending/follow-up/future measurements or medications to be started/changed.\n"
        "  - uncertain: not enough information to decide.\n\n"
        "Use the transcript context (turn text) to make the decision. Be conservative: if unclear, choose uncertain.\n"
        "Output strict JSON with key 'temporal_contexts' as a list of objects:\n"
        "[ { \"index\": <entity_index>, \"temporal_context\": \"past|current|future|uncertain\" } ]\n"
    )

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Entities:\n"
            + json.dumps(relevant, ensure_ascii=False, indent=2)
            + "\n\nTurns:\n"
            + json.dumps(turn_payload, ensure_ascii=False, indent=2),
        },
    ]

    try:
        raw = call_llm_for_extraction(messages, cfg, label="temporal_context_llm")
        contexts = []
        if isinstance(raw, dict):
            maybe_list = raw.get("temporal_contexts") or raw.get("contexts") or raw.get("items")
            if isinstance(maybe_list, list):
                contexts = maybe_list
        elif isinstance(raw, list):
            contexts = raw

        for item in contexts:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            ctx = item.get("temporal_context")
            if isinstance(idx, int) and 0 <= idx < len(entities) and isinstance(ctx, str):
                if ctx.lower() in {"past", "current", "future", "uncertain"}:
                    entities[idx]["temporal_context"] = ctx.lower()
    except Exception as exc:
        print(f"[coref] temporal_context LLM failed; leaving contexts unchanged: {exc}")


def _run_grouping_llm(
    mentions: List[Mention],
    turn_texts: List[str],
    cfg: PipelineConfig,
    label: str,
) -> Tuple[List[dict], List[dict]]:
    mention_payload = [_mention_to_dict(m) for m in mentions]
    messages = [
        {"role": "system", "content": GROUPING_PROMPT},
        {
            "role": "user",
            "content": "Turns:\n"
            + "\n".join(turn_texts)
            + "\n\nMentions (JSON):\n"
            + json.dumps(mention_payload, ensure_ascii=False, indent=2),
        },
    ]

    entities: List[dict] = []
    mention_assignment_log: List[dict] = []
    try:
        raw_result = call_llm_for_extraction(messages, cfg, label=label)
        if isinstance(raw_result, dict):
            mention_assignment_log = raw_result.get("mention_assignment_log", []) or []
            maybe_entities = raw_result.get("entities", [])
            if isinstance(maybe_entities, list):
                entities = maybe_entities
        elif isinstance(raw_result, list):
            entities = raw_result
    except Exception:
        entities = []
    return entities, mention_assignment_log


def _suggest_merges(
    entities: List[dict],
    cfg: PipelineConfig,
    transcript_payload: Optional[object] = None,
) -> List[dict]:
    payload = [
        {
            "index": idx,
            "canonical_name": ent.get("canonical_name"),
            "entity_type": ent.get("entity_type"),
            "mentions": ent.get("mentions", []),
            "temporal_context": ent.get("temporal_context"),
        }
        for idx, ent in enumerate(entities)
    ]
    transcript_block = ""
    if transcript_payload is not None:
        transcript_block = "Transcript:\n" + json.dumps(transcript_payload, ensure_ascii=False, indent=2) + "\n\n"
    messages = [
        {"role": "system", "content": MERGE_PROMPT},
        {
            "role": "user",
            "content": transcript_block + "Entities:\n" + json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]
    try:
        raw = call_llm_for_extraction(messages, cfg, label="coref_merge_suggestions")
        if isinstance(raw, dict):
            merge_sets = raw.get("merge_sets")
            if isinstance(merge_sets, list):
                return merge_sets
    except Exception:
        pass
    return []


def _apply_merge_sets(entities: List[dict], merge_sets: List[dict]) -> List[dict]:
    merged_indices: set = set()
    result: List[dict] = []

    for merge in merge_sets:
        idxs = merge.get("entity_indices") or []
        if not isinstance(idxs, list):
            continue
        valid_indices = [i for i in idxs if isinstance(i, int) and 0 <= i < len(entities)]
        if len(valid_indices) < 2:
            continue
        # collect entities to merge
        to_merge = [entities[i] for i in valid_indices]
        if not to_merge:
            continue
        temporal_contexts = {
            ent.get("temporal_context")
            for ent in to_merge
            if isinstance(ent, dict)
            and (ent.get("entity_type") or "").upper() in {"LAB_TEST", "MEDICATION"}
            and ent.get("temporal_context")
        }
        if len(temporal_contexts) > 1:
            # Conflicting temporal contexts for labs/meds; skip this merge suggestion.
            continue
        merged_indices.update(valid_indices)
        canonical_name = merge.get("canonical_name") or to_merge[0].get("canonical_name")
        entity_type = to_merge[0].get("entity_type")

        merged_mentions: List[dict] = []
        seen_mentions: set = set()
        merged_turns: set = set()
        for ent in to_merge:
            for tid in ent.get("turn_ids", []):
                merged_turns.add(tid)
            for m in ent.get("mentions", []):
                mid = m.get("mention_id")
                key = mid or (m.get("turn_id"), m.get("text"))
                if key in seen_mentions:
                    continue
                seen_mentions.add(key)
                merged_mentions.append(m)
                if m.get("turn_id"):
                    merged_turns.add(m["turn_id"])

        merged_entity = {
            "canonical_name": canonical_name,
            "entity_type": entity_type,
            "turn_ids": sorted(merged_turns),
            "mentions": merged_mentions,
        }
        result.append(merged_entity)

    for idx, ent in enumerate(entities):
        if idx in merged_indices:
            continue
        result.append(ent)

    return result


def add_coref_clusters(
    mentions: List[Mention],
    turns: List[Turn],
    cfg: Optional[PipelineConfig] = None,
    use_llm_refinement: bool = True,
    preprocessed_transcript: Optional[str] = None,
) -> List[Mention]:
    """
    Entry point for grouping. Calls an LLM with the grouping prompt.
    Currently returns mentions unchanged; consume grouping output separately if needed.

    preprocessed_transcript: optional full transcript string (already preprocessed); if absent,
    a transcript is reconstructed from the provided turns.
    """
    if not use_llm_refinement:
        return mentions

    cfg = cfg or load_config()
    mention_assignment_log: List[dict] = []  # captured for debugging visibility
    turn_texts = [f"{t.turn_id} ({t.speaker}): {t.text}" for t in turns]
    transcript_payload = (
        preprocessed_transcript
        if preprocessed_transcript is not None
        else [{"turn_id": t.turn_id, "speaker": t.speaker, "text": t.text} for t in turns]
    )

    batches = _batch_mentions_by_turn(mentions, target_size=20)
    batch_entities: List[dict] = []
    for idx, batch in enumerate(batches, start=1):
        entities, log = _run_grouping_llm(
            batch,
            turn_texts=turn_texts,
            cfg=cfg,
            label=f"coref_grouping_batch_{idx}_of_{len(batches)}",
        )
        if log:
            mention_assignment_log.extend(log)
        if entities:
            batch_entities.extend(entities)

    if not batch_entities:
        return mentions

    _annotate_temporal_context(batch_entities, turns, cfg)

    merge_sets = _suggest_merges(batch_entities, cfg, transcript_payload=transcript_payload)
    merged_entities = _apply_merge_sets(batch_entities, merge_sets) if merge_sets else batch_entities

    return merged_entities

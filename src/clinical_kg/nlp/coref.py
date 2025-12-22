"""
Grouping via LLM (replaces previous coref logic).
"""

import json
from typing import List, Optional

from clinical_kg.config import PipelineConfig, load_config
from clinical_kg.data_models import Mention, Turn
from clinical_kg.nlp.llm_client import call_llm_for_extraction


GROUPING_PROMPT = """You are a clinical entity normalization assistant.

Your input will be:
1. A transcript of a clinician–patient conversation split into turns.
2. A JSON list of cleaned span-level mentions (already filtered to clinical content).

Each mention has:
- mention_id
- turn_id
- text
- type (coarse clinical label)

Your task is to group mentions that refer to the same underlying entity and produce a list of entity objects.

Non-dropping requirement (critical)
- The mentions list has already been filtered to clinical content.
- Therefore, you must NOT discard mentions based on your own judgment of relevance.
- Every input mention must appear in exactly one output entity’s mentions list.
- Only exclude a mention if it is clearly a greeting or small talk AND the mention type is OTHER (or an equivalent explicitly non-clinical type). Otherwise keep it.

Special retention rules (do not violate)
A) Physical exam and observed findings must be preserved
- Any mention with type OBS_VALUE is clinically meaningful and must be retained.
- Create an entity for each distinct finding concept, even if it appears only once.
- Examples of what counts: lung exam findings, abdominal exam findings, mental status observations, normal findings, negative findings, and descriptive exam phrases.

B) Plan and follow-up content must be preserved
- Mentions representing the plan going forward must be retained, even if short or generic.
- This includes follow-up timing, referrals, counseling, orders, vaccinations, medications to start or stop, and monitoring instructions.
- Mentions typed PROCEDURE, LAB_TEST, MEDICATION, and PERSON_CLINICIAN can all be plan-related. Do not drop them.

Grouping rules
- Group only when two mentions refer to the same entity concept.
- Prefer granular entities. Do not create umbrella entities unless the umbrella term is explicitly stated.
- Separate distinct symptoms or findings even if they plausibly share a cause.
- If a diagnosis term is explicitly stated, create a separate diagnosis entity for it, and keep symptom entities separate.
- Keep qualifiers (severity, onset, course, negation, improving, worsening) attached to the entity they describe. Qualifiers are not a reason to merge different entities.
- Preserve test and measurement entities:
  - If a test name and its value occur (for example a measurement name plus a numeric result), group them into the same test entity.
  - The canonical_name should be the test or measurement name, not the value.

People separation rule
- Do NOT group people with non-people.
- The patient entity should contain only person references and demographic descriptors about the patient (age, sex, pregnancy status, ethnicity when present).
- The clinician entity should contain only the encounter clinician’s references or roles.
- If a specialist role appears as part of a referral plan (for example a therapist role), keep it as its own person-role entity and do not merge it with the encounter clinician.

Demographic attachment rule (required)
- Do not create separate entities for patient demographics.
- Any mention that is an age, sex, pregnancy or postpartum status, or race or ethnicity term must be grouped into the patient person entity.
- This is the only exception to “do NOT group people with non-people”.

Patient entity rule (required)
- Always output exactly one patient entity per encounter.
- Put every PERSON_PATIENT mention and every patient demographic mention into that same patient entity whenever they refer to the patient.
- If no patient name is present, use canonical_name = "patient".

Entity creation requirement
- If a mention does not match any other mention, it still must form a single-mention entity by itself.
- When in doubt, keep separate entities rather than merging.

Entity fields
For each entity, create:
- canonical_name: concise name optimized for clinical terminologies (SNOMED CT, RxNorm, LOINC). Prefer standard phrasing.
- entity_type: choose a coarse category based on the types of its mentions. Use this deterministic rule:
  - If any mention type is PERSON_PATIENT -> entity_type = PERSON_PATIENT
  - Else if any mention type is PERSON_CLINICIAN -> entity_type = PERSON_CLINICIAN
  - Else if any mention type is MEDICATION -> entity_type = MEDICATION
  - Else if any mention type is LAB_TEST -> entity_type = LAB_TEST
  - Else if any mention type is PROCEDURE -> entity_type = PROCEDURE
  - Else if any mention type is OBS_VALUE -> entity_type = OBS_VALUE
  - Else if any mention type is PROBLEM -> entity_type = PROBLEM
  - Else if any mention type is ACTIVITY -> entity_type = ACTIVITY
  - Else entity_type = OTHER
- turn_ids: list of turn_id values where this entity is mentioned.
- mentions: list of mention objects exactly as provided:
  - mention_id
  - turn_id
  - text
  - type

Output STRICT JSON ONLY:
[
  {
    "canonical_name": "...",
    "entity_type": "<category>",
    "turn_ids": ["t1", "t3"],
    "mentions": [
      {"mention_id": "m0001", "turn_id": "t1", "text": "...", "type": "..."},
      {"mention_id": "m0002", "turn_id": "t3", "text": "...", "type": "..."}
    ]
  }
]

Output rules
- Every input mention must appear exactly once inside the mentions list of exactly one output entity.
- Do not output any commentary, explanations, or markdown.
- Return only the JSON array.
"""


def add_coref_clusters(
    mentions: List[Mention],
    turns: List[Turn],
    cfg: Optional[PipelineConfig] = None,
    use_llm_refinement: bool = True,
) -> List[Mention]:
    """
    Entry point for grouping. Calls an LLM with the grouping prompt.
    Currently returns mentions unchanged; consume grouping output separately if needed.
    """
    if not use_llm_refinement:
        return mentions

    cfg = cfg or load_config()
    turn_texts = [f"{t.turn_id} ({t.speaker}): {t.text}" for t in turns]
    mention_payload = [
        {"mention_id": m.mention_id, "turn_id": m.turn_id, "text": m.text, "type": m.type}
        for m in mentions
    ]

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

    try:
        mentions = call_llm_for_extraction(messages, cfg, label="coref_grouping")
    except Exception:
        pass

    return mentions

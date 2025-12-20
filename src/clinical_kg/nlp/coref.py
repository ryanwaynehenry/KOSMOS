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

Your task is to group mentions that refer to the same entity and produce a list of entity objects.

Rules:
- Do NOT group people with non-people. The patient entity should only contain person references and demographic descriptors about the patient (age, sex, pregnancy status). The clinician entity should only contain clinician references/roles. Do not merge the patient with symptoms, medications, allergies, or the clinician.
- Preserve test and measurement entities. If a test like “blood pressure” is mentioned and a value “120 over 80” appears, they belong to the same test entity (multiple mentions for the same test). Keep both the test name and its value mentions.
- Prefer granular symptom/finding grouping. Only group mentions that refer to the same clinical concept; avoid creating umbrella entities like “<condition> symptoms” unless that umbrella concept is explicitly stated in the transcript.
- Separate distinct symptoms/findings even when they plausibly share a cause. Create one entity per distinct symptom/finding, and let downstream relationship steps connect them if appropriate.
- If a diagnosis/disorder name is explicitly stated, create a separate diagnosis entity for that term, and keep symptom/finding entities separate.
- Keep qualifiers (severity, onset, course/progression, negation, “improving/worsening”, etc.) attached to the specific entity they describe; do not use qualifiers as a reason to merge different symptoms into one entity.
- Keep all clinically meaningful mentions; do not drop mentions just because they repeat earlier ones.
- Distinguish different entities even if they share similar words.

For each entity, create:
- canonical_name: a concise, human-readable name optimized for clinical terminologies (SNOMED CT/RxNorm/LOINC). Prefer standard clinical phrasing over colloquial wording.
- entity_type: a coarse category based on the types of its mentions.
- turn_ids: list of turn_id values where this entity is mentioned.
- mentions: a list of the individual references:
    - mention_id
    - turn_id
    - text
    - type

Ignore non-clinical entities (pure greetings/small talk).

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
  },
  ...
]

Return only the JSON list."""


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

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

Follow these steps internally, but only output the final JSON:

1. Group mentions into entities:
- Mentions that refer to the same condition, symptom, problem, diagnosis,
  medication, test, or other clinical item should belong to the same entity.
- Include pronouns and short references when the referent is clear
  (e.g., “it”, “this medication”, “that problem”, “your blood pressure”).
- Distinguish different entities even if they share similar words.

2. For each entity, create:
- canonical_name: a concise, human-readable name optimized for clinical terminologies (SNOMED CT/RxNorm/LOINC). Prefer standard clinical phrasing over colloquial or subjective wording (e.g., use “depressive disorder” instead of “feeling down”; “shortness of breath” instead of “winded”; “blood pressure” instead of “my pressure”; “lisinopril” instead of “BP pill”). For non-clinical items like names or professions, use the clearest literal noun phrase (e.g., “librarian”) even if it will not be in a terminology.
- entity_type: a coarse category based on the types of its mentions.
- turn_ids: list of turn_id values where this entity is mentioned.
- mentions: a list of the individual references:
    - mention_id
    - turn_id
    - text
    - type


3. Ignore non-clinical entities (pure greetings/small talk).

4. Output STRICT JSON ONLY:
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
        mentions = call_llm_for_extraction(messages, cfg)
    except Exception:
        pass

    return mentions

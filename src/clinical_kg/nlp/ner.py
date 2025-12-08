"""
Lightweight NER placeholder.

This module turns raw turns into mentions using simple rules. Replace the
stubbed logic with a real NER model when available.
"""

import re
from typing import Dict, List

from clinical_kg.data_models import Mention, Turn

_MENTION_ID_TEMPLATE = "m{index:04d}"

# Simple label normalization map (model label -> canonical type)
LABEL_MAP: Dict[str, str] = {
    "DISEASE": "PROBLEM",
    "PROBLEM": "PROBLEM",
    "DRUG": "MEDICATION",
    "MED": "MEDICATION",
    "MEDICATION": "MEDICATION",
    "TEST": "LAB_TEST",
    "LAB_TEST": "LAB_TEST",
    "UNIT": "UNIT",
    "DOSE": "DOSE_AMOUNT",
    "DOSE_AMOUNT": "DOSE_AMOUNT",
    "FREQ": "FREQUENCY",
    "FREQUENCY": "FREQUENCY",
    "PERSON_PATIENT": "PERSON_PATIENT",
}

# Very basic keyword patterns to bootstrap mentions before a real model exists.
_KEYWORD_PATTERNS = {
    "MEDICATION": [r"\blisinopril\b", r"\bmotrin\b", r"\bmetformin\b"],
    "PROBLEM": [r"\bchest pain\b", r"\bcough\b", r"\bdiabetes\b"],
    "LAB_TEST": [r"\bhemoglobin\b", r"\bhba1c\b", r"\bcbc\b"],
    "UNIT": [r"\bmg/dl\b", r"\bmg\b", r"\bmcg\b"],
    "DOSE_AMOUNT": [r"\b\d+\s*(mg|mcg|g|units)\b"],
    "FREQUENCY": [r"\bonce daily\b", r"\btwice a day\b", r"\bbid\b", r"\btid\b"],
}


def _normalize_type(label: str) -> str:
    return LABEL_MAP.get(label.upper(), label.upper())


def _find_pattern_mentions(turn: Turn, next_id: int) -> List[Mention]:
    mentions: List[Mention] = []
    for mtype, patterns in _KEYWORD_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, turn.text, flags=re.IGNORECASE):
                mention_id = _MENTION_ID_TEMPLATE.format(index=next_id)
                next_id += 1
                mentions.append(
                    Mention(
                        mention_id=mention_id,
                        encounter_id=turn.encounter_id,
                        turn_id=turn.turn_id,
                        start_char=match.start(),
                        end_char=match.end(),
                        text=turn.text[match.start() : match.end()],
                        type=mtype,
                        confidence=None,
                    )
                )
    return mentions


def extract_mentions(turns: List[Turn]) -> List[Mention]:
    """
    Run NER and simple tagging over the list of Turns.

    Replace the current heuristic logic with a proper model as needed.
    """
    mentions: List[Mention] = []
    counter = 1
    for turn in turns:
        detected = _find_pattern_mentions(turn, counter)
        counter += len(detected)
        mentions.extend(detected)
    return mentions

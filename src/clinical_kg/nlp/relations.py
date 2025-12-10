"""
Heuristic attachment of local attributes to mentions.

This module operates purely at the mention level and does not depend on UMLS
or graph structures.
"""

import re
from typing import List, Optional

from clinical_kg.data_models import Mention, MentionType


def _split_dose(text: str) -> tuple[Optional[str], Optional[str]]:
    match = re.match(r"^\s*(\d+(?:\.\d+)?)(.*)$", text)
    if not match:
        return None, None
    value = match.group(1).strip()
    unit = match.group(2).strip() or None
    return value, unit


def _nearest_host(attr: Mention, hosts: List[Mention]) -> Optional[Mention]:
    same_encounter = [h for h in hosts if h.encounter_id == attr.encounter_id]
    if not same_encounter:
        return None
    # Prefer same turn, then nearest by turn_id ordering (lexicographic fallback)
    same_turn = [h for h in same_encounter if h.turn_id == attr.turn_id]
    candidates = same_turn if same_turn else same_encounter
    candidates.sort(key=lambda h: abs(_turn_index(h.turn_id) - _turn_index(attr.turn_id)))
    return candidates[0] if candidates else None


def _turn_index(turn_id: str) -> int:
    # turn ids look like t0001; extract numeric part
    digits = "".join(ch for ch in turn_id if ch.isdigit())
    return int(digits) if digits else 0


def attach_attributes(mentions: List[Mention]) -> List[Mention]:
    """
    Attach attributes such as dose_value, dose_unit, negation, temporality to host mentions.
    """
    hosts = [m for m in mentions if m.type in {MentionType.MEDICATION, MentionType.LAB_TEST, MentionType.PROBLEM}]
    attrs = [m for m in mentions if m.type in {MentionType.DOSE_AMOUNT, MentionType.UNIT, MentionType.FREQUENCY}]

    for attr in attrs:
        host = _nearest_host(attr, hosts)
        if not host:
            continue

        if attr.type == MentionType.DOSE_AMOUNT:
            value, unit = _split_dose(attr.text)
            if value:
                host.attributes = host.attributes or {}
                host.attributes.setdefault("dose_value", value)
                host.attributes.setdefault("dose_raw", attr.text)
            if unit:
                host.attributes = host.attributes or {}
                host.attributes.setdefault("dose_unit", unit)
        elif attr.type == MentionType.UNIT:
            host.attributes = host.attributes or {}
            host.attributes.setdefault("dose_unit", attr.text)
        elif attr.type == MentionType.FREQUENCY:
            host.attributes = host.attributes or {}
            host.attributes.setdefault("frequency_text", attr.text)

    # Negation / temporality heuristics for PROBLEM and MEDICATION
    negation_patterns = [r"\bno\b", r"\bdenies\b", r"\bwithout\b", r"\bdenied\b"]
    past_patterns = [r"\bused to\b", r"\bno longer\b", r"\bpast\b", r"\bprevious\b"]

    for m in hosts:
        text_lower = m.text.lower()
        # Negation
        if any(re.search(pat, text_lower) for pat in negation_patterns):
            m.attributes = m.attributes or {}
            m.attributes["negation"] = "true"
        # Temporality
        if any(re.search(pat, text_lower) for pat in past_patterns):
            m.attributes = m.attributes or {}
            m.attributes["temporality"] = "past"

    return mentions

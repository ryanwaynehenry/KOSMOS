"""
Local attribute attachment and simple relations between mentions.

Uses proximity-based heuristics to associate dose/units/frequency mentions with
medications and to propagate negation/temporality flags onto problems.
"""

from typing import List, Optional

from clinical_kg.data_models import Mention


def _nearest_host(target: Mention, candidates: List[Mention]) -> Optional[Mention]:
    """
    Pick the nearest candidate in the same encounter.
    """
    same_encounter = [c for c in candidates if c.encounter_id == target.encounter_id]
    if not same_encounter:
        return None
    # Sort by absolute start offset difference within the turn
    same_encounter.sort(key=lambda c: abs(c.start_char - target.start_char))
    return same_encounter[0]


def attach_attributes(mentions: List[Mention]) -> List[Mention]:
    """
    Attach attribute mentions (dose, unit, frequency, negation, temporality)
    to their host mentions where applicable.
    """
    # Separate hosts and attribute mentions
    hosts = [m for m in mentions if m.type.upper() in {"MEDICATION", "PROBLEM", "LAB_TEST"}]
    attrs = [m for m in mentions if m.type.upper() in {"DOSE_AMOUNT", "UNIT", "FREQUENCY", "NEGATION", "TEMPORAL"}]

    for attr in attrs:
        attr_type = attr.type.upper()

        if attr_type in {"DOSE_AMOUNT", "UNIT", "FREQUENCY"}:
            host = _nearest_host(attr, [h for h in hosts if h.type.upper() == "MEDICATION"])
            if not host and hosts:
                host = _nearest_host(attr, hosts)
            if host:
                if attr_type == "DOSE_AMOUNT":
                    host.attributes.setdefault("dose_value", attr.text)
                elif attr_type == "UNIT":
                    host.attributes.setdefault("dose_unit", attr.text)
                elif attr_type == "FREQUENCY":
                    host.attributes.setdefault("frequency", attr.text)
        elif attr_type == "NEGATION":
            host = _nearest_host(attr, [h for h in hosts if h.type.upper() == "PROBLEM"])
            if host:
                host.attributes["negation"] = "true"
        elif attr_type == "TEMPORAL":
            host = _nearest_host(attr, hosts)
            if host:
                host.attributes["temporality"] = attr.text

    return mentions

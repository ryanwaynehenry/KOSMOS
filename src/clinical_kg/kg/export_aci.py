"""
Utilities for exporting or summarizing graph content.

Includes SOAP note generation from nodes and relationships using an LLM.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from clinical_kg.kg.soap_prompts_aci import (
    SUBJECTIVE_SYSTEM_PROMPT_ACI_JSON,
    OBJECTIVE_SYSTEM_PROMPT_ACI_JSON,
    ASSESSMENT_PLAN_SYSTEM_PROMPT_ACI_JSON,
)

from clinical_kg.config import load_config
from clinical_kg.nlp.llm_client import call_llm_for_extraction


SOAP_SECTION_PROMPTS = {
    "subjective": SUBJECTIVE_SYSTEM_PROMPT_ACI_JSON,
    "objective": OBJECTIVE_SYSTEM_PROMPT_ACI_JSON,
    "assessment_and_plan": ASSESSMENT_PLAN_SYSTEM_PROMPT_ACI_JSON,
}


SoapSectionValue = Union[List[Dict[str, Any]], Dict[str, Any]]
DEFAULT_PATIENT_AGREEMENT = "Patient understands and agrees with the recommended medical treatment plan."


def _call_soap_section(
    section_key: str,
    display_name: str,
    payload: Dict[str, Any],
    cfg: Any,
    system_prompt: Optional[str] = None,
) -> SoapSectionValue:
    """
    Generate a single SOAP section via the LLM and return the section value.

    - subjective: list of section objects
    - objective: list of section objects
    - assessment_and_plan: object
    """
    user_instruction = (
        f"Generate only the {display_name} content using the provided nodes and relationships. "
        "Follow the Document Structure and Output Format rules exactly. "
        f'Return a JSON object with a single key "{section_key}" whose value matches the required schema. '
        "Do not include any other keys."
    )

    user_payload = {
        "section_to_generate": display_name,
        "instructions": user_instruction,
        "data": payload,
    }

    system_prompt = system_prompt or SOAP_SECTION_PROMPTS.get(section_key, "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]

    result = call_llm_for_extraction(messages, cfg, label=f"soap_note_generation_{section_key}")

    if isinstance(result, dict) and section_key in result:
        return result[section_key]

    # Fallback empty type based on section
    if section_key in {"subjective", "objective"}:
        return []
    return {}


def _summarize_node(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": node.get("id"),
        "class": node.get("class"),
        "entity_type": node.get("entity_type"),
        "canonical_name": node.get("canonical_name"),
        "attributes": node.get("attributes", {}),
    }


def _extract_patient_agreement(nodes: List[Dict[str, Any]]) -> Optional[str]:
    """
    Look for a patient-specific agreement attribute on the patient node, if present.
    """
    for node in nodes:
        if node.get("entity_type") != "PERSON_PATIENT":
            continue
        attrs = node.get("attributes") or {}
        for key in ("patient_agreement", "patient_agreements", "patient_agreement_text"):
            if key in attrs and attrs.get(key):
                return str(attrs.get(key)).strip()
    return None


def _summarize_relationship(rel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta = rel.get("llm_relation") or {}
    relation = meta.get("relation") or rel.get("relation")
    if not relation or relation == "no_relation":
        return None
    return {
        "id": rel.get("pair_id"),
        "relation": relation,
        "direction": meta.get("direction", "source->target"),
        "source_node_id": rel.get("source_node_id"),
        "target_node_id": rel.get("target_node_id"),
        "source_canonical_name": rel.get("source_canonical_name"),
        "target_canonical_name": rel.get("target_canonical_name"),
        "explanation": meta.get("explanation"),
        "evidence_turn_ids": meta.get("evidence_turn_ids", []),
    }


def _apply_ros_guard(subjective_sections: List[Dict[str, Any]]) -> None:
    """
    Drop ROS items that lack supporting evidence unless they are valid placeholders.
    If all items are removed, insert a single placeholder item.
    """
    ros_section: Optional[Dict[str, Any]] = None
    for section in subjective_sections:
        if (section.get("section") or "").strip().lower() == "review of systems":
            ros_section = section
            break

    if not ros_section:
        return

    items = ros_section.get("items") or []
    allowed_placeholders = {"Not assessed.", "None reported.", "No data available."}

    kept: List[Dict[str, Any]] = []
    for item in items:
        node_ids = item.get("node_ids") or []
        relationship_ids = item.get("relationship_ids") or []
        text = (item.get("text") or "").strip()
        label = item.get("label")

        if node_ids or relationship_ids:
            kept.append(item)
            continue

        if label is None and text in allowed_placeholders:
            kept.append(item)

    if not kept:
        kept = [
            {
                "label": None,
                "text": "No data available.",
                "node_ids": [],
                "relationship_ids": [],
            }
        ]

    ros_section["items"] = kept


def generate_soap_note(
    nodes: List[Dict[str, Any]],
    relationship_candidates: List[Dict[str, Any]],
    cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate SOAP note sections using only the provided nodes and relationships.

    Returns a dict with keys:
    - subjective
    - objective
    - assessment_and_plan
    """
    cfg = cfg or load_config()

    node_summaries = [_summarize_node(n) for n in nodes if n.get("id")]
    rel_summaries: List[Dict[str, Any]] = []
    for rel in relationship_candidates:
        summary = _summarize_relationship(rel)
        if summary:
            rel_summaries.append(summary)

    patient_agreement_pref = _extract_patient_agreement(nodes)

    payload = {
        "nodes": node_summaries,
        "relationships": rel_summaries,
    }

    subjective = _call_soap_section(
        section_key="subjective",
        display_name="Subjective",
        payload=payload,
        cfg=cfg,
        system_prompt=SOAP_SECTION_PROMPTS["subjective"],
    )
    objective = _call_soap_section(
        section_key="objective",
        display_name="Objective",
        payload=payload,
        cfg=cfg,
        system_prompt=SOAP_SECTION_PROMPTS["objective"],
    )
    assessment_and_plan = _call_soap_section(
        section_key="assessment_and_plan",
        display_name="Assessment and Plan",
        payload=payload,
        cfg=cfg,
        system_prompt=SOAP_SECTION_PROMPTS["assessment_and_plan"],
    )

    if isinstance(subjective, list):
        _apply_ros_guard(subjective)

    if isinstance(assessment_and_plan, dict):
        pa = assessment_and_plan.get("patient_agreements")
        pa_dict = pa if isinstance(pa, dict) else {"text": "", "node_ids": [], "relationship_ids": []}
        pa_text = (pa_dict.get("text") or "").strip()
        if not pa_text or pa_text.lower() == "no data available.":
            pa_dict["text"] = patient_agreement_pref or DEFAULT_PATIENT_AGREEMENT
            pa_dict.setdefault("node_ids", [])
            pa_dict.setdefault("relationship_ids", [])
        assessment_and_plan["patient_agreements"] = pa_dict

    return {
        "subjective": subjective if isinstance(subjective, list) else [],
        "objective": objective if isinstance(objective, list) else [],
        "assessment_and_plan": assessment_and_plan if isinstance(assessment_and_plan, dict) else {},
    }


def _format_soap_text(soap: Dict[str, Any]) -> str:
    """
    Convert the structured SOAP JSON into ACI-like plain text with headings.
    """

    def _find_section(sections: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        for s in sections:
            if (s.get("section") or "").strip().lower() == name.strip().lower():
                return s
        return {}

    def _item_text(item: Dict[str, Any]) -> str:
        return (item.get("text") or "").strip()

    def _join_paragraphs(texts: List[str]) -> str:
        texts = [t for t in (t.strip() for t in texts) if t]
        if not texts:
            return "No data available."
        return "\n\n".join(texts)

    def _bullets(items: List[Tuple[str, str]]) -> str:
        out_lines: List[str] = []
        for label, text in items:
            label = (label or "").strip()
            text = (text or "").strip()
            if label and text:
                out_lines.append(f"• {label}: {text}")
            elif text:
                out_lines.append(f"• {text}")
            elif label:
                out_lines.append(f"• {label}")
        return "\n".join(out_lines) if out_lines else "No data available."

    subjective_sections: List[Dict[str, Any]] = soap.get("subjective", []) or []
    objective_sections: List[Dict[str, Any]] = soap.get("objective", []) or []
    ap: Dict[str, Any] = soap.get("assessment_and_plan", {}) or {}

    # SUBJECTIVE
    cc = _find_section(subjective_sections, "Chief Complaint").get("items") or []
    hpi = _find_section(subjective_sections, "History of Present Illness").get("items") or []
    ros = _find_section(subjective_sections, "Review of Systems").get("items") or []

    cc_text = _join_paragraphs([_item_text(i) for i in cc]) if cc else "No data available."
    hpi_text = _join_paragraphs([_item_text(i) for i in hpi]) if hpi else "No data available."

    ros_bullets = _bullets(
        [((i.get("label") or ""), _item_text(i)) for i in ros]
    ) if ros else "No data available."

    # OBJECTIVE
    pe = _find_section(objective_sections, "Physical Examination").get("items") or []
    vitals = _find_section(objective_sections, "Vitals Reviewed").get("items") or []
    results = _find_section(objective_sections, "Results").get("items") or []

    pe_bullets = _bullets(
        [((i.get("label") or ""), _item_text(i)) for i in pe]
    ) if pe else "No data available."

    vitals_bullets = _bullets(
        [((i.get("label") or ""), _item_text(i)) for i in vitals]
    ) if vitals else "No data available."

    # Results tends to read better without bullets in the ACI examples.
    result_lines: List[str] = []
    for r in results:
        label = (r.get("label") or "").strip()
        text = _item_text(r)
        if label and text:
            result_lines.append(f"{label}: {text}")
        elif text:
            result_lines.append(text)
        elif label:
            result_lines.append(label)
    results_text = "\n\n".join([x for x in result_lines if x.strip()]) if result_lines else "No data available."

    # ASSESSMENT AND PLAN
    assessment_items = ap.get("assessment") or []
    plan_items = ap.get("plan") or []
    patient_agreements = ap.get("patient_agreements") or {}
    instructions_items = ap.get("instructions") or []

    assessment_text = _join_paragraphs(
        [(_item_text(i) if isinstance(i, dict) else str(i)) for i in assessment_items]
    ) if assessment_items else "No data available."

    plan_blocks: List[str] = []
    for p in plan_items:
        if not isinstance(p, dict):
            continue
        problem = (p.get("problem") or "").strip()
        components = p.get("components") or []
        if not problem and not components:
            continue

        block_lines: List[str] = []
        if problem:
            block_lines.append(f"{problem}.")
        for c in components:
            if not isinstance(c, dict):
                continue
            comp = (c.get("component") or "").strip()
            txt = (c.get("text") or "").strip()
            if comp and txt:
                block_lines.append(f"• {comp}: {txt}")
            elif txt:
                block_lines.append(f"• {txt}")
        plan_blocks.append("\n".join(block_lines).strip())

    plan_text = "\n\n".join([b for b in plan_blocks if b.strip()]) if plan_blocks else "No data available."

    pa_text = (patient_agreements.get("text") or "").strip() if isinstance(patient_agreements, dict) else ""
    if not pa_text:
        pa_text = "No data available."

    instructions_lines: List[str] = []
    for ins in instructions_items:
        if isinstance(ins, dict):
            t = (ins.get("text") or "").strip()
            if t:
                instructions_lines.append(t)
        else:
            t = str(ins).strip()
            if t:
                instructions_lines.append(t)
    instructions_text = "\n\n".join(instructions_lines) if instructions_lines else ""

    # Compose final text in ACI-like style
    lines: List[str] = []

    lines.append("CHIEF COMPLAINT")
    lines.append("")
    lines.append(cc_text)
    lines.append("")

    lines.append("HISTORY OF PRESENT ILLNESS")
    lines.append("")
    lines.append(hpi_text)
    lines.append("")

    lines.append("REVIEW OF SYSTEMS")
    lines.append("")
    lines.append(ros_bullets)
    lines.append("")

    lines.append("PHYSICAL EXAMINATION")
    lines.append("")
    lines.append(pe_bullets)
    lines.append("")

    lines.append("VITALS REVIEWED")
    lines.append("")
    lines.append(vitals_bullets)
    lines.append("")

    lines.append("RESULTS")
    lines.append("")
    lines.append(results_text)
    lines.append("")

    lines.append("ASSESSMENT AND PLAN")
    lines.append("")
    lines.append(assessment_text)
    lines.append("")
    lines.append(plan_text)
    lines.append("")

    lines.append(f"Patient Agreements: {pa_text}")
    lines.append("")

    if instructions_text.strip():
        lines.append("INSTRUCTIONS")
        lines.append("")
        lines.append(instructions_text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SOAP note from a KG JSON file.")
    parser.add_argument("input", type=Path, help="Path to JSON containing turns, nodes, and relationship_candidates.")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional override for SOAP JSON output path. Defaults to data/processed/soap_<input-stem>.json",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        help="Optional override for SOAP text output path. Defaults to data/processed/soap_<input-stem>.txt",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    relationships = data.get("relationship_candidates", [])

    soap = generate_soap_note(nodes=nodes, relationship_candidates=relationships)

    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    output_json_path = args.output_json or output_dir / f"soap_{stem}.json"
    output_txt_path = args.output_txt or output_dir / f"soap_{stem}.txt"

    output_json_path.write_text(
        json.dumps(soap, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_txt_path.write_text(
        _format_soap_text(soap),
        encoding="utf-8",
    )

    print(f"Wrote {output_json_path}")
    print(f"Wrote {output_txt_path}")


if __name__ == "__main__":
    main()

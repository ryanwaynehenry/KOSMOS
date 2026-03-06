"""
Utilities for exporting or summarizing graph content.

Includes SOAP note generation from nodes and relationships using an LLM.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from clinical_kg.kg.soap_prompts import SUBJECTIVE_SYSTEM_PROMPT, OBJECTIVE_SYSTEM_PROMPT, ASSESSMENT_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT

from clinical_kg.config import load_config
from clinical_kg.nlp.llm_client import call_llm_for_extraction

SOAP_SECTION_PROMPTS = {
    "subjective": SUBJECTIVE_SYSTEM_PROMPT,
    "objective": OBJECTIVE_SYSTEM_PROMPT,
    "assessment": ASSESSMENT_SYSTEM_PROMPT,
    "plan": PLAN_SYSTEM_PROMPT,
}

def _call_soap_section(
    section_name: str, payload: Dict[str, Any], cfg: Any, system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate a single SOAP section via the LLM and return its section objects list.
    """
    section_key = section_name.lower()
    user_instruction = (
        f"Generate only the {section_name} section of the SOAP note using the provided nodes and relationships. "
        "Follow the Document Structure and Output Format rules exactly. "
        f"Return a JSON object with a single key \"{section_key}\" whose value is the ordered list of section objects. "
        "Do not include any other keys."
    )
    user_payload = {
        "section_to_generate": section_name,
        "instructions": user_instruction,
        "data": payload,
    }
    system_prompt = (
        system_prompt
        or SOAP_SECTION_PROMPTS.get(section_key)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]
    result = call_llm_for_extraction(messages, cfg, label=f"soap_note_generation_{section_key}")
    if isinstance(result, dict):
        items = result.get(section_key)
        if isinstance(items, list):
            return items
    return []


def _summarize_node(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": node.get("id"),
        "class": node.get("class"),
        "entity_type": node.get("entity_type"),
        "canonical_name": node.get("canonical_name"),
        "attributes": node.get("attributes", {}),
    }


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


def generate_soap_note(
    nodes: List[Dict[str, Any]],
    relationship_candidates: List[Dict[str, Any]],
    cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate SOAP note sections using only the provided nodes and relationships.

    Returns a dict with keys: subjective, objective, assessment, plan.
    """
    cfg = cfg or load_config()

    node_summaries = [_summarize_node(n) for n in nodes if n.get("id")]
    rel_summaries = []
    for rel in relationship_candidates:
        summary = _summarize_relationship(rel)
        if summary:
            rel_summaries.append(summary)

    payload = {
        "nodes": node_summaries,
        "relationships": rel_summaries,
    }

    section_prompts = {
        "subjective": globals().get("SUBJECTIVE_SYSTEM_PROMPT"),
        "objective": globals().get("OBJECTIVE_SYSTEM_PROMPT"),
        "assessment": globals().get("ASSESSMENT_SYSTEM_PROMPT"),
        "plan": globals().get("PLAN_SYSTEM_PROMPT"),
    }

    subjective = _call_soap_section(
        "Subjective", payload, cfg, system_prompt=section_prompts.get("subjective")
    )
    objective = _call_soap_section(
        "Objective", payload, cfg, system_prompt=section_prompts.get("objective")
    )
    assessment = _call_soap_section(
        "Assessment", payload, cfg, system_prompt=section_prompts.get("assessment")
    )
    plan = _call_soap_section("Plan", payload, cfg, system_prompt=section_prompts.get("plan"))

    return {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "plan": plan,
    }


def _format_soap_text(soap: Dict[str, Any]) -> str:
    """
    Convert the structured SOAP JSON into a plain-text document with section headings.
    """
    def format_items(subsection: str, items: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        if subsection in {"Medication List", "Plan Items"}:
            for idx, item in enumerate(items, start=1):
                text = (item.get("text") or "").strip()
                label = (item.get("label") or "").strip()
                content = text or label or "(none)"
                lines.append(f"{idx}. {content}")
            return lines

        if subsection == "Vital Signs":
            for item in items:
                label = (item.get("label") or "").strip()
                text = (item.get("text") or "").strip()
                if label and text:
                    lines.append(f"{label} {text}")
                elif text:
                    lines.append(text)
                elif label:
                    lines.append(label)
            return lines

        if subsection in {"Review of Systems", "Physical Exam", "Diagnostic Tests"}:
            for item in items:
                label = (item.get("label") or "").strip()
                text = (item.get("text") or "").strip()
                if label and text:
                    lines.append(f"{label}: {text}")
                elif text:
                    lines.append(text)
                elif label:
                    lines.append(label)
            return lines

        for item in items:
            text = (item.get("text") or "").strip()
            label = (item.get("label") or "").strip()
            lines.append(text or label or "(none)")
        return lines

    sections: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("Subjective", soap.get("subjective", [])),
        ("Objective", soap.get("objective", [])),
        ("Assessment", soap.get("assessment", [])),
        ("Plan", soap.get("plan", [])),
    ]
    lines: List[str] = []
    for title, section_items in sections:
        lines.append(title)
        lines.append("")
        if not section_items:
            lines.append("No data available.")
            lines.append("")
        else:
            for section in section_items:
                section_name = section.get("section") or "(unspecified section)"
                lines.append(section_name)
                items = section.get("items") or []
                formatted_items = format_items(section_name, items)
                if not formatted_items:
                    lines.append("No data available.")
                else:
                    lines.extend(formatted_items)
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

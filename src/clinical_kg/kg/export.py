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
    relation = meta.get("relation")
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
    sections: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("Subjective", soap.get("subjective", [])),
        ("Objective", soap.get("objective", [])),
        ("Assessment", soap.get("assessment", [])),
        ("Plan", soap.get("plan", [])),
    ]
    lines: List[str] = []
    for title, section_items in sections:
        lines.append(f"{title}:")
        if not section_items:
            lines.append("  (none)")
        else:
            for section in section_items:
                section_name = section.get("section") or "(unspecified section)"
                lines.append(f"  {section_name}:")
                items = section.get("items") or []
                if not items:
                    lines.append("    - (none)")
                else:
                    for item in items:
                        label = item.get("label")
                        text = item.get("text", "")
                        if label:
                            lines.append(f"    - {label}: {text}")
                        else:
                            lines.append(f"    - {text}")
        lines.append("")  # blank line between sections
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a SOAP note from a KG JSON file.")
    parser.add_argument("input", type=Path, help="Path to JSON containing turns, nodes, and relationship_candidates.")
    parser.add_argument("--output-json", type=Path, help="Where to write the SOAP note JSON (default: stdout).")
    parser.add_argument("--output-txt", type=Path, help="Where to write the SOAP note text with section headings.")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    nodes = data.get("nodes", [])
    relationships = data.get("relationship_candidates", [])

    soap = generate_soap_note(nodes=nodes, relationship_candidates=relationships)

    if args.output_json:
        args.output_json.write_text(json.dumps(soap, indent=2, ensure_ascii=False))
    if args.output_txt:
        args.output_txt.write_text(_format_soap_text(soap))

    if not args.output_json and not args.output_txt:
        print(json.dumps(soap, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

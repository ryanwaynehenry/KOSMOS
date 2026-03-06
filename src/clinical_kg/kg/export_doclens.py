#!/usr/bin/env python3
"""
Generate a full SOAP note in a single LLM call using the DocLens prompt.

The prompt template lives in soap_prompts_doclens.py as a JSON array of chat
messages. The final user message is populated with the transcript (turn-based
when available) plus a summarized knowledge graph (nodes + relationships).
"""

import argparse
import json
import time
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import litellm

from clinical_kg.config import load_config
from clinical_kg.nlp import llm_client


DOCLENS_PROMPT_PATH = Path(__file__).with_name("soap_prompts_doclens.py")
# DOCLENS_PROMPT_PATH = Path(__file__).with_name("soap_prompts_doclens_kg_only.py")
# Prompt context toggle:
# - False: include only nodes in KNOWLEDGE_GRAPH
# - True: include nodes and relationships in KNOWLEDGE_GRAPH
INCLUDE_RELATIONSHIPS_IN_CONTEXT = False


def _load_doclens_prompt() -> List[Dict[str, Any]]:
    prompt_obj = json.loads(DOCLENS_PROMPT_PATH.read_text(encoding="utf-8"))
    if not isinstance(prompt_obj, list):
        raise ValueError("DocLens prompt file must contain a top-level list.")

    messages: List[Dict[str, Any]] = []
    for msg in prompt_obj:
        if not isinstance(msg, dict):
            raise ValueError("Each prompt entry must be an object with role/content.")
        # Shallow copy to avoid mutating the parsed template
        messages.append(dict(msg))

    if not messages or messages[-1].get("role") != "user":
        raise ValueError("DocLens prompt must end with a user message placeholder.")

    return messages


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


def _infer_example_index(input_path: Optional[Path]) -> Optional[int]:
    if not input_path:
        return None
    m = re.search(r"_([0-9]+)$", input_path.stem)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _load_transcript_from_raw_json(raw_path: Path, example_idx: Optional[int]) -> Optional[str]:
    if not raw_path.exists():
        return None
    try:
        obj = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Common shapes: {"data": [...]} or [...]
    data_list: Optional[List[Any]] = None
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        data_list = obj["data"]
    elif isinstance(obj, list):
        data_list = obj

    if data_list is None or example_idx is None or example_idx < 0 or example_idx >= len(data_list):
        return None

    item = data_list[example_idx]
    if not isinstance(item, dict):
        return None

    for key in ("transcript", "transcript_text", "raw_transcript", "src", "input"):
        if key in item and item[key]:
            return str(item[key])
    return None


def _load_transcript_text(
    data_obj: Dict[str, Any],
    transcript_path: Optional[Path],
    input_path: Optional[Path],
) -> Optional[str]:
    """
    Priority:
    1) If transcript_path is provided, use ONLY that source (raw JSON or text).
    2) Otherwise, fall back to transcript fields in the input data object.
    """
    if transcript_path and transcript_path.exists():
        if transcript_path.suffix.lower() == ".json":
            idx = _infer_example_index(input_path)
            return _load_transcript_from_raw_json(transcript_path, idx)
        return transcript_path.read_text(encoding="utf-8")

    for key in ("transcript", "transcript_text", "raw_transcript", "src"):
        if key in data_obj and data_obj[key]:
            return str(data_obj[key])
    return None


def _format_turns_for_prompt(turns: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(turns, list):
        return None
    out: List[Dict[str, Any]] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        out.append({"turn_id": t.get("turn_id"), "speaker": t.get("speaker"), "text": t.get("text")})
    return out or None


def _has_turn_number(line: str) -> bool:
    if not line.startswith("["):
        return False
    end = line.find("]")
    if end <= 1:
        return False
    token = line[1:end]
    return bool(re.match(r"[tT]?\d+$", token))


def _normalize_turn_id(raw_id: Any, fallback: int) -> int:
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        m = re.match(r"[tT]?0*([0-9]+)$", raw_id.strip())
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        try:
            return int(raw_id.strip())
        except (ValueError, TypeError):
            pass
    return fallback


def _number_transcript_lines(transcript_text: str) -> str:
    lines = [ln.strip() for ln in transcript_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    if all(_has_turn_number(ln) for ln in lines):
        return "\n".join(lines)

    numbered: List[str] = []
    for idx, ln in enumerate(lines):
        if _has_turn_number(ln):
            numbered.append(ln)
        elif ln.startswith("["):
            numbered.append(f"[{idx}]{ln}")
        else:
            numbered.append(f"[{idx}] {ln}")
    return "\n".join(numbered)


def _format_transcript(transcript_text: Optional[str], turns: Optional[List[Dict[str, Any]]]) -> str:
    # If a raw transcript string is provided, prefer it (numbered if needed),
    # even when turns are present in the input.
    if transcript_text:
        return _number_transcript_lines(transcript_text)

    if turns:
        lines: List[str] = []
        for idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_id = _normalize_turn_id(turn.get("turn_id"), idx)
            speaker = (turn.get("speaker") or "").strip() or "unknown"
            text = (turn.get("text") or "").strip()
            lines.append(f"[{turn_id}][{speaker.lower()}] {text}".strip())
        if lines:
            return "\n".join(lines)
    return _number_transcript_lines((transcript_text or "").strip())


def _build_user_input(
    transcript_text: Optional[str],
    turns: Optional[List[Dict[str, Any]]],
    node_summaries: List[Dict[str, Any]],
    rel_summaries: List[Dict[str, Any]],
    include_relationships: bool,
) -> str:
    transcript_block = _format_transcript(transcript_text, turns)
    kg_payload: Dict[str, Any] = {"nodes": node_summaries}
    if include_relationships:
        kg_payload["relationships"] = rel_summaries
    kg_block = json.dumps(kg_payload, ensure_ascii=False, indent=2)

    parts: List[str] = []
    if transcript_block:
        parts.append("TRANSCRIPT\n" + transcript_block)
    parts.append("KNOWLEDGE_GRAPH\n" + kg_block)
    return "\n\n".join(parts).strip()


def _call_llm_for_full_note(
    messages: List[Dict[str, Any]],
    cfg: Optional[Any],
    label: str = "soap_note_generation_full_doclens",
    reasoning_effort: str = "none",
) -> str:
    cfg = cfg or load_config()
    llm_client._ensure_api_key(cfg)  # type: ignore[attr-defined]

    model_name = cfg.llm_model_name
    base = llm_client._base_model_name(model_name)  # type: ignore[attr-defined]
    is_gpt5_family = base.startswith("gpt-5")

    kwargs: Dict[str, Any] = {"model": model_name, "messages": messages}

    if reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
        if is_gpt5_family:
            kwargs["allowed_openai_params"] = ["reasoning_effort"]
    else:
        kwargs["temperature"] = cfg.llm_temperature

    if getattr(cfg, "max_llm_tokens", None) is not None:
        kwargs["max_tokens"] = cfg.max_llm_tokens

    start = time.time()
    response = litellm.completion(**kwargs)
    elapsed = time.time() - start
    print(f"[llm] {label} took {elapsed:.2f}s")

    if not getattr(response, "choices", None):
        raise RuntimeError("LLM response contained no choices")

    content = (
        response.choices[0]["message"]["content"]
        if isinstance(response.choices[0], dict)
        else response.choices[0].message.get("content")  # type: ignore[attr-defined]
    )
    if content is None:
        raise RuntimeError("LLM response message content is empty")

    cleaned = str(content).strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        if len(parts) == 3:
            cleaned = parts[1].strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[len("json") :].lstrip()

    return cleaned


def generate_soap_note(
    nodes: List[Dict[str, Any]],
    relationship_candidates: List[Dict[str, Any]],
    cfg: Optional[Any] = None,
    transcript_text: Optional[str] = None,
    turns: Optional[List[Dict[str, Any]]] = None,
    reasoning_effort: str = "none",
    include_relationships: Optional[bool] = None,
) -> Tuple[str, str]:
    cfg = cfg or load_config()
    if include_relationships is None:
        include_relationships = INCLUDE_RELATIONSHIPS_IN_CONTEXT

    node_summaries = [_summarize_node(n) for n in nodes if n.get("id")]
    rel_summaries: List[Dict[str, Any]] = []
    if include_relationships:
        for rel in relationship_candidates:
            summary = _summarize_relationship(rel)
            if summary:
                rel_summaries.append(summary)

    transcript_text = (transcript_text or "").strip() or None
    turns_payload = _format_turns_for_prompt(turns)

    user_input = _build_user_input(
        transcript_text,
        turns_payload,
        node_summaries,
        rel_summaries,
        include_relationships=include_relationships,
    )
    messages = _load_doclens_prompt()
    messages[-1]["content"] = user_input

    note_text = _call_llm_for_full_note(
        messages=messages,
        cfg=cfg,
        label="soap_note_generation_full_doclens",
        reasoning_effort=reasoning_effort,
    )

    return note_text, user_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a SOAP note from a KG JSON file using the DocLens prompt in a single LLM call."
    )
    parser.add_argument("input", type=Path, help="Path to JSON containing turns, nodes, and relationship_candidates.")
    parser.add_argument(
        "--transcript-path",
        type=Path,
        help="Optional path to raw transcript text. If omitted, uses transcript fields in the input JSON if present.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        help="Optional override for SOAP text output path. Defaults to data/processed/soap_<input-stem>.txt",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Pass-through reasoning_effort for GPT-5 family models (default: none).",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    relationships = data.get("relationship_candidates", [])
    turns = data.get("turns", [])
    transcript_text = _load_transcript_text(data, args.transcript_path, args.input)

    note_text, _prompt_input = generate_soap_note(
        nodes=nodes,
        relationship_candidates=relationships,
        transcript_text=transcript_text,
        turns=turns,
        reasoning_effort=args.reasoning_effort,
        include_relationships=INCLUDE_RELATIONSHIPS_IN_CONTEXT,
    )

    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    output_txt_path = args.output_txt or output_dir / f"soap_{stem}.txt"

    output_txt_path.write_text(
        note_text if note_text.endswith("\n") else f"{note_text}\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_txt_path}")


if __name__ == "__main__":
    main()

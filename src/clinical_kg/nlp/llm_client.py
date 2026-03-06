"""
Thin wrapper around LiteLLM for structured extraction calls.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import litellm

from clinical_kg.config import PipelineConfig, load_config

_api_key_set = False


def _ensure_api_key(cfg: PipelineConfig) -> None:
    global _api_key_set
    if _api_key_set:
        return

    key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("No LLM API key found in environment (OPENAI_API_KEY or LLM_API_KEY).")

    os.environ["OPENAI_API_KEY"] = key
    _api_key_set = True


def _base_model_name(model: str) -> str:
    # "openai/responses/gpt-5.2" -> "gpt-5.2"
    # "responses/gpt-5.2" -> "gpt-5.2"
    # "gpt-5.2" -> "gpt-5.2"
    return model.split("/")[-1].strip()


def _extract_json_block(text: str) -> str:
    starts = [pos for pos in (text.find("["), text.find("{")) if pos != -1]
    if not starts:
        return text
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}"))
    if end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _remove_trailing_commas(text: str) -> str:
    out: List[str] = []
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "]}":
                continue
        out.append(ch)
    return "".join(out)


def _fix_unquoted_explanations(text: str) -> str:
    key = '"explanation"'
    next_key = '"evidence_turn_ids"'
    idx = 0
    parts: List[str] = []
    while True:
        pos = text.find(key, idx)
        if pos == -1:
            parts.append(text[idx:])
            break
        parts.append(text[idx:pos])
        colon = text.find(":", pos + len(key))
        if colon == -1:
            parts.append(text[pos:])
            break
        parts.append(text[pos : colon + 1])
        j = colon + 1
        while j < len(text) and text[j].isspace():
            parts.append(text[j])
            j += 1
        if j >= len(text):
            idx = j
            break
        if text[j] in "\"[{":
            idx = j
            continue
        if text.startswith("null", j) or text.startswith("true", j) or text.startswith("false", j):
            idx = j
            continue
        next_pos = text.find(next_key, j)
        if next_pos == -1:
            parts.append(text[j:])
            break
        comma = text.rfind(",", j, next_pos)
        if comma == -1:
            parts.append(text[j:])
            break
        raw = text[j:comma].strip()
        if raw:
            raw = raw.replace('\\"', '"')
        parts.append(json.dumps(raw))
        parts.append(text[comma:next_pos])
        idx = next_pos
    return "".join(parts)


def _repair_llm_json(text: str) -> str:
    fixed = _extract_json_block(text)
    fixed = _fix_unquoted_explanations(fixed)
    fixed = _remove_trailing_commas(fixed)
    return fixed


def call_llm_for_extraction(
    messages: List[Dict[str, str]],
    cfg: Optional[PipelineConfig] = None,
    label: Optional[str] = None,
    reasoning_effort: str = "none",
) -> Dict[str, Any]:
    """
    Call the configured LLM using LiteLLM for a structured extraction task.

    - Uses cfg.llm_model_name as the model name passed to LiteLLM.
    - Uses cfg.llm_temperature only when reasoning_effort == "none".
    - Passes reasoning_effort through, and whitelists it for GPT-5 family to
      avoid LiteLLM UnsupportedParamsError. :contentReference[oaicite:4]{index=4}
    - Returns parsed JSON from the assistant message content.
    """
    cfg = cfg or load_config()
    _ensure_api_key(cfg)

    # litellm._turn_on_debug()
    model_name = cfg.llm_model_name
    base = _base_model_name(model_name)
    is_gpt5_family = base.startswith("gpt-5")

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        # "reasoning_effort": reasoning_effort,
    }

    # GPT-5 family: temperature is only safe when reasoning_effort == "none"
    if reasoning_effort == "none":
        kwargs["temperature"] = cfg.llm_temperature

    if getattr(cfg, "max_llm_tokens", None) is not None:
        kwargs["max_tokens"] = cfg.max_llm_tokens

    # Workaround for LiteLLM rejecting reasoning_effort for some models
    if is_gpt5_family and reasoning_effort != "none":
        kwargs["allowed_openai_params"] = ["reasoning_effort"]

    start = time.time()
    response = litellm.completion(**kwargs)
    elapsed = time.time() - start
    tag = label or "llm_extraction"
    print(f"[llm] {tag} took {elapsed:.2f}s")

    if not getattr(response, "choices", None):
        raise RuntimeError("LLM response contained no choices")

    content = (
        response.choices[0]["message"]["content"]
        if isinstance(response.choices[0], dict)
        else response.choices[0].message.get("content")  # type: ignore[attr-defined]
    )
    if content is None:
        raise RuntimeError("LLM response message content is empty")

    cleaned = content.strip()

    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        if len(parts) == 3:
            cleaned = parts[1].strip()
        else:
            cleaned = content.strip()

    if cleaned.startswith("json"):
        cleaned = cleaned[len("json") :].lstrip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        try:
            return json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            repaired = _repair_llm_json(cleaned)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                try:
                    return json.loads(repaired, strict=False)
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse LLM JSON content: {cleaned}") from exc

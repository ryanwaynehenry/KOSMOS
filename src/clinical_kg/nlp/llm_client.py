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
    """
    Set the provider API key exactly once from environment/config.
    """
    global _api_key_set
    if _api_key_set:
        return

    # Prefer explicit env var, fall back to a generic key if present.
    key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        # If you later add cfg.llm_api_key, you can use it here.
        raise RuntimeError("No LLM API key found in environment (OPENAI_API_KEY or LLM_API_KEY).")

    os.environ["OPENAI_API_KEY"] = key
    _api_key_set = True


def call_llm_for_extraction(
    messages: List[Dict[str, str]],
    cfg: Optional[PipelineConfig] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call the configured LLM using LiteLLM for a structured extraction task.

    Inputs:
      - messages: list of chat messages, each like
        {"role": "system"|"user"|"assistant", "content": "..."}.
      - cfg: optional PipelineConfig. If None, load_config() will be called.
      - label: optional human-readable label for timing/diagnostics.

    Behavior:
      - Uses cfg.llm_model_name as the model name passed to LiteLLM.
      - Uses cfg.llm_temperature for the temperature parameter.
      - Uses litellm.completion to make the request.
      - Returns the parsed JSON content of the assistant's message.
        If parsing fails, raises a descriptive exception.
    """
    cfg = cfg or load_config()
    _ensure_api_key(cfg)

    kwargs = {
        "model": cfg.llm_model_name,
        "messages": messages,
        "temperature": cfg.llm_temperature,
    }
    if getattr(cfg, "max_llm_tokens", None) is not None:
        kwargs["max_tokens"] = cfg.max_llm_tokens

    start = time.time()
    response = litellm.completion(**kwargs)
    elapsed = time.time() - start
    tag = label or "llm_extraction"
    print(f"[llm] {tag} took {elapsed:.2f}s")
    if not getattr(response, "choices", None):
        raise RuntimeError("LLM response contained no choices")

    # litellm returns an OpenAI-compatible response; access message content
    content = (
        response.choices[0]["message"]["content"]
        if isinstance(response.choices[0], dict)
        else response.choices[0].message.get("content")  # type: ignore[attr-defined]
    )
    if content is None:
        raise RuntimeError("LLM response message content is empty")

    # Strip common markdown fences if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        # remove leading fence
        cleaned = cleaned.split("```", 2)
        if len(cleaned) == 3:
            cleaned = cleaned[1]  # content between fences
        else:
            cleaned = content
    cleaned = cleaned.strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[len("json") :].lstrip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM JSON content: {cleaned}") from exc

#!/usr/bin/env python3
"""
Batch LLM evaluation of candidate SOAP notes against transcript + gold.

Uses:
- scripts/eval_soap_llm.py: derive_raw_paths(), load_transcript_and_gold(), SYSTEM_PROMPT
- src/clinical_kg/nlp/llm_client.py: call_llm_for_extraction()
- src/clinical_kg/config.py: load_config() (explicitly loaded here)

Outputs:
- Prompt text: data/evaluation_prompts/<stem>_llm_eval_prompt.txt
- LLM JSON:    data/evaluation/evaluation_<stem>.json

Concurrency:
- ThreadPoolExecutor via --workers
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_src_to_syspath(root: Path) -> None:
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def import_eval_prompt_module(root: Path):
    mod_path = root / "scripts" / "eval_soap_llm.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Missing expected file: {mod_path}")

    spec = importlib.util.spec_from_file_location("eval_soap_llm", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from: {mod_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def normalize_system_prompt(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("SYSTEM "):
        t = t[len("SYSTEM ") :].lstrip()
    return t


def expand_inputs(inputs: List[str], pattern: str) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob(pattern)))
        else:
            out.append(p)

    seen = set()
    uniq: List[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def safe_load_transcript_gold(eval_mod, raw_path: Path, idx: int) -> Tuple[str, str]:
    """
    scripts/eval_soap_llm.py in your paste returns (transcript, gold) even if the
    type annotation suggests otherwise. Handle either shape safely.
    """
    res = eval_mod.load_transcript_and_gold(raw_path, idx)
    if isinstance(res, tuple) and len(res) >= 2:
        return (res[0] or ""), (res[1] or "")
    raise RuntimeError("load_transcript_and_gold returned an unexpected value")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sleep_backoff(attempt: int, base_seconds: float = 1.0, cap_seconds: float = 30.0) -> None:
    delay = min(cap_seconds, base_seconds * (2 ** attempt))
    delay += random.uniform(0.0, 0.25 * delay)
    time.sleep(delay)


def process_one(
    hyp_path: Path,
    eval_mod,
    cfg,
    reasoning_effort: str,
    overwrite: bool,
    max_retries: int,
) -> Tuple[str, bool, str]:
    root = repo_root()

    prompt_dir = root / "data" / "evaluation_prompts"
    eval_dir = root / "data" / "evaluation"

    stem = hyp_path.stem
    prompt_path = prompt_dir / f"{stem}_llm_eval_prompt.txt"
    out_json_path = eval_dir / f"evaluation_{stem}.json"
    out_err_path = eval_dir / f"evaluation_{stem}.error.json"

    if not hyp_path.exists():
        return stem, False, f"File not found: {hyp_path}"

    if out_json_path.exists() and not overwrite:
        return stem, True, f"Skipped (exists): {out_json_path}"

    raw_path, idx = eval_mod.derive_raw_paths(hyp_path)
    transcript, gold = safe_load_transcript_gold(eval_mod, raw_path, idx)
    candidate = hyp_path.read_text(encoding="utf-8")

    system_prompt = normalize_system_prompt(getattr(eval_mod, "SYSTEM_PROMPT", ""))
    user_content = (
        "TRANSCRIPT\n"
        f"{transcript}\n\n"
        "GOLD\n"
        f"{gold}\n\n"
        "CANDIDATE\n"
        f"{candidate}\n"
    )

    # Save prompt text
    prompt_text = f"{getattr(eval_mod, 'SYSTEM_PROMPT', '').strip()}\n\n{user_content}"
    write_text(prompt_path, prompt_text)

    from clinical_kg.nlp.llm_client import call_llm_for_extraction

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = call_llm_for_extraction(
                messages=messages,
                cfg=cfg,
                label=stem,
                reasoning_effort=reasoning_effort,
            )
            write_json(out_json_path, result)
            return stem, True, f"Wrote: {out_json_path}"
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_backoff(attempt)
                continue

    err_obj = {
        "stem": stem,
        "hyp_path": str(hyp_path),
        "error": str(last_exc),
        "traceback": traceback.format_exc(),
    }
    write_json(out_err_path, err_obj)
    return stem, False, f"Failed, wrote error: {out_err_path}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate candidate SOAP notes via LLM and write JSON outputs.")
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Candidate SOAP files and or directories. Directories expanded with --pattern.",
    )
    ap.add_argument(
        "--pattern",
        default="soap_*.txt",
        help="Glob used when an input is a directory (default: soap_*.txt).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of threads to use (default: 1).",
    )
    ap.add_argument(
        "--reasoning_effort",
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help='Reasoning effort passed to LiteLLM (default: "none").',
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation JSON files.",
    )
    ap.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Retries per file on failure (default: 2).",
    )
    args = ap.parse_args()

    root = repo_root()
    add_src_to_syspath(root)

    # Ensure relative paths inside scripts/eval_soap_llm.py resolve correctly.
    os.chdir(root)

    eval_mod = import_eval_prompt_module(root)
    hyp_files = expand_inputs(args.inputs, args.pattern)
    if not hyp_files:
        raise SystemExit("No candidate files found from the provided inputs.")

    # FIX 2:
    # Import config.py BEFORE checking env vars, because config.py calls load_dotenv() at import time.
    from clinical_kg.config import load_config

    cfg = load_config()

    # Now .env has been loaded (if present). Check for API key.
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")):
        raise RuntimeError("Set OPENAI_API_KEY or LLM_API_KEY (env var or in .env) before running.")

    if args.workers <= 1:
        ok = 0
        for p in hyp_files:
            stem, success, msg = process_one(
                hyp_path=p,
                eval_mod=eval_mod,
                cfg=cfg,
                reasoning_effort=args.reasoning_effort,
                overwrite=args.overwrite,
                max_retries=args.max_retries,
            )
            print(f"[{'OK' if success else 'FAIL'}] {stem}: {msg}")
            ok += 1 if success else 0
        print(f"Done. Success {ok}/{len(hyp_files)}")
        return

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                process_one,
                p,
                eval_mod,
                cfg,
                args.reasoning_effort,
                args.overwrite,
                args.max_retries,
            )
            for p in hyp_files
        ]
        for fut in as_completed(futures):
            stem, success, msg = fut.result()
            print(f"[{'OK' if success else 'FAIL'}] {stem}: {msg}")
            ok += 1 if success else 0

    print(f"Done. Success {ok}/{len(hyp_files)}")


if __name__ == "__main__":
    main()
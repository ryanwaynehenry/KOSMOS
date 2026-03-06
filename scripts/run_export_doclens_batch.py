#!/usr/bin/env python3
"""
Batch runner for export_doclens.py.

Given a glob pattern, runs export_doclens on every matching JSON input and
writes one SOAP note text file per input.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_src_to_syspath(root: Path) -> None:
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def glob_inputs(pattern: str) -> List[Path]:
    paths = sorted(Path().glob(pattern))
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def process_one(
    input_path: Path,
    transcript_path: Path,
    output_dir: Path,
    reasoning_effort: str,
) -> Tuple[Path, bool, str]:
    from clinical_kg.kg.export_doclens import (
        _load_transcript_text,
        generate_soap_note,
    )

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - runtime guard
        return input_path, False, f"Failed to read JSON: {exc}"

    nodes = data.get("nodes", [])
    relationships = data.get("relationship_candidates", [])
    turns = data.get("turns", [])

    transcript_text = _load_transcript_text(
        data_obj=data,
        transcript_path=transcript_path,
        input_path=input_path,
    )

    note_text, _ = generate_soap_note(
        nodes=nodes,
        # relationship_candidates=relationships,
        relationship_candidates=[],
        transcript_text=transcript_text,
        turns=turns,
        reasoning_effort=reasoning_effort,
    )

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"soap_{input_path.stem}.txt"
    out_path.write_text(note_text if note_text.endswith("\n") else f"{note_text}\n", encoding="utf-8")
    return input_path, True, f"Wrote {out_path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run export_doclens on all input files matching a glob pattern."
    )
    parser.add_argument("pattern", help="Glob pattern for input JSONs, e.g., 'data/interim/clinicalnlp_taskB_*.json'")
    parser.add_argument(
        "--transcript-path",
        required=True,
        type=Path,
        help="Path to raw transcript JSON/Text; used to fetch transcripts for all inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "processed",
        help="Directory for output SOAP text files (default: data/processed).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Pass-through reasoning_effort for GPT-5 family models (default: none).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    args = parser.parse_args()

    root = repo_root()
    add_src_to_syspath(root)
    os.chdir(root)

    inputs = glob_inputs(args.pattern)
    if not inputs:
        raise SystemExit(f"No input files matched pattern: {args.pattern}")

    transcript_path = args.transcript_path.resolve()
    output_dir = args.output_dir

    print(f"Found {len(inputs)} inputs. Writing outputs to {output_dir}")

    successes = 0
    failures = 0

    if args.workers <= 1:
        for inp in inputs:
            path, ok, msg = process_one(
                input_path=inp,
                transcript_path=transcript_path,
                output_dir=output_dir,
                reasoning_effort=args.reasoning_effort,
            )
            print(f"[{'OK' if ok else 'FAIL'}] {path.name}: {msg}")
            successes += 1 if ok else 0
            failures += 0 if ok else 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    process_one,
                    inp,
                    transcript_path,
                    output_dir,
                    args.reasoning_effort,
                )
                for inp in inputs
            ]
            for fut in as_completed(futures):
                path, ok, msg = fut.result()
                print(f"[{'OK' if ok else 'FAIL'}] {path.name}: {msg}")
                successes += 1 if ok else 0
                failures += 0 if ok else 1

    print(f"Done. Success {successes}/{len(inputs)}; Failures {failures}")


if __name__ == "__main__":
    main()

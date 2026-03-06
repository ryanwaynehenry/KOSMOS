#!/usr/bin/env python3
"""
Batch runner for align_kg_to_umls.py.

Runs UMLS alignment on every JSON file in a directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def _collect_inputs(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    globber = input_dir.rglob if recursive else input_dir.glob
    return sorted([p for p in globber(pattern) if p.is_file()])


def _iter_commands(
    inputs: Iterable[Path],
    input_dir: Path,
    output_dir: Path | None,
    align_script: Path,
    min_score: float,
    no_db: bool,
    use_faiss: bool,
    overwrite: bool,
) -> Iterable[tuple[Path, Path | None, List[str]]]:
    for input_path in inputs:
        output_path = None
        if output_dir is not None:
            output_path = output_dir / input_path.relative_to(input_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(align_script),
            str(input_path),
            "--min-score",
            str(min_score),
        ]
        if output_path is not None:
            cmd.extend(["--output", str(output_path)])
        if no_db:
            cmd.append("--no-db")
        if use_faiss:
            cmd.append("--use-faiss")
        if overwrite:
            cmd.append("--overwrite")

        yield input_path, output_path, cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run align_kg_to_umls.py on all JSON files in a directory."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing JSON files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. If omitted, files are updated in place.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern to match JSON files (default: *.json).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.8,
        help="Minimum match score required to update canonical_name (default: 0.8).",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip MRCONSO database lookup and use FAISS only (if enabled).",
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Enable FAISS fallback lookup (requires local index files).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ontology fields when present.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining files if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir = args.output_dir.resolve() if args.output_dir else None

    align_script = Path(__file__).resolve().with_name("align_kg_to_umls.py")
    if not align_script.is_file():
        raise SystemExit(f"Could not find align script at: {align_script}")

    inputs = _collect_inputs(input_dir, args.pattern, args.recursive)
    if not inputs:
        raise SystemExit(
            f"No files matched pattern '{args.pattern}' in directory: {input_dir}"
        )

    print(f"Found {len(inputs)} files to process.")
    if output_dir:
        print(f"Writing outputs to: {output_dir}")
    else:
        print("Updating files in place.")

    ok_count = 0
    fail_count = 0

    for input_path, output_path, cmd in _iter_commands(
        inputs=inputs,
        input_dir=input_dir,
        output_dir=output_dir,
        align_script=align_script,
        min_score=args.min_score,
        no_db=args.no_db,
        use_faiss=args.use_faiss,
        overwrite=args.overwrite,
    ):
        target = output_path if output_path is not None else input_path
        if args.dry_run:
            print(f"[DRY-RUN] {input_path} -> {target}")
            print("  " + " ".join(cmd))
            ok_count += 1
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            ok_count += 1
            print(f"[OK] {input_path.name} -> {target}")
            if result.stdout.strip():
                print("  " + result.stdout.strip().splitlines()[-1])
        else:
            fail_count += 1
            print(f"[FAIL] {input_path.name}")
            err = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            print("  " + err.splitlines()[-1])
            if not args.continue_on_error:
                break

    total = ok_count + fail_count
    print(f"Done. Success {ok_count}/{total}; Failures {fail_count}")
    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

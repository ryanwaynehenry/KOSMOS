#!/usr/bin/env python3
"""
make_eval_csv.py

Creates a CSV with columns:
dataset,encounter_id,dialogue,note

Inputs:
- --note_path: path to one generated note .txt (must end in _<index>.txt)
  The script scans the same folder for other .txt note files with the same prefix.
- --raw_json: path to raw json containing items with keys: src, file
  where file is "<encounter_id>-<dataset>"

Output:
- CSV (defaults next to notes) with minimal quoting:
  - headers not quoted
  - dataset + encounter_id not quoted
  - dialogue + note quoted (because they contain newlines)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STEM_RE = re.compile(r"^(?P<prefix>.*?)(?P<idx>\d+)$")


def load_raw_items(raw_json_path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(raw_json_path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    if isinstance(obj, list):
        return obj
    raise ValueError(
        f"Unsupported raw json structure in {raw_json_path}. "
        "Expected a list, or a dict with a top-level 'data' list."
    )


def parse_file_field(file_field: str) -> Tuple[str, str]:
    """
    file_field is "<encounter_id>-<dataset>", e.g. "D2N168-virtassist"
    Returns (dataset, encounter_id)
    """
    if "-" not in file_field:
        raise ValueError(f"Unexpected file field (missing '-'): {file_field!r}")
    encounter_id, dataset = file_field.split("-", 1)
    encounter_id = encounter_id.strip()
    dataset = dataset.strip()
    if not encounter_id or not dataset:
        raise ValueError(f"Unexpected file field (empty parts): {file_field!r}")
    return dataset, encounter_id


def get_prefix_and_index(p: Path) -> Optional[Tuple[str, int]]:
    """
    For a filename stem like "soap_clef_taskC_test3_full_11" returns:
    ("soap_clef_taskC_test3_full_", 11)
    """
    m = STEM_RE.match(p.stem)
    if not m:
        return None
    return m.group("prefix"), int(m.group("idx"))


def find_related_txt_notes(seed_note_path: Path) -> List[Tuple[int, Path]]:
    if seed_note_path.suffix.lower() != ".txt":
        raise ValueError(f"--note_path must be a .txt file, got: {seed_note_path.name}")

    seed_pi = get_prefix_and_index(seed_note_path)
    if not seed_pi:
        raise ValueError(
            f"Seed note filename must end with digits before .txt, got: {seed_note_path.name}"
        )

    seed_prefix, _ = seed_pi
    found: List[Tuple[int, Path]] = []

    for p in seed_note_path.parent.glob(f"{seed_prefix}*.txt"):
        pi = get_prefix_and_index(p)
        if not pi:
            continue
        prefix, idx = pi
        if prefix == seed_prefix:
            found.append((idx, p))

    return sorted(found, key=lambda t: t[0])


def default_output_path(seed_note_path: Path) -> Path:
    pi = get_prefix_and_index(seed_note_path)
    assert pi is not None
    prefix, _ = pi
    safe_prefix = prefix[:-1] if prefix.endswith("_") else prefix
    return seed_note_path.parent / f"{safe_prefix}_eval_input.csv"


def build_rows(raw_items: List[Dict[str, Any]], note_files: List[Tuple[int, Path]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for idx, note_path in note_files:
        if idx < 0 or idx >= len(raw_items):
            raise IndexError(
                f"Note index {idx} from file {note_path.name} is out of range for raw json length {len(raw_items)}."
            )

        item = raw_items[idx]
        if not isinstance(item, dict):
            raise ValueError(f"raw_items[{idx}] is not an object/dict.")

        if "file" not in item or "src" not in item:
            raise ValueError(
                f"raw_items[{idx}] missing required keys 'file' and/or 'src'. Found keys: {sorted(item.keys())}"
            )

        dataset, encounter_id = parse_file_field(str(item["file"]))
        dialogue = str(item["src"])
        note_text = note_path.read_text(encoding="utf-8", errors="replace")

        rows.append(
            {
                "dataset": dataset,
                "encounter_id": encounter_id,
                "dialogue": dialogue,
                "note": note_text,
            }
        )

    return rows


def write_csv(out_path: Path, rows: List[Dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "encounter_id", "dialogue", "note"],
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--note_path",
        required=True,
        help="Path to one generated note .txt, e.g. data/processed/soap_clef_taskC_test3_full_0.txt",
    )
    ap.add_argument(
        "--raw_json",
        required=True,
        help="Path to raw json, e.g. data/raw/aci/clef_taskC_test3_full.json",
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path. If omitted, writes next to notes as <prefix>_eval_input.csv",
    )
    args = ap.parse_args()

    seed_note_path = Path(args.note_path)
    raw_json_path = Path(args.raw_json)

    if not seed_note_path.exists():
        raise FileNotFoundError(f"note_path does not exist: {seed_note_path}")
    if not raw_json_path.exists():
        raise FileNotFoundError(f"raw_json does not exist: {raw_json_path}")

    note_files = find_related_txt_notes(seed_note_path)
    raw_items = load_raw_items(raw_json_path)
    rows = build_rows(raw_items, note_files)

    out_path = Path(args.out_csv) if args.out_csv else default_output_path(seed_note_path)
    write_csv(out_path, rows)

    print(f"Wrote {len(rows)} rows to: {out_path}")


if __name__ == "__main__":
    main()

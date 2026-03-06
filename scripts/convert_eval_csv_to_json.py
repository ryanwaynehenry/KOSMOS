#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="Convert make_eval_csv output to JSON list with example_id/input/output."
    )
    ap.add_argument("csv_path", help="Path to *_eval_input.csv produced by make_eval_csv.py")
    ap.add_argument(
        "--out",
        help="Output JSON path (default: same name as CSV, with .json).",
    )
    ap.add_argument(
        "--id-from",
        choices=["dataset-encounter", "encounter", "row"],
        default="encounter",
        help="How to construct example_id (default uses dataset-encounter_id).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".json")

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    examples = []
    for i, row in enumerate(rows):
        dataset = (row.get("dataset") or "").strip()
        encounter = (row.get("encounter_id") or "").strip()
        if args.id_from == "dataset-encounter":
            example_id = f"{dataset}-{encounter}".strip("-") or f"row-{i}"
        elif args.id_from == "encounter":
            example_id = encounter or f"row-{i}"
        else:
            example_id = f"row-{i}"

        examples.append(
            {
                "example_id": example_id,
                "input": row.get("dialogue", ""),
                "output": row.get("note", ""),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {out_path}")

if __name__ == "__main__":
    main()

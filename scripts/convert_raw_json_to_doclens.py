#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_items(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Expected top-level list or dict with a 'data' list.")


def example_id_from_file_field(file_field: str, fallback: str) -> str:
    head = str(file_field).split("-", 1)[0].strip()
    return head or fallback


def default_out_path(raw_path: Path) -> Path:
    return raw_path.with_name(f"{raw_path.stem}_eval.json")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert raw ACI JSON (with src/tgt/file) into list of {example_id, reference, input}."
    )
    ap.add_argument("raw_json", help="Path to raw JSON, e.g. data/raw/aci/clinicalnlp_taskC_test2_full.json")
    ap.add_argument(
        "--out",
        help="Output JSON path. Default: alongside input, with _eval.json suffix.",
    )
    args = ap.parse_args()

    raw_path = Path(args.raw_json)
    out_path = Path(args.out) if args.out else default_out_path(raw_path)

    items = load_items(raw_path)
    examples: List[Dict[str, str]] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not an object.")
        example_id = example_id_from_file_field(item.get("file", ""), f"row-{idx}")
        examples.append(
            {
                "example_id": example_id,
                "reference": str(item.get("tgt", "")),
                "input": str(item.get("src", "")),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()

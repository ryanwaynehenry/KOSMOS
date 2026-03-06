#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple

TAG_RE = re.compile(r"\[(doctor|patient)\]", re.IGNORECASE)


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def add_turn_indices(text: str) -> Tuple[str, int]:
    idx = 0

    def repl(match) -> str:
        nonlocal idx
        out = f"[{idx}]{match.group(0)}"
        idx += 1
        return out

    return TAG_RE.sub(repl, text), idx


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def process_items(items: List[Any]) -> Tuple[List[Any], int, int]:
    new_items: List[Any] = []
    updated = 0
    total_tags = 0

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            new_items.append(item)
            continue

        if "input" not in item:
            new_items.append(item)
            continue

        input_val = item.get("input")
        if not isinstance(input_val, str):
            eprint(f"[warn] Item {i} input is not a string; leaving unchanged.")
            new_items.append(item)
            continue

        new_input, count = add_turn_indices(input_val)
        if count:
            updated += 1
            total_tags += count

        new_item = dict(item)
        new_item["input"] = new_input
        new_items.append(new_item)

    return new_items, updated, total_tags


def default_out_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_turn_ids.json")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Add turn indices like [0] before each [doctor]/[patient] tag in the input field."
        )
    )
    ap.add_argument("input_json", help="Path to JSON file (list or dict with 'data' list).")
    ap.add_argument(
        "--out",
        help="Output JSON path. Default: alongside input with _turn_ids.json suffix.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_json)
    if not input_path.is_file():
        raise ValueError(f"Input JSON not found: {input_path}")

    obj = load_json(input_path)
    if isinstance(obj, list):
        new_items, updated, total_tags = process_items(obj)
        out_obj: Any = new_items
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        new_items, updated, total_tags = process_items(obj["data"])
        out_obj = dict(obj)
        out_obj["data"] = new_items
    else:
        raise ValueError("Expected top-level list or dict with a 'data' list.")

    out_path = Path(args.out) if args.out else default_out_path(input_path)
    save_json(out_path, out_obj)
    print(f"Wrote {updated} items with {total_tags} turn tags to {out_path}")


if __name__ == "__main__":
    main()

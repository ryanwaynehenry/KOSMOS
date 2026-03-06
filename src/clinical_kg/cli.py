import argparse
import json
from pathlib import Path
from typing import List, Optional

from clinical_kg.pipeline import save_processed_transcript


def _parse_index_selection(selection: str, total: int) -> List[int]:
    """
    Parse selection syntax:
      - "7" -> [7]
      - "0-2" -> [0, 1, 2]
      - "3+" -> [3, 4, ..., total-1]
    """
    if selection is None:
        raise ValueError("json-index is required for JSON datasets.")
    sel = str(selection).strip()
    if not sel:
        raise ValueError("json-index cannot be empty.")

    if sel.endswith("+"):
        start_str = sel[:-1]
        start = int(start_str)
        if start < 0 or start >= total:
            raise IndexError(f"json-index start {start} out of range for data array of size {total}")
        return list(range(start, total))

    if "-" in sel:
        parts = sel.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid json-index range: {sel}")
        start = int(parts[0])
        end = int(parts[1])
        if start < 0 or end < start:
            raise ValueError(f"Invalid json-index range: {sel}")
        if end >= total:
            raise IndexError(f"json-index end {end} out of range for data array of size {total}")
        return list(range(start, end + 1))

    idx = int(sel)
    if idx < 0 or idx >= total:
        raise IndexError(f"json-index {idx} out of range for data array of size {total}")
    return [idx]


def main():
    parser = argparse.ArgumentParser(
        description="Process a raw transcript into turns and mentions (no UMLS/KG). "
        "Supports plain text transcripts or a JSON dataset with a 'data' array and 'src' fields."
    )
    parser.add_argument("transcript", help="Path to transcript file or JSON dataset")
    parser.add_argument(
        "--encounter-id",
        help="Override encounter identifier. Defaults to transcript filename stem.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also write turns/mentions to data/interim/{transcript_name}*.json",
    )
    parser.add_argument(
        "--json-index",
        type=str,
        help="If transcript is a JSON dataset, index selection to process. Supports single (e.g., 7), range (e.g., 0-2), or open range (e.g., 3+).",
    )
    args = parser.parse_args()

    encounter_id: Optional[str] = args.encounter_id

    if Path(args.transcript).suffix.lower() == ".json":
        if args.json_index is None:
            raise ValueError("When transcript is a JSON dataset, you must supply --json-index.")
        dataset_path = Path(args.transcript)
        data_obj = json.loads(dataset_path.read_text(encoding="utf-8"))
        items = data_obj.get("data") if isinstance(data_obj, dict) else None
        if not isinstance(items, list):
            raise ValueError("JSON dataset must contain a 'data' array.")
        indices = _parse_index_selection(args.json_index, len(items))

        output_paths = []
        for idx in indices:
            entry = items[idx]
            transcript_text = entry.get("src")
            if not transcript_text:
                raise ValueError(f"No 'src' field found at index {idx} in dataset.")
            base_name = f"{dataset_path.stem}_{idx}"
            output_stem = base_name
            resolved_encounter = encounter_id or base_name

            output_path = None
            try:
                output_path = save_processed_transcript(
                    transcript_path=None,
                    encounter_id=resolved_encounter,
                    save_intermediate=args.save_intermediate,
                    transcript_text=transcript_text,
                    output_stem=output_stem,
                )
                output_paths.append(output_path)
                print(f"Wrote {output_path}")
            except Exception as exc:
                target = output_path or output_stem
                print(f"Failed to create {target}: {exc}")
                continue
    else:
        output_path = save_processed_transcript(
            transcript_path=args.transcript,
            encounter_id=encounter_id,
            save_intermediate=args.save_intermediate,
            transcript_text=None,
            output_stem=None,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

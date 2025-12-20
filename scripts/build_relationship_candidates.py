"""
Build relationship candidates from an interim JSON by delegating to clinical_kg.kg.relations.

Usage:
  python scripts/build_relationship_candidates.py data\interim\altered_session_2348_1.json --output data/interim/file_with_relationships.json
"""

import argparse
import json
from pathlib import Path

from clinical_kg.config import load_config
from clinical_kg.kg.relations import build_relationship_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build relationship candidates from node JSON.")
    parser.add_argument("input", type=Path, help="Path to interim file with turns and nodes.")
    parser.add_argument("--output", type=Path, help="Where to write JSON with relationship candidates.")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of pairs per LLM batch.")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    turns = data.get("turns", [])
    nodes = data.get("nodes", [])

    cfg = load_config()
    relationship_candidates = build_relationship_candidates(
        turns=turns,
        nodes=nodes,
        cfg=cfg,
        batch_size=args.batch_size,
    )

    output = dict(data)
    output["relationship_candidates"] = relationship_candidates

    out_path = args.output or args.input
    out_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

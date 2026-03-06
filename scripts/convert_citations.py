#!/usr/bin/env python3
"""
Convert SOAP note citations like [n0004][pn0001_n0048] into transcript turn citations like [0][12].

Inputs:
  1) --input_json : path to a JSON file (or a directory of JSON files) where each file is a list of objects
     with at least an "output" field (your SOAP note text).
  2) --kg_dir     : path to a directory containing per-encounter KG JSON files (one per index, like *_0.json).

How it works:
  - For each item i in the input list, find the KG JSON file that corresponds to index i.
  - Build maps:
      nXXXX -> set(turn_numbers)
      pnXXXX_nYYYY -> set(turn_numbers)   (from llm_relation.evidence_turn_ids)
  - For each line in the SOAP "output", replace all [n...]/[pn...] citations on that line with a
    de-duplicated set of turn citations [0][1][5] (0-based), sorted ascending.

Example:
  " ... yesterday. [n0001][pn0001_n0012]  "
  -> " ... yesterday. [0][4][10]  "

Notes:
  - Turn ids in KG files are assumed to be 1-based like "t0001"; we convert to 0-based ints.
  - We preserve the original line’s trailing whitespace (including double-spaces used for Markdown hard breaks).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


CIT_RE = re.compile(r"\[(n\d+|pn\d+_n\d+)\]")
NODE_SUFFIX_RE = re.compile(r"(n\d+)$")
IDX_FROM_FILENAME_RE = re.compile(r"_(\d+)\.json$", re.IGNORECASE)


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_prefix_from_primary_filename(primary_path: str) -> str:
    """
    Tries to infer a base prefix so that prefix_{i}.json is a good first guess.
    Example:
      soap_clef_taskC_test3_full_eval_input.json -> clef_taskC_test3_full
    """
    name = os.path.basename(primary_path)
    stem = os.path.splitext(name)[0]

    if stem.startswith("soap_"):
        stem = stem[len("soap_") :]

    # strip common trailing markers
    for suf in (
        "_full_eval_input",
        "_eval_input",
        "_full_eval",
        "_eval",
        "_full",
        "_input",
    ):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    return stem


def build_kg_index(kg_dir: str) -> Dict[int, List[str]]:
    """
    Map index -> list of candidate KG json paths whose filename ends with _{index}.json
    """
    mapping: Dict[int, List[str]] = defaultdict(list)
    for p in glob(os.path.join(kg_dir, "*.json")):
        base = os.path.basename(p)
        m = IDX_FROM_FILENAME_RE.search(base)
        if not m:
            continue
        idx = int(m.group(1))
        mapping[idx].append(p)
    return dict(mapping)


def best_match_by_similarity(prefix: str, candidates: List[str], idx: int) -> Optional[str]:
    """
    Choose candidate whose filename (minus _{idx}.json) is most similar to prefix.
    """
    if not candidates:
        return None

    def score(path: str) -> float:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        stem = re.sub(rf"_{idx}$", "", stem)
        return SequenceMatcher(None, prefix, stem).ratio()

    return max(candidates, key=score)


def resolve_kg_path(
    idx: int,
    kg_dir: str,
    kg_index: Dict[int, List[str]],
    preferred_prefix: Optional[str],
) -> Optional[str]:
    """
    Resolve per-item KG json path using:
      1) preferred_prefix_{idx}.json if exists
      2) if unique candidate for idx in directory
      3) similarity match among candidates for idx
    """
    if preferred_prefix:
        direct = os.path.join(kg_dir, f"{preferred_prefix}_{idx}.json")
        if os.path.isfile(direct):
            return direct

    candidates = kg_index.get(idx, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and preferred_prefix:
        return best_match_by_similarity(preferred_prefix, candidates, idx)

    return None


def turn_id_to_zero_based_int(tid: str) -> Optional[int]:
    """
    "t0001" -> 0, "t0010" -> 9
    """
    if not isinstance(tid, str):
        return None
    m = re.search(r"(\d+)$", tid)
    if not m:
        return None
    n = int(m.group(1)) - 1
    if n < 0:
        return None
    return n


@dataclass
class CitationMaps:
    node_to_turns: Dict[str, Set[int]]
    rel_to_turns: Dict[str, Set[int]]


def extract_nodes_list(kg: Any) -> List[Dict[str, Any]]:
    if isinstance(kg, dict):
        for key in ("nodes", "node_list", "nodeList"):
            if key in kg and isinstance(kg[key], list):
                return kg[key]
    return []


def extract_relationship_candidates_list(kg: Any) -> List[Dict[str, Any]]:
    if isinstance(kg, dict):
        for key in (
            "relationship_candidates",
            "relationshipCandidates",
            "relationships",
            "edges",
        ):
            if key in kg and isinstance(kg[key], list):
                return kg[key]
    return []


def build_citation_maps(kg: Any) -> CitationMaps:
    node_to_turns: Dict[str, Set[int]] = defaultdict(set)
    rel_to_turns: Dict[str, Set[int]] = defaultdict(set)

    # Nodes: map suffix nXXXX -> node.turn_ids
    for node in extract_nodes_list(kg):
        node_id = node.get("id", "")
        if not isinstance(node_id, str):
            continue
        m = NODE_SUFFIX_RE.search(node_id)
        if not m:
            continue
        short_id = m.group(1)  # "n0001"

        turn_ids = node.get("turn_ids", [])
        if not isinstance(turn_ids, list):
            continue
        for tid in turn_ids:
            z = turn_id_to_zero_based_int(tid)
            if z is not None:
                node_to_turns[short_id].add(z)

    # Relationships: map pair_id -> llm_relation.evidence_turn_ids
    for rel in extract_relationship_candidates_list(kg):
        pair_id = rel.get("pair_id")
        if not isinstance(pair_id, str):
            continue
        llm_rel = rel.get("llm_relation") or {}
        evidence = llm_rel.get("evidence_turn_ids", [])
        if not isinstance(evidence, list):
            evidence = []
        for tid in evidence:
            z = turn_id_to_zero_based_int(tid)
            if z is not None:
                rel_to_turns[pair_id].add(z)

    return CitationMaps(node_to_turns=dict(node_to_turns), rel_to_turns=dict(rel_to_turns))


def convert_output_text(
    text: str,
    maps: CitationMaps,
    keep_unresolved: bool = False,
    context: str = "",
) -> str:
    if not isinstance(text, str) or not text:
        return text

    lines = text.splitlines()
    out_lines: List[str] = []

    for li, line in enumerate(lines):
        # Preserve trailing whitespace exactly (including double-spaces used for markdown breaks)
        trailing_ws_match = re.search(r"(\s*)$", line)
        trailing_ws = trailing_ws_match.group(1) if trailing_ws_match else ""
        core = line[: len(line) - len(trailing_ws)] if trailing_ws else line

        citations = CIT_RE.findall(core)
        if not citations:
            out_lines.append(line)
            continue

        turns: Set[int] = set()
        unresolved: List[str] = []

        for cit in citations:
            if cit.startswith("n"):
                s = maps.node_to_turns.get(cit)
            else:
                s = maps.rel_to_turns.get(cit)
            if s:
                turns.update(s)
            else:
                unresolved.append(cit)

        # Remove the old citations from the line core
        core_no_cits = CIT_RE.sub("", core)
        core_no_cits = re.sub(r"\s+$", "", core_no_cits)

        # Build replacement citations
        if turns:
            new_cits = "".join(f"[{t}]" for t in sorted(turns))
            if core_no_cits:
                if not core_no_cits.endswith(" "):
                    core_no_cits += " "
                core_no_cits += new_cits
            else:
                core_no_cits = new_cits

        # Optionally keep unresolved citations (after the new ones), otherwise drop and warn
        if unresolved:
            if keep_unresolved:
                core_no_cits += "".join(f"[{u}]" for u in unresolved)
            else:
                eprint(
                    f"[warn] Unresolved citations dropped at {context} line {li}: {unresolved}"
                )

        out_lines.append(core_no_cits + trailing_ws)

    return "\n".join(out_lines)


def process_primary_file(
    input_path: str,
    kg_dir: str,
    output_path: str,
    keep_unresolved: bool,
) -> None:
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list at top-level in {input_path}")

    preferred_prefix = infer_prefix_from_primary_filename(input_path)
    kg_index = build_kg_index(kg_dir)

    converted = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            converted.append(item)
            continue

        kg_path = resolve_kg_path(i, kg_dir, kg_index, preferred_prefix)
        if not kg_path:
            eprint(f"[warn] No KG file found for index {i} (input: {input_path}); leaving output unchanged.")
            converted.append(item)
            continue

        try:
            kg = load_json(kg_path)
            maps = build_citation_maps(kg)
        except Exception as ex:
            eprint(f"[warn] Failed reading/parsing KG file {kg_path} for index {i}: {ex}. Leaving output unchanged.")
            converted.append(item)
            continue

        new_item = dict(item)
        if "output" in new_item:
            ctx = f"{os.path.basename(input_path)} idx={i} kg={os.path.basename(kg_path)}"
            new_item["output"] = convert_output_text(
                new_item.get("output", ""),
                maps,
                keep_unresolved=keep_unresolved,
                context=ctx,
            )
        converted.append(new_item)

    save_json(output_path, converted)
    print(f"Wrote: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to a JSON file, or a directory of JSON files, to convert.")
    ap.add_argument("--kg_dir", required=True, help="Directory containing per-encounter KG JSON files (like *_0.json).")
    ap.add_argument(
        "--output",
        required=False,
        help=(
            "Output file path (if input_json is a file) or output directory (if input_json is a directory). "
            "Defaults to creating *_turn_cites.json next to each input file."
        ),
    )
    ap.add_argument(
        "--keep_unresolved",
        action="store_true",
        help="If set, keep unresolved [n...]/[pn...] citations instead of dropping them.",
    )
    args = ap.parse_args()

    input_path = args.input_json
    kg_dir = args.kg_dir
    out = args.output
    keep_unresolved = bool(args.keep_unresolved)

    if not os.path.isdir(kg_dir):
        raise ValueError(f"--kg_dir is not a directory: {kg_dir}")

    if os.path.isdir(input_path):
        in_files = sorted(glob(os.path.join(input_path, "*.json")))
        if not in_files:
            raise ValueError(f"No .json files found in input directory: {input_path}")

        out_dir = out if out else input_path
        os.makedirs(out_dir, exist_ok=True)

        for fp in in_files:
            base = os.path.basename(fp)
            stem = os.path.splitext(base)[0]
            out_path = os.path.join(out_dir, f"{stem}_turn_cites.json")
            try:
                process_primary_file(fp, kg_dir, out_path, keep_unresolved)
            except Exception as ex:
                eprint(f"[warn] Failed converting {fp}: {ex}. Skipping.")
        return

    # input_path is a file
    if not os.path.isfile(input_path):
        raise ValueError(f"--input_json not found: {input_path}")

    if out:
        out_path = out
    else:
        base = os.path.basename(input_path)
        stem = os.path.splitext(base)[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(input_path)), f"{stem}_turn_cites.json")

    process_primary_file(input_path, kg_dir, out_path, keep_unresolved)


if __name__ == "__main__":
    main()

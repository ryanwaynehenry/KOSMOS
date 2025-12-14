"""
Merge FAISS index shards and their mappings into a single index/mapping.

Defaults to merging two shards to keep memory reasonable; adjust SHARD_GLOB and
max_shards as needed (e.g., set max_shards=None to merge all matches).

Usage:
    python scripts/merge_faiss_shards.py 
        --index_pattern "LNC_sapbert_part*.faiss" 
        --mapping_pattern "LNC_sapbert_mapping_part*.json" 
        --output_index "LNC_sapbert_merged.faiss" 
        --output_mapping "LNC_sapbert_mapping_merged.json" 
        --max_shards 2
"""

import argparse
import glob
import json
import os

import faiss


def merge_shards(index_paths, mapping_paths, out_index, out_mapping):
    index_paths = sorted(index_paths)
    mapping_paths = sorted(mapping_paths)
    if len(index_paths) != len(mapping_paths):
        raise ValueError("Number of index and mapping shards must match")

    merged_index = None
    all_cuis, all_terms, all_sources = [], [], []

    for ipath, mpath in zip(index_paths, mapping_paths):
        print(f"Merging {ipath} and {mpath}")
        idx = faiss.read_index(ipath)
        if merged_index is None:
            merged_index = faiss.IndexFlatIP(idx.d)
        merged_index.merge_from(idx, 0)

        with open(mpath, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            all_cuis.extend(mapping.get("cuis", []))
            all_terms.extend(mapping.get("terms", []))
            all_sources.extend(mapping.get("sources", []))

    print(f"Writing merged index to {out_index}")
    faiss.write_index(merged_index, out_index)

    print(f"Writing merged mapping to {out_mapping}")
    with open(out_mapping, "w", encoding="utf-8") as f:
        json.dump(
            {"cuis": all_cuis, "terms": all_terms, "sources": all_sources},
            f,
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Merge FAISS shards and mappings.")
    parser.add_argument("--index_pattern", required=True, help="Glob for shard indexes, e.g., LNC_sapbert_part*.faiss")
    parser.add_argument("--mapping_pattern", required=True, help="Glob for shard mappings, e.g., LNC_sapbert_mapping_part*.json")
    parser.add_argument("--output_index", required=True, help="Path for merged FAISS index")
    parser.add_argument("--output_mapping", required=True, help="Path for merged mapping JSON")
    parser.add_argument("--max_shards", type=int, default=2, help="How many shards to merge (None for all)")
    args = parser.parse_args()

    index_paths = sorted(glob.glob(args.index_pattern))
    mapping_paths = sorted(glob.glob(args.mapping_pattern))
    if args.max_shards is not None:
        index_paths = index_paths[: args.max_shards]
        mapping_paths = mapping_paths[: args.max_shards]

    if not index_paths or not mapping_paths:
        raise SystemExit("No shards matched the given patterns.")

    merge_shards(index_paths, mapping_paths, args.output_index, args.output_mapping)


if __name__ == "__main__":
    main()

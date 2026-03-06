#!/usr/bin/env python3
"""
Build a UMLS FAISS index and mapping JSON from MRCONSO in MySQL.

This script:
1) Reads UMLS terms from MRCONSO for selected SABs.
2) Embeds terms with SapBERT.
3) Builds one or more FAISS shards.
4) Merges shards into a final index/mapping by default.

Example:
  python scripts/build_umls_faiss_index.py \
    --index-path UMLS_sapbert.faiss \
    --mapping-path UMLS_sapbert_mapping.json \
    --sabs SNOMEDCT_US,RXNORM,LNC \
    --batch-size 256 \
    --shard-size 2000000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Allow direct execution from repo root without editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


Record = Tuple[str, str, str]  # (cui, term, sab)
Shard = Tuple[Path, Path]  # (index_path, mapping_path)


def import_faiss_numpy():
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'faiss'. Install 'faiss-cpu' or 'faiss-gpu' to build/merge indices."
        ) from exc

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("Missing dependency 'numpy'. Install 'numpy' to build indices.") from exc

    return faiss, np


def parse_sabs(raw: str) -> List[str]:
    values = [v.strip().upper() for v in (raw or "").split(",")]
    values = [v for v in values if v]
    if not values:
        raise ValueError("At least one SAB must be provided via --sabs.")
    return values


def fetch_umls_records(sabs: Sequence[str], limit: int | None) -> List[Record]:
    from clinical_kg.config import load_config
    from clinical_kg.umls.connection import create_connection

    cfg = load_config()
    conn = create_connection(cfg.db)
    cursor = conn.cursor()
    try:
        placeholders = ", ".join(["%s"] * len(sabs))
        sql = (
            "SELECT CUI, STR, SAB "
            "FROM MRCONSO "
            "WHERE LAT = 'ENG' "
            "  AND SUPPRESS = 'N' "
            f"  AND SAB IN ({placeholders})"
        )
        params: List[object] = list(sabs)
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))

        cursor.execute(sql, params)

        seen = set()
        records: List[Record] = []
        for row in cursor:
            if not isinstance(row, tuple) or len(row) != 3:
                continue
            cui, term, sab = row
            if term is None:
                continue
            rec = (str(cui), str(term), str(sab))
            if rec in seen:
                continue
            seen.add(rec)
            records.append(rec)
    finally:
        cursor.close()
        conn.close()

    return records


def write_mapping(path: Path, cuis: List[str], terms: List[str], sources: List[str]) -> None:
    payload = {"cuis": cuis, "terms": terms, "sources": sources}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def merge_shards(shards: Sequence[Shard], out_index: Path, out_mapping: Path) -> None:
    faiss, _np = import_faiss_numpy()

    merged_index = None
    all_cuis: List[str] = []
    all_terms: List[str] = []
    all_sources: List[str] = []

    for idx_path, map_path in shards:
        index_obj = faiss.read_index(str(idx_path))
        if merged_index is None:
            merged_index = faiss.IndexFlatIP(index_obj.d)
        merged_index.merge_from(index_obj, 0)

        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        all_cuis.extend([str(x) for x in mapping.get("cuis", [])])
        all_terms.extend([str(x) for x in mapping.get("terms", [])])
        all_sources.extend([str(x) for x in mapping.get("sources", [])])

    if merged_index is None:
        raise RuntimeError("No shards were provided for merge.")

    out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(merged_index, str(out_index))
    write_mapping(out_mapping, all_cuis, all_terms, all_sources)


def build_shards(
    records: Sequence[Record],
    index_path: Path,
    mapping_path: Path,
    batch_size: int,
    shard_size: int,
) -> List[Shard]:
    faiss, np = import_faiss_numpy()
    from clinical_kg.umls.sapbert_embedder import SapBERTEmbedder

    if not records:
        raise ValueError("No UMLS records found; cannot build FAISS index.")

    embedder = SapBERTEmbedder()
    shard_count = math.ceil(len(records) / shard_size)
    shards: List[Shard] = []

    for shard_idx, start in enumerate(range(0, len(records), shard_size)):
        end = min(start + shard_size, len(records))
        shard = records[start:end]

        cuis = [cui for (cui, _, _) in shard]
        terms = [term for (_, term, _) in shard]
        sources = [sab for (_, _, sab) in shard]

        print(
            f"[{shard_idx + 1}/{shard_count}] Embedding records {start}..{end - 1} "
            f"({len(shard)} terms)"
        )
        vectors = embedder.encode(terms, batch_size=batch_size).astype(np.float32)
        if vectors.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")

        index_obj = faiss.IndexFlatIP(vectors.shape[1])
        index_obj.add(vectors)

        if shard_count == 1:
            shard_index_path = index_path
            shard_mapping_path = mapping_path
        else:
            shard_index_path = index_path.with_name(
                f"{index_path.stem}_part{shard_idx:03d}{index_path.suffix}"
            )
            shard_mapping_path = mapping_path.with_name(
                f"{mapping_path.stem}_part{shard_idx:03d}{mapping_path.suffix}"
            )

        shard_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index_obj, str(shard_index_path))
        write_mapping(shard_mapping_path, cuis, terms, sources)

        print(f"  Wrote index shard:   {shard_index_path}")
        print(f"  Wrote mapping shard: {shard_mapping_path}")
        shards.append((shard_index_path, shard_mapping_path))

    return shards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build UMLS SapBERT FAISS index and mapping from MRCONSO."
    )
    parser.add_argument(
        "--index-path",
        default="UMLS_sapbert.faiss",
        help="Final FAISS index output path (default: UMLS_sapbert.faiss).",
    )
    parser.add_argument(
        "--mapping-path",
        default="UMLS_sapbert_mapping.json",
        help="Final mapping JSON output path (default: UMLS_sapbert_mapping.json).",
    )
    parser.add_argument(
        "--sabs",
        default="SNOMEDCT_US,RXNORM,LNC",
        help="Comma-separated SABs to include (default: SNOMEDCT_US,RXNORM,LNC).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="SapBERT embedding batch size (default: 256).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2_000_000,
        help="Max records per shard before splitting (default: 2,000,000).",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Keep shard files only; do not merge to final output paths.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="If shards are merged, keep shard files instead of deleting them.",
    )
    args = parser.parse_args()

    index_path = Path(args.index_path)
    mapping_path = Path(args.mapping_path)
    sabs = parse_sabs(args.sabs)

    print("Loading UMLS records from MySQL...")
    records = fetch_umls_records(sabs=sabs, limit=args.limit)
    print(f"Loaded {len(records)} unique records for SABs: {', '.join(sabs)}")
    if not records:
        raise SystemExit("No records loaded from MRCONSO. Check DB connection and SAB values.")

    shards = build_shards(
        records=records,
        index_path=index_path,
        mapping_path=mapping_path,
        batch_size=max(1, int(args.batch_size)),
        shard_size=max(1, int(args.shard_size)),
    )

    if len(shards) > 1 and not args.no_merge:
        print("Merging shards into final output files...")
        merge_shards(shards, out_index=index_path, out_mapping=mapping_path)
        print(f"Wrote merged index:   {index_path}")
        print(f"Wrote merged mapping: {mapping_path}")

        if not args.keep_shards:
            for idx_path, map_path in shards:
                if idx_path != index_path and idx_path.exists():
                    idx_path.unlink()
                if map_path != mapping_path and map_path.exists():
                    map_path.unlink()
            print("Deleted shard files after merge.")

    elif len(shards) == 1:
        print(f"Wrote index:   {shards[0][0]}")
        print(f"Wrote mapping: {shards[0][1]}")
    else:
        print("No merge requested; shard files were left on disk.")


if __name__ == "__main__":
    main()

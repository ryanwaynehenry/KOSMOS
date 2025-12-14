# build_umls_faiss_index.py

import json
from typing import List, Tuple

import faiss
import numpy as np

from clinical_kg.config import load_config
from clinical_kg.umls.connection import create_connection
from sapbert_embedder import SapBERTEmbedder

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# 1. Load UMLS terms
# -----------------------------

def load_umls_terms_from_mysql(limit: int = None) -> List[Tuple[str, str, str]]:
    """
    Load (CUI, term_string, ontology) triples from a MySQL UMLS mirror.

    - CUI: UMLS Concept Unique Identifier
    - term_string: STR field from MRCONSO (the human-readable term)
    - ontology: SAB field from MRCONSO (source vocabulary/ontology, e.g. 'SNOMEDCT_US', 'RXNORM')

    Currently:
      - Includes all English (LAT='ENG') and non-suppressed (SUPPRESS='N') strings
        from all source vocabularies.
      - Optionally limited by the 'limit' argument for testing.

    Returns:
        List of (cui, term_string, ontology) tuples.
    """
    cfg = load_config()
    conn = create_connection(cfg.db)
    cursor = conn.cursor()

    sql = """
    SELECT CUI, STR, SAB
    FROM MRCONSO
    WHERE LAT = 'ENG'
      AND SAB IN ('SNOMEDCT_US', 'RXNORM', 'LNC')
      AND SUPPRESS = 'N'
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    print(f"Fetched {len(rows)} rows from MRCONSO.")

    # rows: list of (cui, str, sab)
    # Optionally deduplicate exact triples to avoid redundant entries
    seen = set()
    result: List[Tuple[str, str, str]] = []

    for cui, term, sab in rows:
        key = (cui, term, sab)
        if key in seen:
            continue
        seen.add(key)
        result.append((cui, term, sab))

    return result


# -----------------------------
# 2. Build index
# -----------------------------

def build_and_save_faiss_index(
    umls_records: List[Tuple[str, str, str]],
    index_path: str,
    mapping_path: str,
    batch_size: int = 1024,
    shard_size: int = 2_000_000,
) -> None:
    """
    Build and save a FAISS index over UMLS term embeddings.

    Args:
        umls_records: list of (cui, term_string, ontology) tuples.
        index_path: path to write the FAISS index file (vectors only).
        mapping_path: path to write the JSON mapping file.
        batch_size: batch size for SapBERT embedding.

    This function will:
      - Embed all term_strings with SapBERT.
      - Build an inner-product FAISS index over the normalized embeddings.
      - Save:
          * index_path: FAISS index (vector store)
          * mapping_path: JSON with parallel arrays:
              - "cuis"    : list of CUIs
              - "terms"   : list of term strings (STR)
              - "sources" : list of SAB values (ontology/vocabulary)
    """
    print(f"Loaded {len(umls_records)} UMLS terms (CUI, STR, SAB).")

    embedder = SapBERTEmbedder()

    # Process in shards to avoid excessive memory usage; embed and save each shard independently
    for shard_start in range(0, len(umls_records), shard_size):
        shard_end = min(shard_start + shard_size, len(umls_records))
        shard = umls_records[shard_start:shard_end]
        cuis = [cui for (cui, _, _) in shard]
        terms = [term for (_, term, _) in shard]
        sources = [sab for (_, _, sab) in shard]

        batches = []
        for start in tqdm(range(0, len(terms), batch_size), desc=f"Embedding terms {shard_start}-{shard_end}"):
            batch_terms = terms[start : start + batch_size]
            batch_emb = embedder.encode(batch_terms, batch_size=batch_size)
            batches.append(batch_emb)
        embeddings = np.vstack(batches)  # shape (shard_N, d)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        shard_suffix = f"_part{shard_start // shard_size:03d}"
        shard_index_path = index_path.replace(".faiss", f"{shard_suffix}.faiss")
        shard_mapping_path = mapping_path.replace(".json", f"{shard_suffix}.json")

        faiss.write_index(index, shard_index_path)
        print(f"Saved FAISS index shard to {shard_index_path} ({index.ntotal} vectors).")

        mapping = {
            "cuis": cuis,
            "terms": terms,
            "sources": sources,
        }
        with open(shard_mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)
        print(f"Saved mapping shard to {shard_mapping_path}")


if __name__ == "__main__":
    # Choose which source to embed: "UCUM" uses pyucum; others use MRCONSO.
    TARGET_SAB = "UMLS"  # change to "UCUM" to embed UCUM units via pyucum

    umls_records = load_umls_terms_from_mysql(limit=None)  # set a limit for testing

    # 2. Build and save FAISS index + mapping
    build_and_save_faiss_index(
        umls_records=umls_records,
        index_path=f"{TARGET_SAB}_sapbert.faiss",
        mapping_path=f"{TARGET_SAB}_sapbert_mapping.json",
        batch_size=256,
    )

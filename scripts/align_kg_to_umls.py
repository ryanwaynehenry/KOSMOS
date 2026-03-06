"""
Align KG node/mention canonical names to UMLS concepts and attach ontology metadata.

This script is intended to post-process ChatGPT-generated KG JSON. It updates
canonical_name fields when a UMLS match exceeds a score threshold and adds
ontology metadata to the relevant nodes (and mentions when present).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from clinical_kg.config import load_config, PipelineConfig
from clinical_kg.data_models import Mention
from clinical_kg.umls.lookup import best_concept_for_mention, source_to_sab


SKIP_TYPES = {"PERSON_PATIENT", "PERSON_CLINICIAN", "DOSE_AMOUNT", "FREQUENCY", "OBS_VALUE", "TIME"}


def _type_preferences(mention_type: Optional[str], cfg: PipelineConfig) -> Tuple[Optional[str], Optional[List[str]]]:
    prefs_map = cfg.ontology_preferences or {}
    if not prefs_map:
        return None, None

    if mention_type is None:
        type_key = "OTHER" if "OTHER" in prefs_map else None
        return type_key, prefs_map.get(type_key) if type_key else None

    type_key = mention_type.upper() if isinstance(mention_type, str) else str(mention_type).upper()
    if type_key in SKIP_TYPES:
        return type_key, None

    if type_key not in prefs_map and "OTHER" in prefs_map:
        type_key = "OTHER"
    return type_key, prefs_map.get(type_key)


def _load_faiss_searcher(cfg: PipelineConfig, use_faiss: bool):
    if not use_faiss:
        return None
    try:
        from clinical_kg.umls.umls_faiss_lookup import UmlsFaissSearcher
    except Exception as exc:
        print(f"[umls] FAISS searcher unavailable: {exc}")
        return None
    try:
        return UmlsFaissSearcher(
            index_path=cfg.faiss_index_path or "UMLS_sapbert.faiss",
            mapping_path=cfg.faiss_mapping_path or "UMLS_sapbert_mapping.json",
        )
    except Exception as exc:
        print(f"[umls] Failed to initialize FAISS searcher: {exc}")
        return None


def _faiss_lookup(
    mention_text: str,
    mention_type: Optional[str],
    cfg: PipelineConfig,
    searcher,
    min_score: float,
) -> Optional[Dict[str, Any]]:
    if not searcher or not mention_text:
        return None

    type_key, prefs = _type_preferences(mention_type, cfg)
    if not prefs:
        return None

    best_hit = None
    best_source = None
    for source in prefs:
        sab = source_to_sab(source)
        if not sab:
            continue
        hits = searcher.search(
            mention_text,
            source=sab,
            top_k=5,
            min_score=0.0,
        )
        if not hits:
            continue
        top_hit = hits[0]
        if top_hit["score"] < min_score:
            continue
        if best_hit is None or top_hit["score"] > best_hit["score"]:
            best_hit = top_hit
            best_source = source

    if not best_hit:
        return None

    return {
        "source": str(best_source).upper(),
        "preferred_term": str(best_hit["term"]),
        "cui": str(best_hit["cui"]),
        "score": float(best_hit["score"]),
        "searched_term": mention_text,
        "method": "faiss",
    }


def _db_lookup(
    mention_text: str,
    mention_type: Optional[str],
    cfg: PipelineConfig,
    min_score: float,
) -> Optional[Dict[str, Any]]:
    if not mention_text:
        return None
    mention = Mention(mention_id="m_tmp", turn_id="", text=mention_text, type=mention_type)
    try:
        code = best_concept_for_mention(mention, cfg)
    except Exception as exc:
        print(f"[umls] DB lookup failed for '{mention_text}': {exc}")
        return None
    if not code or code.score < min_score:
        return None
    return {
        "source": code.source,
        "preferred_term": code.preferred_term,
        "cui": code.cui,
        "score": float(code.score),
        "searched_term": mention_text,
        "method": "initial",
    }


def _align_item(
    item: Dict[str, Any],
    cfg: PipelineConfig,
    searcher,
    min_score: float,
    use_db: bool,
    use_faiss: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    if not overwrite and item.get("ontology") is not None:
        return item

    etype = item.get("entity_type")
    if etype and str(etype).upper() in SKIP_TYPES:
        return item

    canonical = item.get("canonical_name")
    if not isinstance(canonical, str) or not canonical.strip():
        return item

    ontology = None
    strategy = None
    if use_db:
        ontology = _db_lookup(canonical, etype, cfg, min_score=min_score)
        if ontology:
            strategy = "initial"

    if not ontology and use_faiss:
        ontology = _faiss_lookup(canonical, etype, cfg, searcher, min_score=min_score)
        if ontology:
            strategy = "faiss"

    if not ontology:
        return item

    updated = dict(item)
    updated["ontology"] = {k: v for k, v in ontology.items() if k != "method"}
    updated["ontology_strategy"] = strategy

    preferred = ontology.get("preferred_term")
    if isinstance(preferred, str) and preferred.strip():
        updated["canonical_name"] = preferred

    return updated


def _update_relationship_names(data: Dict[str, Any], nodes_by_id: Dict[str, Dict[str, Any]]) -> None:
    rels = data.get("relationship_candidates")
    if not isinstance(rels, list):
        return
    for rel in rels:
        if not isinstance(rel, dict):
            continue
        src_id = rel.get("source_node_id")
        tgt_id = rel.get("target_node_id")
        if src_id in nodes_by_id:
            rel["source_canonical_name"] = nodes_by_id[src_id].get("canonical_name")
        if tgt_id in nodes_by_id:
            rel["target_canonical_name"] = nodes_by_id[tgt_id].get("canonical_name")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align KG nodes to UMLS concepts.")
    parser.add_argument("input", type=Path, help="Path to KG JSON (ChatGPT output).")
    parser.add_argument("--output", type=Path, help="Path to write aligned KG JSON.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.8,
        help="Minimum match score required to update canonical_name.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip MRCONSO database lookup and use FAISS only (if enabled).",
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Enable FAISS fallback lookup (requires local index files).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ontology fields when present.",
    )
    args = parser.parse_args()

    cfg = load_config()
    searcher = _load_faiss_searcher(cfg, use_faiss=args.use_faiss)

    data = json.loads(args.input.read_text(encoding="utf-8"))

    nodes = data.get("nodes") or []
    aligned_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            aligned_nodes.append(node)
            continue
        aligned_nodes.append(
            _align_item(
                node,
                cfg,
                searcher,
                min_score=args.min_score,
                use_db=not args.no_db,
                use_faiss=args.use_faiss,
                overwrite=args.overwrite,
            )
        )
    data["nodes"] = aligned_nodes

    # Optionally align grouped mentions if present
    mentions = data.get("mentions")
    if isinstance(mentions, list):
        aligned_mentions = []
        for mention in mentions:
            if not isinstance(mention, dict):
                aligned_mentions.append(mention)
                continue
            aligned_mentions.append(
                _align_item(
                    mention,
                    cfg,
                    searcher,
                    min_score=args.min_score,
                    use_db=not args.no_db,
                    use_faiss=args.use_faiss,
                    overwrite=args.overwrite,
                )
            )
        data["mentions"] = aligned_mentions

    nodes_by_id = {n.get("id"): n for n in data.get("nodes", []) if isinstance(n, dict)}
    _update_relationship_names(data, nodes_by_id)

    out_path = args.output or args.input
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote aligned KG to {out_path}")


if __name__ == "__main__":
    main()

"""
Transcript-only processing pipeline orchestration.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from clinical_kg.config import PipelineConfig, load_config
from clinical_kg.data_models import Mention, OntologyCode, Turn
from clinical_kg.kg.builder import build_nodes
from clinical_kg.kg.relations import build_relationship_candidates
from clinical_kg.kg.schema import shacl_turtle
from clinical_kg.nlp.coref import add_coref_clusters
from clinical_kg.nlp.ner import extract_mentions
from clinical_kg.nlp.preprocessing import load_and_segment, segment_transcript_text
from clinical_kg.nlp.relations import attach_attributes
from clinical_kg.umls.lookup import best_concept_for_mention, source_to_sab

FAISS_FALLBACK_SCORE = 0.5


def _type_preferences(mention_type, cfg: PipelineConfig):
    """
    Resolve mention type to configured ontology preferences, falling back to OTHER.
    """
    prefs_map = cfg.ontology_preferences or {}
    if not prefs_map:
        return None, None

    skip_types = {"PERSON_PATIENT", "PERSON_CLINICIAN", "DOSE_AMOUNT", "FREQUENCY", "OBS_VALUE"}

    if mention_type is None:
        type_key = "OTHER"
    else:
        type_key = mention_type.upper() if isinstance(mention_type, str) else str(mention_type).upper()
        if type_key in skip_types:
            return type_key, None
        if type_key not in prefs_map and "OTHER" in prefs_map:
            type_key = "OTHER"
    return type_key, prefs_map.get(type_key)


def _load_faiss_searcher(cfg: PipelineConfig):
    try:
        from clinical_kg.umls.umls_faiss_lookup import UmlsFaissSearcher
    except Exception as exc:
        print(f"[ontology] Failed to import UmlsFaissSearcher: {exc}")
        return None
    index_path = getattr(cfg, "faiss_index_path", None) or "UMLS_sapbert.faiss"
    mapping_path = getattr(cfg, "faiss_mapping_path", None) or "UMLS_sapbert_mapping.json"
    try:
        return UmlsFaissSearcher(index_path=index_path, mapping_path=mapping_path)
    except Exception as exc:
        print(
            f"[ontology] Failed to init UmlsFaissSearcher "
            f"(index={index_path}, mapping={mapping_path}): {exc}"
        )
        return None


def _ontology_dict_from_code(code: OntologyCode, searched_term: str, method: str) -> dict:
    return {
        "source": code.source,
        "preferred_term": code.preferred_term,
        "cui": code.cui,
        "score": code.score,
        "searched_term": searched_term,
        "method": method,
    }


def _faiss_fallback_lookup(
    mention: Mention,
    cfg: PipelineConfig,
    searcher: "UmlsFaissSearcher | None",
    min_score: float = FAISS_FALLBACK_SCORE,
) -> OntologyCode | None:
    """
    Use the FAISS index as a fallback when exact MRCONSO lookups fail.
    """
    if searcher is None:
        return None

    type_key, prefs = _type_preferences(mention.type, cfg)
    if not prefs:
        return None

    best_hit = None
    best_source = None
    for source in prefs:
        sab = source_to_sab(source)
        if not sab:
            continue
        hits = searcher.search(
            mention.text,
            source=sab,
            top_k=5,
            min_score=0.0,  # apply threshold ourselves below
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

    return OntologyCode(
        cui=str(best_hit["cui"]),
        source=best_source.upper(),
        preferred_term=str(best_hit["term"]),
        score=float(best_hit["score"]),
    )


def _candidate_to_mention(candidate) -> Mention | None:
    """
    Normalize candidate structures (LLM output dicts or Mention dataclasses)
    into a Mention object for ontology lookup.
    """
    if isinstance(candidate, Mention):
        return candidate

    if isinstance(candidate, dict):
        mentions = candidate.get("mentions") or []
        first_mention = mentions[0] if mentions else {}
        text = candidate.get("canonical_name") or first_mention.get("text")
        if not text:
            return None
        mention_id = first_mention.get("mention_id") or f"m_{text}"
        turn_id = first_mention.get("turn_id") or ""
        mtype = candidate.get("entity_type") or first_mention.get("type")
        return Mention(
            mention_id=str(mention_id),
            turn_id=str(turn_id),
            text=str(text),
            type=mtype,
        )

    return None


def _attach_ontology(
    candidates: List,
    cfg: PipelineConfig,
    searcher: "UmlsFaissSearcher | None",
) -> List:
    """
    Attach ontology matches to grouped candidates, using FAISS as a fallback when
    exact lookup returns nothing.
    """
    concepts = []
    for candidate in candidates:
        mention = _candidate_to_mention(candidate)
        ontology = None
        strategy = None
        canonical_override = None

        if mention:
            try:
                code = best_concept_for_mention(mention, cfg)
            except Exception:
                code = None
            if code:
                ontology = _ontology_dict_from_code(code, mention.text, method="initial")
                strategy = "initial"
                canonical_override = code.preferred_term
            else:
                try:
                    fallback_code = _faiss_fallback_lookup(mention, cfg, searcher)
                except Exception:
                    fallback_code = None
                if fallback_code:
                    ontology = _ontology_dict_from_code(
                        fallback_code, mention.text, method="faiss"
                    )
                    strategy = "faiss"
                    canonical_override = fallback_code.preferred_term

        # Ensure we propagate data as dicts (mentions from LLM are already dicts)
        if isinstance(candidate, dict):
            concept = {**candidate, "ontology": ontology}
            if strategy:
                concept["ontology_strategy"] = strategy
            if canonical_override:
                concept["canonical_name"] = canonical_override
        elif isinstance(candidate, Mention):
            concept = {
                **candidate.__dict__,
                "ontology": ontology,
                "ontology_strategy": strategy,
            }
            if canonical_override:
                concept["canonical_name"] = canonical_override
        else:
            concept = candidate
        concepts.append(concept)

    return concepts


def process_transcript_to_mentions(
    transcript_path: Optional[str],
    encounter_id: str,
    save_intermediate: bool = False,
    use_llm_for_ner: bool = True,
    use_llm_for_coref: bool = True,
    transcript_text: Optional[str] = None,
) -> Tuple[List[Turn], List, List[dict], List[dict]]:
    """
    End to end transcript processing for one encounter, up to mention-level output
    with ontology alignment (database first, FAISS fallback) and KG node creation.

    transcript_path or transcript_text must be provided.
    """
    cfg = load_config()  # Ensure env is loaded; config values used by downstream calls
    if transcript_text is None and transcript_path is None:
        raise ValueError("Either transcript_path or transcript_text must be provided.")

    turns = (
        load_and_segment(transcript_path, encounter_id)
        if transcript_text is None
        else segment_transcript_text(transcript_text, encounter_id, cfg)
    )
    mentions = extract_mentions(turns, use_llm_refinement=use_llm_for_ner)
    candidates = add_coref_clusters(mentions, turns, use_llm_refinement=use_llm_for_coref)
    # mentions = attach_attributes(mentions)

    # Attempt ontology alignment: exact lookup first, FAISS embedding fallback
    faiss_searcher = _load_faiss_searcher(cfg)
    concepts = _attach_ontology(candidates, cfg, searcher=faiss_searcher)
    nodes = build_nodes(concepts, encounter_id=encounter_id, turns=turns, use_llm=use_llm_for_coref, batch_size=5)
    relationships = build_relationship_candidates(turns=turns, nodes=nodes, cfg=cfg, batch_size=20)

    if save_intermediate:
        interim_dir = Path("data") / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)
        with open(interim_dir / f"{encounter_id}_turns.json", "w", encoding="utf-8") as f:
            json.dump([t.__dict__ for t in turns], f, ensure_ascii=False, indent=2)
        with open(interim_dir / f"{encounter_id}_mentions.json", "w", encoding="utf-8") as f:
            json.dump([c for c in candidates], f, ensure_ascii=False, indent=2)
        with open(interim_dir / f"{encounter_id}_nodes.json", "w", encoding="utf-8") as f:
            json.dump(nodes, f, ensure_ascii=False, indent=2)

    return turns, concepts, nodes, relationships


def _default_encounter_id(transcript_path: str) -> str:
    """
    Derive an encounter id from the transcript filename.

    Examples:
      altered_session_2347_1.txt -> 2347_1
      session_123.txt -> 123
      anything_else -> filename stem
    """
    if not transcript_path:
        return "encounter"
    stem = Path(transcript_path).stem
    lower = stem.lower()
    if "session_" in lower:
        # take substring after the first occurrence of "session_"
        after = lower.split("session_", 1)[1]
        if after:
            return after
    return stem


def save_processed_transcript(
    transcript_path: Optional[str],
    encounter_id: str | None = None,
    save_intermediate: bool = False,
    use_llm_for_ner: bool = True,
    use_llm_for_coref: bool = True,
    transcript_text: Optional[str] = None,
    output_stem: Optional[str] = None,
) -> Path:
    """
    Process a transcript and write the consolidated JSON (turns, mentions, nodes)
    to data/interim/<source-stem>.json. Returns the output path.

    If transcript_text is provided, transcript_path can be None and output_stem is used
    for naming (defaults to encounter_id).
    """
    resolved_encounter = encounter_id or _default_encounter_id(transcript_path or "")
    turns, concepts, nodes, relationships = process_transcript_to_mentions(
        transcript_path=transcript_path,
        encounter_id=resolved_encounter,
        save_intermediate=save_intermediate,
        use_llm_for_ner=use_llm_for_ner,
        use_llm_for_coref=use_llm_for_coref,
        transcript_text=transcript_text,
    )

    stem = (
        output_stem
        or (Path(transcript_path).stem if transcript_path else resolved_encounter)
    )
    output_path = Path("data") / "interim" / f"{stem}.json"
    os.makedirs(output_path.parent, exist_ok=True)

    payload = {
        "encounter_id": resolved_encounter,
        "turns": [
            {
                "turn_id": t.turn_id,
                "speaker": t.speaker,
                "text": t.text,
            }
            for t in turns
        ],
        "mentions": [
            c for c in concepts
        ],
        "nodes": nodes,
        "relationship_candidates": relationships,
        "shacl_shapes_ttl": shacl_turtle(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path

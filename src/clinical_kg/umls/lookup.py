"""
Lookups against UMLS-backed ontologies (SNOMED CT, RxNorm, LOINC) plus UCUM.

This mirrors the exploratory logic in `notebooks/ontology_test.py`, but packages
it for use inside the pipeline. The main entry point is
`lookup_concepts_for_mention`, which respects ontology preferences and score
thresholds from `PipelineConfig`.
"""

import difflib
from typing import Dict, List, Optional

from ..config import PipelineConfig, load_config
from ..data_models import Mention, OntologyCode
from ..mapping.rules import TYPE_TO_ONTOLOGY
from .connection import create_connection

# Map friendly ontology names to the SAB values used in UMLS MRCONSO
_SAB_MAP = {
    "RXNORM": "RXNORM",
    # SNOMEDCT in UMLS is usually stored as SNOMEDCT_US
    "SNOMEDCT": "SNOMEDCT_US",
    # LOINC is stored as LNC in UMLS
    "LOINC": "LNC",
}


def _score_name(query: str, candidate: str, tty: Optional[str]) -> float:
    """
    Compute a similarity score between the query and candidate name, with a
    small boost for preferred term types.
    """
    query_lower = query.lower()
    candidate_lower = candidate.lower()
    if candidate_lower == query_lower:
        score = 1.0
    else:
        score = difflib.SequenceMatcher(None, query_lower, candidate_lower).ratio()

    if tty in ("PT", "FN", "PX"):
        score += 0.05
    return score


def _lookup_in_mrconso(
    cursor,
    source: str,
    mention_text: str,
    score_threshold: float,
    max_candidates: int = 100,
) -> Optional[OntologyCode]:
    """
    Query MRCONSO for a given source (SAB) and pick the best matching concept.
    """
    sab = _SAB_MAP.get(source.upper())
    if not sab:
        return None

    term = mention_text.strip()
    if not term:
        return None

    exact_query = """
        SELECT CUI, CODE, STR, TTY
        FROM MRCONSO
        WHERE SAB = %s
          AND LAT = 'ENG'
          AND SUPPRESS = 'N'
          AND STR = %s
        LIMIT %s
    """
    cursor.execute(exact_query, (sab, term, max_candidates))
    rows = cursor.fetchall()
    print(f"UMLS lookup rows for source={source} term={term!r} (exact/like): {rows}")

    if not rows:
        like_query = """
            SELECT CUI, CODE, STR, TTY
            FROM MRCONSO
            WHERE SAB = %s
              AND LAT = 'ENG'
              AND SUPPRESS = 'N'
              AND STR LIKE %s
            LIMIT %s
        """
        pattern = f"%{term}%"
        cursor.execute(like_query, (sab, pattern, max_candidates))
        rows = cursor.fetchall()

    best_row = None
    best_score = -1.0
    for cui, code, name, tty in rows:
        if not isinstance(name, str):
            continue
        score = _score_name(term, name, tty)
        if score > best_score:
            best_score = score
            best_row = (cui, code, name)

    if best_row is None or best_score < score_threshold:
        return None

    cui, code, name = best_row
    return OntologyCode(
        cui=str(cui),
        source=source.upper(),
        source_code=str(code),
        preferred_term=str(name),
        score=best_score,
    )


def _source_threshold(source: str, default: float) -> float:
    """
    Allow per-source score thresholds. Defaults to the configured threshold.
    """
    per_source = {
        "RXNORM": default,
        "SNOMEDCT": default,
        # LOINC strings often include qualifiers; allow a slightly lower threshold.
        "LOINC": min(default, 0.5),
    }
    return per_source.get(source.upper(), default)


def _lookup_ucum(unit_text: str) -> Optional[OntologyCode]:
    """
    Normalize a unit string via pyucum. Returns None if pyucum is unavailable
    or parsing fails.
    """
    try:
        from pyucum import UCUM
    except Exception:
        return None

    try:
        ucum = UCUM()
        unit = ucum.parse(unit_text)
        canonical = ucum.to_canonical(unit)
        code = getattr(canonical, "code", None) or unit_text
    except Exception:
        return None

    return OntologyCode(
        cui="",
        source="UCUM",
        source_code=str(code),
        preferred_term=str(code),
        score=1.0,
    )


def lookup_concepts_for_mention(
    mention: Mention,
    cfg: PipelineConfig,
    max_candidates: int = 100,
) -> List[OntologyCode]:
    """
    Resolve a Mention to ontology codes based on configured source preferences.

    For each preferred source listed for the mention type, attempts to find a
    best concept above the score threshold. UCUM lookups are handled via
    pyucum; others query the UMLS MRCONSO table in MySQL.
    """
    results: List[OntologyCode] = []
    prefs = cfg.ontology_preferences.get(mention.type.upper()) if mention.type else None
    if not prefs:
        return results

    # Handle UCUM separately (no DB required)
    ucum_only = all(source.upper() == "UCUM" for source in prefs)
    if ucum_only:
        code = _lookup_ucum(mention.text)
        return [code] if code else results

    conn = create_connection(cfg.db)
    try:
        cursor = conn.cursor()
        for source in prefs:
            if source.upper() == "UCUM":
                code = _lookup_ucum(mention.text)
            else:
                threshold = _source_threshold(source, cfg.score_threshold)
                code = _lookup_in_mrconso(
                    cursor,
                    source=source,
                    mention_text=mention.text,
                    score_threshold=threshold,
                    max_candidates=max_candidates,
                )
            if code:
                results.append(code)
    finally:
        conn.close()

    return results


def align_entities_with_ontology(
    entities: List[Dict],
    cfg: Optional[PipelineConfig] = None,
    max_candidates: int = 50,
) -> List[Dict]:
    """
    Align grouped entities with ontologies based on their entity_type.

    Updates canonical_name to the preferred term when an ontology match is found
    and adds an 'ontology' field with match details and the searched_term.
    """
    cfg = cfg or load_config()
    aligned: List[Dict] = []

    conn = None
    cursor = None
    try:
        conn = create_connection(cfg.db)
        cursor = conn.cursor()

        for entity in entities:
            etype = str(entity.get("entity_type", "")).upper()
            target_source = TYPE_TO_ONTOLOGY.get(etype)
            ontology = None
            canonical = entity.get("canonical_name") or ""

            if target_source:
                if target_source.upper() == "UCUM":
                    code = _lookup_ucum(canonical)
                else:
                    threshold = _source_threshold(target_source, cfg.score_threshold)
                    code = _lookup_in_mrconso(
                        cursor,
                        source=target_source,
                        mention_text=canonical,
                        score_threshold=threshold,
                        max_candidates=max_candidates,
                    )
                if code:
                    ontology = {
                        "source": code.source,
                        "source_code": code.source_code,
                        "preferred_term": code.preferred_term,
                        "cui": code.cui,
                        "score": code.score,
                        "searched_term": canonical,
                    }
                    entity = {**entity, "canonical_name": code.preferred_term}
            entity["ontology"] = ontology
            aligned.append(entity)
    finally:
        if conn:
            conn.close()

    return aligned

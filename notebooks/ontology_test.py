#!/usr/bin/env python3
"""
Terminology lookups as both a CLI and importable module, backed by a UMLS
MySQL database.

Assumes you have loaded the UMLS Metathesaurus RRF files into MySQL using
the NLM scripts, and have a schema containing MRCONSO.

Defaults assume:
    host = localhost
    database = umls2025
    user = umls_user
    password = StrongPassword!

Update MYSQL_* constants below to match your environment.

Examples (CLI):
    python ontology_test.py --ontology rxnorm "lisinopril 10 mg tablet"
    python ontology_test.py -o snomed "type 2 diabetes"
    python ontology_test.py -o loinc "Hemoglobin"
    python ontology_test.py -o ucum "mg/dL"

From another script:
    from ontology_test import lookup_official

    result = lookup_official("rxnorm", "motrin")
    if result is not None:
        name, code = result
        print(name, code)
"""

from dataclasses import dataclass
from typing import Optional, Literal
import argparse
import sys
import difflib

import mysql.connector  # pip install mysql-connector-python


# ---------------------------------------------------------------------------
# Configuration: adjust these to match your MySQL UMLS database
# ---------------------------------------------------------------------------

MYSQL_HOST = "localhost"
MYSQL_USER = "umls_user"
MYSQL_PASSWORD = "StrongPassword!"
MYSQL_DB = "umls2025"

# UMLS source abbreviations for each system
UMLS_SAB_RXNORM = "RXNORM"
UMLS_SAB_SNOMED = "SNOMEDCT_US"
UMLS_SAB_LOINC = "LNC"


CodeSystem = Literal["RXNORM", "SNOMEDCT", "LOINC", "UCUM"]


@dataclass
class Concept:
    system: CodeSystem
    code: str
    display: str


class TerminologyService:
    def lookup_medication(self, mention: str) -> Optional[Concept]:
        raise NotImplementedError

    def lookup_problem(self, mention: str) -> Optional[Concept]:
        raise NotImplementedError

    def lookup_lab_test(self, mention: str) -> Optional[Concept]:
        raise NotImplementedError

    def normalize_unit(self, unit_str: str) -> Optional[Concept]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# UMLS-backed MySQL terminology for RxNorm, SNOMED CT, LOINC
# ---------------------------------------------------------------------------

class UMLSMySQLTerminology(TerminologyService):
    """
    Terminology lookups using the UMLS MRCONSO table in a MySQL database.

    For each ontology, this class:
        - Filters MRCONSO by SAB (source vocabulary)
        - Restricts to English, non-suppressed terms
        - Fetches a small candidate set using STR LIKE '%term%'
        - Ranks candidates in Python by string similarity to the query
        - Returns the best Concept if above min_similarity
    """

    def __init__(
        self,
        host: str = MYSQL_HOST,
        user: str = MYSQL_USER,
        password: str = MYSQL_PASSWORD,
        database: str = MYSQL_DB,
        min_similarity: float = 0.6,
    ):
        self.conn_params = dict(
            host=host,
            user=user,
            password=password,
            database=database,
        )
        self.min_similarity = min_similarity

    def _lookup_in_mrconso(
        self,
        sab: str,
        system: CodeSystem,
        mention: str,
        max_candidates: int = 100,
    ) -> Optional[Concept]:
        """
        Internal helper: search MRCONSO for a given SAB and mention string.
        """
        term = mention.strip()
        if not term:
            return None

        # Connect to MySQL
        try:
            conn = mysql.connector.connect(**self.conn_params)
        except mysql.connector.Error as e:
            raise RuntimeError(f"Could not connect to MySQL: {e}") from e

        try:
            cursor = conn.cursor()

            # First try an exact match (case-insensitive, depending on collation)
            exact_query = """
                SELECT CODE, STR, TTY
                FROM MRCONSO
                WHERE SAB = %s
                  AND LAT = 'ENG'
                  AND SUPPRESS = 'N'
                  AND STR = %s
                LIMIT %s
            """
            cursor.execute(exact_query, (sab, term, max_candidates))
            rows = cursor.fetchall()

            # If no exact match, fall back to a fuzzy LIKE query
            if not rows:
                like_query = """
                    SELECT CODE, STR, TTY
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
        finally:
            conn.close()

        if not rows:
            return None

        term_lower = term.lower()
        best_row = None
        best_score = -1.0

        for code, name, tty in rows:
            if not isinstance(name, str):
                continue
            name_lower = name.lower()

            if name_lower == term_lower:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, term_lower, name_lower).ratio()

            # Slight boost for "preferred" term types
            if tty in ("PT", "FN", "PX"):
                score += 0.05

            if score > best_score:
                best_score = score
                best_row = (code, name)

        if best_row is None or best_score < self.min_similarity:
            return None

        code, name = best_row
        return Concept(system=system, code=str(code), display=str(name))

    # Domain-specific methods that dispatch to the helper

    def lookup_medication(self, mention: str) -> Optional[Concept]:
        # RxNorm medications
        return self._lookup_in_mrconso(UMLS_SAB_RXNORM, "RXNORM", mention)

    def lookup_problem(self, mention: str) -> Optional[Concept]:
        # SNOMED CT clinical problems / findings
        return self._lookup_in_mrconso(UMLS_SAB_SNOMED, "SNOMEDCT", mention)

    def lookup_lab_test(self, mention: str) -> Optional[Concept]:
        # LOINC lab tests and observations
        return self._lookup_in_mrconso(UMLS_SAB_LOINC, "LOINC", mention)


# ---------------------------------------------------------------------------
# UCUM via pyucum (unchanged)
# ---------------------------------------------------------------------------

class UCUMTerminology(TerminologyService):
    def __init__(self):
        try:
            from pyucum import UCUM
        except ImportError as e:
            raise RuntimeError(
                "pyucum is required for UCUM lookups. Install pyucum."
            ) from e

        self.ucum = UCUM()

    def normalize_unit(self, unit_str: str) -> Optional[Concept]:
        try:
            unit = self.ucum.parse(unit_str)
        except Exception:
            return None

        try:
            canonical = self.ucum.to_canonical(unit)
        except Exception:
            canonical = None

        code = None
        if canonical is not None:
            code = getattr(canonical, "code", None)

        if not code:
            code = unit_str

        return Concept(
            system="UCUM",
            code=code,
            display=code,
        )

    # For completeness, but you probably will not call these for UCUM
    def lookup_medication(self, mention: str) -> Optional[Concept]:
        return None

    def lookup_problem(self, mention: str) -> Optional[Concept]:
        return None

    def lookup_lab_test(self, mention: str) -> Optional[Concept]:
        return None


# ---------------------------------------------------------------------------
# Public API: resolve_concept / lookup_official
# ---------------------------------------------------------------------------

def resolve_concept(
    ontology: str,
    term: str,
) -> Optional[Concept]:
    """
    Core lookup function usable from other modules.

    ontology: "rxnorm", "snomed", "loinc", or "ucum"
    term:     free-text term or unit to resolve

    Returns a Concept or None if no acceptable match is found.
    """
    ontology_lower = ontology.lower()
    if ontology_lower in ("rxnorm", "snomed", "loinc"):
        service = UMLSMySQLTerminology()
        if ontology_lower == "rxnorm":
            return service.lookup_medication(term)
        elif ontology_lower == "snomed":
            return service.lookup_problem(term)
        else:  # "loinc"
            return service.lookup_lab_test(term)
    elif ontology_lower == "ucum":
        service = UCUMTerminology()
        return service.normalize_unit(term)
    else:
        raise RuntimeError(f"Unsupported ontology '{ontology}'")


def lookup_official(
    ontology: str,
    term: str,
) -> Optional[tuple[str, str]]:
    """
    Convenience function for other scripts.

    Returns (official_name, code) for the best match, or None if no
    good match is found.
    """
    concept = resolve_concept(ontology, term)
    if concept is None:
        return None
    return concept.display, concept.code


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lookup a term in a clinical terminology (via UMLS MySQL) and print the best match."
    )
    parser.add_argument(
        "-o",
        "--ontology",
        required=True,
        choices=["rxnorm", "snomed", "loinc", "ucum"],
        help="Which terminology to use",
    )
    parser.add_argument(
        "term",
        help="The term or unit string to look up",
    )

    args = parser.parse_args()
    ontology = args.ontology
    term = args.term.strip()

    try:
        concept = resolve_concept(ontology, term)
        print(concept)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if concept is None:
        print(f"No good match found for '{term}' in {ontology.upper()}")
        sys.exit(1)

    print(f"Best match in {concept.system}:")
    print(f"  code:    {concept.code}")
    print(f"  display: {concept.display}")


if __name__ == "__main__":
    main()
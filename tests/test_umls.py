# tests/test_umls.py

import sys
import types

import pytest

from clinical_kg.config import DBConfig, PipelineConfig, load_config
from clinical_kg.data_models import Mention
from clinical_kg.umls import lookup as lookup_module
from clinical_kg.umls.connection import create_connection
from clinical_kg.umls.lookup import (
    _lookup_in_mrconso,
    _lookup_ucum,
    lookup_concepts_for_mention,
)


@pytest.mark.integration
def test_umls_connection_can_select_1():
    """
    Basic integration test that verifies we can connect to the UMLS MySQL
    database and run a trivial query.
    """
    cfg = load_config()

    # cfg.db should be a DBConfig instance with host, port, user, password, database
    conn = create_connection(cfg.db)

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()

        assert result is not None, "No result returned from SELECT 1"
        assert result[0] == 1, f"Expected 1 from SELECT 1, got {result[0]!r}"
    finally:
        conn.close()


class FakeCursor:
    def __init__(self, rows_map):
        self.rows_map = rows_map
        self.last_params = None

    def execute(self, query, params):
        self.last_params = params

    def fetchall(self):
        if not self.last_params:
            return []
        sab = self.last_params[0]
        term = self.last_params[1]
        # Normalize LIKE patterns by stripping wildcards for lookup
        term_key = term.strip("%")
        return self.rows_map.get((sab, term_key.lower()), [])


class FakeConnection:
    def __init__(self, rows_map):
        self.rows_map = rows_map
        self.closed = False

    def cursor(self):
        return FakeCursor(self.rows_map)

    def close(self):
        self.closed = True


def test_lookup_in_mrconso_picks_best_match():
    rows = {
        ("SNOMEDCT_US", "type 2 diabetes"): [
            ("CUI1", "Type 2 diabetes mellitus", "PT"),
            ("CUI2", "Type 1 diabetes mellitus", "PT"),
        ],
    }
    cursor = FakeCursor(rows)
    result = _lookup_in_mrconso(
        cursor=cursor,
        source="SNOMEDCT",
        mention_text="type 2 diabetes",
        score_threshold=0.6,
    )
    assert result is not None
    assert result.source == "SNOMEDCT"
    assert result.preferred_term == "Type 2 diabetes mellitus"
    assert result.score >= 0.6


def test_lookup_in_mrconso_respects_threshold():
    rows = {
        ("RXNORM", "lipnsprl"): [("CUI3", "lisinopril", "PT")],
    }
    cursor = FakeCursor(rows)
    # Set a high threshold so the fuzzy match fails
    result = _lookup_in_mrconso(
        cursor=cursor,
        source="RXNORM",
        mention_text="lipnsprl",
        score_threshold=0.95,
    )
    assert result is None


def test_lookup_ucum_with_stub(monkeypatch):
    fake_module = types.SimpleNamespace()

    class FakeUnit:
        code = "mg/dL"

    class FakeUCUM:
        def parse(self, text):
            return FakeUnit()

        def to_canonical(self, unit):
            return unit

    fake_module.UCUM = FakeUCUM
    monkeypatch.setitem(sys.modules, "pyucum", fake_module)

    result = _lookup_ucum("mg/dL")
    assert result is not None
    assert result.source == "UCUM"
    assert result.preferred_term == "mg/dL"


def test_lookup_concepts_for_mention_respects_preferences(monkeypatch):
    rows = {
        ("SNOMEDCT_US", "type 2 diabetes"): [
            ("CUI1", "Type 2 diabetes mellitus", "PT"),
        ],
        ("RXNORM", "lisinopril"): [
            ("CUI2", "Lisinopril", "PT"),
        ],
        ("LNC", "hemoglobin"): [
            ("CUI3", "Hemoglobin [Mass/volume] in Blood", "PT"),
        ],
    }
    fake_conn = FakeConnection(rows)
    monkeypatch.setattr(lookup_module, "create_connection", lambda _: fake_conn)

    cfg = PipelineConfig(
        db=DBConfig(host="localhost", port=3306, user="user", password="pass", database="db"),
        ontology_preferences={
            "PROBLEM": ["SNOMEDCT"],
            "MEDICATION": ["RXNORM"],
            "LAB_TEST": ["LOINC"],
        },
        score_threshold=0.6,
    )

    problem_codes = lookup_concepts_for_mention(
        Mention("m1", "t1", "type 2 diabetes", "PROBLEM"), cfg
    )
    med_codes = lookup_concepts_for_mention(
        Mention("m2", "t1", "lisinopril", "MEDICATION"), cfg
    )
    lab_codes = lookup_concepts_for_mention(
        Mention("m3", "t1", "hemoglobin", "LAB_TEST"), cfg
    )

    assert problem_codes and problem_codes[0].preferred_term == "Type 2 diabetes mellitus"
    assert med_codes and med_codes[0].preferred_term == "Lisinopril"
    assert lab_codes and lab_codes[0].preferred_term == "Hemoglobin [Mass/volume] in Blood"


def test_lookup_concepts_for_mention_ucum_only(monkeypatch):
    # Ensure UCUM path is used and DB is not called
    monkeypatch.setattr(lookup_module, "create_connection", lambda _: pytest.fail("DB should not be called"))

    fake_module = types.SimpleNamespace()

    class FakeUnit:
        code = "mg/dL"

    class FakeUCUM:
        def parse(self, text):
            return FakeUnit()

        def to_canonical(self, unit):
            return unit

    fake_module.UCUM = FakeUCUM
    monkeypatch.setitem(sys.modules, "pyucum", fake_module)

    cfg = PipelineConfig(
        db=DBConfig(host="localhost", port=3306, user="user", password="pass", database="db"),
        ontology_preferences={"UNIT": ["UCUM"]},
        score_threshold=0.6,
    )

    codes = lookup_concepts_for_mention(
        Mention("m4", "e1", "t1", 0, 0, "mg/dL", "UNIT"), cfg
    )
    assert codes and codes[0].source == "UCUM"

def test_ontology_preferences_include_required_vocabularies():
    """
    Ensure the configuration includes the expected ontology preferences.
    """
    cfg = load_config()
    prefs = cfg.ontology_preferences

    assert "PROBLEM" in prefs, "PROBLEM ontology preferences are missing"
    assert prefs["PROBLEM"] == ["SNOMEDCT"], "PROBLEM ontology should use SNOMEDCT"

    assert "MEDICATION" in prefs, "MEDICATION ontology preferences are missing"
    assert prefs["MEDICATION"] == ["RXNORM"], "MEDICATION ontology should use RXNORM"

    assert "LAB_TEST" in prefs, "LAB_TEST ontology preferences are missing"
    assert prefs["LAB_TEST"] == ["LOINC"], "LAB_TEST ontology should use LOINC"

    assert "UNIT" in prefs, "UNIT ontology preferences are missing"
    assert prefs["UNIT"] == ["UCUM"], "UNIT ontology should use UCUM"

    assert "ACTIVITY" in prefs, "ACTIVITY ontology preferences are missing"
    assert prefs["ACTIVITY"] == ["SNOMEDCT"], "ACTIVITY ontology should use SNOMEDCT"

# tests/test_umls.py

import pytest

from clinical_kg.config import load_config
from clinical_kg.umls.connection import create_connection


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

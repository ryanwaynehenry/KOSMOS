try:
    from clinical_kg.umls.umls_faiss_lookup import UmlsFaissSearcher
except Exception as exc:
    print(f"[ontology] Failed to import UmlsFaissSearcher: {exc}")
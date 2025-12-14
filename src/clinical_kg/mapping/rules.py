"""
Mapping rules between internal entity types and ontology sources.
"""

# Maps entity_type values to ontology source abbreviations used in UMLS lookups.
TYPE_TO_ONTOLOGY = {
    # Primary clinical concepts
    "PROBLEM": "SNOMEDCT",
    "MEDICATION": "RXNORM",
    "LAB_TEST": "LOINC",
    "UNIT": "UCUM",
    "ACTIVITY": "SNOMEDCT",   # newly enabled
    "OTHER": "SNOMEDCT",

    # Modifiers/values that are properties by default
    "DOSE_AMOUNT": None,      # numeric; tied to MEDICATION + UNIT
    "FREQUENCY": None,        # text; could later map to SNOMED regimen codes
    "OBS_VALUE": None,        # numeric/text; could later distinguish interpretation codes

    # Local entities or miscellaneous
    "PERSON_PATIENT": None,   # local Patient node, not a UMLS lookup
    
}


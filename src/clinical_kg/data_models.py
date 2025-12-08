from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class Turn:
    encounter_id: str
    turn_id: str
    speaker: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

@dataclass
class Mention:
    mention_id: str
    encounter_id: str
    turn_id: str
    start_char: int
    end_char: int
    text: str
    type: str                # PROBLEM, MEDICATION, etc.
    coref_cluster_id: Optional[str] = None
    attributes: Dict[str, str] = None

@dataclass
class OntologyCode:
    cui: str
    source: str              # SNOMEDCT, RXNORM, etc.
    source_code: str
    preferred_term: str
    score: float

@dataclass
class Instance:
    instance_id: str
    encounter_id: str
    cls: str                 # ConditionInstance, MedicationStatement, etc.
    coref_cluster_id: Optional[str]
    ontology_code: Optional[OntologyCode]
    properties: Dict[str, str]

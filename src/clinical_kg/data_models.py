from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    type: str  # PROBLEM, MEDICATION, DOSE_AMOUNT, UNIT, FREQUENCY, etc.
    confidence: Optional[float] = None
    coref_cluster_id: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class RelationMention:
    relation_id: str
    encounter_id: str
    source_mention_id: str
    target_mention_id: str
    relation_type: str  # e.g., "treats", "caused_by", "about"
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class TranscriptMetadata:
    encounter_id: str
    patient_id: Optional[str] = None
    clinician_id: Optional[str] = None
    additional: Dict[str, str] = field(default_factory=dict)


@dataclass
class OntologyCode:
    cui: str
    source: str  # SNOMEDCT, RXNORM, etc.
    source_code: str
    preferred_term: str
    score: float


@dataclass
class Instance:
    instance_id: str
    encounter_id: str
    cls: str  # ConditionInstance, MedicationStatement, etc.
    coref_cluster_id: Optional[str]
    ontology_code: Optional[OntologyCode]
    properties: Dict[str, str]

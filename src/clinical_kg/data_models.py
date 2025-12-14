"""
Core data models used across the transcript processing pipeline.

Only standard-library imports are used here so the models stay lightweight and
reusable.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union


class Speaker(str, Enum):
    PATIENT = "PATIENT"
    CLINICIAN = "CLINICIAN"
    UNKNOWN = "UNKNOWN"


@dataclass
class Turn:
    encounter_id: str
    turn_id: str
    speaker: Union[str, Speaker]
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class MentionType(str, Enum):
    PROBLEM = "PROBLEM"
    MEDICATION = "MEDICATION"
    LAB_TEST = "LAB_TEST"
    UNIT = "UNIT"
    DOSE_AMOUNT = "DOSE_AMOUNT"
    FREQUENCY = "FREQUENCY"
    PERSON_PATIENT = "PERSON_PATIENT"
    PERSON_CLINICIAN = "PERSON_CLINICIAN"
    OBS_VALUE = "OBS_VALUE"
    ACTIVITY = "ACTIVITY"
    OTHER = "OTHER"


@dataclass
class Mention:
    mention_id: str
    turn_id: str
    text: str
    type: Optional[Union[MentionType, str]] = None


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

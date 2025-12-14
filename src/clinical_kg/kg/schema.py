"""
Schema definitions for knowledge-graph nodes derived from grouped entities.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

SH_NS = "http://www.w3.org/ns/shacl#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
KG_NS = "http://example.org/kg#"
KG_SHAPE_NS = "http://example.org/kg/shapes#"


@dataclass(frozen=True)
class ShaclProperty:
    path: str
    datatype: str = "xsd:string"
    description: Optional[str] = None
    min_count: Optional[int] = None
    max_count: Optional[int] = 1

    def to_turtle(self, indent: str = "    ") -> str:
        lines = [
            f"[ sh:path {self.path};",
            f"{indent}sh:datatype {self.datatype};",
        ]
        if self.description:
            lines.append(f"{indent}sh:description \"{self.description}\";")
        if self.min_count is not None:
            lines.append(f"{indent}sh:minCount {self.min_count};")
        if self.max_count is not None:
            lines.append(f"{indent}sh:maxCount {self.max_count};")
        lines[-1] = lines[-1].rstrip(";")
        lines.append("]")
        return "\n".join(indent + line for line in lines)


@dataclass(frozen=True)
class ShaclShape:
    shape_id: str
    target_class: str
    properties: List[ShaclProperty]

    def to_turtle(self, indent: str = "    ") -> str:
        lines = [
            f"{self.shape_id}",
            f"{indent}a sh:NodeShape ;",
            f"{indent}sh:targetClass {self.target_class} ;",
        ]
        for prop in self.properties:
            lines.append(f"{indent}sh:property {prop.to_turtle(indent + indent)} ;")
        lines[-1] = lines[-1].rstrip(" ;")
        lines.append(".")
        return "\n".join(lines)


def _shape_from_options(shape_name: str, class_name: str, attrs: List[str]) -> ShaclShape:
    props = [ShaclProperty(path=f"kg:{a}", datatype="xsd:string") for a in attrs]
    return ShaclShape(
        shape_id=f"kgsh:{shape_name}Shape",
        target_class=f"kg:{class_name}",
        properties=props,
    )


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    definition: str
    examples: List[str]


@dataclass(frozen=True)
class NodeSchema:
    class_name: str
    attribute_options: List[str]
    attribute_definitions: Dict[str, AttributeSpec]
    shacl_shape: ShaclShape


# Attribute specs consolidate definitions and examples to remove ambiguity.
ATTRIBUTE_SPECS: Dict[str, AttributeSpec] = {
    # Person
    "name": AttributeSpec(
        name="name",
        definition="Full name or identifying label for the person.",
        examples=["Sophia Brown", "Dr. Rafael Gomez"],
    ),
    "age": AttributeSpec(
        name="age",
        definition="Age of the individual as stated or implied; preserve units.",
        examples=["54 years", "6 months"],
    ),
    "sex": AttributeSpec(
        name="sex",
        definition="Sex descriptor when explicitly stated.",
        examples=["female", "male", "intersex"],
    ),
    "role": AttributeSpec(
        name="role",
        definition="Care-team or relational role relevant to the encounter.",
        examples=["primary care doctor", "psychiatrist", "patient"],
    ),
    "hobbies": AttributeSpec(
        name="hobbies",
        definition="Leisure activities or interests when mentioned.",
        examples=["gardening", "reading"],
    ),
    "location": AttributeSpec(
        name="location",
        definition="Location relevant to the person if stated.",
        examples=["Jacksonville, Florida", "St. Louis, Missouri", "Sydney, Austrailia"],
    ),
    "relationship": AttributeSpec(
        name="relationship",
        definition="Relationship status or key family link if explicitly mentioned.",
        examples=["divorced", "mother of two"],
    ),
    "notes": AttributeSpec(
        name="notes",
        definition="Other brief person-relevant details not captured by other fields.",
        examples=[],
    ),
    # Condition
    "status": AttributeSpec(
        name="status",
        definition="Condition state or activity level.",
        examples=["active", "chronic", "resolved"],
    ),
    "onset": AttributeSpec(
        name="onset",
        definition="When the condition started or first appeared.",
        examples=["about a month ago", "ten years ago", "past few weeks"],
    ),
    "resolution": AttributeSpec(
        name="resolution",
        definition="When the condition ended or symptoms stopped being present (if known).",
        examples=["last week", "two days ago", "after finishing antibiotics", "in 2019", "still ongoing"],
    ),
    "severity": AttributeSpec(
        name="severity",
        definition="Severity descriptor for the condition.",
        examples=["mild", "moderate", "severe"],
    ),
    "temporality": AttributeSpec(
        name="temporality",
        definition="Pattern over time.",
        examples=["intermittent", "persistent", "episodic"],
    ),
    "body_site": AttributeSpec(
        name="body_site",
        definition="Anatomical site affected.",
        examples=["head", "chest", "thyroid"],
    ),
    "course": AttributeSpec(
        name="course",
        definition="Trajectory or progression of the condition.",
        examples=["getting worse", "stable", "improving"],
    ),
    "duration": AttributeSpec(
        name="duration",
        definition="How long the condition has been present or lasting.",
        examples=["past month", "ten years", "past few weeks"],
    ),
    # Medication
    "dose_value": AttributeSpec(
        name="dose_value",
        definition="Numeric/quantity component of a medication dose.",
        examples=["100", "81", "10"],
    ),
    "dose_unit": AttributeSpec(
        name="dose_unit",
        definition="Unit associated with the dose.",
        examples=["mcg", "mg", "mg/day"],
    ),
    "dose_raw": AttributeSpec(
        name="dose_raw",
        definition="Full free-text dose expression when present.",
        examples=["100 mcg", "10 mg twice daily"],
    ),
    "frequency_text": AttributeSpec(
        name="frequency_text",
        definition="Frequency or schedule of medication or activity.",
        examples=["once daily", "twice a day", "weekly", "as needed", "once a week"],
    ),
    "route": AttributeSpec(
        name="route",
        definition="Route of administration.",
        examples=["oral", "topical", "injection"],
    ),
    "start_date": AttributeSpec(
        name="start_date",
        definition="When the medication was started, if stated.",
        examples=["today", "2017", "5 years ago"],
    ),
    "end_date": AttributeSpec(
        name="end_date",
        definition="When the medication stopped or is planned to stop.",
        examples=["stop next week", "in two months", "March, 2025"],
    ),
    "indication": AttributeSpec(
        name="indication",
        definition="Reason or condition the medication treats.",
        examples=["for hypothyroidism", "for depression"],
    ),
    # Lab/Observation/Activity
    "result_value": AttributeSpec(
        name="result_value",
        definition="Measured or reported value of a test or observation.",
        examples=["140/90", "98.6", "63"],
    ),
    "result_unit": AttributeSpec(
        name="result_unit",
        definition="Unit for the measured value.",
        examples=["mmHg", "mg/dL", "bpm"],
    ),
    "interpretation": AttributeSpec(
        name="interpretation",
        definition="Clinical interpretation of a result.",
        examples=["normal", "elevated", "low"],
    ),
    "specimen": AttributeSpec(
        name="specimen",
        definition="Specimen type for the test.",
        examples=["blood", "urine"],
    ),
    "collection_time": AttributeSpec(
        name="collection_time",
        definition="When the specimen/test was collected.",
        examples=["this morning", "today", "last week", "March 23, 2016"],
    ),
    "description": AttributeSpec(
        name="description",
        definition="What is being observed/tested or the qualitative label for it.",
        examples=["therapy discussion noted", "mood observation", "blood pressure reading"],
    ),
    "value_text": AttributeSpec(
        name="value_text",
        definition="The actual observed/result value expressed in words.",
        examples=["anxious mood", "negative for harm thoughts", "elevated blood pressure"],
    ),
    "context": AttributeSpec(
        name="context",
        definition="Situation/setting that qualifies the observation or test.",
        examples=["during visit", "after exercise", "at work", "socially", "in the morning"],
    ),
    "intensity": AttributeSpec(
        name="intensity",
        definition="Intensity of an activity.",
        examples=["light", "moderate", "vigorous"],
    ),
    # Activity-specific (separate from medication)
    "activity_type": AttributeSpec(
        name="activity_type",
        definition="What the activity or behavior is (occupation, exercise, substance use).",
        examples=["smoking cigarettes", "drinking alcohol", "running", "plays soccer", "night-shift nurse", "librarian"],
    ),
    "activity_frequency": AttributeSpec(
        name="activity_frequency",
        definition="How often the activity/behavior occurs.",
        examples=["daily", "once a week", "weekends", "5 days per week"],
    ),
    "activity_amount": AttributeSpec(
        name="activity_amount",
        definition="Quantity per instance or typical amount when stated (single session/use).",
        examples=["2 packs", "3 beers", "5 miles", "30 minutes per session"],
    ),
    "activity_intensity": AttributeSpec(
        name="activity_intensity",
        definition="Intensity level of the activity.",
        examples=["light", "moderate", "vigorous"],
    ),
    "activity_duration": AttributeSpec(
        name="activity_duration",
        definition="Overall time span or cumulative duration of the activity pattern.",
        examples=["past month", "10 years", "for years", "since childhood"],
    ),
    "activity_context": AttributeSpec(
        name="activity_context",
        definition="Setting or situation for the activity.",
        examples=["at work", "socially", "after exercise", "during weekends"],
    ),
    "activity_status": AttributeSpec(
        name="activity_status",
        definition="Current state of the activity/behavior.",
        examples=["current", "former", "never", "cutting down"],
    ),
    "activity_start_date": AttributeSpec(
        name="activity_start_date",
        definition="When the activity or behavior began.",
        examples=["started in college", "began last year", "started recently"],
    ),
    "activity_end_date": AttributeSpec(
        name="activity_end_date",
        definition="When the activity or behavior stopped or was quit.",
        examples=["quit 5 years ago", "stopped last month", "ended recently"],
    ),
}


def _attribute_def_map(attrs: List[str]) -> Dict[str, AttributeSpec]:
    return {a: ATTRIBUTE_SPECS[a] for a in attrs if a in ATTRIBUTE_SPECS}


# Attribute menus are intentionally permissive so downstream steps can choose
# which values to populate while staying type-aware.
PERSON_ATTRS = [
    "name",
    "age",
    "gender",
    "job",
    "occupation",
    "role",
    "hobbies",
    "location",
    "relationship",
    "notes",
]
PERSON_SCHEMA = NodeSchema(
    class_name="Person",
    attribute_options=PERSON_ATTRS,
    attribute_definitions=_attribute_def_map(PERSON_ATTRS),
    shacl_shape=_shape_from_options("Person", "Person", PERSON_ATTRS),
)

CONDITION_ATTRS = [
    "status",
    "onset",
    "severity",
    "temporality",
    "negation",
    "body_site",
    "course",
    "duration",
    "notes",
]
CONDITION_SCHEMA = NodeSchema(
    class_name="Condition",
    attribute_options=CONDITION_ATTRS,
    attribute_definitions=_attribute_def_map(CONDITION_ATTRS),
    shacl_shape=_shape_from_options("Condition", "Condition", CONDITION_ATTRS),
)

MEDICATION_ATTRS = [
    "dose_value",
    "dose_unit",
    "dose_raw",
    "frequency_text",
    "route",
    "status",
    "start_date",
    "end_date",
    "indication",
    "notes",
]
MEDICATION_SCHEMA = NodeSchema(
    class_name="MedicationStatement",
    attribute_options=MEDICATION_ATTRS,
    attribute_definitions=_attribute_def_map(MEDICATION_ATTRS),
    shacl_shape=_shape_from_options("MedicationStatement", "MedicationStatement", MEDICATION_ATTRS),
)

LAB_TEST_ATTRS = [
    "result_value",
    "result_unit",
    "interpretation",
    "specimen",
    "collection_time",
    "status",
    "notes",
]
LAB_TEST_SCHEMA = NodeSchema(
    class_name="LabTest",
    attribute_options=LAB_TEST_ATTRS,
    attribute_definitions=_attribute_def_map(LAB_TEST_ATTRS),
    shacl_shape=_shape_from_options("LabTest", "LabTest", LAB_TEST_ATTRS),
)

ACTIVITY_ATTRS = [
    "activity_type",
    "activity_frequency",
    "activity_amount",
    "activity_intensity",
    "activity_duration",
    "activity_context",
    "activity_status",
    "activity_start_date",
    "activity_end_date",
    "notes",
]
ACTIVITY_SCHEMA = NodeSchema(
    class_name="Activity",
    attribute_options=ACTIVITY_ATTRS,
    attribute_definitions=_attribute_def_map(ACTIVITY_ATTRS),
    shacl_shape=_shape_from_options("Activity", "Activity", ACTIVITY_ATTRS),
)

OBSERVATION_ATTRS = [
    "description",
    "value_text",
    "context",
    "notes",
]
OBSERVATION_SCHEMA = NodeSchema(
    class_name="Observation",
    attribute_options=OBSERVATION_ATTRS,
    attribute_definitions=_attribute_def_map(OBSERVATION_ATTRS),
    shacl_shape=_shape_from_options("Observation", "Observation", OBSERVATION_ATTRS),
)


ENTITY_TYPE_TO_SCHEMA: Dict[str, NodeSchema] = {
    "PERSON_PATIENT": PERSON_SCHEMA,
    "PERSON_CLINICIAN": PERSON_SCHEMA,
    "PROBLEM": CONDITION_SCHEMA,
    "ACTIVITY": ACTIVITY_SCHEMA,
    "MEDICATION": MEDICATION_SCHEMA,
    "LAB_TEST": LAB_TEST_SCHEMA,
    "OBS_VALUE": OBSERVATION_SCHEMA,
    "OTHER": OBSERVATION_SCHEMA,
    "UNIT": OBSERVATION_SCHEMA,
    "DOSE_AMOUNT": OBSERVATION_SCHEMA,
    "FREQUENCY": OBSERVATION_SCHEMA,
}

DEFAULT_SCHEMA = OBSERVATION_SCHEMA
ALL_SHACL_SHAPES: List[ShaclShape] = [
    PERSON_SCHEMA.shacl_shape,
    CONDITION_SCHEMA.shacl_shape,
    MEDICATION_SCHEMA.shacl_shape,
    LAB_TEST_SCHEMA.shacl_shape,
    ACTIVITY_SCHEMA.shacl_shape,
    OBSERVATION_SCHEMA.shacl_shape,
]


def shacl_turtle() -> str:
    header = [
        f"@prefix sh: <{SH_NS}> .",
        f"@prefix xsd: <{XSD_NS}> .",
        f"@prefix kg: <{KG_NS}> .",
        f"@prefix kgsh: <{KG_SHAPE_NS}> .",
    ]
    body = "\n\n".join(shape.to_turtle() for shape in ALL_SHACL_SHAPES)
    return "\n".join(header) + "\n\n" + body


def schema_for_entity_type(entity_type: Optional[Union[str, object]]) -> NodeSchema:
    if entity_type is None:
        return DEFAULT_SCHEMA
    etype = entity_type.upper() if isinstance(entity_type, str) else str(entity_type).upper()
    return ENTITY_TYPE_TO_SCHEMA.get(etype, DEFAULT_SCHEMA)


def attribute_specs_for_options(options: List[str]) -> List[AttributeSpec]:
    """
    Return AttributeSpec objects (unique, ordered by first occurrence) for the given options.
    """
    seen = set()
    specs: List[AttributeSpec] = []
    for opt in options:
        if opt in seen:
            continue
        spec = ATTRIBUTE_SPECS.get(opt)
        if spec:
            specs.append(spec)
            seen.add(opt)
    return specs

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
    "patient_agreement": AttributeSpec(
        name="patient_agreement",
        definition="Summary of the patient's expressed willingness or unwillingness to follow the care plan or specific instructions.",
        examples=[
            "agrees with the recommended medical treatment plan",
            "will call in 2 days with weight update",
            "will follow up in 2 weeks or sooner if symptoms worsen",
            "declines to start the new medication",
        ],
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
    "reaction": AttributeSpec(
        name="reaction",
        definition="Observed reaction or manifestation when the condition is an allergy or adverse response.",
        examples=["rash", "hives", "anaphylaxis", "nausea"],
    ),
    "duration": AttributeSpec(
        name="duration",
        definition="How long the condition has been present or lasting.",
        examples=["past month", "ten years", "past few weeks"],
    ),
    "impact": AttributeSpec(
        name="impact",
    definition=(
        "Functional or clinical effect the condition has on the person, including limits on activities, "
        "work/school or daily living, quality of life, sleep, and downstream effects like worsening other "
        "conditions or triggering complications."
    ),
    examples=[
        "can't walk up stairs without stopping",
        "missed work for two days",
        "stops me from exercising",
        "wakes me up at night",
        "hard to concentrate at work",
        "needs help with bathing and dressing",
        "worsens my asthma",
        "triggers migraines",
        "makes my blood sugar harder to control",
        "causes me to avoid driving",
    ],
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
    "medication_status": AttributeSpec(
        name="medication_status",
        definition="Whether the patient used to take, is currently taking, or is about to start taking the medication (prescribed/planned).",
        examples=[
            "used to take",
            "currently taking",
            "about to start",
        ],
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
    # Time
    "time_text": AttributeSpec(
        name="time_text",
        definition="Verbatim timing expression as stated in the transcript.",
        examples=["today", "yesterday", "next week", "in 2 weeks", "for 3 months", "last visit", "post-op"],
    ),
    "time_kind": AttributeSpec(
        name="time_kind",
        definition="Coarse type of timing expression.",
        examples=["date", "time_of_day", "duration", "relative", "range", "visit_reference"],
    ),
    "time_normalized": AttributeSpec(
        name="time_normalized",
        definition="Normalized representation when possible (ISO-like text or a structured shorthand). Leave blank if not possible.",
        examples=["2025-12-21", "2025-01", "PT2W", "P3M", "TODAY", "NEXT_WEEK"],
    ),
    "time_granularity": AttributeSpec(
        name="time_granularity",
        definition="Precision level of the time expression when stated or inferable from the phrasing alone.",
        examples=["year", "month", "day", "hour", "minute"],
    ),
    "time_value": AttributeSpec(
        name="time_value",
        definition="Numeric component for durations or quantities of time when explicitly stated.",
        examples=["2", "3", "10"],
    ),
    "time_unit": AttributeSpec(
        name="time_unit",
        definition="Unit for durations when explicitly stated.",
        examples=["days", "weeks", "months", "years", "hours", "minutes"],
    ),
    "time_start": AttributeSpec(
        name="time_start",
        definition="Start of a time range when explicitly stated.",
        examples=["January 2024", "2025-03-01", "last Monday"],
    ),
    "time_end": AttributeSpec(
        name="time_end",
        definition="End of a time range when explicitly stated.",
        examples=["March 2024", "2025-03-15", "next Friday"],
    ),
    "time_anchor": AttributeSpec(
        name="time_anchor",
        definition="What the time expression is anchoring to in the clinical narrative, when explicitly indicated.",
        examples=["onset", "resolution", "follow_up", "appointment", "procedure_date", "referral", "med_start", "med_stop"],
    ),
    "time_reference": AttributeSpec(
        name="time_reference",
        definition="Reference point used by a relative expression when explicitly stated.",
        examples=["last visit", "previous appointment", "since surgery", "after the procedure"],
    ),
    # Procedure
    "procedure_type": AttributeSpec(
        name="procedure_type",
        definition="What procedure or intervention was performed or planned.",
        examples=["appendectomy", "colonoscopy", "psychotherapy session", "physical therapy"],
    ),
    "procedure_subject": AttributeSpec(
    name="procedure_subject",
    definition=(
        "Who the procedure or intervention was performed on or is planned for. "
        "Use this to explicitly state the recipient of the procedure when it is not the patient, "
        "or when multiple people are discussed."
    ),
    examples=[
        "the patient",
        "the patient's mother",
    ],
),
    "procedure_status": AttributeSpec(
        name="procedure_status",
        definition="State of the procedure.",
        examples=["planned", "in progress", "completed", "canceled"],
    ),
    "procedure_intent": AttributeSpec(
        name="procedure_intent",
        definition="Clinical reason or goal of the procedure.",
        examples=["diagnostic", "therapeutic", "screening", "pain relief"],
    ),
    "procedure_body_site": AttributeSpec(
        name="procedure_body_site",
        definition="Anatomical site targeted by the procedure, if applicable.",
        examples=["knee", "colon", "lumbar spine"],
    ),
    "procedure_approach": AttributeSpec(
        name="procedure_approach",
        definition="Approach or modality used.",
        examples=["laparoscopic", "open", "endoscopic", "cognitive behavioral therapy"],
    ),
    "procedure_performer": AttributeSpec(
        name="procedure_performer",
        definition="Who performed the procedure (role/title) if stated.",
        examples=["surgeon", "therapist", "psychiatrist", "primary care doctor"],
    ),
    "procedure_location": AttributeSpec(
        name="procedure_location",
        definition="Location or care setting of the procedure.",
        examples=["outpatient clinic", "operating room", "physical therapy clinic"],
    ),
    "procedure_date": AttributeSpec(
        name="procedure_date",
        definition="When the procedure occurred or is scheduled.",
        examples=["today", "last week", "scheduled for next month", "in 2019"],
    ),
    "procedure_duration": AttributeSpec(
        name="procedure_duration",
        definition="How long the procedure session or course lasted.",
        examples=["30 minutes", "2 hours", "6-week course"],
    ),
    "procedure_outcome": AttributeSpec(
        name="procedure_outcome",
        definition="Result or immediate outcome of the procedure when stated.",
        examples=["no complications", "improved pain", "tolerated well"],
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
    "patient_agreement",
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
    "reaction",
    "duration",
    "impact",
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
    "medication_status",
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

TIME_ATTRS = [
    "time_text",
    "time_kind",
    "time_normalized",
    "time_granularity",
    "time_value",
    "time_unit",
    "time_start",
    "time_end",
    "time_anchor",
    "time_reference",
    "notes",
]
TIME_SCHEMA = NodeSchema(
    class_name="Time",
    attribute_options=TIME_ATTRS,
    attribute_definitions=_attribute_def_map(TIME_ATTRS),
    shacl_shape=_shape_from_options("Time", "Time", TIME_ATTRS),
)

PROCEDURE_ATTRS = [
    "procedure_type",
    "procedure_subject",
    "procedure_status",
    "procedure_intent",
    "procedure_body_site",
    "procedure_approach",
    "procedure_performer",
    "procedure_location",
    "procedure_date",
    "procedure_duration",
    "procedure_outcome",
    "notes",
]
PROCEDURE_SCHEMA = NodeSchema(
    class_name="Procedure",
    attribute_options=PROCEDURE_ATTRS,
    attribute_definitions=_attribute_def_map(PROCEDURE_ATTRS),
    shacl_shape=_shape_from_options("Procedure", "Procedure", PROCEDURE_ATTRS),
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
    "PROCEDURE": PROCEDURE_SCHEMA,
    "TIME": TIME_SCHEMA,
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
    PROCEDURE_SCHEMA.shacl_shape,
    TIME_SCHEMA.shacl_shape,
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

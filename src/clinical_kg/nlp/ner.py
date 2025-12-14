"""
NER and mention extraction utilities.

This module combines a base spaCy (or compatible) NER model with optional LLM
refinement for richer mention detection and normalization.
"""

from typing import Any, Dict, List, Optional

import spacy

from clinical_kg.config import PipelineConfig, load_config
from clinical_kg.data_models import Mention, MentionType, Turn
from clinical_kg.nlp.llm_client import call_llm_for_extraction

# Cache the loaded base model
_nlp_model = None

def get_base_ner_model(cfg: Optional[PipelineConfig] = None):
    """
    Load and cache the spaCy (or other) base NER model.
    Uses cfg.base_ner_model_name to load the model.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
    cfg = cfg or load_config()
    _nlp_model = spacy.load(cfg.base_ner_model_name)
    return _nlp_model

def extract_mentions_base(
    turns: List[Turn],
    cfg: Optional[PipelineConfig] = None,
) -> List[Mention]:
    """
    Run the base NER model on each Turn and return a list of Mention objects.
    """
    print("you are in extract mentions base")
    model = get_base_ner_model(cfg)
    mentions: List[Mention] = []
    counter = 1

    for turn in turns:
        doc = model(turn.text)
        for ent in doc.ents:
            mention_id = f"m{counter:04d}"
            counter += 1
            # Try to pull confidence if provided by pipeline
            confidence = None
            if hasattr(ent, "kb_id_") and isinstance(ent.kb_id_, (float, int)):
                confidence = float(ent.kb_id_)
            elif hasattr(ent, "_") and hasattr(ent._, "confidence"):
                try:
                    confidence = float(ent._.confidence)  # type: ignore[attr-defined]
                except Exception:
                    confidence = None

            mentions.append(
                Mention(
                    mention_id=mention_id,
                    turn_id=turn.turn_id,
                    text=ent.text,
                )
            )

    return mentions


def refine_mentions_with_llm(
    turns: List[Turn],
    base_mentions: List[Mention],
    cfg: Optional[PipelineConfig] = None,
) -> List[Mention]:
    """
    Use an LLM to refine the base mentions.
    """
    cfg = cfg or load_config()

    # Build a single prompt covering all turns for simplicity
    turn_texts = [
        f"{t.turn_id} ({t.speaker}): {t.text}"
        for t in turns
    ]
    mention_payload = [
        {
            "turn_id": m.turn_id,
            "text": m.text,
        }
        for m in base_mentions
    ]

    system_prompt = (
        """You are a clinical information extraction assistant.
        Your input will be:
        1. A transcript of a conversation between a clinician and a patient, split into turns.
        2. A JSON list of span-level "mentions" previously extracted by a base NER model.

        Each mention has:
        - turn_id: string identifier of the turn
        - text: substring of the turn

        Your task in this phase is to clean and refine the mention list while keeping it span-based:

        1. Keep only clinically meaningful mentions:
        - Conditions, diagnoses, diseases, problems, symptoms, signs, complaints.
        - Procedures, interventions, surgeries.
        - Lab tests, imaging studies, vital signs, and other tests.
        - Test results, numeric or qualitative observation values.
        - Medications, doses, frequencies, routes, units, and related prescribing details.
        - Allergies, risk factors, and relevant social or lifestyle factors that affect care.
        - Patient descriptors (age, sex, pregnancy status) and clinician roles.

        2. Remove or correct mentions that are not clinically useful:
        - Greetings and small talk (e.g., “good morning”, “have a nice day”).
        - Generic nouns without clinical content (e.g., “changes”, “things”, “today”),
            unless they appear as part of a medically meaningful phrase such as
            “changes in vision” or “change in blood pressure”.
        - Fragments that look like truncations or obvious parsing errors.

        3. Add the type of entity being mentioned:

        For each mention, assign a "type" value using the following categories. Use the context to determine what the type is. Only use one of the types provided below.

        - PROBLEM: Any clinical issue affecting the patient, including diseases, diagnoses, chronic conditions, acute problems, symptoms, signs, and complaints.
            Examples: “type 2 diabetes”, “chest pain”, “shortness of breath”, "allergic to peanuts", “high blood pressure”.

        - MEDICATION: Any drug or therapeutic product, including brand and generic names, combinations, and formulations.
            Examples: “lisinopril”, “metformin”, “ibuprofen”, “aspirin 81 mg”.

        - LAB_TEST: Any test, measurement, or clinical investigation (labs, imaging, vital sign types). This is the name of what is measured, not the result.
            Examples: “hemoglobin A1c”, “CBC”, “MRI of the knee”, “blood pressure”, “heart rate”.

        - UNIT: Units of measurement used with lab values, doses, or vital signs.
            Examples: “mg”, “mg/dL”, “mmHg”, “bpm”, “degrees Celsius”.

        - DOSE_AMOUNT: The quantity or strength of a medication, including numeric amounts and dose size.
            Examples: “10 mg”, “500 mg”, “one tablet”, “2 puffs”, “0.5 units”.

        - FREQUENCY: How often a medication or treatment is taken or performed.
            Examples: “once daily”, “twice a day”, “every 8 hours”, “three times a week”, “as needed”.

        - PERSON_PATIENT: References to the patient as a person or their key demographic descriptors when clinically relevant.
            Examples: “Jane Doe”, “she”, “he”, “this 45-year-old woman”, “your son”.
        
        ` PERSON_CLINICIAN: References to the clinician (doctor/provider) involved in the encounter, including their name, title, role on the care team, and specialty when it is tied to that clinician mention.
            Include: the doctor as a person, clinician roles (attending, resident, fellow), and specialty descriptors attached to them (for example “cardiologist” meaning the doctor).
            Examples: “Dr. Smith”, “Doctor Patel”, “my cardiologist”, “your primary care doctor”, “the attending”, “the resident”, “the fellow”, “Dr. Lee from orthopedics”, “the ER doctor”, “your surgeon”, “my

        - OBS_VALUE: The observed or measured result of a test or sign. This is the value, not the test name, and can be numeric or qualitative.
            Examples: “7.2”, “one forty over ninety”, “98.6”, “elevated”, “normal”, “very low”.

        - ACTIVITY: Behaviors, actions, lifestyle patterns, occupations, and hobbies that are clinically relevant
            (for example, affecting risk, exposure, or functional status).
            Examples: "smoking", "drinking alcohol", "running", "plays soccer", "works as a construction worker", "nurse on night shifts".

        - OTHER: Any clinically relevant mention that does not clearly fit the categories above, but should still be kept for downstream processing.

        4. Do NOT merge mentions in this phase. Each mention should still correspond
        to a single turn.

        5. Output STRICT, machine-parsable JSON ONLY:
        A JSON list of mention objects in the same schema as the input:
        [
            {
            "turn_id": "...",
            "text": "...",
            "type": "<one of the allowed types>",
            },
            ...
        ]

        Do not include any explanations or comments.
        Return only the JSON list."""

    )
    user_prompt = (
        "Turns:\n" + "\n".join(turn_texts) + "\n\n"
        "Existing mentions (JSON):\n" + json_dumps(mention_payload) +
        "\nReturn refined mentions as pure JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        llm_output = call_llm_for_extraction(messages, cfg)
    except Exception:
        # Fallback to base mentions on any LLM failure
        return base_mentions

    refined: List[Mention] = []
    counter = 1
    seen = set()

    if not isinstance(llm_output, list):
        return base_mentions

    for item in llm_output:
        if not isinstance(item, dict):
            continue
        turn_id = item.get("turn_id")
        text = item.get("text")
        mtype = item.get("type")

        if turn_id is None or text is None or mtype is None:
            continue

        key = (turn_id, text, str(mtype))
        if key in seen:
            continue
        seen.add(key)

        mention_id = f"m{counter:04d}"
        counter += 1

        refined.append(
            Mention(
                mention_id=mention_id,
                turn_id=str(turn_id),
                text=str(text),
                type=str(mtype),
            )
        )

    return refined


def json_dumps(obj: Any) -> str:
    """Small helper to avoid importing json at top level twice."""
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def extract_mentions(
    turns: List[Turn],
    cfg: Optional[PipelineConfig] = None,
    use_llm_refinement: bool = True,
) -> List[Mention]:
    """
    High level function to extract mentions from turns.
    """
    cfg = cfg or load_config()
    base_mentions = extract_mentions_base(turns, cfg)
    if not use_llm_refinement:
        return base_mentions
    return refine_mentions_with_llm(turns, base_mentions, cfg)

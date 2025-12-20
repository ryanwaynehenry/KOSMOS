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

    def _chunk(seq: List[Any], size: int = 50):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    turn_texts = [f"{t.turn_id} ({t.speaker}): {t.text}" for t in turns]

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
        - Test results, numeric or qualitative observation values (include normal/negative findings when stated).
        - Medications, doses, frequencies, routes, units, and related prescribing details.
        - Allergies, risk factors, and relevant social or lifestyle factors that affect care.
        - Patient descriptors (age, sex, pregnancy status) and clinician roles.

        2. Remove or correct mentions that are not clinically useful:
        - Greetings and small talk (e.g., “good morning”, “have a nice day”).
        - Generic nouns without clinical content (e.g., “changes”, “things”, “today”),
            unless they appear as part of a medically meaningful phrase such as
            “changes in vision” or “change in blood pressure”.
        - Fragments that look like truncations or obvious parsing errors.
        Do NOT exclude mentions solely because they repeat earlier mentions; keep all clinically meaningful repeats so downstream clustering can see every turn they appear in.

        3. Add the type of entity being mentioned:

        For each mention, assign a "type" value using the following categories. Use the context to determine what the type is. Only use one of the types provided below.

        - PROBLEM: Any clinical issue affecting the patient, including diseases, diagnoses, chronic conditions, acute problems, symptoms, signs, and complaints.
            Examples: “type 2 diabetes”, “chest pain”, “shortness of breath”, "allergic to peanuts", “high blood pressure”.

        - MEDICATION: Any drug or therapeutic product, including brand and generic names, combinations, and formulations.
            Examples: “lisinopril”, “metformin”, “ibuprofen”, “aspirin 81 mg”.

        - LAB_TEST: Any test, measurement, or clinical investigation (labs, imaging, vital sign types). This is the name of what is measured, not the result.
            Examples: “hemoglobin A1c”, “CBC”, “MRI of the knee”, “blood pressure”, “heart rate”.

        - PROCEDURE: Therapeutic or diagnostic interventions performed on or for the patient, including surgeries, injections, procedures, rehabilitative therapies, and psychotherapy/behavioral treatments.
            Examples: “psychotherapy”, “CBT session”, “physical therapy”, “knee replacement”, “colonoscopy”, “appendectomy”, “lumbar epidural injection”, "MMRI vaccine".

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

        5. Output STRICT, machine-parsable JSON ONLY with two top-level keys:
        {
          "kept": [
            {"turn_id": "...", "text": "...", "type": "<one of the allowed types>"},
            ...
          ],
          "excluded": [
            {"turn_id": "...", "text": "...", "reason": "..."},
            ...
          ]
        }

        - List every input mention either in kept or excluded.
        - For excluded items, include a brief reason.
        Return only the JSON."""

    )

    refined: List[Mention] = []
    counter = 1
    seen = set()

    for batch in _chunk(base_mentions, 50):
        mention_payload = [
            {
                "turn_id": m.turn_id,
                "text": m.text,
            }
            for m in batch
        ]

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
            llm_output = call_llm_for_extraction(messages, cfg, label="ner_refine_mentions_llm")
        except Exception:
            llm_output = None

        kept_items: List[Dict[str, Any]] = []
        if isinstance(llm_output, dict):
            kept_items = llm_output.get("kept") or []
        elif isinstance(llm_output, list):
            kept_items = llm_output  # backward compatibility if model still returns list

        if not kept_items:
            # Fallback: keep batch as-is
            for m in batch:
                key = (m.turn_id, m.text, str(m.type))
                if key in seen:
                    continue
                seen.add(key)
                mention_id = f"m{counter:04d}"
                counter += 1
                refined.append(Mention(mention_id=mention_id, turn_id=str(m.turn_id), text=str(m.text), type=m.type))
            continue

        for item in kept_items:
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


def extract_mentions_llm(
    turns: List[Turn],
    cfg: Optional[PipelineConfig] = None,
) -> List[Mention]:
    """
    Use an LLM to directly extract mentions from turns.
    Returns the same format as the spaCy-based extractor (mention_id, turn_id, text, optional type).
    """
    cfg = cfg or load_config()

    turn_texts = [f"{t.turn_id} ({t.speaker}): {t.text}" for t in turns]
    system_prompt = (
        """You are a clinical mention extractor.

Extract ALL clinically relevant mention spans from the transcript turns, including:
- patient and clinician references (names, roles, pronouns)
- conditions, symptoms, complaints
- medications (drug names), doses, frequencies, routes, units
- tests and measurements (labs, imaging, vitals), observed values
- activities/behaviors (smoking, drinking, exercise, jobs) and related quantities
- any other clinically relevant spans
Do NOT merge or group; keep one entry per span and per turn. A turn may contain multiple mentions.

            Output STRICT JSON list where each item has:
            - turn_id: the turn containing the text span
            - text: the exact mention text

            Format:
            [
            {"turn_id": "t0001", "text": "..."},
            {"turn_id": "t0002", "text": "..."},
            ...
            ]
            Return only the JSON list."""
    )

    user_prompt = "Transcript turns:\n" + "\n".join(turn_texts) + "\nReturn the JSON list."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        llm_output = call_llm_for_extraction(messages, cfg, label="ner_extract_mentions_llm")
    except Exception:
        return []

    mentions: List[Mention] = []
    if not isinstance(llm_output, list):
        return mentions

    counter = 1
    seen = set()
    for item in llm_output:
        if not isinstance(item, dict):
            continue
        turn_id = item.get("turn_id")
        text = item.get("text")
        if turn_id is None or text is None:
            continue
        key = (turn_id, text)
        if key in seen:
            continue
        seen.add(key)
        mention_id = f"m{counter:04d}"
        counter += 1
        mentions.append(Mention(mention_id=mention_id, turn_id=str(turn_id), text=str(text), type=None))

    return mentions


def json_dumps(obj: Any) -> str:
    """Small helper to avoid importing json at top level twice."""
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def extract_mentions(
    turns: List[Turn],
    cfg: Optional[PipelineConfig] = None,
    use_llm_refinement: bool = True,
    use_llm_direct: bool = True,
) -> List[Mention]:
    """
    High level function to extract mentions from turns.
    """
    cfg = cfg or load_config()
    if use_llm_direct:
        base_mentions = extract_mentions_llm(turns, cfg)
        # fallback to base if LLM fails
    else:
        base_mentions = extract_mentions_base(turns, cfg)
    if not use_llm_refinement:
        return base_mentions
    return refine_mentions_with_llm(turns, base_mentions, cfg)

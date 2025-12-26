"""
NER and mention extraction utilities.

This module combines a base spaCy (or compatible) NER model with optional LLM
refinement for richer mention detection and normalization.
"""

from typing import Any, Dict, List, Optional
import re

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

    # Pre-seed patient and clinician mentions (one each) before running base model.
    if turns:
        patient_mention = Mention(
            mention_id=f"m{counter:04d}",
            turn_id=turns[0].turn_id,
            text="patient",
            type=MentionType.PERSON_PATIENT.value if hasattr(MentionType, "PERSON_PATIENT") else None,
        )
        counter += 1
        clinician_mention = Mention(
            mention_id=f"m{counter:04d}",
            turn_id=turns[0].turn_id,
            text="clinician",
            type=MentionType.PERSON_CLINICIAN.value if hasattr(MentionType, "PERSON_CLINICIAN") else None,
        )
        counter += 1
        mentions.extend([patient_mention, clinician_mention])

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
        - A transcript of a conversation between a clinician and a patient, split into turns.
        - A JSON list of span-level mentions previously extracted by a base NER model.

        Each mention has:
        - turn_id: string identifier of the turn
        - text: substring of the turn

        Your task in this phase is to refine the mention list while preserving span-level structure and turn-level completeness. This phase prioritizes structural fidelity over clinical judgment.

        Core Operating Principles (Read Carefully)
        - This task is not about summarization, deduplication, or clinical triage.
        - Do not apply subjective judgments about clinical importance, relevance, or significance.
        - If a mention fits an allowed clinical category, it must be retained even if it is repeated, normal, negative, or boilerplate.
        - Downstream stages depend on seeing every occurrence of a mention in every turn.

        1. Keep Clinically Scoped Mentions

        Retain any mention that fits any of the categories below, regardless of repetition or perceived importance:
        - Conditions, diagnoses, diseases, problems, symptoms, signs, and complaints.
        - Procedures, interventions, therapies, or counseling modalities.
        - Lab tests, imaging studies, vital sign types, and other clinical measurements.
        - Test results and observation values, including normal, negative, or absent findings when explicitly stated.
        - Medications, doses, frequencies, routes, units, and prescribing details.
        - Allergies, risk factors, and relevant social or lifestyle factors.
        - Patient descriptors and references, including names, pronouns, and demographic phrases.
        - Clinician references, roles, or specialties when mentioned.

        Explicit Overrides
        - PERSON_PATIENT name mentions are always clinically meaningful and must be kept, even if repeated without added context.
        - Normal or negative findings (e.g., “no acute distress”, “normal exam”, “denies pain”) are always clinically meaningful and must be kept.
        - Repetition is never a valid reason for exclusion.

        2. Remove Only Clearly Non-Clinical Mentions

        Exclude a mention only if it clearly falls into one of the categories below:
        - Greetings, closings, or conversational filler with no clinical content.
        - Generic nouns or phrases with no medical meaning.
        - Obvious truncations or parsing artifacts that do not form a meaningful phrase.

        Forbidden Exclusion Reasons
        Do not exclude mentions for any of the following reasons:
        - “Redundant”
        - “Not clinically significant”
        - “Normal finding”
        - “Boilerplate”
        - “Already mentioned earlier”
        - “Lacks additional context”

        3. Assign an Entity Type

        For each kept mention, you must assign exactly one type from the allowed set below.
        You are not permitted to invent, rename, expand, or output any other type value.
        If a mention does not cleanly match a specific category, you must still choose one of the allowed types, and in that case use OTHER.

        Allowed type values (must match exactly, including capitalization and underscores):
        - PROBLEM: Diseases, diagnoses, symptoms, signs, allergies, complaints, whether present or not.
        - MEDICATION: Drugs, therapeutic products, formulations.
        - LAB_TEST: Names of tests or measurements (not results).
        - PROCEDURE: Clinical actions and care-plan items, including therapeutic or diagnostic interventions, psychotherapy, procedures, surgeries, appointments and visits, follow-ups, referrals and consultations, and care coordination events.
        - UNIT: Units of measurement.
        - DOSE_AMOUNT: Numeric or descriptive dose amounts.
        - FREQUENCY: Timing or frequency of medications or treatments.
        - TIME: Any timing information, including dates, times, durations, and relative time expressions (e.g., “today”, “yesterday”, “next week”, “in 2 days”, “for 3 months”, “last visit”, “post-op”, “pre-op”).
        - PERSON_PATIENT: Patient names, pronouns, or demographic references.
        - PERSON_CLINICIAN: Clinician names, roles, or specialties.
        - OBS_VALUE: Observed or measured results, including normal or negative values.
        - ACTIVITY: Clinically relevant behaviors or lifestyle factors.
        - OTHER: Clinically relevant mentions not fitting the above categories.

        Type Output Constraint
        - The "type" field must be exactly one of the allowed type values above.
        - Do not invent new types or modify the allowed labels.
        - If a mention does not cleanly match a specific category, choose the closest allowed type, or OTHER if no closer fit exists.


        3A. Demographics Attribution Rule (Critical)

        Demographic references MUST be typed as PERSON_PATIENT when they describe the patient (or another explicitly referenced person in the encounter).
        This includes demographics even when they are not a name or pronoun.

        Demographic spans that should be PERSON_PATIENT include:
        - Age and age ranges (e.g., “__ years old”, “in their __s”)
        - Sex and gender terms when used to describe the patient
        - Race and ethnicity terms when used to describe the patient
        - Pregnancy or postpartum descriptors when used to describe the patient
        - Physical descriptors that function as demographics in context (e.g., “right-handed”) when clearly describing the patient

        Attribution guidance
        - If a demographic mention appears in a turn where the patient is the subject (e.g., introductions, HPI framing, “the patient is …”), treat it as describing the patient and type it PERSON_PATIENT.
        - If the demographic mention explicitly refers to a different person (e.g., a family member), still type it as PERSON_PATIENT because it is a person descriptor, unless your system requires separate family typing (not allowed here).
        - Do not type demographics as OTHER, PROBLEM, or ACTIVITY when they are describing a person.

        Edge cases
        - If the demographic is part of a longer mention span provided by the base NER, do not merge or expand, but still assign PERSON_PATIENT to that span if it is primarily demographic.
        - If a demographic term is used generically (not describing any person in the encounter), you may exclude it as non-clinical only if it clearly has no patient linkage.

        4. Structural Constraints
        - Do not merge mentions.
        - Do not deduplicate across turns.
        - Each mention must remain associated with its original turn_id.
        - Preserve every clinically scoped mention occurrence, even if repeated verbatim.

        5. Output Format (STRICT)

        Return only valid, machine-parsable JSON with exactly two top-level keys:
        {
        "kept": [
            {"turn_id": "...", "text": "...", "type": "<allowed type>"},
            ...
        ],
        "excluded": [
            {"turn_id": "...", "text": "...", "reason": "..."},
            ...
        ]
        }

        Output Rules
        - Every input mention must appear exactly once in either kept or excluded.
        - Exclusion reasons must reference only the allowed removal criteria.
        - Do not include explanations, commentary, or markdown.
        - Return only the JSON.
        """

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

    def word_count(text: str) -> int:
        return len(re.findall(r"\w+", text))

    batches: List[List[str]] = []
    current: List[str] = []
    current_words = 0
    for tt in turn_texts:
        wc = word_count(tt)
        if current and current_words + wc > 300:
            batches.append(current)
            current = [tt]
            current_words = wc
        else:
            current.append(tt)
            current_words += wc
    if current:
        batches.append(current)

    system_prompt = (
        """You are a clinical entity mention extractor.

        Your job is to extract ONLY atomic entity spans from each transcript turn. Do not extract full questions, sentences, or clauses. If a turn contains clinically relevant content, still extract only the smallest meaningful spans.

        Core rule (atomic spans)
        - Output the smallest span that still uniquely identifies the entity.
        - Do not include surrounding filler words (e.g., “what brings”, “in”, “for”, “here”, “okay”, “so”, “and”).
        - Do not bundle multiple entities into one span. Split them into separate entries.

        What counts as an entity mention span
        Extract spans that match any of these categories, when present in the transcript:

        A) People, roles, and identifiers
        - Person names (patient, clinician, caregiver)
        - Family member roles (parent, sibling, grandparent, etc.)
        - Care team roles (nurse, therapist, pharmacist, etc.)
        - Relationship descriptors when explicitly stated (spouse, roommate, child)

        B) Demographics and person attributes
        Extract these as separate atomic spans:
        - Age and age ranges (a number with “year(s) old”, a decade range, “in their thirties”)
        - Sex or gender terms (male, female, man, woman, nonbinary, etc.) when explicitly stated
        - Pregnancy or postpartum status when explicitly stated
        - Race or ethnicity terms only if explicitly stated in the transcript
        - Height, weight, BMI, and other body measurements when explicitly stated
        - Occupation and employment status when explicitly stated
        - Marital or relationship status terms (married, divorced, widowed) when explicitly stated

        C) Problems and symptoms
        - Diagnoses, conditions, diseases
        - Symptoms, complaints, signs
        - Relevant negatives as atomic concepts (see “Negation handling”)

        D) Medications and medication attributes
        - Drug names
        - Dose or strength
        - Frequency or schedule words
        - Route
        - Form (tablet, inhaler) when explicitly stated

        E) Tests, measurements, and observed values
        - Test names
        - Measurement names (vital sign labels)
        - Numeric values and units
        - Physical exam findings as atomic concepts

        F) Time and care-plan entities (appointments, referrals, procedures, surgeries)
        - Dates, times, durations, relative times (weekday names, months, years, “next week”, “yesterday”)
        - Onset or timeframe expressions when they are time entities
        - Appointment-related spans when explicitly stated:
            follow-up or recheck or return visit or post-op visit or next appointment
            prior visit or last visit or previous appointment
        - Referral-related spans when explicitly stated:
            referral or referred or consult or consultation
            and also extract the destination or specialty as its own span when present
                examples: orthopedics, ENT, cardiology, physical therapy, pain clinic
        - Procedure and surgery-related spans when explicitly stated:
            procedure or surgery or operation
            scheduled procedure or scheduled surgery
            pre-op or post-op
            and also extract the procedure or surgery name as its own span when present
                examples: appendectomy, knee replacement, arthroscopy
        - Care-plan action cues as atomic spans when explicitly stated:
            start or stop or order or prescribe or schedule or reschedule or cancel

        G) Social and exposure entities
        - Substances (tobacco, alcohol, vaping, etc.)
        - Quantities and patterns (frequency phrases, counts)
        - Living situation or household context as short spans
        - Exposures (travel, outdoor exposure, workplace exposure) as short spans

        H) Vaccines
        - Vaccines and immunizations as atomic spans when explicitly stated

        Appointment, referral, and surgery extraction rules (atomic spans)
        - Always split action or event from the time reference.
            Bad: "follow up in 2 weeks"
            Good: "follow up", "2 weeks"

            Bad: "scheduled for surgery on January 10"
            Good: "scheduled for surgery", "January 10"

            Bad: "referred to cardiology last visit"
            Good: "referred", "cardiology", "last visit"

        - Always extract the destination or specialty as its own span when present.
            Examples: "orthopedics", "ENT", "cardiology", "physical therapy", "pain clinic"

        - Always extract the procedure or surgery name as its own span when present.
            Examples: "appendectomy", "knee replacement", "arthroscopy"

        - If the transcript implies a planned event without a specific date, still extract the planning cue.
            Examples: "scheduled", "booked", "set up", "plan for surgery", "will refer", "needs referral", "follow-up"

        - If a past event is mentioned, extract both the event cue and the time reference separately if present.
            Example: "had surgery last year" -> "had surgery", "last year"

        Span splitting rules (examples are illustrative patterns only)
        - Split person name and time reference
        - Bad: "Please return next month, [NAME]"
        - Good: "[NAME]", "next month"

        - Split age, sex, and role or description
        - Bad: "[AGE]-year-old [SEX] with ..."
        - Good: "[AGE]-year-old", "[SEX]"

        - Split drug name, dose, and schedule
        - Bad: "Take [DRUG] [DOSE] [SCHEDULE]"
        - Good: "[DRUG]", "[DOSE]", "[SCHEDULE]"

        - Split measurement label and value
        - Bad: "[MEASUREMENT] [VALUE]"
        - Good: "[MEASUREMENT]", "[VALUE]"

        Negation handling
        - If a clinical concept is explicitly negated, extract the concept as its own span.
        - Optionally extract the negation cue as a separate span ONLY if your downstream pipeline uses it.
        - Example pattern: "denies [SYMPTOM]" -> "[SYMPTOM]" and optionally "denies"
        - Example pattern: "no [FINDING]" -> "[FINDING]" and optionally "no"
        - Do not output the full phrase as one chunk.

        Do not extract
        - Full sentences, full questions, or clauses.
        - Conversational scaffolding (greetings, filler, confirmations).
        - Combined multi-entity chunks.

        Output STRICT JSON list
        - Each item must include:
        - turn_id: the turn containing the entity span
        - text: the exact extracted span (verbatim substring)

        Format:
        [
        {"turn_id": "t0001", "text": "[NAME]"},
        {"turn_id": "t0001", "text": "[RELATIVE TIME]"}
        ]

        Return only the JSON list.
"""
    )

    counter = 1
    seen = set()
    mentions: List[Mention] = []

    # Pre-seed patient and clinician mentions (one each) before batching.
    if turns:
        patient_mention = Mention(
            mention_id=f"m{counter:04d}",
            turn_id=turns[0].turn_id,
            text="patient",
            type=MentionType.PERSON_PATIENT.value if hasattr(MentionType, "PERSON_PATIENT") else None,
        )
        counter += 1
        clinician_mention = Mention(
            mention_id=f"m{counter:04d}",
            turn_id=turns[0].turn_id,
            text="clinician",
            type=MentionType.PERSON_CLINICIAN.value if hasattr(MentionType, "PERSON_CLINICIAN") else None,
        )
        counter += 1
        mentions.extend([patient_mention, clinician_mention])
        seen.update({(patient_mention.turn_id, patient_mention.text), (clinician_mention.turn_id, clinician_mention.text)})

    for idx, batch in enumerate(batches, start=1):
        user_prompt = "Transcript turns:\n" + "\n".join(batch) + "\nReturn the JSON list."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            llm_output = call_llm_for_extraction(messages, cfg, label=f"ner_extract_mentions_llm_batch_{idx}")
        except Exception:
            continue

        if not isinstance(llm_output, list):
            continue

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
    # Seed patient/clinician mentions will be added inside extraction functions.
    if use_llm_direct:
        base_mentions = extract_mentions_llm(turns, cfg)
        # fallback to base if LLM fails
    else:
        base_mentions = extract_mentions_base(turns, cfg)
    if not use_llm_refinement:
        return base_mentions
    return refine_mentions_with_llm(turns, base_mentions, cfg)

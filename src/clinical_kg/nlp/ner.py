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

    # Pre-seed patient, clinician, and follow-up visit mentions before running base model.
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
        followup_mention = Mention(
            mention_id=f"m{counter:04d}",
            turn_id=turns[0].turn_id,
            text="follow-up visit",
            type=MentionType.PROCEDURE.value if hasattr(MentionType, "PROCEDURE") else None,
        )
        counter += 1
        mentions.extend([patient_mention, clinician_mention, followup_mention])

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

            Your task in this phase is to ASSIGN ENTITY TYPES to the existing mention spans while preserving span-level structure and turn-level completeness.

            Core Operating Principles (Read Carefully)
            - This task is not about summarization, deduplication, triage, or filtering.
            - Do not apply subjective judgments about clinical importance, relevance, or significance.
            - Do NOT remove mentions. Keep every input mention and assign it a type.
            - Downstream stages depend on seeing every occurrence of a mention in every turn.

            1) Keep All Mentions (No Exclusions)
            - Every input mention must appear exactly once in the output.
            - There is no "excluded" list in this phase.

            2) Assign an Entity Type
            For each input mention, assign exactly one type from the allowed set below.
            You are not permitted to invent, rename, expand, merge, split, deduplicate, or output any other type value.
            If a mention does not cleanly match a specific category, choose OTHER.

            Allowed type values (must match exactly, including capitalization and underscores):
            - PROBLEM: Diseases, diagnoses, symptoms, signs, allergies, complaints, whether present or not, including explicitly denied symptoms/conditions.
            - MEDICATION: Drugs, therapeutic products, formulations.
            - LAB_TEST: Names of tests or measurements (not results).
            - PROCEDURE: Clinical actions and care-plan items, including interventions, counseling, procedures, surgeries, appointments and visits, follow-ups, referrals and consultations, and care coordination events.
            - UNIT: Units of measurement.
            - DOSE_AMOUNT: Numeric or descriptive dose amounts.
            - FREQUENCY: Timing or frequency of medications or treatments.
            - TIME: Any timing information, including dates, times, durations, and relative time expressions.
            - PERSON_PATIENT: Patient names, pronouns, demographic references, and demographic descriptors about a person in the encounter.
            - PERSON_CLINICIAN: Clinician names, clinician role references, and specialties.
            - OBS_VALUE: Observed or measured results, including normal or negative values when stated.
            - ACTIVITY: Clinically relevant behaviors, triggers, sensitivities, avoidance behaviors, exposures, or lifestyle factors.
            - OTHER: Clinically relevant mentions not fitting the above categories, including contextual spans that still support clinical meaning.

            Typing guidance (use only if consistent with the span)
            - Clinician role words (unnamed) should be PERSON_CLINICIAN when they refer to the care provider in the encounter.
            - Care-event or visit-type phrases should be PROCEDURE.
            - Standalone timing words or duration phrases should be TIME.
            - Triggers, sensitivities, exposures, and avoidance contexts should usually be ACTIVITY (or OTHER if they do not clearly describe a behavior/exposure).
            - If the span is a greeting fragment, conversational filler, or a meaningless fragment, still KEEP it and type it as OTHER.

            Demographics Attribution Rule (Critical)
            Demographic references MUST be typed as PERSON_PATIENT when they describe the patient (or another explicitly referenced person in the encounter).
            This includes demographics even when they are not a name or pronoun.
            Examples of demographics: age phrases, sex/gender terms, race/ethnicity terms, pregnancy/postpartum descriptors, and physical descriptors functioning as demographics when clearly describing a person.

            Structural Constraints
            - Do not merge or split mentions.
            - Do not deduplicate across turns.
            - Each mention must remain associated with its original turn_id.
            - Preserve every mention occurrence, even if repeated verbatim.

            Output Format (STRICT)
            Return only valid, machine-parsable JSON with exactly one top-level key:
            {
            "typed_mentions": [
                {"turn_id": "...", "text": "...", "type": "<allowed type>"},
                ...
            ]
            }

            Output Rules
            - The output list must contain the same number of items as the input mention list.
            - Each input mention must appear exactly once in the output, with the same turn_id and text, plus a type.
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
            kept_items = llm_output.get("typed_mentions") or []
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
        - Person names (Jim, Dr. Smith, patient, clinician, doctor, caregiver)
        - Family member roles (parent, sibling, grandparent, etc.)
        - Care team role mentions (extract even when generic)
            - Extract clinician role references even when unnamed:
            - "the doctor", "my doctor", "your doctor" -> extract "doctor"
            - "the clinician", "provider" -> extract "clinician" or "provider"
            - "the nurse" -> extract "nurse"
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

    # Pre-seed patient, clinician, and follow-up visit mentions (one each) before batching.
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
        followup_mention = Mention(
            mention_id=f"m{counter:04d}",
            turn_id=turns[0].turn_id,
            text="follow-up visit",
            type=MentionType.PROCEDURE.value if hasattr(MentionType, "PROCEDURE") else None,
        )
        counter += 1
        mentions.extend([patient_mention, clinician_mention, followup_mention])
        seen.update(
            {
                (patient_mention.turn_id, patient_mention.text),
                (clinician_mention.turn_id, clinician_mention.text),
                (followup_mention.turn_id, followup_mention.text),
            }
        )

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

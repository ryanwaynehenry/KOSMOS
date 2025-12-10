"""
Preprocessing utilities for raw transcript files.

Converts transcript lines into structured Turn objects, with a required
pronoun-resolution pass via LLM.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from clinical_kg.config import PipelineConfig, load_config
from clinical_kg.data_models import Turn
from clinical_kg.nlp.llm_client import call_llm_for_extraction

_TURN_ID_TEMPLATE = "t{index:04d}"


def _resolve_pronouns_with_llm(transcript_text: str, cfg: Optional[PipelineConfig] = None) -> str:
    """
    Required: use LLM to replace pronouns with explicit noun phrases while
    preserving line structure and speaker prefixes.
    """
    cfg = cfg or load_config()
    messages = [
        {
            "role": "system",
            "content": (
                "You will receive a clinical conversation transcript between a doctor and a patient.\n\n"
                "Your goal is to rewrite the transcript by replacing pronouns with the explicit nouns or "
                "noun phrases they refer to, whenever the referent is reasonably clear from context.\n"
                "You MUST do this for BOTH:\n"
                "  (a) pronouns that refer to the doctor or patient, and\n"
                "  (b) pronouns that refer to clinical concepts (conditions, medications, tests, results, plans, etc.).\n\n"
                "1) Preserve the structure:\n"
                "   - Keep all line breaks exactly as in the input.\n"
                "   - Keep the speaker prefixes exactly as in the input (for example, 'P:' or 'D:').\n"
                "   - Do not reorder, add, or remove turns.\n\n"
                "2) Replace pronouns referring to people (doctor or patient):\n"
                "   - Replace personal pronouns such as 'I', 'me', 'my', 'mine', 'you', 'your', 'he', 'she', "
                "     'him', 'her', 'we', 'us', 'our', 'they', 'them' when they clearly refer to the doctor "
                "     or the patient.\n"
                "   - Use the full name or role based on the transcript, for example: 'Sophia Brown', "
                "     'Sophia Brown's', 'Dr. Rafael Gomez', 'Dr. Rafael Gomez's'.\n"
                "   - Example: 'P: Yes, that's me.' -> 'P: Yes, that's Sophia Brown.'\n"
                "   - Example: 'D: How are you feeling today?' -> "
                "              'D: How is Sophia Brown feeling today?'\n\n"
                "3) Replace pronouns referring to clinical concepts:\n"
                "   - For pronouns such as 'it', 'this', 'that', 'they', 'them', 'these', 'those' that refer "
                "     to specific clinical entities, replace the pronoun with an explicit phrase.\n"
                "   - Clinical entities include problems/diagnoses/symptoms (for example, depression, chest pain), "
                "     medications (for example, penicillin, sertraline), allergies, lab tests and imaging, test "
                "     results and vital signs, treatment plans, and follow-up appointments.\n"
                "   - Example: 'I'm allergic to penicillin – it gives me a rash.' -> "
                "              'I'm allergic to penicillin – penicillin gives me a rash.'\n"
                "   - Example: 'We'll start a new medication and see how it goes.' -> "
                "              'We'll start a new medication and see how the new medication goes.'\n"
                "   - Example: 'It has been getting worse.' when 'it' refers to low mood -> "
                "              'The low mood has been getting worse.'\n\n"
                "4) When NOT to replace:\n"
                "   - If the referent of a pronoun is genuinely unclear or ambiguous, leave the pronoun as is.\n"
                "   - Do not invent or guess new medical facts that are not implied by the transcript.\n\n"
                "5) Do not otherwise rewrite or summarize:\n"
                "   - Do not change the meaning of any sentence.\n"
                "   - Do not add new sentences or commentary.\n"
                "   - Only modify the text as needed to expand pronouns into explicit noun phrases.\n\n"
                "Output format:\n"
                "Return STRICT JSON with a single key 'rewritten_transcript' whose value is the full rewritten "
                "transcript as a single string, including all line breaks.\n"
                "Example:\n"
                "{ \"rewritten_transcript\": \"D: ...\\nP: ...\\n...\" }"
            ),
        },
        {"role": "user", "content": transcript_text},
    ]
    try:
        output = call_llm_for_extraction(messages, cfg)
        if isinstance(output, dict) and "rewritten_transcript" in output:
            return str(output["rewritten_transcript"])
    except Exception:
        pass
    return transcript_text


def load_and_segment(transcript_path: str, encounter_id: str, cfg: Optional[PipelineConfig] = None) -> List[Turn]:
    """
    Read a raw transcript file and return a list of Turns.

    Assumptions:
      - transcript_path points to a UTF-8 text file.
      - Each line is either empty or looks like 'SPEAKER: content'.
        For example: 'PATIENT: I have chest pain.'

    Steps:
      - Read all lines.
      - Skip empty lines.
      - Split each line at the first ':' to get speaker and text.
      - Trim whitespace from both.
      - Assign turn_id values like 't0001', 't0002', ... in order.
      - Use the given encounter_id for all Turns.
      - Leave start_time and end_time as None for now.

    Returns:
      A list of Turn objects in the same order as the lines.
    """
    path = Path(transcript_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    raw_content = path.read_text(encoding="utf-8")
    rewritten = _resolve_pronouns_with_llm(raw_content, cfg)

    turns: List[Turn] = []
    for idx, line in enumerate(rewritten.splitlines()):
        raw = line.strip()
        if not raw:
            continue
        if ":" in raw:
            speaker_part, text_part = raw.split(":", 1)
            speaker = speaker_part.strip()
            text = text_part.strip()
        else:
            speaker = "UNKNOWN"
            text = raw
        turn_id = _TURN_ID_TEMPLATE.format(index=idx + 1)
        turns.append(
            Turn(
                encounter_id=encounter_id,
                turn_id=turn_id,
                speaker=speaker,
                text=text,
                start_time=None,
                end_time=None,
            )
        )
    return turns


def save_turns_to_json(turns: List[Turn], output_path: str) -> None:
    """
    Serialize the turns to JSON for debugging.
    Represent each Turn as a dict with its fields.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [asdict(t) for t in turns]
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

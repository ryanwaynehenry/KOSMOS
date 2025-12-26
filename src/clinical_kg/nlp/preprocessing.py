"""
Preprocessing utilities for raw transcript files.

Converts transcript lines into structured Turn objects, with a required
pronoun-resolution pass via LLM.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from clinical_kg.config import PipelineConfig, load_config
from clinical_kg.data_models import Turn
from clinical_kg.nlp.llm_client import call_llm_for_extraction

_TURN_ID_TEMPLATE = "t{index:04d}"


def _chunk_transcript_to_turn_dicts(transcript_text: str) -> List[Dict[str, str]]:
    """
    Normalize the transcript into a list of turn dicts (speaker/text) while preserving
    contiguous speaker runs. This is used before sending structured content to the LLM.
    """
    turns: List[Dict[str, str]] = []
    speaker_pattern = re.compile(r"^(?P<speaker>[A-Za-z]{1,5})\.?\s*:\s*(?P<text>.*)$")

    current_speaker: Optional[str] = None
    current_chunks: List[str] = []

    def _flush_turn():
        if current_speaker is None:
            return
        text = " ".join(chunk for chunk in current_chunks if chunk)
        if not text:
            return
        turns.append({"speaker": current_speaker, "text": text})

    for line in transcript_text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        match = speaker_pattern.match(raw)
        if match:
            speaker_candidate = match.group("speaker").strip()
            text_part = match.group("text").strip()

            if speaker_candidate == current_speaker or current_speaker is None:
                current_speaker = speaker_candidate
                current_chunks.append(text_part)
            else:
                _flush_turn()
                current_speaker = speaker_candidate
                current_chunks = [text_part]
        else:
            if current_speaker is None:
                current_speaker = "UNKNOWN"
                current_chunks = [raw]
            else:
                current_chunks.append(raw)

    _flush_turn()
    return turns


def _resolve_pronouns_with_llm(turns: List[Dict[str, str]], cfg: Optional[PipelineConfig] = None) -> List[Dict[str, str]]:
    """
    Required: use LLM to replace pronouns with explicit noun phrases while preserving
    structure and speaker prefixes. Expects and returns a list of turn dicts.
    """
    cfg = cfg or load_config()
    payload = {"turns": turns}
    messages = [
        {
            "role": "system",
            "content": (
                "You will receive a clinical conversation transcript between a doctor and a patient as JSON.\n\n"
                "Input format:\n"
                "{ \"turns\": [ { \"speaker\": \"D\", \"text\": \"How are you feeling?\" }, ... ] }\n\n"
                "Your goal is to rewrite ONLY the 'text' fields by replacing pronouns with the explicit nouns or "
                "noun phrases they refer to, whenever the referent is reasonably clear from context.\n"
                "You MUST do this for BOTH:\n"
                "  (a) pronouns that refer to the doctor or patient, and\n"
                "  (b) pronouns that refer to clinical concepts (conditions, medications, tests, results, plans, etc.).\n\n"
                "Structural requirements:\n"
                "  - Keep the same number of turns, in the same order.\n"
                "  - Do not change the 'speaker' values.\n"
                "  - Do not add or remove keys.\n\n"
                "2) Replace pronouns referring to people (doctor or patient):\n"
                "   - Replace personal pronouns such as 'I', 'me', 'my', 'mine', 'you', 'your', 'he', 'she', "
                "     'him', 'her', 'we', 'us', 'our', 'they', 'them' when they clearly refer to the doctor "
                "     or the patient.\n"
                "   - Detect the patient name and doctor name if they appear anywhere in the transcript (even once) "
                "     and use those names consistently for every pronoun reference to that person across all turns.\n"
                "   - Use the full name or role when it is explicitly provided (for example: 'Sophia Brown', "
                "     'Sophia Brown's', 'Dr. Rafael Gomez', 'Dr. Rafael Gomez's').\n"
                "   - If a name is not provided, DO NOT invent one. Use 'the patient' / 'the doctor' instead.\n"
                "   - Example: 'P: Yes, that's me.' -> 'P: Yes, that's Sophia Brown.' (when name given)\n"
                "   - Example: 'P: Yes, that's me.' -> 'P: Yes, that's the patient.' (when name not given)\n"
                "   - Example: 'D: How are you feeling today?' -> "
                "              'D: How is Sophia Brown feeling today?' (when patient name given)\n"
                "   - Example: 'D: How are you feeling today?' -> "
                "              'D: How is the patient feeling today?' (when patient name not given)\n\n"
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
                "Return STRICT JSON with a single key 'turns' holding the modified turns list.\n"
                "Example:\n"
                "{ \"turns\": [ { \"speaker\": \"D\", \"text\": \"How is Sophia Brown feeling today?\" }, ... ] }"
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    try:
        output = call_llm_for_extraction(messages, cfg, label="preprocess_llm")
        if isinstance(output, dict):
            if isinstance(output.get("turns"), list):
                rewritten_turns: List[Dict[str, str]] = []
                returned_turns = output["turns"]
                for idx, turn in enumerate(returned_turns):
                    base = turns[idx] if idx < len(turns) else {}
                    speaker = (turn.get("speaker") if isinstance(turn, dict) else None) or base.get("speaker") or ""
                    text = (turn.get("text") if isinstance(turn, dict) else None) or base.get("text") or ""
                    rewritten_turns.append({"speaker": str(speaker), "text": str(text)})
                if len(rewritten_turns) == len(turns):
                    return rewritten_turns
            if isinstance(output.get("rewritten_transcript"), str):
                return _chunk_transcript_to_turn_dicts(output["rewritten_transcript"])
    except Exception as exc:
        print(f"[preprocess] pronoun rewrite failed; using original turns: {exc}")
    return turns


def _normalize_speaker_tokens(transcript_text: str) -> str:
    """
    Normalize bracketed speaker tags to the expected colon format (D:/P:).
    """
    normalized = re.sub(r"(?mi)^\s*\[doctor\]\s*", "D: ", transcript_text)
    normalized = re.sub(r"(?mi)^\s*\[patient\]\s*", "P: ", normalized)
    return normalized


def _segment_transcript_text(
    transcript_text: str, encounter_id: str, cfg: Optional[PipelineConfig] = None
) -> List[Turn]:
    """
    Shared segmentation logic for both on-disk and in-memory transcripts.
    """
    normalized_text = _normalize_speaker_tokens(transcript_text)
    base_turns = _chunk_transcript_to_turn_dicts(normalized_text)
    rewritten_turns = _resolve_pronouns_with_llm(base_turns, cfg)
    turns_for_segment = rewritten_turns if rewritten_turns else base_turns
    turns: List[Turn] = []
    for turn_dict in turns_for_segment:
        speaker = turn_dict.get("speaker") or "UNKNOWN"
        text = (turn_dict.get("text") or "").strip()
        if not text:
            continue
        turn_id = _TURN_ID_TEMPLATE.format(index=len(turns) + 1)
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


def segment_transcript_text(
    transcript_text: str, encounter_id: str, cfg: Optional[PipelineConfig] = None
) -> List[Turn]:
    """
    Segment an in-memory transcript string into Turns (same behavior as load_and_segment).
    """
    return _segment_transcript_text(transcript_text, encounter_id, cfg)


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
    return _segment_transcript_text(raw_content, encounter_id, cfg)


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

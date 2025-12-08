"""
Preprocessing utilities for raw transcript files.

This module is the only place that knows about the raw transcript format. It
converts transcript lines into structured Turn objects and can optionally
split long turns into sentence-level turns and save intermediate JSON for
debugging.
"""

import json
import os
import re
from typing import List

from clinical_kg.data_models import Turn

_TURN_ID_TEMPLATE = "t{index:04d}"


def _parse_line(line: str) -> tuple[str, str]:
    """
    Split a raw transcript line into (speaker, text).

    Expected format is `D: text` or `P: text` (doctor/patient). Falls back to
    an unknown speaker if the prefix is missing.
    """
    stripped = line.strip()
    if not stripped:
        return ("UNKNOWN", "")

    if ":" in stripped:
        speaker_prefix, text = stripped.split(":", 1)
        speaker_prefix = speaker_prefix.strip().upper()
        text = text.strip()
        if speaker_prefix in ("D", "DOCTOR", "CLINICIAN"):
            speaker = "CLINICIAN"
        elif speaker_prefix in ("P", "PATIENT"):
            speaker = "PATIENT"
        else:
            speaker = speaker_prefix or "UNKNOWN"
        return (speaker, text)

    return ("UNKNOWN", stripped)


def load_and_segment(transcript_path: str, encounter_id: str) -> List[Turn]:
    """
    Read a raw transcript file and return a list of Turns in chronological
    order. Currently supports simple txt files with `SPEAKER: text` per line.
    """
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    turns: List[Turn] = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            speaker, text = _parse_line(line)
            turn_id = _TURN_ID_TEMPLATE.format(index=idx + 1)
            turns.append(Turn(encounter_id=encounter_id, turn_id=turn_id, speaker=speaker, text=text))
    return turns


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_turns_into_sentences(turns: List[Turn]) -> List[Turn]:
    """
    Optionally split turns into sentence-level chunks. Uses a simple regex
    split; this can be swapped out for a more robust sentence splitter later.
    """
    split_turns: List[Turn] = []
    counter = 1
    for turn in turns:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(turn.text) if s.strip()]
        for sent in sentences:
            turn_id = _TURN_ID_TEMPLATE.format(index=counter)
            split_turns.append(
                Turn(
                    encounter_id=turn.encounter_id,
                    turn_id=turn_id,
                    speaker=turn.speaker,
                    text=sent,
                    start_time=turn.start_time,
                    end_time=turn.end_time,
                )
            )
            counter += 1
    return split_turns


def save_turns_to_json(turns: List[Turn], output_path: str) -> None:
    """
    Serialize a list of turns to JSON for debugging or later reuse.
    """
    serializable = [
        {
            "encounter_id": t.encounter_id,
            "turn_id": t.turn_id,
            "speaker": t.speaker,
            "text": t.text,
            "start_time": t.start_time,
            "end_time": t.end_time,
        }
        for t in turns
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

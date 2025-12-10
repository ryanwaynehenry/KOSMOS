"""
Preprocessing utilities for raw transcript files.

Converts transcript lines into structured Turn objects. Keeps dependencies to
the standard library plus clinical_kg.data_models.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from clinical_kg.data_models import Turn

_TURN_ID_TEMPLATE = "t{index:04d}"


def load_and_segment(transcript_path: str, encounter_id: str) -> List[Turn]:
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

    turns: List[Turn] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
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

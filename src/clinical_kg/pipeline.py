"""
Transcript-only processing pipeline orchestration.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

from clinical_kg.data_models import Mention, Turn
from clinical_kg.nlp import coref, ner, preprocessing, relations


def _maybe_save(obj, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _default_encounter_id(transcript_path: str) -> str:
    """
    Derive an encounter id from the transcript filename.

    Examples:
      altered_session_2347_1.txt -> 2347_1
      session_123.txt -> 123
      anything_else -> filename stem
    """
    stem = Path(transcript_path).stem
    lower = stem.lower()
    if "session_" in lower:
        # take substring after the first occurrence of "session_"
        after = lower.split("session_", 1)[1]
        if after:
            return after
    return stem


def process_transcript_to_mentions(
    transcript_path: str,
    encounter_id: str | None = None,
    save_intermediate: bool = False,
) -> Tuple[List[Turn], List[Mention], str]:
    """
    End to end transcript processing for one encounter, up to mention-level
    output. Returns turns, mentions, and the resolved encounter_id.

    Steps:
      1. Load and segment transcript into Turns.
      2. Run NER to get Mentions.
      3. Run coref to assign coref_cluster_id.
      4. Attach local attributes (dose, unit, negation, temporality).
    """
    transcript_path = str(transcript_path)
    resolved_encounter = encounter_id or _default_encounter_id(transcript_path)

    turns = preprocessing.load_and_segment(transcript_path, resolved_encounter)
    mentions = ner.extract_mentions(turns)
    mentions = coref.add_coref_clusters(mentions, turns)
    mentions = relations.attach_attributes(mentions)

    if save_intermediate:
        interim_dir = Path("data") / "interim"
        _maybe_save(
            [
                {
                    "encounter_id": t.encounter_id,
                    "turn_id": t.turn_id,
                    "speaker": t.speaker,
                    "text": t.text,
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                }
                for t in turns
            ],
            interim_dir / f"{resolved_encounter}_turns.json",
        )
        _maybe_save(
            [
                {
                    "mention_id": m.mention_id,
                    "encounter_id": m.encounter_id,
                    "turn_id": m.turn_id,
                    "start_char": m.start_char,
                    "end_char": m.end_char,
                    "text": m.text,
                    "type": m.type,
                    "confidence": m.confidence,
                    "coref_cluster_id": m.coref_cluster_id,
                    "attributes": m.attributes,
                }
                for m in mentions
            ],
            interim_dir / f"{resolved_encounter}_mentions.json",
        )

    return turns, mentions, resolved_encounter


def save_processed_transcript(
    transcript_path: str,
    encounter_id: str | None = None,
    save_intermediate: bool = False,
) -> Path:
    """
    Process a transcript and write the consolidated JSON to
    data/interim/<source-stem>.json. Returns the output path.
    """
    turns, mentions, resolved_encounter = process_transcript_to_mentions(
        transcript_path=transcript_path,
        encounter_id=encounter_id,
        save_intermediate=save_intermediate,
    )

    transcript_stem = Path(transcript_path).stem
    output_path = Path("data") / "interim" / f"{transcript_stem}.json"
    os.makedirs(output_path.parent, exist_ok=True)

    payload = {
        "encounter_id": resolved_encounter,
        "turns": [
            {
                "turn_id": t.turn_id,
                "speaker": t.speaker,
                "text": t.text,
                "start_time": t.start_time,
                "end_time": t.end_time,
            }
            for t in turns
        ],
        "mentions": [
            {
                "mention_id": m.mention_id,
                "turn_id": m.turn_id,
                "start_char": m.start_char,
                "end_char": m.end_char,
                "text": m.text,
                "type": m.type,
                "confidence": m.confidence,
                "coref_cluster_id": m.coref_cluster_id,
                "attributes": m.attributes,
            }
            for m in mentions
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path

"""
Transcript-only processing pipeline orchestration.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

from clinical_kg.config import PipelineConfig, load_config
from clinical_kg.data_models import Mention, Turn
from clinical_kg.nlp.coref import add_coref_clusters
from clinical_kg.nlp.ner import extract_mentions
from clinical_kg.nlp.preprocessing import load_and_segment
from clinical_kg.nlp.relations import attach_attributes


def process_transcript_to_mentions(
    transcript_path: str,
    encounter_id: str,
    save_intermediate: bool = False,
    use_llm_for_ner: bool = True,
    use_llm_for_coref: bool = True,
) -> Tuple[List[Turn], List[Mention]]:
    """
    End to end transcript processing for one encounter, up to mention-level output.
    """
    _ = load_config()  # Ensure env is loaded; config values used by downstream calls

    turns = load_and_segment(transcript_path, encounter_id)
    mentions = extract_mentions(turns, use_llm_refinement=use_llm_for_ner)
    candidates = add_coref_clusters(mentions, turns, use_llm_refinement=use_llm_for_coref)
    # mentions = attach_attributes(mentions)

    if save_intermediate:
        interim_dir = Path("data") / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)
        with open(interim_dir / f"{encounter_id}_turns.json", "w", encoding="utf-8") as f:
            json.dump([t.__dict__ for t in turns], f, ensure_ascii=False, indent=2)
        with open(interim_dir / f"{encounter_id}_mentions.json", "w", encoding="utf-8") as f:
            json.dump([c for c in candidates], f, ensure_ascii=False, indent=2)

    return turns, candidates


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


def save_processed_transcript(
    transcript_path: str,
    encounter_id: str | None = None,
    save_intermediate: bool = False,
    use_llm_for_ner: bool = True,
    use_llm_for_coref: bool = True,
) -> Path:
    """
    Process a transcript and write the consolidated JSON to
    data/interim/<source-stem>.json. Returns the output path.
    """
    resolved_encounter = encounter_id or _default_encounter_id(transcript_path)
    turns, candidates = process_transcript_to_mentions(
        transcript_path=transcript_path,
        encounter_id=resolved_encounter,
        save_intermediate=save_intermediate,
        use_llm_for_ner=use_llm_for_ner,
        use_llm_for_coref=use_llm_for_coref,
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
            c for c in candidates
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path

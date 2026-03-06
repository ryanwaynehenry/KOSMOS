#!/usr/bin/env python3
"""
Prompt generator for LLM-based evaluation of SOAP notes.

Given a candidate SOAP note file, this script:
- infers the corresponding raw ACI JSON (transcript + gold) from the filename
- loads the transcript (src) and gold note (tgt) at the matching data index
- writes a ready-to-send LLM prompt to a .txt file
"""

import argparse
import json
import re
from pathlib import Path
from typing import Tuple


SYSTEM_PROMPT = """SYSTEM You are a clinical documentation evaluator. Your job is to compare a candidate SOAP note against (1) the encounter transcript and (2) a gold SOAP note.
Goal Produce a single structured evaluation that identifies:
1) Facts missed in the candidate that are present in the gold and supported by the transcript.
2) Facts hallucinated in the candidate that are not supported by the transcript.
3) Facts wrong or misrepresented in the candidate compared to the transcript (and optionally compared to gold).
4) Facts correct in the candidate that are supported by the transcript but not present in the gold.
Also:
- Rate each fact for clinical importance (criticality).
- Handle facts that appear in different sections (do not mark as missing if the fact appears elsewhere, but record a section placement issue).
- Anchor truth to the transcript. The transcript is the source of truth for what was actually said or documented during the encounter.

Inputs you will receive
- Transcript text with speaker tags.
- Gold SOAP note text with headings.
- Candidate SOAP note text with headings.

Hard constraints
- Do not infer or add new clinical facts. Only use what is explicitly supported by the transcript.
- When you quote evidence, keep each quote to 20 words or fewer.
- Output strict JSON only. No prose outside JSON.

Definitions
Atomic fact A single claim that can be checked against the transcript, for example:
- symptom present or absent
- condition or history item
- medication use or change, including dose and frequency
- test performed or reviewed and its result
- exam finding
- clinician impression or assessment statement
- order, referral, follow up timing, or instruction
- patient agreement or refusal

Fact fields to normalize when relevant
- polarity: affirmed or negated
- certainty: definite, possible, suspected, unclear
- key modifiers: timing, duration, severity, laterality
- for meds: drug name, route if present, dose, frequency, as needed status, start or continue or stop

Section handling
- Parse each note into sections based on headings.
- A fact counts as present in a note if it appears anywhere in that note.
- If present in both gold and candidate but under different sections, record a section mismatch item.
- Do not count a section mismatch as a miss.

Transcript support status for any fact
- supported: clearly stated in transcript
- contradicted: transcript clearly states the opposite
- notFound: transcript does not contain enough evidence either way

Fact categories you must assign for reporting
Use these rules, with transcript as ground truth:
A) missed
- Fact is present in gold
- Fact transcriptSupport is supported
- Fact is absent in candidate

B) hallucinated
- Fact is present in candidate
- Fact transcriptSupport is notFound or contradicted
- Subtype:
  - unsupported if notFound
  - contradicted if contradicted

C) misrepresented
- Candidate expresses the same general concept as transcript, but one or more key attributes are wrong.
  Common misrepresentation types:
  - polarity error (denies vs endorses)
  - numeric or dose error
  - timeframe error
  - certainty inflation or deflation (possible vs definite)
  - attribution error (patient reported vs clinician observed)
  - scope error (applies a detail too broadly)
For misrepresented facts, include:
- the candidate claim
- what the transcript supports
- the specific mismatch type

D) correctExtra
- Fact is present in candidate
- Fact transcriptSupport is supported
- Fact is absent in gold

E) correct
- Fact is present in candidate
- Fact transcriptSupport is supported
- Fact is also present in gold

Also produce two auxiliary lists
- goldUnsupported: facts present in gold whose transcriptSupport is notFound or contradicted
- sectionMismatches: facts present in both gold and candidate but section differs

Clinical importance labeling
For every fact, assign:
- domain: clinical or contextual
- criticality: high, medium, or low
Guidance
High
- diagnoses and clinician impressions
- new or changed meds or stop or continue directions
- orders (labs, imaging), referrals
- abnormal findings that drive decisions
- follow up timing, return precautions, safety instructions
Medium
- stable chronic problems
- relevant negatives (denies fever) and routine ROS items
- normal results that are explicitly cited as part of reasoning
- routine exam findings
Low
- narrative or logistics not affecting care (who drove, greetings)
- minor storytelling details without clinical impact
Note: contextual details can be medium or high if they change risk or management.

Patient agreement rule
Assume agreement is true unless transcript contains explicit counter evidence (refusal, disagreement, confusion, unwillingness).
- If transcript has explicit agreement or explicit refusal, treat it like any other fact with transcript evidence.
- If transcript is silent, you may still include a patientAgreements summary as assumed, but it should not be counted as hallucination and it should have transcriptSupport status notFound with a special flag assumedTrue.

Matching and deduplication
- Extract atomic facts independently from gold and candidate.
- Normalize and merge duplicates so the same fact is evaluated once.
- Use best effort matching using synonyms and paraphrases.
- For matched facts, prefer the more specific version (includes dose, timing, polarity) as the representative fact.

Output JSON schema
Return exactly this JSON object:
{
  "evaluation": {
    "encounter": {
      "dataset": string or null,
      "encounterId": string or null,
      "patientName": string or null,
      "patientAge": string or null,
      "patientGender": string or null,
      "chiefComplaint": string or null
    },
    "summary": {
      "counts": {
        "missed": number,
        "hallucinated": number,
        "misrepresented": number,
        "correctExtra": number,
        "correct": number,
        "goldUnsupported": number,
        "sectionMismatches": number
      },
      "countsByCriticality": {
        "high": { "missed": number, "hallucinated": number, "misrepresented": number, "correctExtra": number, "correct": number },
        "medium": { ... },
        "low": { ... }
      },
      "scores": {
        "goldSupportedRecallHigh": number,
        "candidateFaithfulnessPrecisionHigh": number,
        "hallucinationRateAll": number,
        "misrepresentationRateAll": number
      }
    },
    "facts": [
      {
        "factId": "F001",
        "fact": string,
        "domain": "clinical" or "contextual",
        "criticality": "high" or "medium" or "low",
        "category": "missed" or "hallucinated" or "misrepresented" or "correctExtra" or "correct" or "goldUnsupported",
        "hallucinationSubtype": "unsupported" or "contradicted" or null,
        "misrepresentationTypes": [string] or [],
        "assumedTrue": true or false,
        "transcriptSupport": { "status": "supported" or "contradicted" or "notFound", "speaker": "doctor" or "patient" or "either" or "unknown", "quotes": [string] },
        "goldPresence": { "present": true or false, "section": string or null, "quotes": [string] },
        "candidatePresence": { "present": true or false, "section": string or null, "quotes": [string] },
        "notes": string or null
      }
    ],
    "sectionMismatches": [
      { "factId": string, "fact": string, "goldSection": string, "candidateSection": string }
    ]
  }
}

Scoring formulas
- goldSupportedRecallHigh:
  numerator: count of facts with criticality high where category is correct
  denominator: count of facts with criticality high where goldPresence present is true and transcriptSupport status is supported
- candidateFaithfulnessPrecisionHigh:
  numerator: count of facts with criticality high where candidatePresence present is true and transcriptSupport status is supported
  denominator: count of facts with criticality high where candidatePresence present is true
- hallucinationRateAll:
  numerator: count of facts where category is hallucinated
  denominator: count of facts where candidatePresence present is true
- misrepresentationRateAll:
  numerator: count of facts where category is misrepresented
  denominator: count of facts where candidatePresence present is true and transcriptSupport status is supported or contradicted

ID formatting note
When you include ids, you must preserve any ids exactly as provided in the input. Typical formats:
- node ids look like "clef_taskC_test3_full_1_n00052"
- relationship ids look like "pn0001_n0034"
If the inputs do not provide node ids or relationship ids, omit them entirely and rely on quotes. Do not invent ids.

Now perform the evaluation in a single pass using the user provided transcript, gold note, candidate note, and metadata.
Output strict JSON only."""


def derive_raw_paths(hyp: Path) -> Tuple[Path, int]:
    """
    From a hypothesis filename like soap_<stem>_<idx>.txt, derive the raw ACI JSON path and index.
    Includes a fallback that strips a trailing hyphenated run suffix from the stem if the first path is missing.
    """
    name = hyp.stem  # e.g., soap_clef_taskC_test3_full_1 or soap_clef_taskC_test3_full_1-5_2
    m = re.match(r"soap_(.+)_([0-9]+)$", name)
    if not m:
        raise SystemExit("Cannot derive reference: hyp filename must look like soap_<stem>_<idx>.txt")
    stem = m.group(1)
    idx = int(m.group(2))

    def candidate_paths(s: str) -> list[Path]:
        base = Path("data") / "raw" / "aci"
        paths = [base / f"{s}.json"]
        if "-" in s:
            parts = s.rsplit("-", 1)
            if parts[1].isdigit():
                paths.append(base / f"{parts[0]}.json")
        return paths

    for path in candidate_paths(stem):
        if path.exists():
            return path, idx
    raise SystemExit(f"Cannot find derived reference JSON for stem '{stem}' under data/raw/aci")


def load_transcript_and_gold(raw_path: Path, idx: int) -> tuple[str, str, dict]:
    raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
    data_list = raw_data.get("data") if isinstance(raw_data, dict) else raw_data
    if not isinstance(data_list, list):
        raise SystemExit(f"Reference JSON at {raw_path} has unexpected structure; expected list under 'data'.")
    try:
        entry = data_list[idx]
    except Exception as exc:
        raise SystemExit(f"Failed to load data[{idx}] from {raw_path}: {exc}") from exc
    transcript = entry.get("src") or ""
    gold = entry.get("tgt") or ""
    return transcript, gold


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an LLM evaluation prompt for a candidate SOAP note.")
    ap.add_argument("--hyp", required=True, type=Path, help="Path to candidate SOAP text file.")
    ap.add_argument("--out", type=Path, help="Output prompt txt path. Defaults to <hyp>_llm_eval_prompt.txt")
    args = ap.parse_args()

    raw_path, idx = derive_raw_paths(args.hyp)
    transcript, gold = load_transcript_and_gold(raw_path, idx)
    candidate = args.hyp.read_text(encoding="utf-8")

    prompt_text = (
        f"{SYSTEM_PROMPT}\n\n"
        "TRANSCRIPT\n"
        f"{transcript}\n\n"
        "GOLD\n"
        f"{gold}\n\n"
        "CANDIDATE\n"
        f"{candidate}\n"
    )

    if args.out:
        out_path = args.out
    else:
        stem = args.hyp.stem  # drop original extension
        out_dir = Path("data") / "evaluation_prompts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_llm_eval_prompt.txt"
    out_path.write_text(prompt_text, encoding="utf-8")
    print(f"Wrote prompt to {out_path}")

    # Also create an empty evaluation JSON stub in data/evaluation/<stem>.json if not already present.
    eval_dir = Path("data") / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"evaluation_{stem}.json"
    if not eval_path.exists():
        eval_path.write_text("{}", encoding="utf-8")
        print(f"Created empty evaluation stub at {eval_path}")
    else:
        print(f"Evaluation stub already exists at {eval_path}")


if __name__ == "__main__":
    main()

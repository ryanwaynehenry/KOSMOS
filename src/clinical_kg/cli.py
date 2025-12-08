import argparse

from clinical_kg.pipeline import save_processed_transcript


def main():
    parser = argparse.ArgumentParser(
        description="Process a raw transcript into turns and mentions (no UMLS/KG)."
    )
    parser.add_argument("transcript", help="Path to transcript file")
    parser.add_argument(
        "--encounter-id",
        help="Override encounter identifier. Defaults to transcript filename stem.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also write turns/mentions to data/interim/{transcript_name}*.json",
    )
    args = parser.parse_args()

    output_path = save_processed_transcript(
        transcript_path=args.transcript,
        encounter_id=args.encounter_id,
        save_intermediate=args.save_intermediate,
    )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

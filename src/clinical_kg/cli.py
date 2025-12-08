import argparse
from clinical_kg.pipeline import run_pipeline_for_transcript

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", help="Path to transcript file")
    parser.add_argument("--encounter-id", required=True)
    parser.add_argument("--out", required=True, help="Path to Turtle file")
    args = parser.parse_args()

    run_pipeline_for_transcript(
        transcript_path=args.transcript,
        encounter_id=args.encounter_id,
        output_path=args.out,
    )

if __name__ == "__main__":
    main()

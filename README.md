# KOSMOS

KOSMOS is a clinical knowledge graph and SOAP note generation pipeline.

Given a raw encounter transcript, KOSMOS:
1. Segments transcript turns and rewrites pronouns for clearer context.
2. Extracts mentions/entities.
3. Builds a knowledge graph (`nodes` + `relationship_candidates`).
4. Generates a SOAP note using the DocLens exporter.

## What You Get

For each processed encounter, KOSMOS writes:
- `data/interim/<stem>.json`
  - Includes `turns`, `mentions`, `nodes`, `relationship_candidates`, and `soap_note`.
- `data/processed/soap_<stem>.txt`
  - Final generated SOAP note text.
- `data/processed/soap_<stem>.json`
  - DocLens output payload (`note_text`, `prompt_input`, exporter metadata).

## Project Layout

- `src/clinical_kg/cli.py`: main CLI entry point.
- `src/clinical_kg/pipeline.py`: end-to-end processing pipeline.
- `src/clinical_kg/kg/export_doclens.py`: DocLens SOAP note generator.
- `scripts/`: batch/evaluation/utilities.
- `data/`: raw, interim, processed, and analysis artifacts.

## Requirements

- Python 3.10+
- An OpenAI-compatible API key (`OPENAI_API_KEY` or `LLM_API_KEY`)
- UMLS MySQL database (required)
- Runtime packages used by the codebase (at minimum):
  - `python-dotenv`
  - `mysql-connector-python`
  - `litellm`
  - `spacy`
  - `faiss-cpu` (or `faiss-gpu`)
  - `torch`
  - `transformers`

## Setup

### Option A (Recommended): Conda Environment

Create and activate the same environment style used in this repo:

```bash
conda create -n kosmos python=3.10 -y
conda activate kosmos
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Option B: Python venv

```bash
python -m venv .venv
# PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Environment Variables

Create `.env` in the repo root.

Example `.env`:

```env
# LLM
OPENAI_API_KEY=your_key_here
LLM_MODEL_NAME=gpt-5.2
LLM_TEMPERATURE=0.0
MAX_LLM_TOKENS=4096

# UMLS DB (required)
UMLS_DB_HOST=localhost
UMLS_DB_PORT=3306
UMLS_DB_USER=umls_user
UMLS_DB_PASSWORD=...
UMLS_DB_NAME=umls2025

# FAISS files (required for configured fallback search)
UMLS_FAISS_INDEX_PATH=UMLS_sapbert.faiss
UMLS_FAISS_MAPPING_PATH=UMLS_sapbert_mapping.json
```

## UMLS Database Setup (Required)

You must have a licensed UMLS release and load the `MRCONSO` table into MySQL.

General setup flow:
1. Request a UMLS license and follow the official download/install instructions from NLM:
   https://www.nlm.nih.gov/research/umls/index.html
2. Create a MySQL database (example name used by this repo: `umls2025`).
3. Load UMLS tables from the release into MySQL (including `MRCONSO`).
4. Set `.env` values (`UMLS_DB_*`) to point KOSMOS at that database.
5. Verify the DB contains data:

```sql
SELECT COUNT(*) FROM MRCONSO;
SELECT COUNT(*) FROM MRCONSO WHERE SAB IN ('SNOMEDCT_US', 'RXNORM', 'LNC');
```

If either query returns `0`, ontology alignment and FAISS build will not work correctly.

## Build FAISS Index (Required)

KOSMOS uses SapBERT embeddings + FAISS for ontology fallback retrieval.

Build the default combined index/mapping:

```bash
python scripts/build_umls_faiss_index.py \
  --index-path UMLS_sapbert.faiss \
  --mapping-path UMLS_sapbert_mapping.json \
  --sabs SNOMEDCT_US,RXNORM,LNC \
  --batch-size 256 \
  --shard-size 2000000
```

Quick smoke test (small sample):

```bash
python scripts/build_umls_faiss_index.py --limit 50000
```

If you intentionally keep shards only:

```bash
python scripts/build_umls_faiss_index.py --no-merge
```

You can also merge shard files manually with:

```bash
python scripts/merge_faiss_shards.py \
  --index_pattern "UMLS_sapbert_part*.faiss" \
  --mapping_pattern "UMLS_sapbert_mapping_part*.json" \
  --output_index UMLS_sapbert.faiss \
  --output_mapping UMLS_sapbert_mapping.json \
  --max_shards 9999
```

## Quick Start

Run from repo root.

### 1) Process a plain transcript text file

```bash
python -m clinical_kg.cli path/to/transcript.txt
```

Optional:

```bash
python -m clinical_kg.cli path/to/transcript.txt --encounter-id encounter_001 --save-intermediate
```

### 2) Process a JSON dataset (`data` array with `src`)

Single index:

```bash
python -m clinical_kg.cli data/raw/aci/clef_taskC_test4_full.json --json-index 1
```

Range:

```bash
python -m clinical_kg.cli data/raw/aci/clef_taskC_test4_full.json --json-index 0-2
```

Open range:

```bash
python -m clinical_kg.cli data/raw/aci/clef_taskC_test4_full.json --json-index 3+
```

## DocLens Exporter (Direct Usage)

If you already have an interim JSON and want only note generation:

```bash
python -m clinical_kg.kg.export_doclens data/interim/clef_taskC_test4_full_1.json
```

To force transcript source from a separate file/json:

```bash
python -m clinical_kg.kg.export_doclens data/interim/clef_taskC_test4_full_1.json --transcript-path data/raw/aci/clef_taskC_test4_full.json
```

## Useful Scripts

- `scripts/run_export_doclens_batch.py`: batch SOAP export from many interim JSON files.
- `scripts/make_eval_csv.py`: build eval CSV from generated SOAP notes. (For ACI-BENCH)
- `scripts/convert_eval_csv_to_json.py`: convert eval CSV to JSON format. (For DocLens)

## Notes

- Current pipeline path uses `export_doclens.py` for SOAP generation.
- If model/provider settings change, update `.env` and rerun.
- UMLS DB and FAISS artifacts are required for the expected ontology behavior in KOSMOS.

## Evaluation Repository Links

If you want to reproduce or compare against the external evaluation setups used in this project, use the original benchmark repositories below:

- DocLens: https://github.com/yiqingxyq/DocLens
- ACI-Bench: https://github.com/wyim/aci-bench

## UMLS Citation

UMLS Knowledge Sources [dataset on the Internet]. Release 2024AA. Bethesda (MD): National Library of Medicine (US); 2024 May 6 [cited 2024 Jul 15]. Available from: http://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

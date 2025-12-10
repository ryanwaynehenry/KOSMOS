from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present.
load_dotenv()


@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class PipelineConfig:
    db: DBConfig
    ontology_preferences: dict
    score_threshold: float
    base_ner_model_name: str
    llm_model_name: str
    llm_temperature: float
    max_llm_tokens: int | None = None


def load_config() -> PipelineConfig:
    return PipelineConfig(
        db=DBConfig(
            host=os.getenv("UMLS_DB_HOST", "localhost"),
            port=int(os.getenv("UMLS_DB_PORT", "3306")),
            user=os.getenv("UMLS_DB_USER", "umls_user"),
            password=os.getenv("UMLS_DB_PASSWORD", ""),
            database=os.getenv("UMLS_DB_NAME", "umls2025"),
        ),
        ontology_preferences={
            "PROBLEM": ["SNOMEDCT"],
            "MEDICATION": ["RXNORM"],
            "LAB_TEST": ["LOINC"],
            "UNIT": ["UCUM"],
        },
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.7")),
        base_ner_model_name=os.getenv("BASE_NER_MODEL_NAME", "en_core_sci_lg"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_llm_tokens=int(os.getenv("MAX_LLM_TOKENS")) if os.getenv("MAX_LLM_TOKENS") else None,
    )

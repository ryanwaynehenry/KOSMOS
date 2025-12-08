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

def load_config() -> PipelineConfig:
    return PipelineConfig(
        db=DBConfig(
            host=os.getenv("UMLS_DB_HOST", "localhost"),
            port=int(os.getenv("UMLS_DB_PORT", "3306")),
            user=os.getenv("UMLS_DB_USER", "umls_user"),
            password=os.getenv("UMLS_DB_PASSWORD", ""),
            database=os.getenv("UMLS_DB_NAME", "umls2025")
        ),
        ontology_preferences={
            "PROBLEM": ["SNOMEDCT"],
            "MEDICATION": ["RXNORM"],
            "LAB_TEST": ["LOINC"],
            "UNIT": ["UCUM"],
        },
        score_threshold=0.7,
    )

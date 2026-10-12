from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env into os.environ too: libraries such as LangChain and LangSmith read their keys
# (ANTHROPIC_API_KEY, LANGSMITH_*) from the process environment, not from Settings.
load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    triage_model: str = "claude-haiku-5-5"
    investigator_model: str = "claude-opus-5-5"
    reporter_model: str = "claude-opus-5-5"
    judge_model: str = "claude-haiku-5-5"

    kafka_bootstrap: str = "localhost:19092"
    topic_transactions: str = "transactions"
    topic_flagged: str = "flagged-transactions"

    pg_dsn: str = "postgresql://fraud:fraud@localhost:5433/fraud"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "fraudgraph"
    redis_url: str = "redis://localhost:6379/0"
    api_url: str = "http://localhost:8000"

    # Detector
    detector_model_path: str = "models/isolation_forest.joblib"
    iforest_threshold: float = -0.05   # decision_function below this => anomalous

    # Agent guardrails
    max_investigator_steps: int = 12
    semantic_cache_threshold: float = 0.92

    @property
    def sqlalchemy_dsn(self) -> str:
        # langchain-postgres uses SQLAlchemy + psycopg3
        return self.pg_dsn.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()

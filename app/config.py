"""Central configuration.

Every threshold and weight used by the pipeline lives here so it can be tuned
through environment variables (or `.env`) without code changes. All numeric
thresholds are PROVISIONAL defaults; the experiments in `experiments/` are
what justify (or change) them.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App -------------------------------------------------------------
    APP_NAME: str = "Telecom Support Resolution Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # --- Database --------------------------------------------------------
    DATABASE_URL: str = "postgresql+asyncpg://support:support@localhost:5432/support"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # --- LLM (any OpenAI-compatible endpoint: Groq, Gemini, OpenAI, vLLM) --
    # LLM_PROVIDER: "openai_compatible" | "extractive" (deterministic, no network)
    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "qwen/qwen3.8-27b"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1200
    LLM_TIMEOUT_S: float = 30.0
    LLM_MAX_RETRIES: int = 3
    # Fall back to the extractive generator when the LLM is unavailable.
    LLM_FALLBACK_TO_EXTRACTIVE: bool = True
    # Force IPv4 for LLM calls (workaround for networks with a broken IPv6 route).
    LLM_FORCE_IPV4: bool = False
    # Reasoning models only (e.g. "low" for openai/gpt-oss-*); None = don't send the parameter.
    LLM_REASONING_EFFORT: str | None = None

    # --- Models ----------------------------------------------------------
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # Lighter than roberta-large-mnli (see ARCHITECTURE.md, "NLI model choice").
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Understanding ---------------------------------------------------
    # "knn" (embedding k-NN over labelled history) | "llm"
    INTENT_CLASSIFIER: str = "knn"
    INTENT_KNN_K: int = 15
    # Below either of these the intent is reported as unknown_intent.
    # Operating point chosen on the dev split (evaluation/understanding_eval.py sweep):
    # low false-unknown rate; novel classes are mainly caught downstream + by drift monitoring.
    INTENT_MIN_CONFIDENCE: float = 0.35
    INTENT_MIN_SIMILARITY: float = 0.40
    PRODUCT_NEIGHBOUR_MIN_SHARE: float = 0.6
    PRODUCT_LLM_FALLBACK: bool = False
    # Polarity (p_pos - p_neg) thresholds, calibrated on the dev split. None => argmax label.
    SENTIMENT_NEGATIVE_THRESHOLD: float | None = -0.85
    SENTIMENT_POSITIVE_THRESHOLD: float | None = 0.0
    # Plan §16.5 bumps medium->high when sentiment < -0.7. Disabled (-1.0): on the dev split it cut severity
    # accuracy from 0.749 to 0.558, because severity is about impact, not tone. Set e.g. -0.7 to re-enable.
    SEVERITY_NEGATIVE_SENTIMENT_BUMP: float = -1.0

    # --- Retrieval -------------------------------------------------------
    RETRIEVAL_CANDIDATES_TICKETS: int = 50
    RETRIEVAL_CANDIDATES_KB: int = 20
    RETRIEVAL_TOP_K: int = 8
    RETRIEVAL_KB_MIN_SLOTS: int = 2
    RRF_K: int = 60
    # Selected on the dev split in Experiment 1 (grid 0.3-0.9; plateau 0.6-0.9, best 0.8).
    RRF_SEMANTIC_WEIGHT: float = 0.8
    RRF_LEXICAL_WEIGHT: float = 0.2
    # "or" builds an OR-of-terms tsquery; "and" uses plainto_tsquery (all terms).
    LEXICAL_QUERY_MODE: str = "or"
    IVFFLAT_PROBES: int = 10
    USE_RERANKING: bool = False  # decided by Experiment 2
    RERANK_CANDIDATES: int = 30

    # --- Generation ------------------------------------------------------
    MAX_CONTEXT_CHARS: int = 8000

    # --- Validation ------------------------------------------------------
    # Tuned on the dev half of the groundedness set in Experiment 3 (v2: sentence premises,
    # top-similarity contradiction, quote rule; experiments/results/e3_groundedness.json).
    GROUNDED_SIM_THRESHOLD: float = 0.60
    GROUNDED_SIM_WEAK_THRESHOLD: float = 0.45
    GROUNDED_ENTAIL_THRESHOLD: float = 0.30
    GROUNDED_ENTAIL_WEAK_THRESHOLD: float = 0.15
    GROUNDED_LEXICAL_THRESHOLD: float = 0.50
    GROUNDED_LEXICAL_WEAK_THRESHOLD: float = 0.30
    GROUNDED_CONTRADICTION_THRESHOLD: float = 0.50
    WEAK_SUPPORT_CREDIT: float = 0.5
    # NLI premise handling (E3 v2): sentence premises, contradiction read from the most similar premise,
    # verbatim quotes short-circuit to supported.
    NLI_PREMISE_UNIT: str = "sentence"
    NLI_CONTRADICTION_AGG: str = "top_similarity"
    NLI_TOP_PREMISES: int = 3
    GROUNDED_USE_QUOTE_MATCH: bool = True

    # --- Evidence sufficiency / decision --------------------------------
    EVIDENCE_MIN_TOP_RELEVANCE: float = 0.35
    EVIDENCE_MIN_AVG_RELEVANCE: float = 0.40
    EVIDENCE_MIN_GROUNDEDNESS: float = 0.60
    CONF_W_RETRIEVAL: float = 0.30
    CONF_W_EVIDENCE: float = 0.40
    CONF_W_UNDERSTANDING: float = 0.20
    CONF_W_SUFFICIENCY: float = 0.10
    CONF_RETRIEVAL_NORMALISER: float = 0.80
    CONF_INSUFFICIENT_CREDIT: float = 0.3
    DECISION_RESOLVE_THRESHOLD: float = 0.75
    DECISION_REVIEW_THRESHOLD: float = 0.55
    CRITICAL_SEVERITY_ALWAYS_REVIEW: bool = True

    # --- Monitoring ------------------------------------------------------
    DRIFT_WINDOW_HOURS: int = 24
    DRIFT_UNKNOWN_RATE_ALERT: float = 0.15
    DRIFT_LOW_SIMILARITY_ALERT: float = 0.45
    # Intent-mix shift alert: TVD > coef / sqrt(min window size) ~ 99th percentile of sampling noise
    # (measured: p99 = 0.467 / 0.317 / 0.217 / 0.138 at n = 30 / 60 / 120 / 240). Recalibrate on real traffic.
    DRIFT_TVD_ALERT_COEF: float = 2.5
    DRIFT_MIN_WINDOW: int = 30

    # --- Paths -----------------------------------------------------------
    TAXONOMY_DIR: Path = PROJECT_ROOT / "data" / "taxonomies"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

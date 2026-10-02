"""Service container: models are loaded once at startup and shared across requests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.core.database import get_session_factory
from app.core.llm import LLMProvider, build_llm_provider
from app.models.embeddings import EmbeddingService, get_embedding_service
from app.services.decision import DecisionService
from app.services.generation import ExtractiveGenerator, GenerationService, LLMGenerator
from app.services.ingestion import IngestionService
from app.services.pipeline import ResolutionPipeline
from app.services.retrieval import HybridRetriever, Reranker, RetrievalService
from app.services.understanding import (
    PgNeighbourIndex,
    SentimentAnalyzer,
    UnderstandingService,
    load_taxonomy_db,
    load_taxonomy_files,
)
from app.services.validation import GroundednessChecker, NLIModel, ValidationService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    embedder: EmbeddingService
    llm: LLMProvider | None
    understanding: UnderstandingService
    pipeline: ResolutionPipeline
    ingestion: IngestionService | None
    session_factory: Any  # async_sessionmaker in production; None in API tests

    async def refresh_taxonomy(self) -> None:
        async with self.session_factory() as session:
            intents, products = await load_taxonomy_db(session)
        if intents:
            self.understanding.set_taxonomy(intents, products or self.understanding.products)


_container: Container | None = None


async def build_container() -> Container:
    session_factory = get_session_factory()
    embedder = get_embedding_service()
    llm = build_llm_provider()
    intents, products = load_taxonomy_files()
    understanding = UnderstandingService(
        embedder,
        PgNeighbourIndex(session_factory),
        intents,
        products,
        SentimentAnalyzer(),
        llm,
    )
    retriever = HybridRetriever(session_factory, embedder)
    reranker = Reranker() if settings.USE_RERANKING else None
    llm_generator = LLMGenerator(llm) if llm is not None else None
    pipeline = ResolutionPipeline(
        embedder,
        understanding,
        RetrievalService(retriever, reranker),
        GenerationService(llm_generator, ExtractiveGenerator()),
        ValidationService(GroundednessChecker(embedder, NLIModel())),
        DecisionService(),
        session_factory,
    )
    container = Container(
        embedder, llm, understanding, pipeline, IngestionService(session_factory, embedder), session_factory
    )
    try:
        await container.refresh_taxonomy()
    except Exception as exc:  # DB may not be initialised yet; file taxonomy is used meanwhile
        logger.warning("taxonomy not loaded from DB (%s); using files", exc)
    return container


async def init_container() -> Container:
    global _container
    _container = await build_container()
    return _container


def get_container() -> Container:
    if _container is None:
        raise RuntimeError("Service container not initialised")
    return _container


def set_container(container: Container | None) -> None:
    """Test hook."""
    global _container
    _container = container

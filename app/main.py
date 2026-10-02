"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.dependencies import init_container
from app.api.v1.routes import router
from app.config import settings
from app.core.database import dispose_engine
from app.core.logging import configure_logging, request_id_var

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    await init_container()
    logger.info("startup complete", extra={"extra_fields": {"startup_s": round(time.perf_counter() - started, 1)}})
    yield
    await dispose_engine()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Semantic resolution assistant for telecom support agents: understand a complaint, retrieve similar "
    "resolved tickets and KB articles (hybrid search), draft a cited resolution, validate it, and decide "
    "RESOLVE / REVIEW / ESCALATE from evidence.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error")
        response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
    finally:
        request_id_var.reset(token)
    response.headers["x-request-id"] = rid
    response.headers["x-process-time-ms"] = str(int((time.perf_counter() - started) * 1000))
    return response


app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": settings.APP_NAME, "docs": "/docs", "health": "/api/v1/health"}

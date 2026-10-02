FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel.
RUN pip install --timeout 120 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --timeout 120 --retries 5 -r requirements.txt

# Bake models into the image so containers start without network access.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from transformers import pipeline; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/nli-deberta-v3-small'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"

COPY . .
RUN chmod +x scripts/docker_entrypoint.sh && useradd --create-home appuser && chown -R appuser /app /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["scripts/docker_entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

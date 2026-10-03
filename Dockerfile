FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel.
RUN pip install --timeout 120 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
# Whole-step retries survive short DNS/network outages that exhaust pip's own quick retries.
RUN for i in 1 2 3 4 5; do \
      pip install --timeout 120 --retries 10 -r requirements.txt && exit 0; \
      echo "pip attempt $i failed; retrying in 20s"; sleep 20; \
    done; exit 1

# Bake models into the image so containers start without network access.
RUN for i in 1 2 3 4 5; do \
      python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from transformers import pipeline; \
SentenceTransformer('sentence-transformers/all-mpnet-base-v2'); \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/nli-deberta-v3-small'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')" && exit 0; \
      echo "model download attempt $i failed; retrying in 20s"; sleep 20; \
    done; exit 1

COPY . .
RUN chmod +x scripts/docker_entrypoint.sh && useradd --create-home appuser && chown -R appuser /app /models
USER appuser

EXPOSE 8000
# No curl/apt layer: the health check uses Python's stdlib (fewer network dependencies at build time).
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=5)" || exit 1

ENTRYPOINT ["scripts/docker_entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

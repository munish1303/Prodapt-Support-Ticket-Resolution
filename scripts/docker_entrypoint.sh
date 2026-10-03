#!/bin/sh
# Container entrypoint: apply schema (idempotent), ingest the corpus on first boot, then start the API.
set -e

echo "applying migrations..."
python scripts/init_db.py

# Vectors must come from the configured embedding model: resize/re-embed if the model changed (no-op otherwise).
python scripts/reembed.py

COUNT=$(python - <<'EOF'
import asyncio
from sqlalchemy import text
from app.core.database import get_engine, dispose_engine
async def main():
    async with get_engine().connect() as c:
        n = (await c.execute(text("SELECT count(*) FROM tickets"))).scalar_one()
    await dispose_engine()
    print(n)
asyncio.run(main())
EOF
)

if [ "$COUNT" = "0" ] && [ "${AUTO_INGEST:-true}" = "true" ]; then
    echo "empty corpus: generating data (if missing) and ingesting..."
    [ -f data/processed/tickets.jsonl ] || python scripts/generate_synthetic.py
    python scripts/ingest.py
else
    echo "corpus already has $COUNT tickets; skipping ingestion"
fi

exec "$@"

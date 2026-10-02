"""Telecom Support Ticket Resolution Assistant."""

import os
from pathlib import Path

# Keep model downloads inside the project (overridable via env, e.g. in Docker).
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

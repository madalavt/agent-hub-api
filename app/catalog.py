"""Standard ML model catalog, added on startup if missing.

Ids are derived from the model name (uuid5), so the catalog is idempotent across restarts and
environments, and seeds/imports can point at "GPT-4.1" without knowing a random id. Existing
rows are never overwritten, so edits made in the UI stick.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.models import MlModel

_NS = uuid.UUID("6f1c6c1e-6a52-4a7e-9a4c-2d3c6f0b8a11")

DEFAULT_GPT_MODEL = "GPT-4.1"


def catalog_id(name: str) -> str:
    return str(uuid.uuid5(_NS, name))


GPT_MODELS: list[dict[str, Any]] = [
    {
        "name": "GPT-4.1",
        "type": "gpt",
        "parameters": {
            "model_type": "azure_openai",
            "model_name": "gpt-4.1",
            "max_completion_tokens": 32768,
            "max_context_tokens": 1_000_000,
        },
    },
    {
        "name": "GPT-4.1 mini",
        "type": "gpt",
        "parameters": {
            "model_type": "azure_openai",
            "model_name": "gpt-4.1-mini",
            "max_completion_tokens": 32768,
            "max_context_tokens": 1_000_000,
            "fallback_mlmodel_id": catalog_id("GPT-4.1"),
        },
    },
    {
        "name": "GPT-5 mini (reasoning)",
        "type": "azure_openai_reasoning",
        "parameters": {
            "model_type": "azure_openai",
            "model_name": "gpt-5-mini",
            "reasoning_effort": "low",
            "max_completion_tokens": 32768,
            "max_context_tokens": 400_000,
            "fallback_mlmodel_id": catalog_id("GPT-4.1"),
        },
    },
]


def ollama_model(tag: str) -> dict[str, Any]:
    return {
        "name": f"Ollama · {tag}",
        "type": "ollama",
        "parameters": {
            "model_type": "ollama",
            "model_name": tag,
            "max_completion_tokens": 8192,
            "local_only": True,
        },
    }


def ensure_catalog(session: Session, local_models: list[str] | None) -> list[str]:
    """Insert missing catalog models. Returns the names that were added."""
    added = []
    for spec in GPT_MODELS + [ollama_model(t) for t in (local_models or [])]:
        mid = catalog_id(spec["name"])
        if session.get(MlModel, mid) is None:
            session.add(MlModel(id=mid, standard=True, **spec))
            added.append(spec["name"])
    session.flush()
    return added

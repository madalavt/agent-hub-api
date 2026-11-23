"""ML model parameter rules shared by the REST and GraphQL APIs."""

from __future__ import annotations

from typing import Any

from app import importer


class ModelParamsError(ValueError):
    pass


def validate_model_params(params: dict[str, Any] | None, supported: list[str]) -> dict[str, Any]:
    """Return cleaned parameters or raise ModelParamsError (never store secrets on models)."""
    cleaned, dropped = importer._clean_params(params)
    if dropped:
        raise ModelParamsError(
            f"do not store secrets on models ({dropped}); put them in config/credentials.yaml"
        )
    if cleaned.get("model_type") not in supported:
        raise ModelParamsError(f"parameters.model_type must be one of {supported}")
    if not cleaned.get("model_name"):
        raise ModelParamsError("parameters.model_name is required")
    return cleaned

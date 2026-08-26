"""Primitive validation helpers shared by annotation modules."""

from __future__ import annotations

from typing import Any


def require_text(record: dict[str, Any], key: str, context: str) -> str:
    """Return one required non-empty string field."""
    return require_non_empty_text(record.get(key), f"{context} field {key!r}")


def require_non_empty_text(value: object, context: str) -> str:
    """Return a stripped non-empty string or reject it."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{context} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def require_int(value: object, context: str) -> int:
    """Return an integer that is not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{context} must be an integer."
        raise ValueError(msg)
    return value


def require_positive_int(value: object, context: str) -> int:
    """Return a strictly positive integer."""
    parsed = require_int(value, context)
    if parsed <= 0:
        msg = f"{context} must be positive."
        raise ValueError(msg)
    return parsed

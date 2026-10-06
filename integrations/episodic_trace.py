"""Schéma de trace épisodique. Métadonnée du nœud, pas un encodeur.

Les champs structurés ne sont pas concaténés à la légende : le rappel
lexical et le cosine restent sur le texte vécu.
"""

from __future__ import annotations

from typing import Any

SCHEMA_KEYS = (
    "trace_id",
    "entity_id",
    "timestamp",
    "place",
    "participants",
    "context",
    "event",
    "intent",
    "outcome",
    "importance",
    "links",
)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def build_episode_meta(
    episode_id: str,
    text: str,
    created_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble le nœud : légende inchangée, trace structurée à part."""
    src = dict(extra or {})
    nested = src.pop("trace", None)
    if isinstance(nested, dict):
        for key, value in nested.items():
            src.setdefault(key, value)
    picked = {key: src.pop(key) for key in SCHEMA_KEYS if key in src}
    timestamp = picked.get("timestamp")
    if timestamp is None:
        timestamp = created_at
    event = picked["event"] if "event" in picked else text
    trace = {
        "trace_id": str(picked.get("trace_id") or episode_id),
        "entity_id": picked.get("entity_id"),
        "timestamp": timestamp,
        "place": picked.get("place"),
        "participants": _as_list(picked.get("participants")),
        "context": picked.get("context"),
        "event": event,
        "intent": picked.get("intent"),
        "outcome": picked.get("outcome"),
        "importance": picked.get("importance"),
        "links": _as_list(picked.get("links")),
    }
    meta = {"text": text, "created_at": created_at, "trace": trace}
    meta.update(src)
    return meta

"""Shared embedding helpers for semantic search and vector indexing."""

from __future__ import annotations

from typing import Any

EMBEDDING_MODEL = "text-embedding-ada-002"


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_interventions(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    normalized.append(name)
            else:
                text = str(item).strip()
                if text:
                    normalized.append(text)
        return normalized
    return []


def build_embedding_text(trial: dict, *, max_summary_chars: int = 1000) -> str:
    """Create a compact text representation of a trial for embeddings."""
    protocol = trial.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    conditions_module = protocol.get("conditionsModule") or {}
    interventions_module = protocol.get("armsInterventionsModule") or {}
    description_module = protocol.get("descriptionModule") or {}
    status_module = protocol.get("statusModule") or {}

    nct_id = trial.get("nct_id") or identification.get("nctId") or ""
    title = (
        trial.get("title")
        or identification.get("briefTitle")
        or identification.get("officialTitle")
        or ""
    )
    conditions = _normalize_string_list(
        trial.get("conditions") or conditions_module.get("conditions") or []
    )
    interventions = _normalize_interventions(
        trial.get("interventions") or interventions_module.get("interventions") or []
    )
    summary = (
        trial.get("summary")
        or trial.get("description")
        or description_module.get("briefSummary")
        or description_module.get("detailedDescription")
        or ""
    )
    status = trial.get("status") or status_module.get("overallStatus") or ""

    parts: list[str] = []
    if nct_id:
        parts.append(f"NCT ID: {nct_id}")
    if title:
        parts.append(f"Title: {title}")
    if conditions:
        parts.append(f"Conditions: {', '.join(conditions)}")
    if interventions:
        parts.append(f"Interventions: {', '.join(interventions)}")
    if status:
        parts.append(f"Status: {status}")
    if summary:
        parts.append(f"Summary: {summary[:max_summary_chars]}")

    return "\n".join(parts)

"""Dataset helpers for chat and semantic result ordering."""

from __future__ import annotations

from services.trial_documents import extract_essential_fields, get_trial_nct_id, serialize_document
from services.trial_filters import build_query_from_filters

MAX_CHAT_DATASET_STUDIES = 100


def fetch_chat_dataset(collection, filters: dict, *, advanced_mode: bool, limit: int = MAX_CHAT_DATASET_STUDIES) -> dict:
    """Fetch studies for cross-study chat and trim them for the selected mode."""
    query = build_query_from_filters(filters)
    total = collection.count_documents(query)
    provided = min(total, limit)
    studies = list(collection.find(query).limit(provided))

    if advanced_mode:
        processed = [serialize_document(study) for study in studies]
        mode_label = "complete"
    else:
        processed = [extract_essential_fields(study) for study in studies]
        mode_label = "essential"

    return {
        "query": query,
        "total": total,
        "provided": len(processed),
        "studies": processed,
        "mode_label": mode_label,
        "was_truncated": total > len(processed),
    }


def order_trials_by_rank(trials: list[dict], ranked_nct_ids: list[str]) -> list[dict]:
    """Preserve vector-search ranking after fetching documents from Mongo."""
    if not ranked_nct_ids:
        return trials

    indexed_trials = {get_trial_nct_id(trial): trial for trial in trials}
    ordered = [indexed_trials[nct_id] for nct_id in ranked_nct_ids if nct_id in indexed_trials]
    ranked_ids = set(ranked_nct_ids)
    remaining = [trial for trial in trials if get_trial_nct_id(trial) not in ranked_ids]
    return ordered + remaining

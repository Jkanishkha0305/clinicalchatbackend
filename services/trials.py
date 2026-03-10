"""Compatibility exports for trial helpers used across the FastAPI app."""

from services.trial_datasets import MAX_CHAT_DATASET_STUDIES, fetch_chat_dataset, order_trials_by_rank
from services.trial_documents import (
    extract_essential_fields,
    format_study,
    get_study_by_nct,
    get_trial_nct_id,
    serialize_document,
)
from services.trial_filters import (
    build_query_from_filters,
    build_ranked_trial_query,
    get_semantic_query_text,
    normalize_intervention_filter,
)

__all__ = [
    "MAX_CHAT_DATASET_STUDIES",
    "build_query_from_filters",
    "build_ranked_trial_query",
    "extract_essential_fields",
    "fetch_chat_dataset",
    "format_study",
    "get_semantic_query_text",
    "get_study_by_nct",
    "get_trial_nct_id",
    "normalize_intervention_filter",
    "order_trials_by_rank",
    "serialize_document",
]

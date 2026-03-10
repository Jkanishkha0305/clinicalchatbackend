"""Shared data access helpers for the agentic modules."""

from __future__ import annotations

import os

from db_utils import get_mongo_client

_collection = None


def _combine_clauses(clauses: list[dict]) -> dict:
    active_clauses = [clause for clause in clauses if clause]
    if not active_clauses:
        return {}
    if len(active_clauses) == 1:
        return active_clauses[0]
    return {"$and": active_clauses}


def get_trials_collection():
    """Lazily initialize the trials Mongo collection."""
    global _collection
    if _collection is not None:
        return _collection

    db_name = os.getenv("MONGO_DB_NAME", "clinical_trials")
    collection_name = os.getenv("MONGO_COLLECTION_NAME", "studies")
    mongo_client = get_mongo_client(serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    _collection = mongo_client[db_name][collection_name]
    return _collection


def build_agentic_trials_query(condition: str, phase: str | None = None, intervention_type: str | None = None) -> dict:
    """Build a broad match query for agentic landscape analysis."""
    clauses: list[dict] = []

    if condition:
        regex = {"$regex": condition, "$options": "i"}
        clauses.append(
            {
                "$or": [
                    {"conditions": regex},
                    {"title": regex},
                    {"protocolSection.conditionsModule.conditions": regex},
                    {"protocolSection.identificationModule.briefTitle": regex},
                    {"protocolSection.identificationModule.officialTitle": regex},
                ]
            }
        )

    if phase:
        clauses.append(
            {
                "$or": [
                    {"phase": phase},
                    {"phases": phase},
                    {"protocolSection.designModule.phases": phase},
                ]
            }
        )

    if intervention_type:
        regex = {"$regex": intervention_type, "$options": "i"}
        clauses.append(
            {
                "$or": [
                    {"interventions": regex},
                    {"protocolSection.armsInterventionsModule.interventions.type": regex},
                    {"protocolSection.armsInterventionsModule.interventions.name": regex},
                ]
            }
        )

    return _combine_clauses(clauses)


def fetch_similar_trials(
    condition: str,
    phase: str | None = None,
    intervention_type: str | None = None,
    *,
    limit: int,
) -> list[dict]:
    """Fetch similar trials for agentic analysis workflows."""
    query = build_agentic_trials_query(condition, phase, intervention_type)
    return list(get_trials_collection().find(query).limit(limit))

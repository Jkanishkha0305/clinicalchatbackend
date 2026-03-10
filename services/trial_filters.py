"""Query-building helpers for trial search and semantic ranking."""

from __future__ import annotations

from typing import Any


def normalize_intervention_filter(value: Any) -> list[str]:
    """Normalize the intervention filter into a list of non-empty strings."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _combine_clauses(clauses: list[dict]) -> dict:
    active_clauses = [clause for clause in clauses if clause]
    if not active_clauses:
        return {}
    if len(active_clauses) == 1:
        return active_clauses[0]
    return {"$and": active_clauses}


def _regex_or_clause(fields: list[str], value: str) -> dict:
    regex = {"$regex": value, "$options": "i"}
    return {"$or": [{field: regex} for field in fields]}


def _in_or_clause(fields: list[str], values: list[str]) -> dict:
    return {"$or": [{field: {"$in": values}} for field in fields]}


def _equals_or_clause(fields: list[str], value: Any) -> dict:
    return {"$or": [{field: value} for field in fields]}


def _range_or_clause(fields: list[str], *, start: str | None = None, end: str | None = None) -> dict:
    range_query: dict[str, str] = {}
    if start:
        range_query["$gte"] = start
    if end:
        range_query["$lte"] = end
    if not range_query:
        return {}
    return {"$or": [{field: range_query} for field in fields]}


def _normalize_phase_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def build_query_from_filters(filters: dict) -> dict:
    """Build a Mongo query that supports both flat and nested trial documents."""
    clauses: list[dict] = []

    condition = (filters.get("condition") or "").strip()
    if condition:
        clauses.append(
            _regex_or_clause(
                ["conditions", "protocolSection.conditionsModule.conditions"],
                condition,
            )
        )

    intervention = normalize_intervention_filter(filters.get("intervention"))
    if intervention:
        clauses.append(
            _in_or_clause(
                [
                    "interventions",
                    "protocolSection.armsInterventionsModule.interventions.name",
                ],
                intervention,
            )
        )

    status = filters.get("status") or []
    if status:
        clauses.append(
            _in_or_clause(
                ["status", "protocolSection.statusModule.overallStatus"],
                status,
            )
        )

    study_type = filters.get("studyType") or []
    if study_type:
        clauses.append(
            _in_or_clause(
                ["studyType", "protocolSection.designModule.studyType"],
                study_type,
            )
        )

    phase = _normalize_phase_list(filters.get("phase"))
    if phase:
        clauses.append(
            {
                "$or": [
                    {"phase": {"$in": phase}},
                    {"phases": {"$in": phase}},
                    {"protocolSection.designModule.phases": {"$elemMatch": {"$in": phase}}},
                ]
            }
        )

    title = (filters.get("title") or "").strip()
    if title:
        clauses.append(
            _regex_or_clause(
                [
                    "title",
                    "protocolSection.identificationModule.briefTitle",
                    "protocolSection.identificationModule.officialTitle",
                    "protocolSection.identificationModule.acronym",
                ],
                title,
            )
        )

    nct_id = (filters.get("nctId") or "").strip().upper()
    if nct_id:
        clauses.append(
            _equals_or_clause(
                ["nct_id", "protocolSection.identificationModule.nctId"],
                nct_id,
            )
        )

    sex = (filters.get("sex") or "").strip()
    if sex:
        clauses.append(
            _equals_or_clause(
                ["sex", "protocolSection.eligibilityModule.sex"],
                sex,
            )
        )

    age_groups = filters.get("ageGroups") or []
    if age_groups:
        clauses.append(
            {
                "$or": [
                    {"ageGroups": {"$in": age_groups}},
                    {"stdAges": {"$in": age_groups}},
                    {"protocolSection.eligibilityModule.stdAges": {"$elemMatch": {"$in": age_groups}}},
                ]
            }
        )

    if filters.get("healthyVolunteers"):
        clauses.append(
            _equals_or_clause(
                ["healthyVolunteers", "protocolSection.eligibilityModule.healthyVolunteers"],
                True,
            )
        )

    has_results = filters.get("hasResults")
    if has_results == "true":
        clauses.append(_equals_or_clause(["hasResults"], True))
    elif has_results == "false":
        clauses.append(_equals_or_clause(["hasResults"], False))

    if filters.get("hasProtocol"):
        clauses.append(_equals_or_clause(["documentSection.largeDocumentModule.largeDocs.hasProtocol"], True))
    if filters.get("hasSAP"):
        clauses.append(_equals_or_clause(["documentSection.largeDocumentModule.largeDocs.hasSap"], True))
    if filters.get("hasICF"):
        clauses.append(_equals_or_clause(["documentSection.largeDocumentModule.largeDocs.hasIcf"], True))

    funder_type = filters.get("funderType") or []
    if funder_type:
        clauses.append(
            _in_or_clause(
                [
                    "funderType",
                    "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
                ],
                funder_type,
            )
        )

    location = (filters.get("location") or "").strip()
    if location:
        clauses.append(
            _regex_or_clause(
                [
                    "protocolSection.contactsLocationsModule.locations.country",
                    "protocolSection.contactsLocationsModule.locations.city",
                    "protocolSection.contactsLocationsModule.locations.state",
                ],
                location,
            )
        )

    clauses.append(
        _range_or_clause(
            [
                "studyStart",
                "startDate",
                "protocolSection.statusModule.startDateStruct.date",
            ],
            start=filters.get("studyStartFrom"),
            end=filters.get("studyStartTo"),
        )
    )
    clauses.append(
        _range_or_clause(
            [
                "primaryCompletion",
                "primaryCompletionDate",
                "protocolSection.statusModule.primaryCompletionDateStruct.date",
            ],
            start=filters.get("primaryCompletionFrom"),
            end=filters.get("primaryCompletionTo"),
        )
    )

    sponsor = (filters.get("sponsor") or "").strip()
    if sponsor:
        clauses.append(
            _regex_or_clause(
                ["sponsor", "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"],
                sponsor,
            )
        )

    outcome = (filters.get("outcome") or "").strip()
    if outcome:
        clauses.append(
            _regex_or_clause(
                [
                    "primaryOutcome",
                    "protocolSection.outcomesModule.primaryOutcomes.measure",
                    "protocolSection.outcomesModule.secondaryOutcomes.measure",
                ],
                outcome,
            )
        )

    if filters.get("fdaaa801Violation"):
        clauses.append(
            _equals_or_clause(
                ["fdaaa801Violation", "protocolSection.oversightModule.fdaaa801Violation"],
                True,
            )
        )

    return _combine_clauses(clauses)


def build_ranked_trial_query(base_query: dict, ranked_nct_ids: list[str]) -> dict:
    """Combine a base filter query with a ranked NCT-ID subset."""
    if not ranked_nct_ids:
        return base_query

    nct_clause = {
        "$or": [
            {"nct_id": {"$in": ranked_nct_ids}},
            {"protocolSection.identificationModule.nctId": {"$in": ranked_nct_ids}},
        ]
    }
    return _combine_clauses([base_query, nct_clause])


def get_semantic_query_text(filters: dict) -> str:
    """Choose the best free-text input for semantic search."""
    return (
        filters.get("query")
        or filters.get("condition")
        or filters.get("title")
        or ""
    ).strip()

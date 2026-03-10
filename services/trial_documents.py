"""Study formatting and serialization helpers."""

from __future__ import annotations

from typing import Any

from services.trial_filters import normalize_intervention_filter


def _normalize_phase_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [value] if value else []
    return []


def serialize_document(document: dict) -> dict:
    """Convert Mongo ObjectIds to strings without mutating the original document."""
    serialized = dict(document)
    if "_id" in serialized:
        serialized["_id"] = str(serialized["_id"])
    return serialized


def get_trial_nct_id(document: dict) -> str:
    protocol = document.get("protocolSection", {}) or {}
    identification = protocol.get("identificationModule", {}) or {}
    return document.get("nct_id") or identification.get("nctId") or "N/A"


def format_study(document: dict) -> dict:
    """Convert either document shape into the frontend's expected result card."""
    serialized = serialize_document(document)
    protocol = serialized.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    design = protocol.get("designModule") or {}
    sponsor_module = protocol.get("sponsorCollaboratorsModule") or {}

    if not sponsor_module:
        sponsor_module = {"leadSponsor": {"name": serialized.get("sponsor", "N/A")}}

    phases = design.get("phases")
    if not phases:
        phases = _normalize_phase_value(serialized.get("phases") or serialized.get("phase"))

    study_type = design.get("studyType") or serialized.get("studyType") or "N/A"
    sponsor_name = sponsor_module.get("leadSponsor", {}).get("name") or serialized.get("sponsor") or "N/A"

    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": identification.get("nctId") or serialized.get("nct_id") or "N/A",
                "briefTitle": identification.get("briefTitle") or serialized.get("title") or "No title",
            },
            "statusModule": {
                "overallStatus": status.get("overallStatus") or serialized.get("status") or "UNKNOWN",
            },
            "designModule": {
                "studyType": study_type,
                "phases": phases,
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": sponsor_name},
            },
        },
        "hasResults": bool(serialized.get("hasResults", False)),
        "_original": serialized,
    }


def extract_essential_fields(document: dict) -> dict:
    """Reduce a trial document to the most useful fields for multi-study chat."""
    serialized = serialize_document(document)
    protocol = serialized.get("protocolSection") or {}

    if protocol:
        identification = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        conditions = protocol.get("conditionsModule", {}).get("conditions", [])
        interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
        outcomes = protocol.get("outcomesModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        enrollment = design.get("enrollmentInfo", {}).get("count", "N/A")
        eligibility = protocol.get("eligibilityModule", {})
        locations = protocol.get("contactsLocationsModule", {}).get("locations", [])

        return {
            "nctId": identification.get("nctId", serialized.get("nct_id", "N/A")),
            "title": identification.get("briefTitle", serialized.get("title", "N/A")),
            "status": status_module.get("overallStatus", serialized.get("status", "N/A")),
            "studyType": design.get("studyType", serialized.get("studyType", "N/A")),
            "phases": design.get("phases", []),
            "conditions": conditions[:5],
            "interventions": [item.get("name", "") for item in interventions[:3] if item.get("name")],
            "primaryOutcome": (
                outcomes.get("primaryOutcomes", [{}])[0].get("measure", "N/A")
                if outcomes.get("primaryOutcomes")
                else serialized.get("primaryOutcome", "N/A")
            ),
            "sponsor": sponsor_module.get("leadSponsor", {}).get("name", serialized.get("sponsor", "N/A")),
            "enrollment": enrollment,
            "eligibility": {
                "sex": eligibility.get("sex", serialized.get("sex", "N/A")),
                "ageRange": f"{eligibility.get('minimumAge', 'N/A')} to {eligibility.get('maximumAge', 'N/A')}",
            },
            "countries": sorted(
                {location.get("country", "") for location in locations if location.get("country")}
            )[:5],
        }

    return {
        "nctId": serialized.get("nct_id", "N/A"),
        "title": serialized.get("title", "N/A"),
        "status": serialized.get("status", "N/A"),
        "studyType": serialized.get("studyType", "N/A"),
        "phases": _normalize_phase_value(serialized.get("phases") or serialized.get("phase")),
        "conditions": (serialized.get("conditions") or [])[:5],
        "interventions": normalize_intervention_filter(serialized.get("interventions"))[:3],
        "primaryOutcome": serialized.get("primaryOutcome", "N/A"),
        "sponsor": serialized.get("sponsor", "N/A"),
        "enrollment": serialized.get("enrollment", "N/A"),
        "eligibility": {
            "sex": serialized.get("sex", "N/A"),
            "ageRange": f"{serialized.get('minimumAge', 'N/A')} to {serialized.get('maximumAge', 'N/A')}",
        },
        "countries": (serialized.get("countries") or [])[:5],
    }


def get_study_by_nct(collection, nct_id: str) -> dict | None:
    """Find a trial regardless of whether the document is flat or nested."""
    normalized_id = (nct_id or "").upper()
    if not normalized_id:
        return None

    return collection.find_one(
        {
            "$or": [
                {"nct_id": normalized_id},
                {"protocolSection.identificationModule.nctId": normalized_id},
            ]
        }
    )

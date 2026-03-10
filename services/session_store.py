"""Persistence helpers for session and study-chat history."""

import secrets
from datetime import datetime
from typing import Iterable


def append_chat_session_messages(collection, session_id: str | None, messages: Iterable[dict]) -> None:
    """Append chat messages to a saved cross-study session if one exists."""
    if not session_id:
        return

    message_list = [message for message in messages if message]
    if not message_list:
        return

    collection.update_one(
        {"_id": session_id},
        {
            "$push": {"messages": {"$each": message_list}},
            "$set": {"updated_at": datetime.now().isoformat()},
        },
    )


def append_study_chat_messages(
    collection,
    study_id: str | None,
    chat_session_id: str | None,
    messages: Iterable[dict],
) -> None:
    """Append chat messages to a saved single-study thread if one exists."""
    if not study_id or not chat_session_id:
        return

    message_list = [message for message in messages if message]
    if not message_list:
        return

    now = datetime.now().isoformat()
    collection.update_one(
        {"study_id": study_id, "chat_session_id": chat_session_id},
        {
            "$setOnInsert": {
                "study_id": study_id,
                "chat_session_id": chat_session_id,
                "messages": [],
                "reports": [],
                "custom_questions": None,
                "created_at": now,
            },
            "$push": {"messages": {"$each": message_list}},
            "$set": {"updated_at": now},
        },
        upsert=True,
    )


def append_chat_session_report(
    collection,
    session_id: str | None,
    report_doc: dict,
    *,
    last_report_filters: dict | None = None,
) -> dict | None:
    """Append a generated report to an existing saved chat session."""
    if not session_id or not report_doc:
        return None

    session = collection.find_one({"_id": session_id})
    if not session:
        return None

    reports = session.get("reports", [])
    reports.append(report_doc)

    update_payload = {
        "reports": reports,
        "updated_at": datetime.now().isoformat(),
    }
    if last_report_filters is not None:
        update_payload["last_report_filters"] = last_report_filters

    collection.update_one({"_id": session_id}, {"$set": update_payload})
    return {
        "id": session["_id"],
        "title": session.get("title", "Chat Session"),
        "description": session.get("description", ""),
    }


def create_chat_session_with_report(
    collection,
    *,
    title: str,
    description: str,
    last_filters: dict | None,
    last_report_filters: dict | None,
    report_doc: dict,
    custom_questions=None,
) -> dict:
    """Create a new chat session pre-populated with a generated report."""
    session_id = secrets.token_urlsafe(16)
    now = datetime.now().isoformat()
    session = {
        "_id": session_id,
        "title": title,
        "description": description,
        "last_filters": last_filters or {},
        "last_report_filters": last_report_filters,
        "messages": [],
        "reports": [report_doc],
        "custom_questions": custom_questions,
        "created_at": now,
        "updated_at": now,
    }
    collection.insert_one(session)
    return {
        "id": session_id,
        "title": title,
        "description": description,
    }


def append_study_chat_report(collection, query: dict, report_doc: dict) -> None:
    """Append a generated report to an existing study-chat thread."""
    if not query or not report_doc:
        return

    study_chat = collection.find_one(query)
    if not study_chat:
        return

    reports = study_chat.get("reports", [])
    reports.append(report_doc)
    collection.update_one(
        query,
        {
            "$set": {
                "reports": reports,
                "updated_at": datetime.now().isoformat(),
            }
        },
    )

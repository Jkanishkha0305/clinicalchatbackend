from fastapi import APIRouter
from fastapi.responses import JSONResponse

import dependencies as deps
from models import ChatReportRequest, ProtocolReportRequest, StudyChatReportRequest
from services.report_generation import (
    generate_chat_report,
    generate_protocol_report,
    generate_study_chat_report,
)
from services.session_store import (
    append_chat_session_report,
    append_study_chat_report,
    create_chat_session_with_report,
)
from services.trials import build_query_from_filters, get_study_by_nct

router = APIRouter()


@router.post("/generate-protocol-report")
def generate_protocol_report_endpoint(body: ProtocolReportRequest):
    condition = body.condition or ""
    intervention = body.intervention or ""

    if not condition:
        return JSONResponse({"success": False, "error": "Condition is required"}, status_code=400)

    query = build_query_from_filters({"condition": condition, "intervention": intervention})
    total_count = deps.collection.count_documents(query)
    if total_count == 0:
        return JSONResponse({"success": False, "error": f"No trials found for {condition}"}, status_code=404)

    similar_trials = list(deps.collection.find(query).limit(min(total_count, 100)))

    try:
        result = generate_protocol_report(
            openai_client=deps.openai_client,
            count_tokens=deps.count_tokens,
            condition=condition,
            intervention=intervention,
            similar_trials=similar_trials,
            total_count=total_count,
            format_type=body.format or "styled",
        )

        last_report_filters = {
            "condition": condition,
            "intervention": intervention or None,
        }
        if body.sessionId:
            session_info = append_chat_session_report(
                deps.chat_sessions_collection,
                body.sessionId,
                result["report_doc"],
                last_report_filters=last_report_filters,
            )
        else:
            session_info = create_chat_session_with_report(
                deps.chat_sessions_collection,
                title=f"Protocol Report: {condition}",
                description=(
                    f"Protocol research report for {condition}"
                    f"{f' with {intervention}' if intervention else ''}"
                ),
                last_filters={"condition": condition, "intervention": intervention},
                last_report_filters=last_report_filters,
                report_doc=result["report_doc"],
            )

        response = {
            "success": True,
            "report": result["report"],
            "metadata": result["metadata"],
        }
        if session_info:
            response["sessionInfo"] = session_info
        return response
    except Exception as e:
        return JSONResponse({"success": False, "error": f"AI error: {e}"}, status_code=500)


@router.post("/generate-chat-report")
def generate_chat_report_endpoint(body: ChatReportRequest):
    session_id = body.sessionId or ""
    if not session_id:
        return JSONResponse({"success": False, "error": "Session ID is required"}, status_code=400)

    session = deps.chat_sessions_collection.find_one({"_id": session_id})
    if not session:
        return JSONResponse({"success": False, "error": "Chat session not found"}, status_code=404)

    messages = session.get("messages", [])
    last_filters = session.get("last_filters", {})
    condition = last_filters.get("condition", "Unknown")
    intervention = last_filters.get("intervention", "")

    query = build_query_from_filters(last_filters) if last_filters else {}
    total_count = deps.collection.count_documents(query)
    studies = list(deps.collection.find(query).limit(min(total_count, 50)))

    try:
        result = generate_chat_report(
            openai_client=deps.openai_client,
            count_tokens=deps.count_tokens,
            condition=condition,
            intervention=intervention,
            messages=messages,
            studies=studies,
            total_count=total_count,
            format_type=body.format or "styled",
        )
        append_chat_session_report(
            deps.chat_sessions_collection,
            session_id,
            result["report_doc"],
        )
        return {
            "success": True,
            "report": result["report"],
            "metadata": result["metadata"],
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": f"AI error: {e}"}, status_code=500)


@router.post("/generate-study-chat-report")
def generate_study_chat_report_endpoint(body: StudyChatReportRequest):
    study_id = body.studyId or ""
    if not study_id:
        return JSONResponse({"success": False, "error": "Study ID is required"}, status_code=400)

    study = get_study_by_nct(deps.collection, study_id)
    if not study:
        return JSONResponse({"success": False, "error": "Study not found"}, status_code=404)

    query = {"study_id": study_id}
    if body.chatSessionId:
        query["chat_session_id"] = body.chatSessionId

    study_chat = deps.study_chats_collection.find_one(query)
    messages = study_chat.get("messages", []) if study_chat else []

    try:
        result = generate_study_chat_report(
            openai_client=deps.openai_client,
            count_tokens=deps.count_tokens,
            study_id=study_id,
            study_title=study.get("title", "Unknown Study"),
            study=study,
            messages=messages,
            format_type=body.format or "styled",
        )
        append_study_chat_report(
            deps.study_chats_collection,
            query,
            result["report_doc"],
        )
        return {
            "success": True,
            "report": result["report"],
            "metadata": result["metadata"],
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": f"AI error: {e}"}, status_code=500)

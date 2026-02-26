import secrets
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import dependencies as deps
from models import SessionCreateRequest, SessionUpdateRequest, SessionQuestionsRequest

router = APIRouter()


@router.get("/chat-sessions")
def get_chat_sessions():
    try:
        sessions = list(deps.chat_sessions_collection.find().sort("created_at", -1))
        for session in sessions:
            if "_id" in session:
                session["id"] = str(session["_id"])
                del session["_id"]
        return {"success": True, "sessions": sessions}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.post("/chat-sessions")
def create_chat_session(body: SessionCreateRequest):
    try:
        session = {
            "_id": secrets.token_urlsafe(16),
            "title": body.title or "New Chat Session",
            "description": body.description or "",
            "last_filters": body.last_filters or {},
            "messages": [],
            "reports": [],
            "custom_questions": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        deps.chat_sessions_collection.insert_one(session)
        session["id"] = session["_id"]
        del session["_id"]
        return {"success": True, "session": session}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.get("/chat-sessions/{session_id}")
def get_chat_session(session_id: str):
    try:
        session = deps.chat_sessions_collection.find_one({"_id": session_id})
        if not session:
            return JSONResponse({"success": False, "error": "Session not found"}, status_code=404)
        session["id"] = session["_id"]
        del session["_id"]
        return {"success": True, "session": session}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.patch("/chat-sessions/{session_id}")
def update_chat_session(session_id: str, body: SessionUpdateRequest):
    try:
        update_data = {"updated_at": datetime.now().isoformat()}
        if body.title is not None:
            update_data["title"] = body.title
        if body.description is not None:
            update_data["description"] = body.description
        if body.last_filters is not None:
            update_data["last_filters"] = body.last_filters
        if body.custom_questions is not None:
            update_data["custom_questions"] = body.custom_questions

        result = deps.chat_sessions_collection.update_one({"_id": session_id}, {"$set": update_data})
        if result.matched_count == 0:
            return JSONResponse({"success": False, "error": "Session not found"}, status_code=404)

        session = deps.chat_sessions_collection.find_one({"_id": session_id})
        session["id"] = session["_id"]
        del session["_id"]
        return {"success": True, "session": session}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.delete("/chat-sessions/{session_id}")
def delete_chat_session(session_id: str):
    try:
        result = deps.chat_sessions_collection.delete_one({"_id": session_id})
        if result.deleted_count == 0:
            return JSONResponse({"success": False, "error": "Session not found"}, status_code=404)
        return {"success": True, "message": "Session deleted successfully"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.patch("/chat-sessions/{session_id}/questions")
def update_session_questions(session_id: str, body: SessionQuestionsRequest):
    try:
        if body.questions is None or not isinstance(body.questions, list):
            return JSONResponse({"success": False, "error": "questions must be an array"}, status_code=400)

        result = deps.chat_sessions_collection.update_one(
            {"_id": session_id},
            {"$set": {"custom_questions": body.questions, "updated_at": datetime.now().isoformat()}},
        )
        if result.matched_count == 0:
            return JSONResponse({"success": False, "error": "Session not found"}, status_code=404)

        session = deps.chat_sessions_collection.find_one({"_id": session_id})
        session["id"] = session["_id"]
        del session["_id"]
        return {"success": True, "message": "Session questions updated successfully", "session": session}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

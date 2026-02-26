from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import dependencies as deps
from models import (
    PreferencesUpdateRequest,
    SettingsUpdateRequest,
    ChatQuestionsUpdateRequest,
    StudyChatQuestionsUpdateRequest,
)

router = APIRouter()

DEFAULT_QUESTIONS = [
    "What are the eligibility criteria?",
    "What is the study design?",
    "What are the primary outcomes?",
]


# ── User Preferences ──────────────────────────────────────────────────────────

@router.get("/user-preferences")
def get_user_preferences():
    try:
        prefs = deps.user_preferences_collection.find_one({}) or {
            "default_chat_questions": DEFAULT_QUESTIONS,
            "ai_provider": "openai",
            "ai_model": "gpt-4o-mini",
        }
        prefs.pop("_id", None)
        return {"success": True, "preferences": prefs}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.patch("/user-preferences")
def update_user_preferences(body: PreferencesUpdateRequest):
    try:
        update = {k: v for k, v in body.dict(exclude_none=True).items()}
        deps.user_preferences_collection.update_one({}, {"$set": update}, upsert=True)
        prefs = deps.user_preferences_collection.find_one({})
        prefs.pop("_id", None)
        return {"success": True, "preferences": prefs}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── User Settings ─────────────────────────────────────────────────────────────

@router.get("/user-settings")
def get_user_settings():
    try:
        settings = deps.user_settings_collection.find_one({}) or {
            "theme": "light",
            "visible_models": ["gpt-4o-mini", "gpt-4o", "gemini-1.5-flash"],
            "report_format": "styled",
        }
        settings.pop("_id", None)
        return {"success": True, "settings": settings}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.patch("/user-settings")
def update_user_settings(body: SettingsUpdateRequest):
    try:
        update = {k: v for k, v in body.dict(exclude_none=True).items()}
        deps.user_settings_collection.update_one({}, {"$set": update}, upsert=True)
        settings = deps.user_settings_collection.find_one({})
        settings.pop("_id", None)
        return {"success": True, "settings": settings}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── Chat Questions ────────────────────────────────────────────────────────────

@router.get("/chat-questions")
def get_chat_questions(sessionId: Optional[str] = Query(None)):
    try:
        if sessionId:
            session = deps.chat_sessions_collection.find_one({"_id": sessionId})
            if session and session.get("custom_questions"):
                return {"success": True, "questions": session["custom_questions"], "source": "session"}

        prefs = deps.user_preferences_collection.find_one({})
        questions = prefs.get("default_chat_questions", DEFAULT_QUESTIONS) if prefs else DEFAULT_QUESTIONS
        return {"success": True, "questions": questions, "source": "default"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.patch("/chat-questions")
def update_chat_questions(body: ChatQuestionsUpdateRequest):
    try:
        if body.saveAsDefault:
            deps.user_preferences_collection.update_one(
                {}, {"$set": {"default_chat_questions": body.questions}}, upsert=True
            )
        if body.sessionId:
            deps.chat_sessions_collection.update_one(
                {"_id": body.sessionId}, {"$set": {"custom_questions": body.questions}}
            )
        return {"success": True, "message": "Questions updated successfully"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── Study Chat Questions ──────────────────────────────────────────────────────

@router.get("/study-chat-questions")
def get_study_chat_questions(
    studyId: Optional[str] = Query(None),
    chatSessionId: Optional[str] = Query(None),
):
    try:
        if studyId and chatSessionId:
            study_chat = deps.study_chats_collection.find_one(
                {"study_id": studyId, "chat_session_id": chatSessionId}
            )
            if study_chat and study_chat.get("custom_questions"):
                return {"success": True, "questions": study_chat["custom_questions"], "source": "study_chat"}

        prefs = deps.user_preferences_collection.find_one({})
        questions = prefs.get("default_chat_questions", DEFAULT_QUESTIONS) if prefs else DEFAULT_QUESTIONS
        return {"success": True, "questions": questions, "source": "default"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.patch("/study-chat-questions")
def update_study_chat_questions(body: StudyChatQuestionsUpdateRequest):
    try:
        if body.saveAsDefault:
            deps.user_preferences_collection.update_one(
                {}, {"$set": {"default_chat_questions": body.questions}}, upsert=True
            )
        if body.studyId and body.chatSessionId:
            deps.study_chats_collection.update_one(
                {"study_id": body.studyId, "chat_session_id": body.chatSessionId},
                {"$set": {"custom_questions": body.questions}},
                upsert=True,
            )
        return {"success": True, "message": "Study chat questions updated successfully"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── Study Chats ───────────────────────────────────────────────────────────────

@router.get("/study-chats")
def get_study_chats(chatSessionId: Optional[str] = Query(None)):
    try:
        query = {}
        if chatSessionId:
            query["chat_session_id"] = chatSessionId
        study_chats = list(deps.study_chats_collection.find(query))
        for chat in study_chats:
            if "_id" in chat:
                chat["id"] = str(chat["_id"])
                del chat["_id"]
        return {"success": True, "studyChats": study_chats}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.get("/study-chats/{study_id}")
def get_study_chat(study_id: str, chatSessionId: Optional[str] = Query(None)):
    try:
        query = {"study_id": study_id}
        if chatSessionId:
            query["chat_session_id"] = chatSessionId

        study_chat = deps.study_chats_collection.find_one(query)
        if not study_chat:
            study_chat = {
                "study_id": study_id,
                "chat_session_id": chatSessionId,
                "messages": [],
                "reports": [],
                "custom_questions": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            deps.study_chats_collection.insert_one(study_chat)

        study_chat.pop("_id", None)
        return {"success": True, "studyChat": study_chat}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.delete("/study-chats/{study_id}/{session_id}")
def delete_study_chat(study_id: str, session_id: str):
    try:
        result = deps.study_chats_collection.delete_one(
            {"study_id": study_id, "chat_session_id": session_id}
        )
        if result.deleted_count == 0:
            return JSONResponse({"success": False, "error": "Study chat not found"}, status_code=404)
        return {"success": True, "message": "Study chat deleted successfully"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

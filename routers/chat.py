import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

import dependencies as deps
from models import ChatRequest, ChatAllRequest
from services.rendering import render_markdown
from services.session_store import append_chat_session_messages, append_study_chat_messages
from services.trials import fetch_chat_dataset, get_study_by_nct, serialize_document

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/chat")
def chat_single_study(body: ChatRequest):
    study = get_study_by_nct(deps.collection, body.nctId)
    if not study:
        return JSONResponse({"error": "Study not found"}, status_code=404)

    study_copy = serialize_document(study)
    system_message = f"""You are a clinical trials expert. Answer questions about this study.

STUDY DATA:
{json.dumps(study_copy, indent=2)}

Instructions:
- Answer based only on the provided data
- Be precise and concise"""

    try:
        completion = deps.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": body.question},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        answer = completion.choices[0].message.content
        answer_html = render_markdown(answer)
        append_study_chat_messages(
            deps.study_chats_collection,
            body.nctId,
            body.chatSessionId,
            [
                {"role": "user", "content": body.question},
                {"role": "assistant", "content": answer},
            ],
        )
        return {"answer": answer_html}
    except Exception as e:
        return JSONResponse({"error": f"AI error: {e}"}, status_code=500)


@router.post("/chat-stream")
def chat_stream(body: ChatRequest):
    study = get_study_by_nct(deps.collection, body.nctId)
    if not study:
        return JSONResponse({"error": "Study not found"}, status_code=404)

    study_copy = serialize_document(study)
    system_message = f"""You are a clinical trials expert. Answer questions about this study.

STUDY DATA:
{json.dumps(study_copy, indent=2)}

Instructions:
- Answer based only on the provided data
- Be precise and concise"""

    question = body.question

    def generate():
        full_answer = ""
        try:
            stream = deps.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=1000,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_answer += delta.content
                    yield f"data: {json.dumps({'type': 'content', 'chunk': delta.content})}\n\n"
            answer_html = render_markdown(full_answer)
            append_study_chat_messages(
                deps.study_chats_collection,
                body.nctId,
                body.chatSessionId,
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": full_answer},
                ],
            )
            yield f"data: {json.dumps({'type': 'done', 'answer': answer_html})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/chat-all")
def chat_all_studies(body: ChatAllRequest):
    filters = body.filters or {}
    question = body.question
    selected_model = body.model or "openai"
    dataset = fetch_chat_dataset(
        deps.collection,
        filters,
        advanced_mode=bool(body.advancedMode),
    )

    system_message = f"""You are a clinical trials research analyst.

DATASET SUMMARY:
- Total studies in filtered results: {dataset['total']}
- Studies provided for analysis: {dataset['provided']}
- Data mode: {"COMPLETE (all fields)" if body.advancedMode else "ESSENTIAL (key fields only)"}

STUDY DATA:
{json.dumps(dataset['studies'], indent=1)}

Answer the question by analyzing the provided studies. Provide statistics and insights."""

    try:
        answer, model_info = deps.call_llm(selected_model, system_message, question, max_tokens=2000)
        answer_html = render_markdown(answer, tables=True)
        append_chat_session_messages(
            deps.chat_sessions_collection,
            body.sessionId,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        )

        info = (
            f"Analyzing {dataset['provided']} studies using {model_info} "
            f"({dataset['mode_label']} data)"
        )
        if dataset["was_truncated"]:
            info += f"; {dataset['total']} studies matched in total"
        return {
            "answer": answer_html,
            "info": info,
        }
    except Exception as e:
        return JSONResponse({"error": f"AI error: {e}"}, status_code=500)


@router.post("/chat-all-stream")
def chat_all_stream(body: ChatAllRequest):
    filters = body.filters or {}
    question = body.question
    dataset = fetch_chat_dataset(
        deps.collection,
        filters,
        advanced_mode=bool(body.advancedMode),
    )

    system_message = f"""You are a clinical trials research analyst.

DATASET SUMMARY:
- Total studies in filtered results: {dataset['total']}
- Studies provided for analysis: {dataset['provided']}
- Data mode: {"COMPLETE (all fields)" if body.advancedMode else "ESSENTIAL (key fields only)"}

STUDY DATA:
{json.dumps(dataset['studies'], indent=1)}

Answer the question by analyzing the provided studies. Provide statistics and insights."""

    def generate():
        full_answer = ""
        try:
            stream = deps.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=2000,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_answer += delta.content
                    yield f"data: {json.dumps({'type': 'content', 'chunk': delta.content})}\n\n"
            answer_html = render_markdown(full_answer, tables=True)
            append_chat_session_messages(
                deps.chat_sessions_collection,
                body.sessionId,
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": full_answer},
                ],
            )
            yield f"data: {json.dumps({'type': 'done', 'answer': answer_html})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)

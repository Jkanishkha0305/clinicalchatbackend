"""
Pydantic request body models for all FastAPI routes.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ── Auth ──────────────────────────────────────────────────────────────────────

class SignUpRequest(BaseModel):
    username: str
    password: str
    confirmPassword: str

class LoginRequest(BaseModel):
    username: str
    password: str


# ── Search ────────────────────────────────────────────────────────────────────

class SearchFilters(BaseModel):
    condition: Optional[str] = None
    intervention: Optional[Any] = None   # str or list[str]
    status: Optional[List[str]] = None
    phase: Optional[List[str]] = None
    title: Optional[str] = None
    nctId: Optional[str] = None
    page: Optional[int] = 1
    per_page: Optional[int] = 20
    useSemanticSearch: Optional[bool] = False
    sessionId: Optional[str] = None
    query: Optional[str] = None          # free-text semantic query
    advancedMode: Optional[bool] = False
    model: Optional[str] = None

    class Config:
        extra = "allow"   # pass unknown filter keys through


class StatisticsRequest(BaseModel):
    condition: Optional[str] = None
    intervention: Optional[Any] = None
    status: Optional[List[str]] = None
    title: Optional[str] = None
    nctId: Optional[str] = None

    class Config:
        extra = "allow"


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    nctId: str
    question: str
    chatSessionId: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None

class ChatAllRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = {}
    question: str
    advancedMode: Optional[bool] = False
    sessionId: Optional[str] = None
    model: Optional[str] = "openai"
    provider: Optional[str] = None


# ── Reports ───────────────────────────────────────────────────────────────────

class ProtocolReportRequest(BaseModel):
    condition: str
    intervention: Optional[str] = ""
    sessionId: Optional[str] = None
    format: Optional[str] = "styled"

class ChatReportRequest(BaseModel):
    sessionId: str
    format: Optional[str] = "styled"

class StudyChatReportRequest(BaseModel):
    studyId: str
    chatSessionId: Optional[str] = ""
    format: Optional[str] = "styled"


# ── Sessions ──────────────────────────────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    title: Optional[str] = "New Chat Session"
    description: Optional[str] = ""
    last_filters: Optional[Dict[str, Any]] = {}

class SessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    last_filters: Optional[Dict[str, Any]] = None
    custom_questions: Optional[Any] = None   # list or None

class SessionQuestionsRequest(BaseModel):
    questions: List[str]


# ── Preferences / Settings / Questions ───────────────────────────────────────

class PreferencesUpdateRequest(BaseModel):
    default_chat_questions: Optional[List[str]] = None
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None

    class Config:
        extra = "allow"

class SettingsUpdateRequest(BaseModel):
    theme: Optional[str] = None
    visible_models: Optional[List[str]] = None
    report_format: Optional[str] = None

    class Config:
        extra = "allow"

class ChatQuestionsUpdateRequest(BaseModel):
    sessionId: Optional[str] = None
    questions: List[str]
    saveAsDefault: Optional[bool] = False

class StudyChatQuestionsUpdateRequest(BaseModel):
    studyId: Optional[str] = None
    chatSessionId: Optional[str] = None
    questions: List[str]
    saveAsDefault: Optional[bool] = False


# ── Agents ────────────────────────────────────────────────────────────────────

class CompareTrialsRequest(BaseModel):
    nctIds: List[str]

class AgentSearchRequest(BaseModel):
    query: str

class AgentNctRequest(BaseModel):
    nctId: str

class AgentConditionRequest(BaseModel):
    condition: str
    phase: Optional[str] = None
    interventionType: Optional[str] = None


# ── Documents (ChromaDB) ──────────────────────────────────────────────────────

class AddDocumentsRequest(BaseModel):
    ids: List[str]
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = []
    embeddings: Optional[List[Any]] = None

"""
FastAPI entry point for the ClinicalChat backend.
Run with: uvicorn main:app --reload --port 8081
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

# Import routers
from routers import auth, search, chat, reports, sessions, preferences, agents


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Trigger all client initialisations at startup
    import dependencies  # noqa: F401
    yield
    # (cleanup if needed)


app = FastAPI(
    title="ClinicalChat API",
    version="2.0.0",
    description="Clinical trials AI-powered search and analysis backend",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
CLIENT_HOST = os.getenv("CLIENT_HOST", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        CLIENT_HOST,
        "http://localhost:3000",
        "https://clinicalchat.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files & templates ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/agentic-demo", response_class=HTMLResponse)
def agentic_demo(request: Request):
    return templates.TemplateResponse("agentic_demo.html", {"request": request})


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,        prefix="/api/auth",  tags=["Auth"])
app.include_router(search.router,      prefix="/api",       tags=["Search"])
app.include_router(chat.router,        prefix="/api",       tags=["Chat"])
app.include_router(reports.router,     prefix="/api",       tags=["Reports"])
app.include_router(sessions.router,    prefix="/api",       tags=["Sessions"])
app.include_router(preferences.router, prefix="/api",       tags=["Preferences"])
app.include_router(agents.router,      prefix="/api",       tags=["Agents"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8081))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

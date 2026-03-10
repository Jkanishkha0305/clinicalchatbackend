# ClinicalChat Backend

FastAPI backend and single-page frontend for clinical trial search, chat, and protocol-analysis workflows.

## Active app layout

- `main.py`: FastAPI entry point and route wiring.
- `routers/`: HTTP endpoints grouped by domain.
- `services/`: shared business logic used by routers and scripts.
- `agentic/`: LLM-driven analysis modules used by the agent endpoints.
- `dependencies.py`: shared client setup for MongoDB, OpenAI, Chroma, and Qdrant.
- `models.py`: request and response models.
- `templates/` and `static/`: frontend UI.
- `tests/`: lightweight automated coverage for core service helpers.

## Frontend structure

- `static/js/app.js`: module entrypoint that exposes UI actions to the page.
- `static/js/core/`: small shared browser utilities (`api`, `dom`, `state`).
- `static/js/features/`: feature-focused modules for search, chat, protocol reports, tabs, PDF export, and agent tools.

## Offline / support scripts

- `import_data.py`: import trial data into MongoDB.
- `generate_embeddings.py`: build local Chroma embeddings.
- `upload_to_chroma_cloud.py`: upload embeddings to Chroma Cloud.
- `upload_to_qdrant.py`: upload embeddings to Qdrant Cloud.
- `demo_agentic_features.py`: manual API demo runner.
- `test_evaluation.py`: manual evaluation harness.

## Legacy code

Legacy Flask-era and superseded files live under `legacy/`. The FastAPI app in this repo root is the current code path.

## Local setup with `uv`

`uv` is now the recommended way to install and run the backend. You do not need Conda for this repo.

1. Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync the project dependencies:

```bash
cd /Users/j_kanishkha/Projects/clinicalchatbackend
uv sync
```

3. Make sure your `.env` includes the backend secrets and connections you need, especially:

- `OPENAI_API_KEY`
- Mongo connection settings such as `MONGODB_URI` / `MONGO_DB_NAME`
- Optional vector store settings if you want Chroma or Qdrant features

4. Run the backend:

```bash
uv run uvicorn main:app --reload --port 8081
```

5. Check health:

```bash
open http://127.0.0.1:8081/health
```

## Common `uv` commands

Run tests:

```bash
uv run python -m unittest discover -s tests -v
```

Run lint:

```bash
uv run ruff check .
```

Run a script:

```bash
uv run python import_data.py
uv run python generate_embeddings.py
```

## Fallback `pip` install

If you still want the older flow, `requirements.txt` is still present:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8081
```

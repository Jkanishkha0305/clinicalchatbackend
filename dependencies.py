"""
Shared client singletons for the FastAPI application.
Imported by routers — all clients are initialised at module load time.
"""
import os
import json
import re
import shutil
import tiktoken
import markdown
import bcrypt
import secrets
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

load_dotenv()

from db_utils import get_mongo_client

# =============================================================================
# TOKEN COUNTING
# =============================================================================

def count_tokens(messages, model="gpt-4o-mini"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    total = 4
    for msg in messages:
        total += 4
        if isinstance(msg.get("content"), str):
            total += len(encoding.encode(msg["content"]))
    return total

# =============================================================================
# MONGODB
# =============================================================================

print("🔌 Connecting to MongoDB Atlas...")
try:
    mongo_client = get_mongo_client(serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    print("✓ MongoDB connected successfully!")
except Exception as e:
    raise RuntimeError(f"✗ MongoDB connection failed: {str(e)}")

MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'clinical_trials')
MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'studies')

db = mongo_client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]
users_collection = db['users']
chat_sessions_collection = db['chat_sessions']
user_preferences_collection = db['user_preferences']
user_settings_collection = db['user_settings']
study_chats_collection = db['study_chats']

# =============================================================================
# OPENAI
# =============================================================================

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    for key_path in ['data1/key/openai_key.txt', 'openai_key.txt']:
        try:
            with open(key_path) as f:
                openai_key = f.readline().strip()
                os.environ["OPENAI_API_KEY"] = openai_key
                print(f"✓ Loaded OpenAI API key from: {key_path}")
                break
        except FileNotFoundError:
            continue

if not openai_key:
    print("\n⚠️  ERROR: OpenAI API key not found!")
    raise RuntimeError("OpenAI API key is required. Set OPENAI_API_KEY in .env")

openai_client = OpenAI()
print(f"✓ OpenAI client initialised")

# =============================================================================
# GEMINI (optional)
# =============================================================================

gemini_api_key = os.environ.get("GEMINI_API_KEY")
gemini_client = None
if gemini_api_key:
    try:
        from google import genai as _genai
        gemini_client = _genai.Client(api_key=gemini_api_key)
        print("✓ Gemini API key loaded (google-genai SDK)")
    except Exception as _e:
        print(f"⚠️  Gemini init failed: {_e}")
        gemini_api_key = None
else:
    print("⚠️  Gemini API key not found (optional)")

# =============================================================================
# GROQ (optional)
# =============================================================================

groq_api_key = os.environ.get("GROQ_API_KEY")
groq_client = None
if groq_api_key:
    try:
        from groq import Groq
        groq_client = Groq(api_key=groq_api_key)
        print("✓ Groq API key loaded")
    except Exception:
        pass
else:
    print("⚠️  Groq API key not found (optional)")

# =============================================================================
# CHROMADB (optional, multiple modes)
# =============================================================================

chroma_collection = None
chroma_client = None

try:
    import chromadb
    chroma_api_key = os.getenv('CHROMA_API_KEY')
    chroma_tenant = os.getenv('CHROMA_TENANT')
    chroma_database = os.getenv('CHROMA_DATABASE', 'clinicalchat')

    if chroma_api_key and chroma_tenant:
        chroma_client = chromadb.CloudClient(
            api_key=chroma_api_key, tenant=chroma_tenant, database=chroma_database
        )
        try:
            chroma_collection = chroma_client.get_or_create_collection(name='clinical_trials_embeddings')
            print(f"✓ ChromaDB Cloud connected: {chroma_collection.count()} embeddings")
        except Exception as e:
            print(f"⚠️  ChromaDB Cloud error: {e}")
            chroma_collection = None
    else:
        chroma_host = os.getenv('CHROMA_HOST')
        if chroma_host:
            chroma_port = os.getenv('CHROMA_PORT', '8000')
            auth_token = os.getenv('CHROMA_AUTH_TOKEN')
            if auth_token:
                chroma_client = chromadb.HttpClient(
                    host=chroma_host, port=int(chroma_port),
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
            else:
                chroma_client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
            try:
                chroma_collection = chroma_client.get_collection(name='clinical_trials_embeddings')
                print(f"✓ ChromaDB connected to {chroma_host}:{chroma_port}")
            except Exception as e:
                print(f"⚠️  ChromaDB collection not found: {e}")
                chroma_collection = None
        else:
            chroma_path = os.getenv('CHROMADB_PATH')
            if chroma_path:
                # Only initialise local PersistentClient when path is explicitly configured
                chroma_client = chromadb.PersistentClient(path=chroma_path)
                try:
                    chroma_collection = chroma_client.get_collection(name='clinical_trials_embeddings')
                    print(f"✓ ChromaDB loaded from {chroma_path}: {chroma_collection.count()} embeddings")
                except Exception as e:
                    print(f"⚠️  ChromaDB collection not found at {chroma_path}: {e}")
                    chroma_collection = None
            else:
                print("⚠️  ChromaDB not configured (set CHROMA_API_KEY+CHROMA_TENANT, CHROMA_HOST, or CHROMADB_PATH)")
except Exception as e:
    print(f"⚠️  ChromaDB initialisation failed: {e}")
    chroma_collection = None

# =============================================================================
# QDRANT (optional)
# =============================================================================

qdrant_client = None
QDRANT_COLLECTION_NAME = 'clinical_trials'

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')

    if qdrant_url and qdrant_api_key:
        print("\n🔌 Connecting to Qdrant Cloud...")
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5)
        try:
            info = qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
            count = getattr(info, 'vectors_count', None) or getattr(info, 'points_count', 0) or 0
            print(f"✓ Qdrant Cloud connected: {count:,} vectors in '{QDRANT_COLLECTION_NAME}'")
        except Exception as e:
            print(f"⚠️  Qdrant collection '{QDRANT_COLLECTION_NAME}' not found: {e}")
            qdrant_client = None
    else:
        print("⚠️  Qdrant not configured (QDRANT_URL / QDRANT_API_KEY not set)")
except Exception as e:
    print(f"⚠️  Qdrant initialisation failed: {e}")
    qdrant_client = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def call_llm(model_name, system_message, user_message, max_tokens=2000):
    """Unified multi-LLM call. Returns (response_text, model_info)."""
    if model_name == "openai":
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content, "OpenAI GPT-4o"

    elif model_name == "gemini":
        if not gemini_client:
            raise Exception("Gemini API key not configured")
        prompt = f"{system_message}\n\n{user_message}"
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text, "Google Gemini 2.0 Flash"

    elif model_name == "groq":
        if not groq_client:
            raise Exception("Groq API key not configured")
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content, "Groq Llama 3.1 70B"

    else:
        raise Exception(f"Unknown model: {model_name}")


def build_query_from_filters(filters: dict) -> dict:
    """Build MongoDB query from search filters."""
    query = {}
    if filters.get('condition'):
        query['conditions'] = {'$regex': filters['condition'], '$options': 'i'}
    intervention = filters.get('intervention')
    if intervention:
        if isinstance(intervention, str):
            intervention = [i.strip() for i in intervention.split(',') if i.strip()]
        if isinstance(intervention, list) and intervention:
            query['interventions'] = {'$in': intervention}
    if filters.get('status') and len(filters['status']) > 0:
        query['status'] = {'$in': filters['status']}
    if filters.get('title'):
        query['title'] = {'$regex': filters['title'], '$options': 'i'}
    if filters.get('nctId'):
        query['nct_id'] = filters['nctId'].upper()
    return query


def format_study(result: dict) -> dict:
    """Convert flat MongoDB study doc to the nested format the frontend expects."""
    result['_id'] = str(result['_id'])
    return {
        'protocolSection': {
            'identificationModule': {
                'nctId': result.get('nct_id', 'N/A'),
                'briefTitle': result.get('title', 'No title')
            },
            'statusModule': {'overallStatus': result.get('status', 'UNKNOWN')},
            'designModule': {'studyType': 'INTERVENTIONAL', 'phases': []},
            'sponsorCollaboratorsModule': {'leadSponsor': {'name': 'N/A'}}
        },
        'hasResults': False,
        '_original': result
    }


NCT_LINK_RE = re.compile(r'(NCT\d{8})')
NCT_LINK_REPLACE = r'<a href="https://clinicaltrials.gov/study/\1" target="_blank" style="color:#4f46e5;text-decoration:underline;">\1</a>'


def render_report_html(raw_markdown: str) -> str:
    """Convert markdown report to HTML with NCT ID links."""
    html = markdown.markdown(raw_markdown, extensions=['extra', 'nl2br', 'tables'])
    return NCT_LINK_RE.sub(NCT_LINK_REPLACE, html)

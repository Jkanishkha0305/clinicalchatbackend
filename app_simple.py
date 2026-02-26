from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime
import json
import markdown
import tiktoken
import os
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
import chromadb
import bcrypt
import secrets
import itertools
import re
import shutil

from db_utils import get_mongo_client

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# TOKEN COUNTING UTILITIES
# =============================================================================

def count_tokens(messages, model="gpt-4o-mini"):
    """Count tokens in a message list."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 4  # overhead
    for message in messages:
        total_tokens += 4  # role overhead
        if isinstance(message.get("content"), str):
            total_tokens += len(encoding.encode(message["content"]))
    
    return total_tokens

# =============================================================================
# FLASK APP INITIALIZATION
# =============================================================================

app = Flask(__name__)

# CORS configuration - allow frontend origin
CLIENT_HOST = os.getenv('CLIENT_HOST', 'http://localhost:3000')
CORS(app, origins=[CLIENT_HOST, 'http://localhost:3000', 'https://clinicalchat.vercel.app'])

print("🔌 Connecting to MongoDB Atlas...")

try:
    client = get_mongo_client(serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    print("✓ MongoDB connected successfully!")
except Exception as e:
    raise RuntimeError(f"✗ MongoDB connection failed: {str(e)}")

# Use environment variables for database and collection names
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'clinical_trials')
MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'studies')

db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]
users_collection = db['users']  # For authentication
chat_sessions_collection = db['chat_sessions']  # For chat sessions
user_preferences_collection = db['user_preferences']  # For user preferences
user_settings_collection = db['user_settings']  # For user settings
study_chats_collection = db['study_chats']  # For study-specific chats

# OpenAI setup - prioritize .env file
openai_key = os.environ.get("OPENAI_API_KEY")

if openai_key:
    print(f"✓ Loaded OpenAI API key from .env file (length: {len(openai_key)})")
else:
    # Fallback to file-based keys
    key_paths = ['data1/key/openai_key.txt', 'openai_key.txt']
    for key_path in key_paths:
        try:
            with open(key_path) as f:
                openai_key = f.readline().strip()
                print(f"✓ Loaded OpenAI API key from: {key_path}")
                os.environ["OPENAI_API_KEY"] = openai_key
                break
        except FileNotFoundError:
            continue

if not openai_key:
    print("\n⚠️  ERROR: OpenAI API key not found!")
    print("Please create a .env file with OPENAI_API_KEY=your-key-here")
    exit(1)

openai_client = OpenAI()

# Initialize Gemini (optional)
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    print(f"✓ Gemini API key loaded")
else:
    print("⚠️  Gemini API key not found (optional)")

# Initialize Groq (optional)
groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print(f"✓ Groq API key loaded")
else:
    print("⚠️  Groq API key not found (optional)")

# Initialize ChromaDB for semantic search
# Try to use ChromaDB Cloud first, fallback to other methods
chroma_collection = None
chroma_client = None

try:
    # Check if ChromaDB Cloud is configured (preferred for production)
    chroma_api_key = os.getenv('CHROMA_API_KEY')
    chroma_tenant = os.getenv('CHROMA_TENANT')
    chroma_database = os.getenv('CHROMA_DATABASE', 'clinicalchat')
    
    if chroma_api_key and chroma_tenant:
        # Use ChromaDB Cloud (recommended for production)
        chroma_client = chromadb.CloudClient(
            api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_database
        )
        try:
            chroma_collection = chroma_client.get_or_create_collection(name='clinical_trials_embeddings')
            count = chroma_collection.count()
            print(f"✓ ChromaDB Cloud connected (database: {chroma_database}): {count} embeddings")
        except Exception as e:
            print(f"⚠️  ChromaDB Cloud connection error: {str(e)}")
            print("   Collection 'clinical_trials_embeddings' needs to be created with embeddings")
            chroma_collection = None
    else:
        # Fallback to client-server mode (for self-hosted ChromaDB)
        chroma_host = os.getenv('CHROMA_HOST')
        chroma_port = os.getenv('CHROMA_PORT', '8000')
        
        if chroma_host:
            # Use client-server mode (for production with separate ChromaDB service)
            chroma_auth_token = os.getenv('CHROMA_AUTH_TOKEN')
            
            # Create HttpClient with optional authentication
            if chroma_auth_token:
                chroma_client = chromadb.HttpClient(
                    host=chroma_host, 
                    port=int(chroma_port),
                    headers={"Authorization": f"Bearer {chroma_auth_token}"}
                )
            else:
                chroma_client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
            
            try:
                chroma_collection = chroma_client.get_collection(name='clinical_trials_embeddings')
                print(f"✓ ChromaDB connected to {chroma_host}:{chroma_port}: {chroma_collection.count()} embeddings")
            except Exception as e:
                print(f"⚠️  ChromaDB collection not found: {str(e)}")
                print("   Collection 'clinical_trials_embeddings' needs to be created with embeddings")
                chroma_collection = None
        else:
            # Use persistent client mode (local or with writable path)
            # Try multiple paths: custom path, /tmp (production writable), then local
            chroma_path = os.getenv('CHROMADB_PATH')
            is_production = os.getenv('FLASK_ENV') == 'production' or os.getenv('NODE_ENV') == 'production'
            
            if not chroma_path:
                # In production, use /tmp which is writable (ephemeral but works)
                # For local development, use ./chromadb_data
                if is_production:
                    chroma_path = '/tmp/chromadb_data'
                    print(f"🔧 Using /tmp for ChromaDB (production mode)")
                    
                    # Try to copy embeddings from read-only location if available
                    readonly_paths = ['./chromadb_data', '/app/chromadb_data', '/chromadb_data']
                    for readonly_path in readonly_paths:
                        if os.path.exists(readonly_path) and os.path.isdir(readonly_path):
                            try:
                                if os.path.exists(chroma_path):
                                    shutil.rmtree(chroma_path)
                                shutil.copytree(readonly_path, chroma_path)
                                print(f"✓ Copied ChromaDB data from {readonly_path} to {chroma_path}")
                                break
                            except Exception as copy_error:
                                print(f"⚠️  Could not copy from {readonly_path}: {str(copy_error)}")
                                continue
                else:
                    chroma_path = './chromadb_data'
            
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            try:
                chroma_collection = chroma_client.get_collection(name='clinical_trials_embeddings')
                print(f"✓ ChromaDB loaded from {chroma_path}: {chroma_collection.count()} embeddings")
            except Exception as e:
                print(f"⚠️  ChromaDB collection not found at {chroma_path}: {str(e)}")
                print("   Collection 'clinical_trials_embeddings' needs to be created with embeddings")
                print("   Run generate_embeddings.py to create embeddings")
                chroma_collection = None
except Exception as e:
    print(f"⚠️  ChromaDB initialization failed: {str(e)}")
    print("   Semantic search will be disabled. To enable:")
    print("   - Option 1: Set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE for ChromaDB Cloud (recommended)")
    print("   - Option 2: Set CHROMA_HOST and CHROMA_PORT for client-server mode")
    print("   - Option 3: Set CHROMADB_PATH to a writable directory")
    chroma_collection = None

# =============================================================================
# MULTI-LLM HELPER FUNCTION
# =============================================================================

def call_llm(model_name, system_message, user_message, max_tokens=2000):
    """
    Unified function to call different LLM providers
    Returns: (response_text, model_info)
    """
    try:
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
            if not gemini_api_key:
                raise Exception("Gemini API key not configured. Add GEMINI_API_KEY to .env file")

            model = genai.GenerativeModel('gemini-pro')
            prompt = f"{system_message}\n\n{user_message}"
            response = model.generate_content(prompt)
            return response.text, "Google Gemini Pro"

        elif model_name == "groq":
            if not groq_api_key:
                raise Exception("Groq API key not configured. Add GROQ_API_KEY to .env file")

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

    except Exception as e:
        raise Exception(f"Error calling {model_name}: {str(e)}")

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/agentic-demo')
def agentic_demo():
    """Render the agentic AI features demo page"""
    return render_template('agentic_demo.html')


@app.route('/api/interventions', methods=['GET'])
def get_interventions():
    """Get list of unique interventions from database"""
    try:
        pipeline = [
            {'$unwind': '$interventions'},
            {'$group': {'_id': '$interventions'}},
            {'$sort': {'_id': 1}},
            {'$limit': 500}
        ]

        interventions = collection.aggregate(pipeline)
        intervention_list = [doc['_id'] for doc in interventions if doc['_id']]

        return jsonify({'interventions': intervention_list})
    except Exception as e:
        print(f"Error fetching interventions: {str(e)}")
        return jsonify({'interventions': []})


# =============================================================================
# AUTHENTICATION ROUTES
# =============================================================================

@app.route('/api/auth/sign-up', methods=['POST'])
def signup():
    """User signup - simple username/password"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')

        # Validation
        if not username or len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400

        if not password or len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400

        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'}), 400

        # Check if username already exists
        existing_user = users_collection.find_one({'username': username})
        if existing_user:
            return jsonify({'success': False, 'message': 'Username is already taken'}), 409

        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create user
        user = {
            'username': username,
            'password_hash': password_hash,
            'created_at': datetime.utcnow(),
            'status': 'active'
        }

        result = users_collection.insert_one(user)
        user_id = str(result.inserted_id)

        # Generate simple token (just a random string for demo)
        token = secrets.token_urlsafe(32)

        return jsonify({
            'success': True,
            'message': 'Account created successfully!',
            'data': {
                'id': user_id,
                'username': username,
                'is_guest': False
            },
            'token': token
        }), 200

    except Exception as e:
        print(f"Signup error: {str(e)}")
        return jsonify({'success': False, 'message': f'Signup failed: {str(e)}'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login - simple username/password"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400

        # Find user
        user = users_collection.find_one({'username': username})
        if not user:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

        # Check password
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

        # Check if active
        if user.get('status') != 'active':
            return jsonify({'success': False, 'message': 'User account is not active'}), 401

        # Generate token
        token = secrets.token_urlsafe(32)

        return jsonify({
            'success': True,
            'message': 'Successfully logged in',
            'data': {
                'id': str(user['_id']),
                'username': user['username'],
                'is_guest': False
            },
            'token': token
        }), 200

    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'success': False, 'message': f'Login failed: {str(e)}'}), 500


@app.route('/api/auth/guest', methods=['POST'])
def guest():
    """Guest mode - no authentication required"""
    return jsonify({
        'success': True,
        'message': 'Continuing as Guest',
        'data': {
            'id': None,
            'username': 'Guest',
            'is_guest': True
        }
    }), 200


@app.route('/api/search', methods=['POST'])
def search_studies():
    """Search for clinical trials based on filters"""
    filters = request.json
    
    # Extract session ID if provided
    session_id = filters.get('sessionId')
    
    # Handle intervention - convert string to array if needed
    if filters.get('intervention'):
        if isinstance(filters['intervention'], str):
            # Split comma-separated values
            filters['intervention'] = [i.strip() for i in filters['intervention'].split(',') if i.strip()]
        elif not isinstance(filters['intervention'], list):
            filters['intervention'] = []
    
    use_semantic = filters.get('useSemanticSearch', False)

    # If semantic search is requested and available
    if use_semantic and chroma_collection is not None:
        return semantic_search_studies(filters)

    # Otherwise use traditional keyword search
    query = build_query_from_filters(filters)

    page = filters.get('page', 1)
    per_page = min(filters.get('per_page', 20), 100)
    skip = (page - 1) * per_page

    total = collection.count_documents(query)
    results = list(collection.find(query).skip(skip).limit(per_page))

    # Convert MongoDB documents to simplified format for frontend
    simplified_results = []
    for result in results:
        result['_id'] = str(result['_id'])
        # Wrap in expected structure for frontend
        simplified = {
            'protocolSection': {
                'identificationModule': {
                    'nctId': result.get('nct_id', 'N/A'),
                    'briefTitle': result.get('title', 'No title')
                },
                'statusModule': {
                    'overallStatus': result.get('status', 'UNKNOWN')
                },
                'designModule': {
                    'studyType': 'INTERVENTIONAL',
                    'phases': []
                },
                'sponsorCollaboratorsModule': {
                    'leadSponsor': {'name': 'N/A'}
                }
            },
            'hasResults': False,
            '_original': result  # Keep original data
        }
        simplified_results.append(simplified)

    # Handle session management
    session_info = None
    if session_id:
        # Update existing session
        chat_sessions_collection.update_one(
            {'_id': session_id},
            {
                '$set': {
                    'last_filters': filters,
                    'updated_at': datetime.now().isoformat()
                }
            }
        )
        session = chat_sessions_collection.find_one({'_id': session_id})
        if session:
            session_info = {
                'id': session['_id'],
                'title': session.get('title', 'Search Session'),
                'description': session.get('description', '')
            }
    
    response_data = {
        'success': True,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
        'results': simplified_results,
        'searchType': 'keyword'
    }
    
    if session_info:
        response_data['sessionInfo'] = session_info
    
    return jsonify(response_data)


def semantic_search_studies(filters):
    """Semantic search using RAG/embeddings"""
    try:
        # Get the condition search query for semantic search
        condition = filters.get('condition', '')
        if not condition:
            return jsonify({'error': 'Please enter a condition for semantic search'}), 400

        # Generate embedding for the search query
        query_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[condition]
        )
        query_embedding = query_response.data[0].embedding

        # Get pagination settings
        page = filters.get('page', 1)
        per_page = min(filters.get('per_page', 20), 100)

        # Query ChromaDB for similar trials
        # Get more results than needed to allow for filtering
        n_results = min(page * per_page + 100, 1000)

        chroma_results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Get NCT IDs from semantic search
        nct_ids = chroma_results['ids'][0] if chroma_results['ids'] else []

        if not nct_ids:
            return jsonify({
                'total': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0,
                'results': [],
                'searchType': 'semantic'
            })

        # Build query with semantic results + other filters
        query = {'nct_id': {'$in': nct_ids}}

        # Apply additional filters
        if filters.get('status') and len(filters['status']) > 0:
            query['status'] = {'$in': filters['status']}
        if filters.get('intervention') and len(filters['intervention']) > 0:
            query['interventions'] = {'$in': filters['intervention']}

        # Get matching studies from MongoDB
        total = collection.count_documents(query)

        # Apply pagination
        skip = (page - 1) * per_page
        results = list(collection.find(query).skip(skip).limit(per_page))

        # Convert to frontend format
        simplified_results = []
        for result in results:
            result['_id'] = str(result['_id'])
            simplified = {
                'protocolSection': {
                    'identificationModule': {
                        'nctId': result.get('nct_id', 'N/A'),
                        'briefTitle': result.get('title', 'No title')
                    },
                    'statusModule': {
                        'overallStatus': result.get('status', 'UNKNOWN')
                    },
                    'designModule': {
                        'studyType': 'INTERVENTIONAL',
                        'phases': []
                    },
                    'sponsorCollaboratorsModule': {
                        'leadSponsor': {'name': 'N/A'}
                    }
                },
                'hasResults': False,
                '_original': result
            }
            simplified_results.append(simplified)

        return jsonify({
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page,
            'results': simplified_results,
            'searchType': 'semantic'
        })

    except Exception as e:
        print(f"Semantic search error: {str(e)}")
        return jsonify({'error': f'Semantic search failed: {str(e)}'}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for individual study"""
    data = request.json
    nct_id = data.get('nctId')
    question = data.get('question')
    
    study = collection.find_one({'nct_id': nct_id})
    if not study:
        return jsonify({'error': 'Study not found'}), 404
    
    study_copy = {k: v for k, v in study.items() if k != '_id'}
    study_context = json.dumps(study_copy, indent=2)
    
    system_message = f"""You are a clinical trials expert. Answer questions about this study.

STUDY DATA:
{study_context}

Instructions:
- Answer based only on the provided data
- Be precise and concise"""
    
    try:
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_list,
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = completion.choices[0].message.content
        answer_html = markdown.markdown(answer, extensions=['extra', 'nl2br'])
        
        return jsonify({'answer': answer_html})
        
    except Exception as e:
        return jsonify({'error': f'AI error: {str(e)}'}), 500


@app.route('/api/chat-stream', methods=['POST'])
def chat_stream():
    """Streaming chat for individual study (SSE)"""
    data = request.json
    nct_id = data.get('nctId')
    question = data.get('question')
    
    study = collection.find_one({'nct_id': nct_id})
    if not study:
        return jsonify({'error': 'Study not found'}), 404
    
    study_copy = {k: v for k, v in study.items() if k != '_id'}
    study_context = json.dumps(study_copy, indent=2)
    
    system_message = f"""You are a clinical trials expert. Answer questions about this study.

STUDY DATA:
{study_context}

Instructions:
- Answer based only on the provided data
- Be precise and concise"""

    def generate():
        full_answer = ""
        try:
            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
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
            # Final answer as HTML
            answer_html = markdown.markdown(full_answer, extensions=['extra', 'nl2br'])
            yield f"data: {json.dumps({'type': 'done', 'answer': answer_html})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), headers=headers)


@app.route('/api/chat-all', methods=['POST'])
def chat_all():
    """Chat endpoint for all filtered studies"""
    data = request.json
    filters = data.get('filters', {})
    question = data.get('question', '')
    selected_model = data.get('model', 'openai')  # Get selected model

    query = build_query_from_filters(filters)
    total_count = collection.count_documents(query)

    limit = min(total_count, 100)
    studies = list(collection.find(query).limit(limit))

    # Remove _id from studies
    processed_studies = []
    for study in studies:
        study_copy = {k: v for k, v in study.items() if k != '_id'}
        processed_studies.append(study_copy)

    studies_json = json.dumps(processed_studies, indent=1)

    system_message = f"""You are a clinical trials research analyst.

DATASET: {len(processed_studies)} clinical trial studies
{studies_json}

Answer the question by analyzing the provided studies. Provide statistics and insights."""

    try:
        # Call the selected LLM
        answer, model_info = call_llm(selected_model, system_message, question, max_tokens=2000)

        # Convert to HTML
        answer_html = markdown.markdown(answer, extensions=['extra', 'nl2br', 'tables'])

        mode_info = f"Analyzing {len(processed_studies)} studies using {model_info}"

        return jsonify({
            'answer': answer_html,
            'info': mode_info
        })

    except Exception as e:
        print(f"Error in chat-all: {str(e)}")
        return jsonify({'error': f'AI error: {str(e)}'}), 500


@app.route('/api/chat-all-stream', methods=['POST'])
def chat_all_stream():
    """Streaming chat for all filtered studies (SSE)"""
    data = request.json
    filters = data.get('filters', {})
    question = data.get('question', '')
    advanced_mode = data.get('advancedMode', False)

    query = build_query_from_filters(filters)
    total_count = collection.count_documents(query)
    limit = min(total_count, 100)
    studies = list(collection.find(query).limit(limit))

    processed_studies = []
    for study in studies:
        study_copy = {k: v for k, v in study.items() if k != '_id'}
        processed_studies.append(study_copy)

    studies_json = json.dumps(processed_studies, indent=1)

    system_message = f"""You are a clinical trials research analyst.

DATASET: {len(processed_studies)} clinical trial studies
{studies_json}

Answer the question by analyzing the provided studies. Provide statistics and insights."""

    def generate():
        full_answer = ""
        try:
            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
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
            answer_html = markdown.markdown(full_answer, extensions=['extra', 'nl2br', 'tables'])
            yield f"data: {json.dumps({'type': 'done', 'answer': answer_html})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), headers=headers)


@app.route('/api/generate-protocol-report', methods=['POST'])
def generate_protocol_report():
    """Generate a protocol research report based on similar trials"""
    data = request.json
    condition = data.get('condition', '')
    intervention = data.get('intervention', '')
    session_id = data.get('sessionId')
    format_type = data.get('format', 'styled')

    if not condition:
        return jsonify({'success': False, 'error': 'Condition is required'}), 400

    # Build query to find similar trials
    query = {}
    if condition:
        query['conditions'] = {'$regex': condition, '$options': 'i'}
    if intervention:
        query['interventions'] = {'$regex': intervention, '$options': 'i'}

    # Find similar trials
    total_count = collection.count_documents(query)
    if total_count == 0:
        return jsonify({'success': False, 'error': f'No trials found for {condition}'}), 404

    # Limit to 100 trials for analysis
    limit = min(total_count, 100)
    similar_trials = list(collection.find(query).limit(limit))

    # Prepare data for AI
    trials_summary = []
    for trial in similar_trials:
        trials_summary.append({
            'nct_id': trial.get('nct_id'),
            'title': trial.get('title'),
            'status': trial.get('status'),
            'conditions': trial.get('conditions', []),
            'interventions': trial.get('interventions', []),
            'summary': trial.get('summary', '')[:500]  # Limit summary length
        })

    trials_json = json.dumps(trials_summary, indent=1)

    # AI prompt for protocol report generation
    system_message = f"""You are a clinical trial protocol design expert. Generate a comprehensive protocol research report with detailed statistics.

DATASET: {len(trials_summary)} similar clinical trials
Condition: {condition}
{f"Intervention: {intervention}" if intervention else ""}

TRIALS DATA:
{trials_json}

Generate a detailed protocol research report with these sections. Include QUANTITATIVE STATISTICS in every section:

1. ELIGIBILITY CRITERIA RECOMMENDATIONS
   - Analyze the most common inclusion criteria across trials with percentages (e.g., "Age ≥18: 85% of trials")
   - Analyze the most common exclusion criteria with frequencies
   - Provide specific recommendations with statistical support
   - Include: prevalence (%), counts, and ranges where applicable

2. STUDY DESIGN PATTERNS
   - Identify common study designs with distribution (e.g., "Randomized: 60%, Single-arm: 25%")
   - Typical duration ranges with median and mean values
   - Common sample sizes: provide min, max, median, and quartiles
   - Phase distribution with percentages
   - Include specific counts and statistical breakdowns

3. KEY INTERVENTIONS ANALYSIS
   - Most common interventions with usage percentages
   - Typical dosing/treatment approaches with frequency data
   - Combination vs monotherapy statistics
   - Include prevalence data for each intervention type

4. SIMILAR TRIALS REFERENCE
   - List top 5-8 most relevant trial NCT IDs with brief descriptions
   - Include trial phase, status, and key characteristics
   - Format NCT IDs exactly as: NCT00000000 (they will be converted to links)

5. STUDY LIMITATIONS
   - Discuss data completeness and quality issues
   - Scope of analysis: date ranges, trial selection criteria
   - Potential biases in the dataset
   - Recommendations for interpreting these results
   - Statistical limitations and confidence considerations

IMPORTANT:
- Include specific numbers, percentages, and statistical measures in EVERY section
- Use quantitative evidence to support all recommendations
- Provide counts alongside percentages (e.g., "45% (18/40 trials)")
- Format the report professionally with clear sections and bullet points"""

    try:
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate a protocol research report for designing a new {condition} trial{f' using {intervention}' if intervention else ''}."}
        ]

        token_count = count_tokens(message_list, model="gpt-4o")
        print(f"📊 Protocol Report - Token count: {token_count:,} tokens")

        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_list,
            temperature=0.3,
            max_tokens=2500
        )

        report = completion.choices[0].message.content
        report_html = markdown.markdown(report, extensions=['extra', 'nl2br', 'tables'])

        # Convert NCT IDs to clickable links
        nct_pattern = r'(NCT\d{8})'
        report_html = re.sub(
            nct_pattern,
            r'<a href="https://clinicaltrials.gov/study/\1" target="_blank" style="color: #4f46e5; text-decoration: underline;">\1</a>',
            report_html
        )

        # Add metadata header
        header = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">📋 Clinical Trial Protocol Research Report</h2>
            <p style="margin: 5px 0;"><strong>Indication:</strong> {condition}</p>
            {f'<p style="margin: 5px 0;"><strong>Intervention:</strong> {intervention}</p>' if intervention else ''}
            <p style="margin: 5px 0;"><strong>Analysis Based On:</strong> {len(trials_summary)} similar trials (out of {total_count} total)</p>
            <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        """

        full_report = header + report_html
        
        # Handle session management - create or update session
        session_info = None
        if session_id:
            # Update existing session with report
            session = chat_sessions_collection.find_one({'_id': session_id})
            if session:
                if 'reports' not in session:
                    session['reports'] = []
                session['reports'].append({
                    'type': 'protocol',
                    'content': full_report,
                    'created_at': datetime.now().isoformat(),
                    'format': format_type
                })
                chat_sessions_collection.update_one(
                    {'_id': session_id},
                    {'$set': {'reports': session['reports'], 'updated_at': datetime.now().isoformat()}}
                )
                session_info = {
                    'id': session['_id'],
                    'title': session.get('title', 'Protocol Report Session'),
                    'description': session.get('description', '')
                }
        else:
            # Create new session for this report
            new_session_id = secrets.token_urlsafe(16)
            session = {
                '_id': new_session_id,
                'title': f'Protocol Report: {condition}',
                'description': f'Protocol research report for {condition}' + (f' with {intervention}' if intervention else ''),
                'last_filters': {'condition': condition, 'intervention': intervention},
                'messages': [],
                'reports': [{
                    'type': 'protocol',
                    'content': full_report,
                    'created_at': datetime.now().isoformat(),
                    'format': format_type
                }],
                'custom_questions': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            chat_sessions_collection.insert_one(session)
            session_info = {
                'id': new_session_id,
                'title': session['title'],
                'description': session['description']
            }

        response_data = {
            'success': True,
            'report': full_report,
            'metadata': {
                'trials_analyzed': len(trials_summary),
                'total_matching': total_count,
                'condition': condition,
                'intervention': intervention
            }
        }
        
        if session_info:
            response_data['sessionInfo'] = session_info
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Error generating protocol report: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'AI error: {str(e)}'}), 500


@app.route('/api/generate-protocol-report-stream', methods=['POST'])
def generate_protocol_report_stream():
    """Generate protocol report with SSE streaming to avoid timeouts"""
    data = request.json
    condition = data.get('condition', '')
    intervention = data.get('intervention', '')
    session_id = data.get('sessionId')
    format_type = data.get('format', 'styled')

    def error_stream(msg):
        yield f"data: {json.dumps({'type': 'error', 'error': msg})}\n\n"

    if not condition:
        return Response(stream_with_context(error_stream('Condition is required')),
                        content_type='text/event-stream')

    query = {}
    if condition:
        query['conditions'] = {'$regex': condition, '$options': 'i'}
    if intervention:
        query['interventions'] = {'$regex': intervention, '$options': 'i'}

    total_count = collection.count_documents(query)
    if total_count == 0:
        return Response(stream_with_context(error_stream(f'No trials found for {condition}')),
                        content_type='text/event-stream')

    limit = min(total_count, 100)
    similar_trials = list(collection.find(query).limit(limit))

    trials_summary = []
    for trial in similar_trials:
        trials_summary.append({
            'nct_id': trial.get('nct_id'),
            'title': trial.get('title'),
            'status': trial.get('status'),
            'conditions': trial.get('conditions', []),
            'interventions': trial.get('interventions', []),
            'summary': trial.get('summary', '')[:500]
        })

    trials_json = json.dumps(trials_summary, indent=1)

    system_message = f"""You are a clinical trial protocol design expert. Generate a comprehensive protocol research report with detailed statistics.

DATASET: {len(trials_summary)} similar clinical trials
Condition: {condition}
{f"Intervention: {intervention}" if intervention else ""}

TRIALS DATA:
{trials_json}

Generate a detailed protocol research report with these sections. Include QUANTITATIVE STATISTICS in every section:

1. ELIGIBILITY CRITERIA RECOMMENDATIONS
   - Analyze the most common inclusion criteria across trials with percentages (e.g., "Age ≥18: 85% of trials")
   - Analyze the most common exclusion criteria with frequencies
   - Provide specific recommendations with statistical support
   - Include: prevalence (%), counts, and ranges where applicable

2. STUDY DESIGN PATTERNS
   - Identify common study designs with distribution (e.g., "Randomized: 60%, Single-arm: 25%")
   - Typical duration ranges with median and mean values
   - Common sample sizes: provide min, max, median, and quartiles
   - Phase distribution with percentages
   - Include specific counts and statistical breakdowns

3. KEY INTERVENTIONS ANALYSIS
   - Most common interventions with usage percentages
   - Typical dosing/treatment approaches with frequency data
   - Combination vs monotherapy statistics
   - Include prevalence data for each intervention type

4. SIMILAR TRIALS REFERENCE
   - List top 5-8 most relevant trial NCT IDs with brief descriptions
   - Include trial phase, status, and key characteristics
   - Format NCT IDs exactly as: NCT00000000 (they will be converted to links)

5. STUDY LIMITATIONS
   - Discuss data completeness and quality issues
   - Scope of analysis: date ranges, trial selection criteria
   - Potential biases in the dataset
   - Recommendations for interpreting these results
   - Statistical limitations and confidence considerations

IMPORTANT:
- Include specific numbers, percentages, and statistical measures in EVERY section
- Use quantitative evidence to support all recommendations
- Provide counts alongside percentages (e.g., "45% (18/40 trials)")
- Format the report professionally with clear sections and bullet points"""

    def generate():
        full_report_text = ""
        try:
            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Generate a protocol research report for designing a new {condition} trial{f' using {intervention}' if intervention else ''}."}
                ],
                temperature=0.3,
                max_tokens=2500,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_report_text += delta.content
                    yield f"data: {json.dumps({'type': 'content', 'chunk': delta.content})}\n\n"

            # Post-process the complete report
            report_html = markdown.markdown(full_report_text, extensions=['extra', 'nl2br', 'tables'])
            nct_pattern = r'(NCT\d{8})'
            report_html = re.sub(
                nct_pattern,
                r'<a href="https://clinicaltrials.gov/study/\1" target="_blank" style="color: #4f46e5; text-decoration: underline;">\1</a>',
                report_html
            )
            header = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0;">📋 Clinical Trial Protocol Research Report</h2>
                <p style="margin: 5px 0;"><strong>Indication:</strong> {condition}</p>
                {f'<p style="margin: 5px 0;"><strong>Intervention:</strong> {intervention}</p>' if intervention else ''}
                <p style="margin: 5px 0;"><strong>Analysis Based On:</strong> {len(trials_summary)} similar trials (out of {total_count} total)</p>
                <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
            full_html = header + report_html

            # Session management
            session_info = None
            if session_id:
                session = chat_sessions_collection.find_one({'_id': session_id})
                if session:
                    if 'reports' not in session:
                        session['reports'] = []
                    session['reports'].append({
                        'type': 'protocol',
                        'content': full_html,
                        'created_at': datetime.now().isoformat(),
                        'format': format_type
                    })
                    chat_sessions_collection.update_one(
                        {'_id': session_id},
                        {'$set': {'reports': session['reports'], 'updated_at': datetime.now().isoformat()}}
                    )
                    session_info = {
                        'id': session['_id'],
                        'title': session.get('title', 'Protocol Report Session'),
                        'description': session.get('description', '')
                    }
            else:
                new_session_id = secrets.token_urlsafe(16)
                session = {
                    '_id': new_session_id,
                    'title': f'Protocol Report: {condition}',
                    'description': f'Protocol research report for {condition}' + (f' with {intervention}' if intervention else ''),
                    'last_filters': {'condition': condition, 'intervention': intervention},
                    'messages': [],
                    'reports': [{
                        'type': 'protocol',
                        'content': full_html,
                        'created_at': datetime.now().isoformat(),
                        'format': format_type
                    }],
                    'custom_questions': None,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                chat_sessions_collection.insert_one(session)
                session_info = {
                    'id': new_session_id,
                    'title': session['title'],
                    'description': session['description']
                }

            yield f"data: {json.dumps({'type': 'done', 'report': full_html, 'metadata': {'trials_analyzed': len(trials_summary), 'total_matching': total_count, 'condition': condition, 'intervention': intervention}, 'sessionInfo': session_info})}\n\n"

        except Exception as e:
            print(f"Error in protocol report stream: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), headers=headers)


@app.route('/api/generate-chat-report', methods=['POST'])
def generate_chat_report():
    """Generate a report based on a chat session conversation"""
    data = request.json
    session_id = data.get('sessionId', '')
    format_type = data.get('format', 'styled')
    
    if not session_id:
        return jsonify({'success': False, 'error': 'Session ID is required'}), 400
    
    try:
        # Get chat session from database
        session = chat_sessions_collection.find_one({'_id': session_id})
        if not session:
            return jsonify({'success': False, 'error': 'Chat session not found'}), 404
        
        # Extract messages and filters
        messages = session.get('messages', [])
        last_filters = session.get('last_filters', {})
        condition = last_filters.get('condition', 'Unknown')
        intervention = last_filters.get('intervention', '')
        
        # Get studies from the session context
        query = build_query_from_filters(last_filters) if last_filters else {}
        total_count = collection.count_documents(query)
        limit = min(total_count, 50)
        studies = list(collection.find(query).limit(limit))
        
        # Prepare conversation summary
        conversation_summary = []
        for msg in messages[-10:]:  # Last 10 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if len(content) > 500:
                content = content[:500] + '...'
            conversation_summary.append(f"{role.upper()}: {content}")
        
        conversation_text = '\n\n'.join(conversation_summary)
        
        # AI prompt for chat report generation
        system_message = f"""You are a clinical research analyst. Generate a comprehensive research report based on this chat conversation.

CONVERSATION SUMMARY:
{conversation_text}

PRIMARY FOCUS: {condition}
{f"INTERVENTION: {intervention}" if intervention else ""}
STUDIES AVAILABLE: {len(studies)} studies analyzed (out of {total_count} matching)

Generate a detailed research report with these sections:

1. CONVERSATION OVERVIEW
   - Summarize the key questions and topics discussed
   - Highlight main findings from the conversation

2. KEY INSIGHTS
   - Extract the most important insights from the conversation
   - Provide evidence-based conclusions

3. STUDY LANDSCAPE
   - Summarize the relevant clinical trials landscape
   - Provide statistics on study designs, phases, and interventions

4. RECOMMENDATIONS
   - Based on the conversation, provide actionable recommendations
   - Suggest areas for further investigation

5. REFERENCES
   - List relevant NCT IDs mentioned or related to the discussion
   - Format NCT IDs as: NCT00000000

Format the report professionally with clear sections and bullet points."""
        
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate a comprehensive research report based on this chat conversation about {condition}."}
        ]
        
        token_count = count_tokens(message_list, model="gpt-4o")
        print(f"📊 Chat Report - Token count: {token_count:,} tokens")
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_list,
            temperature=0.3,
            max_tokens=2500
        )
        
        report = completion.choices[0].message.content
        report_html = markdown.markdown(report, extensions=['extra', 'nl2br', 'tables'])
        
        # Convert NCT IDs to clickable links
        nct_pattern = r'(NCT\d{8})'
        report_html = re.sub(
            nct_pattern,
            r'<a href="https://clinicaltrials.gov/study/\1" target="_blank" style="color: #4f46e5; text-decoration: underline;">\1</a>',
            report_html
        )
        
        # Add metadata header based on format
        if format_type == 'styled':
            header = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0;">💬 Chat Conversation Research Report</h2>
                <p style="margin: 5px 0;"><strong>Primary Focus:</strong> {condition}</p>
                {f'<p style="margin: 5px 0;"><strong>Intervention:</strong> {intervention}</p>' if intervention else ''}
                <p style="margin: 5px 0;"><strong>Messages Analyzed:</strong> {len(messages)}</p>
                <p style="margin: 5px 0;"><strong>Studies Referenced:</strong> {len(studies)} studies (out of {total_count} matching)</p>
                <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        elif format_type == 'professional':
            header = f"""
            <div style="border-bottom: 3px solid #667eea; padding-bottom: 15px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0; color: #333;">Chat Conversation Research Report</h2>
                <p style="margin: 5px 0; color: #666;"><strong>Primary Focus:</strong> {condition}</p>
                {f'<p style="margin: 5px 0; color: #666;"><strong>Intervention:</strong> {intervention}</p>' if intervention else ''}
                <p style="margin: 5px 0; color: #666;"><strong>Messages:</strong> {len(messages)} | <strong>Studies:</strong> {len(studies)}/{total_count}</p>
                <p style="margin: 5px 0; font-size: 12px; color: #999;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        else:  # standard
            header = f"""
            <div style="margin-bottom: 20px;">
                <h2>Chat Conversation Research Report</h2>
                <p><strong>Primary Focus:</strong> {condition}</p>
                {f'<p><strong>Intervention:</strong> {intervention}</p>' if intervention else ''}
                <p><strong>Messages:</strong> {len(messages)} | <strong>Studies:</strong> {len(studies)}/{total_count}</p>
                <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        
        full_report = header + report_html
        
        # Save report to session
        if 'reports' not in session:
            session['reports'] = []
        session['reports'].append({
            'type': 'chat',
            'content': full_report,
            'created_at': datetime.now().isoformat(),
            'format': format_type
        })
        chat_sessions_collection.update_one(
            {'_id': session_id},
            {'$set': {'reports': session['reports']}}
        )
        
        return jsonify({
            'success': True,
            'report': full_report,
            'metadata': {
                'messages_count': len(messages),
                'studies_analyzed': len(studies),
                'total_matching': total_count,
                'condition': condition,
                'intervention': intervention if intervention else None
            }
        })
        
    except Exception as e:
        print(f"Error generating chat report: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'AI error: {str(e)}'}), 500


@app.route('/api/generate-study-chat-report', methods=['POST'])
def generate_study_chat_report():
    """Generate a report based on a specific study chat conversation"""
    data = request.json
    study_id = data.get('studyId', '')
    chat_session_id = data.get('chatSessionId', '')
    format_type = data.get('format', 'styled')
    
    if not study_id:
        return jsonify({'success': False, 'error': 'Study ID is required'}), 400
    
    try:
        # Get study from database
        study = collection.find_one({'nct_id': study_id})
        if not study:
            return jsonify({'success': False, 'error': 'Study not found'}), 404
        
        study_title = study.get('title', 'Unknown Study')
        
        # Get study chat messages
        query = {'study_id': study_id}
        if chat_session_id:
            query['chat_session_id'] = chat_session_id
        
        study_chat = study_chats_collection.find_one(query)
        messages = study_chat.get('messages', []) if study_chat else []
        
        # Prepare conversation summary
        conversation_summary = []
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if len(content) > 500:
                content = content[:500] + '...'
            conversation_summary.append(f"{role.upper()}: {content}")
        
        conversation_text = '\n\n'.join(conversation_summary)
        
        # Prepare study summary
        study_summary = {
            'nct_id': study.get('nct_id'),
            'title': study.get('title'),
            'status': study.get('status'),
            'conditions': study.get('conditions', []),
            'interventions': study.get('interventions', []),
            'summary': study.get('summary', '')[:1000]
        }
        study_json = json.dumps(study_summary, indent=2)
        
        # AI prompt for study chat report generation
        system_message = f"""You are a clinical research analyst. Generate a comprehensive report based on the conversation about this specific clinical trial.

STUDY INFORMATION:
{study_json}

CONVERSATION SUMMARY:
{conversation_text}

Generate a detailed report with these sections:

1. STUDY OVERVIEW
   - Summarize the key aspects of this clinical trial
   - Highlight the study design, intervention, and objectives

2. CONVERSATION INSIGHTS
   - Summarize the key questions and topics discussed about this study
   - Extract important insights from the conversation

3. DETAILED ANALYSIS
   - Provide in-depth analysis based on the conversation
   - Address specific questions or concerns raised

4. KEY FINDINGS
   - List the most important findings discussed
   - Provide evidence-based conclusions

5. CLINICAL IMPLICATIONS
   - Discuss the clinical relevance of this study
   - Suggest practical applications or considerations

Format the report professionally with clear sections and bullet points.
Always reference the study by its NCT ID: {study_id}"""
        
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate a comprehensive report about study {study_id} based on our conversation."}
        ]
        
        token_count = count_tokens(message_list, model="gpt-4o")
        print(f"📊 Study Chat Report - Token count: {token_count:,} tokens")
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_list,
            temperature=0.3,
            max_tokens=2500
        )
        
        report = completion.choices[0].message.content
        report_html = markdown.markdown(report, extensions=['extra', 'nl2br', 'tables'])
        
        # Convert NCT IDs to clickable links
        nct_pattern = r'(NCT\d{8})'
        report_html = re.sub(
            nct_pattern,
            r'<a href="https://clinicaltrials.gov/study/\1" target="_blank" style="color: #4f46e5; text-decoration: underline;">\1</a>',
            report_html
        )
        
        # Add metadata header based on format
        if format_type == 'styled':
            header = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0;">🔬 Study Chat Research Report</h2>
                <p style="margin: 5px 0;"><strong>Study:</strong> {study_id}</p>
                <p style="margin: 5px 0;"><strong>Title:</strong> {study_title}</p>
                <p style="margin: 5px 0;"><strong>Messages Analyzed:</strong> {len(messages)}</p>
                <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        elif format_type == 'professional':
            header = f"""
            <div style="border-bottom: 3px solid #667eea; padding-bottom: 15px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0; color: #333;">Study Chat Research Report</h2>
                <p style="margin: 5px 0; color: #666;"><strong>Study:</strong> {study_id}</p>
                <p style="margin: 5px 0; color: #666;"><strong>Title:</strong> {study_title}</p>
                <p style="margin: 5px 0; color: #666;"><strong>Messages:</strong> {len(messages)}</p>
                <p style="margin: 5px 0; font-size: 12px; color: #999;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        else:  # standard
            header = f"""
            <div style="margin-bottom: 20px;">
                <h2>Study Chat Research Report</h2>
                <p><strong>Study:</strong> {study_id}</p>
                <p><strong>Title:</strong> {study_title}</p>
                <p><strong>Messages:</strong> {len(messages)}</p>
                <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        
        full_report = header + report_html
        
        # Save report to study chat
        if study_chat:
            if 'reports' not in study_chat:
                study_chat['reports'] = []
            study_chat['reports'].append({
                'type': 'study_chat',
                'content': full_report,
                'created_at': datetime.now().isoformat(),
                'format': format_type
            })
            study_chats_collection.update_one(
                query,
                {'$set': {'reports': study_chat['reports']}}
            )
        
        return jsonify({
            'success': True,
            'report': full_report,
            'metadata': {
                'messages_count': len(messages),
                'study_id': study_id,
                'study_title': study_title,
                'report_type': 'study_chat'
            }
        })
        
    except Exception as e:
        print(f"Error generating study chat report: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'AI error: {str(e)}'}), 500


@app.route('/api/compare-trials', methods=['POST'])
def compare_trials():
    """Multi-agent comparison of multiple trials"""
    from agentic_comparison import multi_agent_comparison

    data = request.json
    nct_ids = data.get('nctIds', [])

    if not nct_ids or len(nct_ids) < 2:
        return jsonify({'error': 'At least 2 NCT IDs required'}), 400

    # Get trials from database
    trials = []
    for nct_id in nct_ids[:5]:  # Limit to 5 trials
        trial = collection.find_one({'nct_id': nct_id})
        if trial:
            trial_copy = {k: v for k, v in trial.items() if k != '_id'}
            trials.append(trial_copy)

    if len(trials) < 2:
        return jsonify({'error': 'Not enough valid trials found'}), 404

    try:
        print(f"\n🔬 Starting multi-agent comparison of {len(trials)} trials...")

        # Run comparison
        result = multi_agent_comparison(trials)

        # Format for frontend
        comparisons_html = {}
        for key, content in result['comparisons'].items():
            comparisons_html[key] = markdown.markdown(content, extensions=['extra', 'nl2br'])

        synthesis_html = markdown.markdown(result['strategic_synthesis'], extensions=['extra', 'nl2br', 'tables'])

        return jsonify({
            'success': True,
            'trials': result['trials'],
            'comparisons': comparisons_html,
            'strategic_synthesis': synthesis_html,
            'metadata': result['metadata']
        })

    except Exception as e:
        print(f"Error in trial comparison: {str(e)}")
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500


@app.route('/api/agentic-search', methods=['POST'])
def agentic_search():
    """Enhanced search using multiple AI agents"""
    from agentic_search import agentic_search_enhancement

    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    try:
        # Run agentic search enhancement
        result = agentic_search_enhancement(query)

        return jsonify({
            'success': True,
            'original_query': result['original_query'],
            'terminology_expansion': result['terminology_expansion'],
            'search_strategy': result['search_strategy'],
            'enhanced_search_terms': result['enhanced_search_terms']
        })

    except Exception as e:
        print(f"Error in agentic search: {str(e)}")
        return jsonify({'error': f'Agentic search failed: {str(e)}'}), 500


@app.route('/api/multi-agent-analysis', methods=['POST'])
def multi_agent_protocol_analysis():
    """Multi-agent analysis of a clinical trial"""
    from agentic_analysis import multi_agent_analysis

    data = request.json
    nct_id = data.get('nctId')

    if not nct_id:
        return jsonify({'error': 'NCT ID is required'}), 400

    # Get trial from database
    trial = collection.find_one({'nct_id': nct_id})
    if not trial:
        return jsonify({'error': 'Trial not found'}), 404

    # Remove _id for JSON serialization
    trial_copy = {k: v for k, v in trial.items() if k != '_id'}

    try:
        print(f"\n🤖 Starting multi-agent analysis for {nct_id}...")

        # Run multi-agent analysis
        result = multi_agent_analysis(trial_copy)

        # Format for frontend
        analyses_html = []
        for analysis in result['agent_analyses']:
            analysis_html = markdown.markdown(analysis['analysis'], extensions=['extra', 'nl2br'])
            analyses_html.append({
                'agent': analysis['agent'],
                'focus_areas': analysis['focus_areas'],
                'content': analysis_html
            })

        executive_html = markdown.markdown(result['executive_summary'], extensions=['extra', 'nl2br', 'tables'])

        return jsonify({
            'success': True,
            'trial': result['trial'],
            'agent_analyses': analyses_html,
            'executive_summary': executive_html,
            'metadata': result['metadata']
        })

    except Exception as e:
        print(f"Error in multi-agent analysis: {str(e)}")
        return jsonify({'error': f'Multi-agent analysis failed: {str(e)}'}), 500


# =============================================================================
# NEW CAPSTONE FEATURES - AMENDMENT, PATTERNS, SOA
# =============================================================================

@app.route('/api/amendment-risk', methods=['POST'])
def amendment_risk_prediction():
    """Predict amendment risk for a clinical trial"""
    from agentic_amendment import amendment_risk_analysis

    data = request.json
    nct_id = data.get('nctId')

    if not nct_id:
        return jsonify({'error': 'NCT ID is required'}), 400

    # Get trial from database
    trial = collection.find_one({'nct_id': nct_id})
    if not trial:
        return jsonify({'error': 'Trial not found'}), 404

    try:
        print(f"\n⚠️ Starting amendment risk analysis for {nct_id}...")

        # Extract relevant trial data
        protocol = trial.get('protocolSection', {})
        identification = protocol.get('identificationModule', {})
        design = protocol.get('designModule', {})
        eligibility = protocol.get('eligibilityModule', {})
        outcomes = protocol.get('outcomesModule', {})

        trial_data = {
            'nct_id': trial.get('nct_id', 'N/A'),
            'title': identification.get('briefTitle', 'N/A'),
            'phase': ', '.join(design.get('phases', ['N/A'])),
            'status': trial.get('status', 'N/A'),
            'enrollment': design.get('enrollmentInfo', {}).get('count', 'N/A'),
            'eligibility': eligibility.get('eligibilityCriteria', 'Not specified'),
            'outcomes': f"Primary: {outcomes.get('primaryOutcomes', [{}])[0].get('measure', 'N/A')}\nSecondary: {', '.join([o.get('measure', 'N/A') for o in outcomes.get('secondaryOutcomes', [])[:3]])}",
            'design': f"{design.get('studyType', 'N/A')} | {design.get('designInfo', {}).get('allocation', 'N/A')} | {design.get('designInfo', {}).get('maskingInfo', {}).get('masking', 'N/A')}"
        }

        # Run amendment risk analysis
        result = amendment_risk_analysis(trial_data)

        if not result['success']:
            return jsonify({'error': result.get('error', 'Analysis failed')}), 500

        return jsonify({
            'success': True,
            'trial': result['trial'],
            # Pass through rich and plain variants
            'agent_analyses': result['agent_analyses'],
            'risk_assessment': result.get('risk_assessment'),  # backward-compatible (HTML)
            'risk_assessment_raw': result.get('risk_assessment_raw'),
            'risk_assessment_html': result.get('risk_assessment_html'),
            'risk_assessment_text': result.get('risk_assessment_text')
        })

    except Exception as e:
        print(f"Error in amendment risk analysis: {str(e)}")
        return jsonify({'error': f'Amendment risk analysis failed: {str(e)}'}), 500


@app.route('/api/design-patterns', methods=['POST'])
def design_pattern_discovery_endpoint():
    """Discover design patterns across similar trials"""
    from agentic_patterns import design_pattern_discovery

    data = request.json
    condition = data.get('condition')
    phase = data.get('phase')
    intervention_type = data.get('interventionType')

    if not condition:
        return jsonify({'error': 'Condition is required'}), 400

    try:
        print(f"\n🔍 Starting design pattern discovery for {condition}...")

        # Run pattern discovery
        result = design_pattern_discovery(condition, phase, intervention_type)

        if not result['success']:
            return jsonify({'error': result.get('error', 'Analysis failed')}), 500

        return jsonify({
            'success': True,
            'query': result['query'],
            'agent_analyses': result['agent_analyses'],
            'strategic_insights': result.get('strategic_insights'),  # backward-compatible (HTML)
            'strategic_insights_raw': result.get('strategic_insights_raw'),
            'strategic_insights_html': result.get('strategic_insights_html'),
            'strategic_insights_text': result.get('strategic_insights_text'),
            'trials_by_phase': result['trials_summary']['trials_by_phase']
        })

    except Exception as e:
        print(f"Error in design pattern discovery: {str(e)}")
        return jsonify({'error': f'Design pattern discovery failed: {str(e)}'}), 500


@app.route('/api/soa-composer', methods=['POST'])
def soa_composer_endpoint():
    """Generate Schedule of Assessments"""
    from agentic_soa import soa_composer

    data = request.json
    condition = data.get('condition')
    phase = data.get('phase')
    intervention_type = data.get('interventionType')

    if not condition:
        return jsonify({'error': 'Condition is required'}), 400

    try:
        print(f"\n📋 Starting SoA composition for {condition}...")

        # Run SoA composition
        result = soa_composer(condition, phase, intervention_type)

        if not result['success']:
            return jsonify({'error': result.get('error', 'Analysis failed')}), 500

        return jsonify({
            'success': True,
            'query': result['query'],
            'agent_analyses': result['agent_analyses'],
            'complete_soa': result.get('complete_soa'),  # backward-compatible (HTML)
            'complete_soa_raw': result.get('complete_soa_raw'),
            'complete_soa_html': result.get('complete_soa_html'),
            'complete_soa_text': result.get('complete_soa_text'),
            'reference_trials': result['reference_trials']
        })

    except Exception as e:
        print(f"Error in SoA composition: {str(e)}")
        return jsonify({'error': f'SoA composition failed: {str(e)}'}), 500


@app.route('/api/documents', methods=['POST'])
def add_documents():
    """Add documents to ChromaDB Cloud collection"""
    try:
        request_body = request.get_json()
        ids = request_body.get('ids')
        documents = request_body.get('documents')
        metadatas = request_body.get('metadatas', [])
        embeddings = request_body.get('embeddings')  # Optional
        
        if not ids or not documents:
            return jsonify({'error': 'ids and documents are required'}), 400
        
        if len(ids) != len(documents):
            return jsonify({'error': 'ids and documents must have the same length'}), 400
        
        # Use ChromaDB Cloud if available
        if chroma_collection is None:
            return jsonify({'error': 'ChromaDB not available. Please configure CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE'}), 503
        
        # Prepare add parameters
        add_params = {
            'ids': ids,
            'documents': documents
        }
        
        if metadatas and len(metadatas) == len(ids):
            add_params['metadatas'] = metadatas
        
        if embeddings and len(embeddings) == len(ids):
            add_params['embeddings'] = embeddings
        
        # Add to collection
        chroma_collection.add(**add_params)
        
        return jsonify({
            'message': 'Documents added successfully',
            'ids': ids,
            'count': len(ids)
        }), 200
        
    except Exception as e:
        print(f"Error adding documents: {str(e)}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# CHAT SESSIONS API
# =============================================================================

@app.route('/api/chat-sessions', methods=['GET'])
def get_chat_sessions():
    """Get all chat sessions for the current user"""
    try:
        # TODO: Add user authentication and filter by user_id
        sessions = list(chat_sessions_collection.find().sort('created_at', -1))
        
        # Convert ObjectId to string
        for session in sessions:
            if '_id' in session:
                session['id'] = str(session['_id'])
                del session['_id']
        
        return jsonify({'success': True, 'sessions': sessions})
    except Exception as e:
        print(f"Error getting chat sessions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat-sessions', methods=['POST'])
def create_chat_session():
    """Create a new chat session"""
    try:
        data = request.json or {}
        
        session = {
            '_id': secrets.token_urlsafe(16),
            'title': data.get('title', 'New Chat Session'),
            'description': data.get('description', ''),
            'last_filters': data.get('last_filters', {}),
            'messages': [],
            'reports': [],
            'custom_questions': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        chat_sessions_collection.insert_one(session)
        
        # Convert for response
        session['id'] = session['_id']
        del session['_id']
        
        return jsonify({'success': True, 'session': session})
    except Exception as e:
        print(f"Error creating chat session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat-sessions/<session_id>', methods=['GET'])
def get_chat_session(session_id):
    """Get a specific chat session"""
    try:
        session = chat_sessions_collection.find_one({'_id': session_id})
        
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        # Convert for response
        session['id'] = session['_id']
        del session['_id']
        
        return jsonify({'success': True, 'session': session})
    except Exception as e:
        print(f"Error getting chat session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat-sessions/<session_id>', methods=['PATCH'])
def update_chat_session(session_id):
    """Update a chat session"""
    try:
        data = request.json or {}
        
        update_data = {
            'updated_at': datetime.now().isoformat()
        }
        
        if 'title' in data:
            update_data['title'] = data['title']
        if 'description' in data:
            update_data['description'] = data['description']
        if 'last_filters' in data:
            update_data['last_filters'] = data['last_filters']
        if 'custom_questions' in data:
            update_data['custom_questions'] = data['custom_questions']
        
        result = chat_sessions_collection.update_one(
            {'_id': session_id},
            {'$set': update_data}
        )
        
        if result.matched_count == 0:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        # Get updated session
        session = chat_sessions_collection.find_one({'_id': session_id})
        session['id'] = session['_id']
        del session['_id']
        
        return jsonify({'success': True, 'session': session})
    except Exception as e:
        print(f"Error updating chat session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat-sessions/<session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    """Delete a chat session"""
    try:
        result = chat_sessions_collection.delete_one({'_id': session_id})
        
        if result.deleted_count == 0:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        return jsonify({'success': True, 'message': 'Session deleted successfully'})
    except Exception as e:
        print(f"Error deleting chat session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# USER PREFERENCES AND SETTINGS API
# =============================================================================

@app.route('/api/user-preferences', methods=['GET'])
def get_user_preferences():
    """Get user preferences"""
    try:
        # TODO: Add user authentication and filter by user_id
        # For now, return default preferences
        prefs = user_preferences_collection.find_one({}) or {
            'default_chat_questions': [
                'What are the eligibility criteria?',
                'What is the study design?',
                'What are the primary outcomes?'
            ],
            'ai_provider': 'openai',
            'ai_model': 'gpt-4o-mini'
        }
        
        return jsonify({'success': True, 'preferences': prefs})
    except Exception as e:
        print(f"Error getting user preferences: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user-preferences', methods=['PATCH'])
def update_user_preferences():
    """Update user preferences"""
    try:
        data = request.json or {}
        
        # TODO: Add user authentication
        result = user_preferences_collection.update_one(
            {},
            {'$set': data},
            upsert=True
        )
        
        # Get updated preferences
        prefs = user_preferences_collection.find_one({})
        
        return jsonify({'success': True, 'preferences': prefs})
    except Exception as e:
        print(f"Error updating user preferences: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user-settings', methods=['GET'])
def get_user_settings():
    """Get user settings"""
    try:
        # TODO: Add user authentication and filter by user_id
        settings = user_settings_collection.find_one({}) or {
            'theme': 'light',
            'visible_models': ['gpt-4o-mini', 'gpt-4o', 'gemini-1.5-flash'],
            'report_format': 'styled'
        }
        
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        print(f"Error getting user settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user-settings', methods=['PATCH'])
def update_user_settings():
    """Update user settings"""
    try:
        data = request.json or {}
        
        # TODO: Add user authentication
        result = user_settings_collection.update_one(
            {},
            {'$set': data},
            upsert=True
        )
        
        # Get updated settings
        settings = user_settings_collection.find_one({})
        
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        print(f"Error updating user settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# CHAT QUESTIONS API
# =============================================================================

@app.route('/api/chat-questions', methods=['GET'])
def get_chat_questions():
    """Get chat questions for a session or default"""
    try:
        session_id = request.args.get('sessionId')
        
        if session_id:
            session = chat_sessions_collection.find_one({'_id': session_id})
            if session and session.get('custom_questions'):
                return jsonify({
                    'success': True,
                    'questions': session['custom_questions'],
                    'source': 'session'
                })
        
        # Return user default questions
        prefs = user_preferences_collection.find_one({})
        default_questions = prefs.get('default_chat_questions', [
            'What are the eligibility criteria?',
            'What is the study design?',
            'What are the primary outcomes?'
        ]) if prefs else [
            'What are the eligibility criteria?',
            'What is the study design?',
            'What are the primary outcomes?'
        ]
        
        return jsonify({
            'success': True,
            'questions': default_questions,
            'source': 'default'
        })
    except Exception as e:
        print(f"Error getting chat questions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat-questions', methods=['PATCH'])
def update_chat_questions():
    """Update chat questions for a session or as default"""
    try:
        data = request.json or {}
        session_id = data.get('sessionId')
        questions = data.get('questions', [])
        save_as_default = data.get('saveAsDefault', False)
        
        if save_as_default:
            # Update user preferences
            user_preferences_collection.update_one(
                {},
                {'$set': {'default_chat_questions': questions}},
                upsert=True
            )
        
        if session_id:
            # Update session
            chat_sessions_collection.update_one(
                {'_id': session_id},
                {'$set': {'custom_questions': questions}}
            )
        
        return jsonify({
            'success': True,
            'message': 'Questions updated successfully'
        })
    except Exception as e:
        print(f"Error updating chat questions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# STUDY CHATS API
# =============================================================================

@app.route('/api/study-chats', methods=['GET'])
def get_study_chats():
    """Get all study chats, optionally filtered by session"""
    try:
        chat_session_id = request.args.get('chatSessionId')
        
        query = {}
        if chat_session_id:
            query['chat_session_id'] = chat_session_id
        
        study_chats = list(study_chats_collection.find(query))
        
        # Convert ObjectId to string
        for chat in study_chats:
            if '_id' in chat:
                chat['id'] = str(chat['_id'])
                del chat['_id']
        
        return jsonify({'success': True, 'studyChats': study_chats})
    except Exception as e:
        print(f"Error getting study chats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/study-chats/<study_id>', methods=['GET'])
def get_study_chat(study_id):
    """Get a specific study chat"""
    try:
        chat_session_id = request.args.get('chatSessionId')
        
        query = {'study_id': study_id}
        if chat_session_id:
            query['chat_session_id'] = chat_session_id
        
        study_chat = study_chats_collection.find_one(query)
        
        if not study_chat:
            # Create a new study chat
            study_chat = {
                'study_id': study_id,
                'chat_session_id': chat_session_id,
                'messages': [],
                'reports': [],
                'custom_questions': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            study_chats_collection.insert_one(study_chat)
        
        # Convert for response
        if '_id' in study_chat:
            del study_chat['_id']
        
        return jsonify({'success': True, 'studyChat': study_chat})
    except Exception as e:
        print(f"Error getting study chat: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/study-chats/<study_id>/<session_id>', methods=['DELETE'])
def delete_study_chat(study_id, session_id):
    """Delete a specific study chat"""
    try:
        result = study_chats_collection.delete_one({
            'study_id': study_id,
            'chat_session_id': session_id
        })
        
        if result.deleted_count == 0:
            return jsonify({'success': False, 'error': 'Study chat not found'}), 404
        
        return jsonify({'success': True, 'message': 'Study chat deleted successfully'})
    except Exception as e:
        print(f"Error deleting study chat: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/study-chat-questions', methods=['GET'])
def get_study_chat_questions():
    """Get study chat questions"""
    try:
        study_id = request.args.get('studyId')
        chat_session_id = request.args.get('chatSessionId')
        
        # First check for study-specific custom questions
        if study_id and chat_session_id:
            query = {'study_id': study_id, 'chat_session_id': chat_session_id}
            study_chat = study_chats_collection.find_one(query)
            if study_chat and study_chat.get('custom_questions'):
                return jsonify({
                    'success': True,
                    'questions': study_chat['custom_questions'],
                    'source': 'study_chat'
                })
        
        # Return user default questions
        prefs = user_preferences_collection.find_one({})
        default_questions = prefs.get('default_chat_questions', [
            'What are the eligibility criteria?',
            'What is the study design?',
            'What are the primary outcomes?'
        ]) if prefs else [
            'What are the eligibility criteria?',
            'What is the study design?',
            'What are the primary outcomes?'
        ]
        
        return jsonify({
            'success': True,
            'questions': default_questions,
            'source': 'default'
        })
    except Exception as e:
        print(f"Error getting study chat questions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/study-chat-questions', methods=['PATCH'])
def update_study_chat_questions():
    """Update study chat questions"""
    try:
        data = request.json or {}
        study_id = data.get('studyId')
        chat_session_id = data.get('chatSessionId')
        questions = data.get('questions', [])
        save_as_default = data.get('saveAsDefault', False)
        
        if save_as_default:
            # Update user preferences
            user_preferences_collection.update_one(
                {},
                {'$set': {'default_chat_questions': questions}},
                upsert=True
            )
        
        if study_id and chat_session_id:
            # Update study chat
            study_chats_collection.update_one(
                {'study_id': study_id, 'chat_session_id': chat_session_id},
                {'$set': {'custom_questions': questions}},
                upsert=True
            )
        
        return jsonify({
            'success': True,
            'message': 'Study chat questions updated successfully'
        })
    except Exception as e:
        print(f"Error updating study chat questions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_query_from_filters(filters):
    """Build MongoDB query from filters - works with simplified structure"""
    query = {}
    
    # Condition search (case-insensitive)
    if filters.get('condition'):
        query['conditions'] = {
            '$regex': filters['condition'], '$options': 'i'
        }
    
    # Intervention search (handle both string and array)
    intervention = filters.get('intervention')
    if intervention:
        # If it's a string, convert to array (supports comma-separated values)
        if isinstance(intervention, str):
            intervention = [i.strip() for i in intervention.split(',') if i.strip()]
        # If it's an array and not empty
        if isinstance(intervention, list) and len(intervention) > 0:
            query['interventions'] = {'$in': intervention}
    
    # Status filter
    if filters.get('status') and len(filters['status']) > 0:
        query['status'] = {'$in': filters['status']}
    
    # Title search
    if filters.get('title'):
        query['title'] = {
            '$regex': filters['title'], '$options': 'i'
        }
    
    # NCT ID search
    if filters.get('nctId'):
        query['nct_id'] = filters['nctId'].upper()
    
    return query


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5034))  # Use 5034 for local testing (5033 is Node backend)
    print("=" * 60)
    print("Clinical Trials Search Application")
    print("=" * 60)
    print(f"MongoDB: {collection.count_documents({}):,} studies loaded")
    print("Starting server...")
    print(f"Open your browser: http://localhost:{port}")
    print("=" * 60)
    app.run(debug=True, port=port)

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from pymongo import MongoClient
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

# MongoDB connection - check multiple environment variable names for compatibility
MONGODB_URI = os.getenv('MONGO_URL') or os.getenv('MONGODB_PROD_URL') or os.getenv('MONGODB_URI')
if not MONGODB_URI:
    print("⚠️  WARNING: MongoDB URI not found in environment variables")
    print(f"Available env vars containing 'MONGO': {[k for k in os.environ.keys() if 'MONGO' in k.upper()]}")
    MONGODB_URI = 'mongodb://localhost:27017/'
    print(f"Falling back to localhost: {MONGODB_URI}")
else:
    print(f"🔌 Using MongoDB URI from environment")

print(f"🔌 Connecting to MongoDB: {MONGODB_URI[:50]}...")

try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    # Validate connection on startup
    client.server_info()
    print("✓ MongoDB connected successfully!")
except Exception as e:
    print(f"✗ MongoDB connection failed: {str(e)}")
    print("⚠️  WARNING: Running without database connection. Some features may not work.")
    # Continue anyway (graceful degradation)

# Use environment variables for database and collection names
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'clinical_trials')
MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'studies')

db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]
users_collection = db['users']  # For authentication

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
chroma_collection = None
chroma_client = None

try:
    # Check if ChromaDB Cloud is configured (preferred for production)
    chroma_cloud_api_key = os.getenv('CHROMA_CLOUD_API_KEY') or os.getenv('CHROMA_API_KEY')
    chroma_cloud_tenant = os.getenv('CHROMA_CLOUD_TENANT')
    chroma_cloud_database = os.getenv('CHROMA_CLOUD_DATABASE', 'clinicalchat')
    
    if chroma_cloud_api_key and chroma_cloud_tenant:
        # Use ChromaDB Cloud (recommended for production)
        chroma_client = chromadb.CloudClient(
            api_key=chroma_cloud_api_key,
            tenant=chroma_cloud_tenant,
            database=chroma_cloud_database
        )
        try:
            chroma_collection = chroma_client.get_collection(name='clinical_trials_embeddings')
            print(f"✓ ChromaDB Cloud connected (database: {chroma_cloud_database}): {chroma_collection.count()} embeddings")
        except Exception as e:
            print(f"⚠️  ChromaDB Cloud collection not found: {str(e)}")
            print("   Collection 'clinical_trials_embeddings' needs to be created with embeddings")
            chroma_collection = None
    else:
        # Fallback to client-server mode (for self-hosted ChromaDB)
        chroma_host = os.getenv('CHROMA_HOST')
        chroma_port = os.getenv('CHROMA_PORT', '8000')
        
        if chroma_host:
            # Use client-server mode (for production with separate ChromaDB service)
            chroma_auth_token = os.getenv('CHROMA_AUTH_TOKEN') or os.getenv('CHROMA_API_KEY')
            
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
    print("   - Option 1: Set CHROMA_HOST and CHROMA_PORT for client-server mode (recommended for production)")
    print("   - Option 2: Set CHROMADB_PATH to a writable directory")
    print("   - Option 3: Ensure /tmp is writable (used automatically in production)")
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

    return jsonify({
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
        'results': simplified_results,
        'searchType': 'keyword'
    })


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

    if not condition:
        return jsonify({'error': 'Condition is required'}), 400

    # Build query to find similar trials
    query = {}
    if condition:
        query['conditions'] = {'$regex': condition, '$options': 'i'}
    if intervention:
        query['interventions'] = {'$regex': intervention, '$options': 'i'}

    # Find similar trials
    total_count = collection.count_documents(query)
    if total_count == 0:
        return jsonify({'error': f'No trials found for {condition}'}), 404

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

        return jsonify({
            'report': full_report,
            'metadata': {
                'trials_analyzed': len(trials_summary),
                'total_matching': total_count,
                'condition': condition,
                'intervention': intervention
            }
        })

    except Exception as e:
        print(f"Error generating protocol report: {str(e)}")
        return jsonify({'error': f'AI error: {str(e)}'}), 500


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
    
    # Intervention search (array match)
    if filters.get('intervention') and len(filters['intervention']) > 0:
        query['interventions'] = {'$in': filters['intervention']}
    
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
    print("=" * 60)
    print("Clinical Trials Search Application")
    print("=" * 60)
    print(f"MongoDB: {collection.count_documents({}):,} studies loaded")
    print("Starting server...")
    print("Open your browser: http://localhost:5033")
    print("=" * 60)
    app.run(debug=True, port=5033)

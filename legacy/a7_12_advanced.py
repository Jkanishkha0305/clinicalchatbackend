from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datetime import datetime
import os
import json, markdown
import tiktoken
import json

from db_utils import get_mongo_client

def count_tokens(messages, model="gpt-4o-mini"):
    """
    Count tokens in a message list.
    Works for text and estimates for images.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    
    for message in messages:
        # Count tokens for role
        total_tokens += 4  # every message has role overhead
        
        if isinstance(message.get("content"), str):
            # Simple text content
            total_tokens += len(encoding.encode(message["content"]))
        
        elif isinstance(message.get("content"), list):
            # Complex content (text + images)
            for content_part in message["content"]:
                if content_part.get("type") == "text":
                    total_tokens += len(encoding.encode(content_part["text"]))
                
                elif content_part.get("type") == "image_url":
                    # Image token estimation
                    total_tokens += estimate_image_tokens(content_part)
    
    total_tokens += 2  # every reply is primed with assistant message
    
    return total_tokens


def estimate_image_tokens(image_content):
    """
    Estimate tokens for an image.
    OpenAI charges ~85-170 tokens per image depending on detail level.
    """
    detail = image_content.get("image_url", {}).get("detail", "auto")
    
    if detail == "low":
        return 85
    else:
        # High detail: 85 base + 170 per 512x512 tile
        # For safety, estimate 255 tokens (assumes ~1 tile)
        return 255

app = Flask(__name__)
CORS(app)

DB_NAME = os.getenv("MONGO_DB_NAME", "clinical_trials")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "studies")

try:
    client = get_mongo_client(serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
except Exception as e:
    raise RuntimeError(f"MongoDB Atlas connection failed: {str(e)}")

with open('data1/key/openai_key.txt') as f:
    openai_key = f.readline()

import openai, getpass, os

os.environ["OPENAI_API_KEY"] = openai_key
openai.api_key = openai_key

from openai import OpenAI
openai_client = OpenAI()

# HTML Template with Sidebar Chat
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Clinical Trials Search</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; padding: 20px; background: #f5f5f5; }
        .main-container { max-width: 1800px; margin: 0 auto; display: flex; gap: 20px; }
        .left-section { flex: 1; min-width: 0; }
        .right-sidebar { width: 400px; flex-shrink: 0; }
        
        h1 { margin-bottom: 20px; color: #333; }
        
        .search-box { background: white; padding: 30px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .filter-section { margin-bottom: 25px; }
        .filter-section h3 { margin-bottom: 12px; color: #0066cc; font-size: 16px; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }
        
        .filter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .filter-grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        
        .form-group { display: flex; flex-direction: column; }
        .form-group label { margin-bottom: 5px; font-size: 14px; font-weight: 500; }
        input, select, textarea { padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        input[type="date"] { padding: 7px; }
        select[multiple] { min-height: 80px; }
        
        .checkbox-group { display: flex; flex-direction: column; gap: 8px; }
        .checkbox-group label { display: flex; align-items: center; gap: 8px; font-weight: normal; }
        .checkbox-group input[type="checkbox"] { width: auto; }
        
        .button-group { margin-top: 20px; display: flex; gap: 10px; }
        button { background: #0066cc; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 15px; }
        button:hover { background: #0052a3; }
        button.secondary { background: #6c757d; }
        button.secondary:hover { background: #5a6268; }
        
        .results { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .study-card { border: 1px solid #e0e0e0; padding: 20px; margin-bottom: 15px; border-radius: 6px; transition: box-shadow 0.2s; }
        .study-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .study-card h3 { color: #0066cc; margin-bottom: 12px; cursor: pointer; }
        .study-card h3:hover { text-decoration: underline; }
        .study-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 14px; }
        .meta-item { display: flex; gap: 5px; }
        .meta-label { font-weight: 600; color: #555; }
        .meta-value { color: #333; }
        
        .pagination { display: flex; justify-content: center; gap: 10px; margin-top: 20px; }
        .pagination button { padding: 8px 16px; }
        .pagination button.active { background: #28a745; }
        .pagination button:disabled { background: #ccc; cursor: not-allowed; }
        
        .badge { display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 12px; font-weight: 600; }
        .badge.recruiting { background: #28a745; color: white; }
        .badge.completed { background: #6c757d; color: white; }
        .badge.has-results { background: #17a2b8; color: white; }
        
        /* Sidebar Chat Styles */
        .chat-sidebar { 
            background: white; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            display: flex; 
            flex-direction: column;
            height: calc(100vh - 40px);
            position: sticky;
            top: 20px;
        }
        
        .chat-header {
            padding: 20px;
            border-bottom: 2px solid #0066cc;
            background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
            color: white;
            border-radius: 8px 8px 0 0;
        }
        
        .chat-header h2 { font-size: 18px; margin-bottom: 5px; }
        .chat-header p { font-size: 13px; opacity: 0.9; }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .chat-message { 
            margin-bottom: 15px; 
            padding: 12px; 
            border-radius: 6px; 
            max-width: 90%;
            word-wrap: break-word;
        }
        .chat-message.user { 
            background: #007bff; 
            color: white; 
            margin-left: auto; 
        }
        .chat-message.assistant { 
            background: white; 
            border: 1px solid #ddd; 
        }
        .chat-message.loading { 
            background: #f0f0f0; 
            font-style: italic; 
        }

        .chat-message.assistant ul, 
        .chat-message.assistant ol {
            margin-left: 20px;
            margin-top: 8px;
            margin-bottom: 8px;
        }

        .chat-message.assistant li {
            margin-bottom: 5px;
            line-height: 1.5;
        }

        .chat-message.assistant p {
            margin-bottom: 10px;
            line-height: 1.6;
        }

        .chat-message.assistant strong {
            font-weight: 600;
        }
        
        .chat-input-container {
            padding: 15px;
            border-top: 1px solid #ddd;
            background: white;
            border-radius: 0 0 8px 8px;
        }
        
        .chat-input-wrapper {
            display: flex;
            gap: 10px;
        }
        
        #sidebarChatInput {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .chat-send-btn {
            padding: 12px 24px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .chat-send-btn:hover {
            background: #218838;
        }
        
        .chat-send-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .chat-placeholder {
            text-align: center;
            color: #666;
            padding: 40px 20px;
        }
        
        .chat-placeholder h3 {
            color: #0066cc;
            margin-bottom: 15px;
        }
        
        .chat-placeholder ul {
            text-align: left;
            list-style: none;
            padding: 0;
        }
        
        .chat-placeholder li {
            padding: 8px 0;
            color: #555;
        }
        
        .chat-placeholder li:before {
            content: "💬 ";
        }
        
        /* Individual study chat modal */
        .study-chat-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
        }
        
        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            width: 90%;
            max-width: 800px;
            height: 80%;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
        }
        
        .modal-header {
            padding: 20px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .close-btn {
            background: #dc3545;
            padding: 8px 16px;
        }
        
        .close-btn:hover {
            background: #c82333;
        }

        @media (max-width: 1400px) {
            .main-container {
                flex-direction: column;
            }
            .right-sidebar {
                width: 100%;
                height: 500px;
            }
            .chat-sidebar {
                position: static;
                height: 500px;
            }
        }

        .advanced-mode-container {
            padding: 10px 15px;
            background: #f8f9fa;
            border-top: 1px solid #ddd;
            border-bottom: 1px solid #ddd;
        }

        .advanced-mode-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            cursor: pointer;
        }

        .mode-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 5px;
        }

        .mode-badge.essential {
            background: #28a745;
            color: white;
        }

        .mode-badge.advanced {
            background: #ff6b6b;
            color: white;
        }

        .warning-message {
            margin-top: 8px;
            padding: 8px;
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            font-size: 12px;
            color: #856404;
            display: none;
        }

        .warning-message.show {
            display: block;
        }


    </style>
</head>
<body>
    <div class="main-container">
        <div class="left-section">
            <h1>Clinical Trials Search</h1>
            
            <div class="search-box">
                <div class="filter-section">
                    <h3>Basic Search</h3>
                    <div class="filter-grid">
                        <div class="form-group">
                            <label>Condition or Disease</label>
                            <input type="text" id="condition" placeholder="e.g., Cancer, Diabetes, COVID-19">
                        </div>
                        <div class="form-group">
                            <label>Intervention/Treatment</label>
                            <select id="intervention" multiple>
                                <option value="" disabled>Loading interventions...</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Location</label>
                            <input type="text" id="location" placeholder="Country, City, or State">
                        </div>
                    </div>
                </div>
                
                <div class="filter-section">
                    <h3>Study Status and Type</h3>
                    <div class="filter-grid">
                        <div class="form-group">
                            <label>Study Status</label>
                            <select id="status" multiple>
                                <option value="RECRUITING">Recruiting</option>
                                <option value="NOT_YET_RECRUITING">Not yet recruiting</option>
                                <option value="ACTIVE_NOT_RECRUITING">Active, not recruiting</option>
                                <option value="COMPLETED">Completed</option>
                                <option value="ENROLLING_BY_INVITATION">Enrolling by invitation</option>
                                <option value="SUSPENDED">Suspended</option>
                                <option value="TERMINATED">Terminated</option>
                                <option value="WITHDRAWN">Withdrawn</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Study Type</label>
                            <select id="studyType" multiple>
                                <option value="INTERVENTIONAL">Interventional</option>
                                <option value="OBSERVATIONAL">Observational</option>
                                <option value="EXPANDED_ACCESS">Expanded Access</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Study Phase</label>
                            <select id="phase" multiple>
                                <option value="EARLY_PHASE1">Early Phase 1</option>
                                <option value="PHASE1">Phase 1</option>
                                <option value="PHASE2">Phase 2</option>
                                <option value="PHASE3">Phase 3</option>
                                <option value="PHASE4">Phase 4</option>
                                <option value="NA">Not Applicable</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="filter-section">
                    <h3>Eligibility Criteria</h3>
                    <div class="filter-grid">
                        <div class="form-group">
                            <label>Sex</label>
                            <select id="sex">
                                <option value="">All</option>
                                <option value="FEMALE">Female</option>
                                <option value="MALE">Male</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Age Groups</label>
                            <div class="checkbox-group">
                                <label><input type="checkbox" name="ageGroups" value="CHILD"> Child (birth - 17)</label>
                                <label><input type="checkbox" name="ageGroups" value="ADULT"> Adult (18 - 64)</label>
                                <label><input type="checkbox" name="ageGroups" value="OLDER_ADULT"> Older Adult (65+)</label>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Accepts Healthy Volunteers</label>
                            <label><input type="checkbox" id="healthyVolunteers"> Yes</label>
                        </div>
                    </div>
                </div>
                
                <div class="filter-section">
                    <h3>Study Results & Documents</h3>
                    <div class="filter-grid">
                        <div class="form-group">
                            <label>Study Results</label>
                            <select id="hasResults">
                                <option value="">Any</option>
                                <option value="true">With results</option>
                                <option value="false">Without results</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Study Documents</label>
                            <div class="checkbox-group">
                                <label><input type="checkbox" id="hasProtocol"> Study Protocol</label>
                                <label><input type="checkbox" id="hasSAP"> Statistical Analysis Plan (SAP)</label>
                                <label><input type="checkbox" id="hasICF"> Informed Consent Form (ICF)</label>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Funder Type</label>
                            <select id="funderType" multiple>
                                <option value="NIH">NIH</option>
                                <option value="FED">Other U.S. Federal Agency</option>
                                <option value="INDUSTRY">Industry</option>
                                <option value="OTHER">All Others</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="filter-section">
                    <h3>Date Ranges</h3>
                    <div class="filter-grid-2">
                        <div class="form-group">
                            <label>Study Start From</label>
                            <input type="date" id="studyStartFrom">
                        </div>
                        <div class="form-group">
                            <label>Study Start To</label>
                            <input type="date" id="studyStartTo">
                        </div>
                        <div class="form-group">
                            <label>Primary Completion From</label>
                            <input type="date" id="primaryCompletionFrom">
                        </div>
                        <div class="form-group">
                            <label>Primary Completion To</label>
                            <input type="date" id="primaryCompletionTo">
                        </div>
                    </div>
                </div>
                
                <div class="filter-section">
                    <h3>More Ways to Search</h3>
                    <div class="filter-grid">
                        <div class="form-group">
                            <label>Title or Acronym</label>
                            <input type="text" id="title" placeholder="Search in study titles">
                        </div>
                        <div class="form-group">
                            <label>Outcome Measure</label>
                            <input type="text" id="outcome" placeholder="Primary or secondary outcome">
                        </div>
                        <div class="form-group">
                            <label>Lead Sponsor</label>
                            <input type="text" id="sponsor" placeholder="Organization name">
                        </div>
                        <div class="form-group">
                            <label>NCT Number</label>
                            <input type="text" id="nctId" placeholder="NCT########">
                        </div>
                    </div>
                    <div class="form-group" style="margin-top: 15px;">
                        <label><input type="checkbox" id="fdaaa801Violation"> FDAAA 801 Violations</label>
                    </div>
                </div>
                
                <div class="button-group">
                    <button onclick="searchStudies()">Search</button>
                    <button class="secondary" onclick="clearFilters()">Clear Filters</button>
                </div>
            </div>
            
            <div class="results">
                <div class="results-header">
                    <h2>Results (<span id="totalCount">0</span> studies found)</h2>
                    <div>
                        Results per page: 
                        <select id="perPage" onchange="searchStudies()">
                            <option value="10">10</option>
                            <option value="20" selected>20</option>
                            <option value="50">50</option>
                            <option value="100">100</option>
                        </select>
                    </div>
                </div>
                <div id="resultsContainer"></div>
                <div id="pagination" class="pagination"></div>
            </div>
        </div>
        
        <!-- Right Sidebar Chat -->
        <div class="right-sidebar">
            <div class="chat-sidebar">
                <div class="chat-header">
                    <h2>Ask About All Results</h2>
                    <p>Chat about your filtered studies</p>
                </div>
                <div class="advanced-mode-container">
                    <label class="advanced-mode-label">
                        <input type="checkbox" id="advancedModeCheckbox" onchange="toggleAdvancedMode()">
                        <span>
                            Advanced Mode (Complete Data)
                            <span id="modeBadge" class="mode-badge essential">ESSENTIAL</span>
                        </span>
                    </label>
                    <div id="advancedWarning" class="warning-message">
                        ⚠️ Advanced mode includes ALL study data. Recommended for 50 or fewer studies for best performance.
                    </div>
                </div>

                <div id="sidebarChatMessages" class="chat-messages">
                    <div class="chat-placeholder">
                        <h3>Search first, then ask questions!</h3>
                        <p style="margin: 15px 0;">After searching, ask questions like:</p>
                        <ul>
                            <li>What are the most common interventions?</li>
                            <li>How many studies are in Phase 3?</li>
                            <li>Which sponsors are funding these trials?</li>
                            <li>What are the primary outcomes?</li>
                            <li>Summarize the recruiting studies</li>
                        </ul>
                    </div>
                </div>
                
                <div class="chat-input-container">
                    <div class="chat-input-wrapper">
                        <input type="text" id="sidebarChatInput" placeholder="Ask about all filtered studies..." 
                               onkeypress="if(event.key==='Enter') sendSidebarMessage()" disabled>
                        <button class="chat-send-btn" onclick="sendSidebarMessage()" id="sidebarSendBtn" disabled>Send</button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Individual Study Chat Modal -->
        <div id="chatModal" class="study-chat-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="chatStudyTitle">Chat about Study</h3>
                    <button onclick="closeChatModal()" class="close-btn">Close</button>
                </div>
                
                <div id="chatMessages" class="chat-messages">
                    <div style="text-align: center; color: #666; padding: 20px;">
                        Ask any question about this clinical trial study.
                    </div>
                </div>
                
                <div class="chat-input-container">
                    <div class="chat-input-wrapper">
                        <input type="text" id="chatInput" placeholder="Ask a question about this study..." 
                               onkeypress="if(event.key==='Enter') sendMessage()">
                        <button class="chat-send-btn" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentChatStudy = null;
        let currentPage = 1;
        let totalPages = 1;
        let currentFilters = null;
        let totalResults = 0;
        
        // Load interventions on page load
        async function loadInterventions() {
            try {
                const response = await fetch('/api/interventions');
                const data = await response.json();
                const select = document.getElementById('intervention');
                
                // Clear loading message
                select.innerHTML = '';
                
                // Add interventions as options
                data.interventions.forEach(intervention => {
                    const option = document.createElement('option');
                    option.value = intervention;
                    option.textContent = intervention;
                    select.appendChild(option);
                });
                
                console.log(`Loaded ${data.interventions.length} interventions`);
            } catch (error) {
                console.error('Failed to load interventions:', error);
                document.getElementById('intervention').innerHTML = '<option value="" disabled>Failed to load</option>';
            }
        }
        
        // Load interventions when page loads
        window.addEventListener('DOMContentLoaded', loadInterventions);
        
        // Individual Study Chat Functions
        function openChatModal(nctId, title) {
            currentChatStudy = nctId;
            document.getElementById('chatStudyTitle').textContent = `Chat about ${nctId}`;
            document.getElementById('chatModal').style.display = 'block';
            document.getElementById('chatMessages').innerHTML = `
                <div style="text-align: center; color: #666; padding: 20px;">
                    <strong>${title}</strong><br><br>
                    Ask any question about this study.
                </div>
            `;
        }
        
        function closeChatModal() {
            document.getElementById('chatModal').style.display = 'none';
            currentChatStudy = null;
        }
        
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const question = input.value.trim();
            
            if (!question || !currentChatStudy) return;
            
            addModalMessage(question, 'user');
            input.value = '';
            
            const loadingId = addModalMessage('Analyzing study data...', 'loading');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        nctId: currentChatStudy,
                        question: question
                    })
                });
                
                const data = await response.json();
                document.getElementById(loadingId).remove();
                
                if (data.error) {
                    addModalMessage('Error: ' + data.error, 'assistant');
                } else {
                    addModalMessage(data.answer, 'assistant');
                }
                
            } catch (error) {
                document.getElementById(loadingId).remove();
                addModalMessage('Error: Failed to get response.', 'assistant');
            }
        }
                
        function addModalMessage(text, type) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            const id = 'msg-' + Date.now();
            messageDiv.id = id;
            messageDiv.className = `chat-message ${type}`;
            
            if (type === 'assistant') {
                messageDiv.innerHTML = text;
            } else {
                messageDiv.textContent = text;
            }
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return id;
        }
        
        async function sendSidebarMessage() {
            const input = document.getElementById('sidebarChatInput');
            const question = input.value.trim();
            
            if (!question || !currentFilters) return;
            
            addSidebarMessage(question, 'user');
            input.value = '';
            
            // Disable input during processing
            const sendBtn = document.getElementById('sidebarSendBtn');
            const inputField = document.getElementById('sidebarChatInput');
            sendBtn.disabled = true;
            inputField.disabled = true;
            
            const modeText = isAdvancedMode ? 'ALL complete study data' : 'essential study data';
            const loadingId = addSidebarMessage(`Analyzing ${modeText} from filtered studies...`, 'loading');

            try {
                const response = await fetch('/api/chat-all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filters: currentFilters,
                        question: question,
                        advancedMode: isAdvancedMode
                    })
                });
                
                const data = await response.json();
                document.getElementById(loadingId).remove();
                
                if (data.error) {
                    addSidebarMessage('Error: ' + data.error, 'assistant');
                } else {
                    let answer = data.answer;
                    
                    // Add info badge if available
                    if (data.info) {
                        const infoBadge = `<div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 10px; margin-bottom: 10px; font-size: 12px;">
                            <strong>ℹ️ Analysis Info:</strong> ${data.info}
                        </div>`;
                        answer = infoBadge + answer;
                    }
                    
                    addSidebarMessage(answer, 'assistant');
                }
                
            } catch (error) {
                document.getElementById(loadingId).remove();
                addSidebarMessage('Error: Failed to get response.', 'assistant');
            } finally {
                sendBtn.disabled = false;
                inputField.disabled = false;
            }
        }
        
        function addSidebarMessage(text, type) {
            const messagesDiv = document.getElementById('sidebarChatMessages');
            
            // Remove placeholder if exists
            const placeholder = messagesDiv.querySelector('.chat-placeholder');
            if (placeholder) {
                placeholder.remove();
            }
            
            const messageDiv = document.createElement('div');
            const id = 'sidebar-msg-' + Date.now();
            messageDiv.id = id;
            messageDiv.className = `chat-message ${type}`;
            
            if (type === 'assistant') {
                messageDiv.innerHTML = text;
            } else {
                messageDiv.textContent = text;
            }
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return id;
        }
        
        function getFilters() {
            return {
                condition: document.getElementById('condition').value,
                intervention: Array.from(document.getElementById('intervention').selectedOptions).map(o => o.value),
                location: document.getElementById('location').value,
                status: Array.from(document.getElementById('status').selectedOptions).map(o => o.value),
                studyType: Array.from(document.getElementById('studyType').selectedOptions).map(o => o.value),
                phase: Array.from(document.getElementById('phase').selectedOptions).map(o => o.value),
                sex: document.getElementById('sex').value,
                ageGroups: Array.from(document.querySelectorAll('input[name="ageGroups"]:checked')).map(cb => cb.value),
                healthyVolunteers: document.getElementById('healthyVolunteers').checked,
                hasResults: document.getElementById('hasResults').value,
                hasProtocol: document.getElementById('hasProtocol').checked,
                hasSAP: document.getElementById('hasSAP').checked,
                hasICF: document.getElementById('hasICF').checked,
                funderType: Array.from(document.getElementById('funderType').selectedOptions).map(o => o.value),
                studyStartFrom: document.getElementById('studyStartFrom').value,
                studyStartTo: document.getElementById('studyStartTo').value,
                primaryCompletionFrom: document.getElementById('primaryCompletionFrom').value,
                primaryCompletionTo: document.getElementById('primaryCompletionTo').value,
                title: document.getElementById('title').value,
                outcome: document.getElementById('outcome').value,
                sponsor: document.getElementById('sponsor').value,
                nctId: document.getElementById('nctId').value,
                fdaaa801Violation: document.getElementById('fdaaa801Violation').checked,
                page: currentPage,
                per_page: parseInt(document.getElementById('perPage').value)
            };
        }
        
        async function searchStudies(page = 1) {
            currentPage = page;
            const filters = getFilters();
            currentFilters = filters; // Store for sidebar chat
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(filters)
                });
                
                if (!response.ok) throw new Error('Search failed');
                
                const data = await response.json();
                totalPages = data.total_pages;
                totalResults = data.total;
                displayResults(data);
                displayPagination(data);
                
                // Enable sidebar chat after first search
                document.getElementById('sidebarChatInput').disabled = false;
                document.getElementById('sidebarSendBtn').disabled = false;

                // Show/hide warning based on results count and mode
                const warning = document.getElementById('advancedWarning');
                if (isAdvancedMode && totalResults > 50) {
                    warning.classList.add('show');
                } else {
                    warning.classList.remove('show');
                }

            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        function displayResults(data) {
            document.getElementById('totalCount').textContent = data.total.toLocaleString();
            
            const container = document.getElementById('resultsContainer');
            container.innerHTML = '';
            
            if (data.results.length === 0) {
                container.innerHTML = '<p>No studies found matching your criteria.</p>';
                return;
            }
            
            data.results.forEach(study => {
                const ps = study.protocolSection;
                const id = ps.identificationModule;
                const status = ps.statusModule;
                const design = ps.designModule;
                const sponsor = ps.sponsorCollaboratorsModule;
                
                const statusClass = status.overallStatus === 'RECRUITING' ? 'recruiting' : 
                                   status.overallStatus === 'COMPLETED' ? 'completed' : '';
                
                const card = document.createElement('div');
                card.className = 'study-card';
                card.innerHTML = `
                    <h3 onclick="window.open('https://clinicaltrials.gov/study/${id.nctId}', '_blank')">
                        ${id.nctId}: ${id.briefTitle || 'No title'}
                    </h3>
                    <div class="study-meta">
                        <div class="meta-item">
                            <span class="meta-label">Status:</span>
                            <span class="badge ${statusClass}">${status.overallStatus}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Type:</span>
                            <span class="meta-value">${design.studyType}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Phase:</span>
                            <span class="meta-value">${design.phases?.join(', ') || 'N/A'}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Sponsor:</span>
                            <span class="meta-value">${sponsor?.leadSponsor?.name || 'N/A'}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Results:</span>
                            ${study.hasResults ? '<span class="badge has-results">Has Results</span>' : '<span class="meta-value">No Results</span>'}
                        </div>
                    </div>
                    <button onclick="openChatModal('${id.nctId}', '${(id.briefTitle || '').replace(/'/g, "\\'")}'); event.stopPropagation();" 
                        style="margin-top: 10px; padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        💬 Ask About This Study
                    </button>
                `;
                container.appendChild(card);
            });
        }
        
        function displayPagination(data) {
            const container = document.getElementById('pagination');
            container.innerHTML = '';
            
            if (totalPages <= 1) return;
            
            const prevBtn = document.createElement('button');
            prevBtn.textContent = '« Previous';
            prevBtn.disabled = currentPage === 1;
            prevBtn.onclick = () => searchStudies(currentPage - 1);
            container.appendChild(prevBtn);
            
            const startPage = Math.max(1, currentPage - 3);
            const endPage = Math.min(totalPages, currentPage + 3);
            
            for (let i = startPage; i <= endPage; i++) {
                const btn = document.createElement('button');
                btn.textContent = i;
                btn.className = i === currentPage ? 'active' : '';
                btn.onclick = () => searchStudies(i);
                container.appendChild(btn);
            }
            
            const nextBtn = document.createElement('button');
            nextBtn.textContent = 'Next »';
            nextBtn.disabled = currentPage === totalPages;
            nextBtn.onclick = () => searchStudies(currentPage + 1);
            container.appendChild(nextBtn);
        }
        
        function clearFilters() {
            document.querySelectorAll('input[type="text"], input[type="date"]').forEach(input => input.value = '');
            document.querySelectorAll('select').forEach(select => {
                if (select.id === 'perPage') return;
                // For multi-select, clear all selections
                Array.from(select.options).forEach(option => option.selected = false);
            });
            document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
            currentPage = 1;
            currentFilters = null;
            document.getElementById('resultsContainer').innerHTML = '';
            document.getElementById('totalCount').textContent = '0';
            
            // Reset sidebar chat
            document.getElementById('sidebarChatMessages').innerHTML = `
                <div class="chat-placeholder">
                    <h3>Search first, then ask questions!</h3>
                    <p style="margin: 15px 0;">After searching, ask questions about all filtered studies</p>
                </div>
            `;
            document.getElementById('sidebarChatInput').disabled = true;
            document.getElementById('sidebarSendBtn').disabled = true;
        }

        let isAdvancedMode = false;

        function toggleAdvancedMode() {
            isAdvancedMode = document.getElementById('advancedModeCheckbox').checked;
            const badge = document.getElementById('modeBadge');
            const warning = document.getElementById('advancedWarning');
            
            if (isAdvancedMode) {
                badge.textContent = 'ADVANCED';
                badge.className = 'mode-badge advanced';
                if (totalResults > 50) {
                    warning.classList.add('show');
                }
            } else {
                badge.textContent = 'ESSENTIAL';
                badge.className = 'mode-badge essential';
                warning.classList.remove('show');
            }
        }


    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/interventions', methods=['GET'])
def get_interventions():
    """Get list of unique interventions from database"""
    try:
        # Aggregate to get unique intervention names
        pipeline = [
            {'$unwind': '$protocolSection.armsInterventionsModule.interventions'},
            {'$group': {
                '_id': '$protocolSection.armsInterventionsModule.interventions.name'
            }},
            {'$sort': {'_id': 1}},
            {'$limit': 500}  # Limit to most common 500
        ]
        
        interventions = collection.aggregate(pipeline)
        intervention_list = [doc['_id'] for doc in interventions if doc['_id']]
        
        return jsonify({'interventions': intervention_list})
    except Exception as e:
        print(f"Error fetching interventions: {str(e)}")
        return jsonify({'interventions': []})

def build_query_from_filters(filters):
    """Build MongoDB query from filters"""
    query = {}
    
    if filters.get('condition'):
        query['protocolSection.conditionsModule.conditions'] = {
            '$regex': filters['condition'], '$options': 'i'
        }
    
    if filters.get('intervention') and len(filters['intervention']) > 0:
        query['protocolSection.armsInterventionsModule.interventions.name'] = {
            '$in': filters['intervention']
        }
    
    if filters.get('status') and len(filters['status']) > 0:
        query['protocolSection.statusModule.overallStatus'] = {'$in': filters['status']}
    
    if filters.get('studyType') and len(filters['studyType']) > 0:
        query['protocolSection.designModule.studyType'] = {'$in': filters['studyType']}
    
    if filters.get('phase') and len(filters['phase']) > 0:
        query['protocolSection.designModule.phases'] = {
            '$elemMatch': {'$in': filters['phase']}
        }
    
    if filters.get('sex') and filters['sex'] != 'ALL':
        query['protocolSection.eligibilityModule.sex'] = filters['sex']
    
    if filters.get('ageGroups') and len(filters['ageGroups']) > 0:
        query['protocolSection.eligibilityModule.stdAges'] = {
            '$elemMatch': {'$in': filters['ageGroups']}
        }
    
    if filters.get('healthyVolunteers'):
        query['protocolSection.eligibilityModule.healthyVolunteers'] = True
    
    if filters.get('hasResults') == 'true':
        query['hasResults'] = True
    elif filters.get('hasResults') == 'false':
        query['hasResults'] = False
    
    if filters.get('hasProtocol'):
        query['documentSection.largeDocumentModule.largeDocs.hasProtocol'] = True
    if filters.get('hasSAP'):
        query['documentSection.largeDocumentModule.largeDocs.hasSap'] = True
    if filters.get('hasICF'):
        query['documentSection.largeDocumentModule.largeDocs.hasIcf'] = True
    
    if filters.get('funderType') and len(filters['funderType']) > 0:
        query['protocolSection.sponsorCollaboratorsModule.leadSponsor.class'] = {
            '$in': filters['funderType']
        }
    
    if filters.get('location'):
        location_regex = {'$regex': filters['location'], '$options': 'i'}
        query['$or'] = [
            {'protocolSection.contactsLocationsModule.locations.country': location_regex},
            {'protocolSection.contactsLocationsModule.locations.city': location_regex},
            {'protocolSection.contactsLocationsModule.locations.state': location_regex}
        ]
    
    date_fields = {
        'studyStart': 'protocolSection.statusModule.startDateStruct.date',
        'primaryCompletion': 'protocolSection.statusModule.primaryCompletionDateStruct.date',
    }
    
    for filter_key, field_path in date_fields.items():
        from_date = filters.get(f'{filter_key}From')
        to_date = filters.get(f'{filter_key}To')
        
        if from_date or to_date:
            query[field_path] = {}
            if from_date:
                query[field_path]['$gte'] = from_date
            if to_date:
                query[field_path]['$lte'] = to_date
    
    if filters.get('title'):
        title_regex = {'$regex': filters['title'], '$options': 'i'}
        if '$or' not in query:
            query['$or'] = []
        query['$or'].extend([
            {'protocolSection.identificationModule.briefTitle': title_regex},
            {'protocolSection.identificationModule.officialTitle': title_regex},
            {'protocolSection.identificationModule.acronym': title_regex}
        ])
    
    if filters.get('sponsor'):
        query['protocolSection.sponsorCollaboratorsModule.leadSponsor.name'] = {
            '$regex': filters['sponsor'], '$options': 'i'
        }
    
    if filters.get('outcome'):
        outcome_regex = {'$regex': filters['outcome'], '$options': 'i'}
        if '$or' not in query:
            query['$or'] = []
        query['$or'].extend([
            {'protocolSection.outcomesModule.primaryOutcomes.measure': outcome_regex},
            {'protocolSection.outcomesModule.secondaryOutcomes.measure': outcome_regex}
        ])
    
    if filters.get('nctId'):
        query['protocolSection.identificationModule.nctId'] = filters['nctId'].upper()
    
    if filters.get('fdaaa801Violation'):
        query['protocolSection.oversightModule.fdaaa801Violation'] = True
    
    return query

@app.route('/api/search', methods=['POST'])
def search_studies():
    filters = request.json
    query = build_query_from_filters(filters)
    
    page = filters.get('page', 1)
    per_page = min(filters.get('per_page', 20), 100)
    skip = (page - 1) * per_page
    
    total = collection.count_documents(query)
    results = list(collection.find(query).skip(skip).limit(per_page))
    
    for result in results:
        result['_id'] = str(result['_id'])
    
    return jsonify({
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
        'results': results
    })

@app.route('/api/study/<nct_id>', methods=['GET'])
def get_study(nct_id):
    study = collection.find_one({
        'protocolSection.identificationModule.nctId': nct_id
    })
    
    if study:
        study['_id'] = str(study['_id'])
        return jsonify(study)
    return jsonify({'error': 'Study not found'}), 404

def extract_essential_fields(study):
    """Extract only essential fields from a study to reduce token usage"""
    ps = study.get('protocolSection', {})
    
    # Identification
    ident = ps.get('identificationModule', {})
    nct_id = ident.get('nctId', 'N/A')
    title = ident.get('briefTitle', 'N/A')
    
    # Status
    status_mod = ps.get('statusModule', {})
    status = status_mod.get('overallStatus', 'N/A')
    
    # Design
    design = ps.get('designModule', {})
    study_type = design.get('studyType', 'N/A')
    phases = design.get('phases', [])
    
    # Conditions
    conditions = ps.get('conditionsModule', {}).get('conditions', [])
    
    # Interventions
    interventions_list = ps.get('armsInterventionsModule', {}).get('interventions', [])
    interventions = [i.get('name', '') for i in interventions_list[:3]]  # Limit to first 3
    
    # Outcomes - just the first primary outcome
    outcomes = ps.get('outcomesModule', {})
    primary_outcomes = outcomes.get('primaryOutcomes', [])
    primary_outcome = primary_outcomes[0].get('measure', 'N/A') if primary_outcomes else 'N/A'
    
    # Sponsor
    sponsor_mod = ps.get('sponsorCollaboratorsModule', {})
    sponsor = sponsor_mod.get('leadSponsor', {}).get('name', 'N/A')
    
    # Enrollment
    design_mod = ps.get('designModule', {})
    enrollment = design_mod.get('enrollmentInfo', {}).get('count', 'N/A')
    
    # Eligibility
    eligibility = ps.get('eligibilityModule', {})
    sex = eligibility.get('sex', 'N/A')
    min_age = eligibility.get('minimumAge', 'N/A')
    max_age = eligibility.get('maximumAge', 'N/A')
    
    # Locations - just countries
    locations = ps.get('contactsLocationsModule', {}).get('locations', [])
    countries = list(set([loc.get('country', '') for loc in locations if loc.get('country')]))[:5]
    
    return {
        'nctId': nct_id,
        'title': title,
        'status': status,
        'studyType': study_type,
        'phases': phases,
        'conditions': conditions[:5],  # Limit to 5
        'interventions': interventions,
        'primaryOutcome': primary_outcome,
        'sponsor': sponsor,
        'enrollment': enrollment,
        'eligibility': {
            'sex': sex,
            'ageRange': f"{min_age} to {max_age}"
        },
        'countries': countries
    }


@app.route('/api/chat-all', methods=['POST'])
def chat_all():
    """Chat endpoint for all filtered studies"""
    data = request.json
    filters = data.get('filters', {})
    question = data.get('question', '')
    advanced_mode = data.get('advancedMode', False)
    
    # Build query from filters
    query = build_query_from_filters(filters)
    
    # Get total count
    total_count = collection.count_documents(query)
    
    # Different limits based on mode
    if advanced_mode:
        max_studies = 100
    else:
        max_studies = 500
    
    limit = min(total_count, max_studies)
    studies = list(collection.find(query).limit(limit))
    
    # Process based on mode
    if advanced_mode:
        # Use COMPLETE data - just remove _id
        processed_studies = []
        for study in studies:
            study_copy = {k: v for k, v in study.items() if k != '_id'}
            processed_studies.append(study_copy)
        mode_info = f"Analyzing {len(processed_studies)} studies with COMPLETE data"
    else:
        # Use essential fields only
        processed_studies = [extract_essential_fields(study) for study in studies]
        mode_info = f"Analyzing {len(processed_studies)} studies with essential data"
    
    # Convert to JSON
    studies_json = json.dumps(processed_studies, indent=1)
    
    # Estimate tokens and handle limits
    estimated_tokens = len(studies_json) / 4
    max_context_tokens = 120000 if advanced_mode else 100000
    
    truncated_message = ""
    if estimated_tokens > max_context_tokens:
        chars_per_study = len(studies_json) / len(processed_studies)
        max_studies_fit = int(max_context_tokens * 4 / chars_per_study)
        
        if max_studies_fit < 1:
            return jsonify({
                'error': f'Studies are too large for {("advanced" if advanced_mode else "essential")} mode. Try {"essential mode or " if advanced_mode else ""}fewer studies.'
            }), 400
        
        processed_studies = processed_studies[:max_studies_fit]
        studies_json = json.dumps(processed_studies, indent=1)
        truncated_message = f"\n\n[Note: Showing {len(processed_studies)} out of {total_count} studies due to context limits]"
        mode_info += f" (truncated to {len(processed_studies)} studies)"
    
    system_message = f"""You are a clinical trials research analyst. Answer questions about this collection of clinical trial studies.

DATASET SUMMARY:
- Total studies in filtered results: {total_count}
- Studies provided for analysis: {len(processed_studies)}
- Data mode: {"COMPLETE (all fields)" if advanced_mode else "ESSENTIAL (key fields only)"}

STUDY DATA:
{studies_json}{truncated_message}

Instructions:
- Analyze the provided studies to answer the question
- Provide statistics, trends, and insights when relevant
- If asked about specific studies, cite NCT IDs
- If information spans many studies, provide summaries and key patterns
- Be precise and data-driven
- Use markdown formatting for readability
{"- You have access to COMPLETE study data including descriptions, outcomes, eligibility criteria, etc." if advanced_mode else "- You have access to essential study data. For detailed information about specific studies, suggest using the individual study chat."}"""
    
    try:
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        # Count tokens for logging
        token_count = count_tokens(message_list, model="gpt-4o")
        print(f"📊 Token count for chat-all ({mode_info}): {token_count:,} tokens")
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_list,
            temperature=0.3,
            max_tokens=2000
        )
        
        answer = completion.choices[0].message.content
        answer_html = markdown.markdown(answer, extensions=['extra', 'nl2br', 'tables'])
        
        return jsonify({
            'answer': answer_html,
            'info': mode_info
        })
        
    except Exception as e:
        print(f"Error in chat-all: {str(e)}")
        return jsonify({'error': f'AI error: {str(e)}'}), 500
@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for individual study"""
    data = request.json
    nct_id = data.get('nctId')
    question = data.get('question')
    
    study = collection.find_one({'protocolSection.identificationModule.nctId': nct_id})
    if not study:
        return jsonify({'error': 'Study not found'}), 404
    
    study_copy = {k: v for k, v in study.items() if k != '_id'}
    study_context = json.dumps(study_copy, indent=2)
    
    estimated_tokens = len(study_context) / 4
    max_context_tokens = 6000
    if estimated_tokens > max_context_tokens:
        max_chars = max_context_tokens * 4
        study_context = study_context[:max_chars] + "\n\n[Context truncated due to length]"
    
    system_message = f"""You are a clinical trials expert. Answer questions about this study based on the data below.

STUDY DATA:
{study_context}

Instructions:
- Answer based only on the provided data
- If information isn't in the data, say so
- Be precise and reference specific fields when relevant
- Keep answers concise but complete"""
    
    try:
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        


        token_count = count_tokens(message_list, model="gpt-4o-mini")
        print(f"📊 Token count: {token_count:,} tokens")
        
        # Check against model limit
        MODEL_LIMITS = {
            "gpt-4o-mini": 128_000,
            "gpt-4.1": 1_000_000,
            "grok-4-fast-reasoning": 2_000_000,
            "gemini-2.5-pro": 1_000_000
        }

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

if __name__ == '__main__':
    print("=" * 60)
    print("Clinical Trials Search Application")
    print("=" * 60)
    print("Starting server...")
    print("Open your browser and go to: http://localhost:5033")
    print("=" * 60)
    app.run(debug=True, port=5033)

'''
how many studies do you see and how many of them have protocol pdf?

NCT01064466

what is the link?
what is the link to the protocol pdf?

what is your name?
id
clinical.repeatme.us

'''

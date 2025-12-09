let currentChatStudy = null;
let currentPage = 1;
let totalPages = 1;
let currentFilters = null;
let totalResults = 0;
let isAdvancedMode = false;

// PDF Download Function
function downloadReportAsPDF(elementId, filename) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    // Use html2pdf library if available, otherwise use window.print
    if (typeof html2pdf !== 'undefined') {
        const opt = {
            margin: 1,
            filename: filename + '.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
        };
        html2pdf().set(opt).from(element).save();
    } else {
        // Fallback: open print dialog
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
            <html>
                <head>
                    <title>${filename}</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.8; }
                        p { margin-bottom: 1em; }
                        h1, h2, h3, h4, h5, h6 { margin-top: 1.5em; margin-bottom: 0.5em; }
                    </style>
                </head>
                <body>
                    ${element.innerHTML}
                </body>
            </html>
        `);
        printWindow.document.close();
        printWindow.print();
    }
}

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
        // Get selected model
        const selectedModel = document.getElementById('llmModelSelector').value;

        const response = await fetch('/api/chat-all', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filters: currentFilters,
                question: question,
                advancedMode: isAdvancedMode,
                model: selectedModel  // Add selected model
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
    // Get intervention as text (split by comma if multiple)
    const interventionText = document.getElementById('intervention').value.trim();
    const intervention = interventionText ? interventionText.split(',').map(i => i.trim()).filter(i => i) : [];
    
    return {
        condition: document.getElementById('condition').value,
        intervention: intervention,
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
        useSemanticSearch: document.getElementById('useSemanticSearch').checked,
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

        // Show search type indicator
        if (data.searchType === 'semantic') {
            document.getElementById('totalCount').innerHTML = `${data.total.toLocaleString()} <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 5px;">🚀 AI SEMANTIC</span>`;
        } else {
            document.getElementById('totalCount').textContent = data.total.toLocaleString();
        }

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
    // totalCount is now set in searchStudies() based on search type
    
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

// =============================================================================
// PROTOCOL DESIGNER FUNCTIONS
// =============================================================================

function switchTab(tab) {
    const searchSection = document.getElementById('searchSection');
    const protocolDesigner = document.getElementById('protocolDesigner');
    const agentsSection = document.getElementById('agentsSection');
    const searchTab = document.getElementById('searchTab');
    const protocolTab = document.getElementById('protocolTab');
    const agentsTab = document.getElementById('agentsTab');

    // Hide all sections
    searchSection.style.display = 'none';
    protocolDesigner.style.display = 'none';
    agentsSection.style.display = 'none';

    // Reset all tab styles
    searchTab.style.background = '#ddd';
    searchTab.style.color = '#333';
    protocolTab.style.background = '#ddd';
    protocolTab.style.color = '#333';
    agentsTab.style.background = '#ddd';
    agentsTab.style.color = '#333';

    // Show selected section and highlight tab
    if (tab === 'search') {
        searchSection.style.display = 'block';
        searchTab.style.background = '#4CAF50';
        searchTab.style.color = 'white';
    } else if (tab === 'protocol') {
        protocolDesigner.style.display = 'block';
        protocolTab.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        protocolTab.style.color = 'white';
    } else if (tab === 'agents') {
        agentsSection.style.display = 'block';
        agentsTab.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        agentsTab.style.color = 'white';
    }
}

async function generateProtocolReport() {
    const condition = document.getElementById('protocolCondition').value.trim();
    const intervention = document.getElementById('protocolIntervention').value.trim();

    if (!condition) {
        alert('Please enter a condition or disease');
        return;
    }

    // Show loading, hide previous report
    document.getElementById('protocolLoading').style.display = 'block';
    document.getElementById('protocolReportContainer').style.display = 'none';

    try {
        const response = await fetch('/api/generate-protocol-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                condition: condition,
                intervention: intervention
            })
        });

        const data = await response.json();

        // Hide loading
        document.getElementById('protocolLoading').style.display = 'none';

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        // Display report with proper spacing
        const reportContent = data.report.replace(/<p>/g, '<p style="margin-bottom: 1em;">').replace(/<\/p>/g, '</p>');
        document.getElementById('protocolReportContent').innerHTML = reportContent;
        document.getElementById('protocolReportContainer').style.display = 'block';

        // Scroll to report
        document.getElementById('protocolReportContainer').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });

    } catch (error) {
        document.getElementById('protocolLoading').style.display = 'none';
        alert('Error generating report: ' + error.message);
    }
}

// =============================================================================
// AGENTIC AI FEATURES FUNCTIONS
// =============================================================================

// Helper function to show loading
function showAgentLoading(resultId) {
    const resultDiv = document.getElementById(resultId);
    resultDiv.innerHTML = '<div style="text-align: center; padding: 40px; color: #667eea;"><div style="font-size: 48px; margin-bottom: 15px;">⏳</div><h3>AI agents working...</h3><p>Multiple specialized agents are analyzing your request</p></div>';
    resultDiv.style.display = 'block';
}

// Helper function to show error
function showAgentError(resultId, message) {
    const resultDiv = document.getElementById(resultId);
    resultDiv.innerHTML = `<div style="background: #fee; color: #c00; padding: 15px; border-radius: 5px; border-left: 4px solid #c00;"><strong>Error:</strong> ${message}</div>`;
    resultDiv.style.display = 'block';
}

// Demo 1: Agentic Search Enhancement
async function runAgenticSearch() {
    const query = document.getElementById('agenticSearchQuery').value.trim();
    if (!query) {
        alert('Please enter a search query');
        return;
    }

    const btn = document.getElementById('agenticSearchBtn');
    btn.disabled = true;
    showAgentLoading('agenticSearchResult');

    try {
        const response = await fetch('/api/agentic-search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });

        const data = await response.json();

        if (data.success) {
            const term = data.terminology_expansion;
            const strategy = data.search_strategy;

            let html = '<div style="background: #efe; color: #060; padding: 15px; border-radius: 5px; border-left: 4px solid #060; margin-bottom: 15px;">✅ Search enhancement complete!</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea;">';
            html += '<h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">📚 Medical Terminology Agent</h3>';
            html += `<p><strong>Synonyms:</strong> ${term.synonyms.join(', ')}</p>`;
            html += `<p><strong>Related Terms:</strong> ${term.related_terms.join(', ')}</p>`;
            html += `<p><strong>Abbreviations:</strong> ${term.abbreviations.join(', ')}</p>`;
            html += '</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea;">';
            html += '<h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">🎯 Search Strategy Agent</h3>';
            html += `<p><strong>Strategy:</strong> ${strategy.boolean_strategy}</p>`;
            html += `<p><strong>Priority Terms:</strong> ${strategy.priority_terms.join(' > ')}</p>`;
            html += '</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea;">';
            html += '<h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">✨ Enhanced Search Terms</h3>';
            html += `<p style="margin-bottom: 1em;">${data.enhanced_search_terms.join(', ')}</p>`;
            html += '</div>';

            html += '<div style="text-align: right; margin-top: 15px;"><button onclick="downloadReportAsPDF(\'agenticSearchResult\', \'Agentic-Search-Report\')" style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">📥 Download as PDF</button></div>';

            document.getElementById('agenticSearchResult').innerHTML = html;
        } else {
            showAgentError('agenticSearchResult', data.error || 'Unknown error');
        }
    } catch (error) {
        showAgentError('agenticSearchResult', error.message);
    } finally {
        btn.disabled = false;
    }
}

// Demo 2: Multi-Agent Protocol Analysis
async function runMultiAgentAnalysis() {
    const nctId = document.getElementById('analysisNctId').value.trim();
    if (!nctId) {
        alert('Please enter an NCT ID');
        return;
    }

    const btn = document.getElementById('analysisBtn');
    btn.disabled = true;
    showAgentLoading('analysisResult');

    try {
        const response = await fetch('/api/multi-agent-analysis', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({nctId})
        });

        const data = await response.json();

        if (data.success) {
            let html = '<div style="background: #efe; color: #060; padding: 15px; border-radius: 5px; border-left: 4px solid #060; margin-bottom: 15px;">✅ Multi-agent analysis complete!</div>';

            html += `<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea;"><h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">Trial: ${data.trial.nct_id}</h3><p>${data.trial.title}</p></div>`;

            data.agent_analyses.forEach(agent => {
                html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea; line-height: 1.8;">';
                html += `<h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">${agent.agent}</h3>`;
                html += `<div style="color: #888; font-size: 12px; margin-bottom: 10px; font-style: italic;">Focus: ${agent.focus_areas.join(', ')}</div>`;
                html += agent.content.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
                html += '</div>';
            });

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #d4af37; line-height: 1.8;">';
            html += '<h3 style="color: #d4af37; font-size: 16px; margin-bottom: 10px;">👔 Executive Summary (Chief Strategist)</h3>';
            html += data.executive_summary.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
            html += '</div>';

            html += '<div style="text-align: right; margin-top: 15px;"><button onclick="downloadReportAsPDF(\'analysisResult\', \'Multi-Agent-Analysis-Report\')" style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">📥 Download as PDF</button></div>';

            document.getElementById('analysisResult').innerHTML = html;
        } else {
            showAgentError('analysisResult', data.error || 'Unknown error');
        }
    } catch (error) {
        showAgentError('analysisResult', error.message);
    } finally {
        btn.disabled = false;
    }
}

// Demo 3: Multi-Agent Trial Comparison
async function runTrialComparison() {
    const nct1 = document.getElementById('compareNct1').value.trim();
    const nct2 = document.getElementById('compareNct2').value.trim();
    const nct3 = document.getElementById('compareNct3').value.trim();

    if (!nct1 || !nct2) {
        alert('Please enter at least 2 NCT IDs');
        return;
    }

    const nctIds = [nct1, nct2];
    if (nct3) nctIds.push(nct3);

    const btn = document.getElementById('compareBtn');
    btn.disabled = true;
    showAgentLoading('compareResult');

    try {
        const response = await fetch('/api/compare-trials', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({nctIds})
        });

        const data = await response.json();

        if (data.success) {
            let html = '<div style="background: #efe; color: #060; padding: 15px; border-radius: 5px; border-left: 4px solid #060; margin-bottom: 15px;">✅ Multi-agent comparison complete!</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea;"><h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">Trials Compared</h3><ul>';
            data.trials.forEach(t => {
                html += `<li><strong>${t.nct_id}</strong>: ${t.title}</li>`;
            });
            html += '</ul></div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea; line-height: 1.8;">';
            html += '<h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">👥 Eligibility Comparison</h3>';
            html += data.comparisons.eligibility.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
            html += '</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea; line-height: 1.8;">';
            html += '<h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">📐 Design Comparison</h3>';
            html += data.comparisons.design.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
            html += '</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #667eea; line-height: 1.8;">';
            html += '<h3 style="color: #667eea; font-size: 16px; margin-bottom: 10px;">🎯 Endpoints Comparison</h3>';
            html += data.comparisons.endpoints.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
            html += '</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #d4af37; line-height: 1.8;">';
            html += '<h3 style="color: #d4af37; font-size: 16px; margin-bottom: 10px;">🧠 Strategic Synthesis</h3>';
            html += data.strategic_synthesis.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
            html += '</div>';

            html += '<div style="text-align: right; margin-top: 15px;"><button onclick="downloadReportAsPDF(\'compareResult\', \'Trial-Comparison-Report\')" style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">📥 Download as PDF</button></div>';

            document.getElementById('compareResult').innerHTML = html;
        } else {
            showAgentError('compareResult', data.error || 'Unknown error');
        }
    } catch (error) {
        showAgentError('compareResult', error.message);
    } finally {
        btn.disabled = false;
    }
}


// =============================================================================
// NEW CAPSTONE FEATURES - AMENDMENT, PATTERNS, SOA
// =============================================================================

// Feature 4: Amendment Risk Predictor
async function runAmendmentRisk() {
    const nctId = document.getElementById('amendmentNctId').value.trim();
    if (!nctId) {
        alert('Please enter an NCT ID');
        return;
    }

    const btn = document.getElementById('amendmentBtn');
    btn.disabled = true;
    showAgentLoading('amendmentResult');

    try {
        const response = await fetch('/api/amendment-risk', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({nctId})
        });

        const data = await response.json();

        if (data.success) {
            let html = '<div style="background: #efe; color: #060; padding: 15px; border-radius: 5px; border-left: 4px solid #060; margin-bottom: 15px;">✅ Amendment risk analysis complete!</div>';

            html += `<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #ff6b6b;"><h3 style="color: #ff6b6b; font-size: 16px; margin-bottom: 10px;">Trial: ${data.trial.nct_id}</h3><p>${data.trial.title}</p></div>`;

            data.agent_analyses.forEach(agent => {
                html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #ff6b6b; line-height: 1.8;">';
                html += `<h3 style="color: #ff6b6b; font-size: 16px; margin-bottom: 10px;">${agent.agent}</h3>`;
                html += `<div style="color: #888; font-size: 12px; margin-bottom: 10px; font-style: italic;">Focus: ${agent.focus_areas.join(', ')}</div>`;
                html += agent.content.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
                html += '</div>';
            });

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #d4af37; line-height: 1.8;">';
            html += '<h3 style="color: #d4af37; font-size: 16px; margin-bottom: 10px;">⚠️ Overall Risk Assessment</h3>';
            html += data.risk_assessment.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
            html += '</div>';

            html += '<div style="text-align: right; margin-top: 15px;"><button onclick="downloadReportAsPDF(\'amendmentResult\', \'Amendment-Risk-Report\')" style="padding: 10px 20px; background: #ff6b6b; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">📥 Download as PDF</button></div>';

            document.getElementById('amendmentResult').innerHTML = html;
        } else {
            showAgentError('amendmentResult', data.error || 'Unknown error');
        }
    } catch (error) {
        showAgentError('amendmentResult', error.message);
    } finally {
        btn.disabled = false;
    }
}

// Feature 5: Design Pattern Discovery
async function runDesignPatterns() {
    const condition = document.getElementById('patternCondition').value.trim();
    const phase = document.getElementById('patternPhase').value;
    const interventionType = document.getElementById('patternIntervention').value.trim();

    if (!condition) {
        alert('Please enter a condition');
        return;
    }

    const btn = document.getElementById('patternBtn');
    btn.disabled = true;
    showAgentLoading('patternResult');

    try {
        const response = await fetch('/api/design-patterns', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({condition, phase: phase || null, interventionType: interventionType || null})
        });

        const data = await response.json();

        if (data.success) {
            let html = '<div style="background: #efe; color: #060; padding: 15px; border-radius: 5px; border-left: 4px solid #060; margin-bottom: 15px;">✅ Design pattern analysis complete!</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #4ecdc4;">';
            html += '<h3 style="color: #4ecdc4; font-size: 16px; margin-bottom: 10px;">Query Summary</h3>';
            html += `<p><strong>Condition:</strong> ${data.query.condition}</p>`;
            html += `<p><strong>Phase:</strong> ${data.query.phase || 'All phases'}</p>`;
            html += `<p><strong>Trials Analyzed:</strong> ${data.query.trials_analyzed}</p>`;
            if (data.trials_by_phase) {
                html += `<p><strong>Distribution:</strong> ${Object.entries(data.trials_by_phase).map(([k,v]) => k + ': ' + v).join(', ')}</p>`;
            }
            html += '</div>';

            data.agent_analyses.forEach(agent => {
                html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #4ecdc4; line-height: 1.8;">';
                html += `<h3 style="color: #4ecdc4; font-size: 16px; margin-bottom: 10px;">${agent.agent}</h3>`;
                html += `<div style="color: #888; font-size: 12px; margin-bottom: 10px; font-style: italic;">Focus: ${agent.focus_areas.join(', ')}</div>`;
                html += agent.content.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
                html += '</div>';
            });

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #d4af37; line-height: 1.8;">';
            html += '<h3 style="color: #d4af37; font-size: 16px; margin-bottom: 10px;">🧠 Strategic Design Blueprint</h3>';
            html += data.strategic_insights.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
            html += '</div>';

            html += '<div style="text-align: right; margin-top: 15px;"><button onclick="downloadReportAsPDF(\'patternResult\', \'Design-Pattern-Report\')" style="padding: 10px 20px; background: #4ecdc4; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">📥 Download as PDF</button></div>';

            document.getElementById('patternResult').innerHTML = html;
        } else {
            showAgentError('patternResult', data.error || 'Unknown error');
        }
    } catch (error) {
        showAgentError('patternResult', error.message);
    } finally {
        btn.disabled = false;
    }
}

// Feature 6: SoA Composer
async function runSoAComposer() {
    const condition = document.getElementById('soaCondition').value.trim();
    const phase = document.getElementById('soaPhase').value;
    const interventionType = document.getElementById('soaIntervention').value.trim();

    if (!condition) {
        alert('Please enter a condition');
        return;
    }

    const btn = document.getElementById('soaBtn');
    btn.disabled = true;
    showAgentLoading('soaResult');

    try {
        const response = await fetch('/api/soa-composer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({condition, phase: phase || null, interventionType: interventionType || null})
        });

        const data = await response.json();

        if (data.success) {
            let html = '<div style="background: #efe; color: #060; padding: 15px; border-radius: 5px; border-left: 4px solid #060; margin-bottom: 15px;">✅ SoA composition complete!</div>';

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #f7b731;">';
            html += '<h3 style="color: #f7b731; font-size: 16px; margin-bottom: 10px;">Query Summary</h3>';
            html += `<p><strong>Condition:</strong> ${data.query.condition}</p>`;
            html += `<p><strong>Phase:</strong> ${data.query.phase || 'All phases'}</p>`;
            html += `<p><strong>Trials Analyzed:</strong> ${data.query.trials_analyzed}</p>`;
            html += '</div>';

            if (data.reference_trials && data.reference_trials.length > 0) {
                html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #f7b731;">';
                html += '<h3 style="color: #f7b731; font-size: 16px; margin-bottom: 10px;">📚 Reference Trials</h3>';
                html += '<ul style="margin: 0; padding-left: 20px;">';
                data.reference_trials.slice(0, 5).forEach(t => {
                    html += `<li><strong>${t.nct_id}</strong>: ${t.primary_outcome} (${t.primary_timeframe})</li>`;
                });
                html += '</ul></div>';
            }

            data.agent_analyses.forEach(agent => {
                html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #f7b731; line-height: 1.8;">';
                html += `<h3 style="color: #f7b731; font-size: 16px; margin-bottom: 10px;">${agent.agent}</h3>`;
                html += `<div style="color: #888; font-size: 12px; margin-bottom: 10px; font-style: italic;">Focus: ${agent.focus_areas.join(', ')}</div>`;
                html += agent.content.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
                html += '</div>';
            });

            html += '<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #d4af37; line-height: 1.8;">';
            html += '<h3 style="color: #d4af37; font-size: 16px; margin-bottom: 10px;">📋 Complete Schedule of Assessments</h3>';
            html += data.complete_soa.replace(/<p>/g, '<p style="margin-bottom: 1em;">');
            html += '</div>';

            html += '<div style="text-align: right; margin-top: 15px;"><button onclick="downloadReportAsPDF(\'soaResult\', \'SoA-Report\')" style="padding: 10px 20px; background: #f7b731; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">📥 Download as PDF</button></div>';

            document.getElementById('soaResult').innerHTML = html;
        } else {
            showAgentError('soaResult', data.error || 'Unknown error');
        }
    } catch (error) {
        showAgentError('soaResult', error.message);
    } finally {
        btn.disabled = false;
    }
}

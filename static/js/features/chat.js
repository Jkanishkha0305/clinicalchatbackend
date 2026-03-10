import { requestJson } from "../core/api.js";
import { byId } from "../core/dom.js";
import { appState } from "../core/state.js";

function removeElementById(id) {
    const element = byId(id);
    if (element) {
        element.remove();
    }
}

export function addModalMessage(text, type) {
    const messagesDiv = byId("chatMessages");
    const messageDiv = document.createElement("div");
    const id = `msg-${Date.now()}`;
    messageDiv.id = id;
    messageDiv.className = `chat-message ${type}`;
    if (type === "assistant") {
        messageDiv.innerHTML = text;
    } else {
        messageDiv.textContent = text;
    }
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return id;
}

export function addSidebarMessage(text, type) {
    const messagesDiv = byId("sidebarChatMessages");
    const placeholder = messagesDiv.querySelector(".chat-placeholder");
    if (placeholder) {
        placeholder.remove();
    }

    const messageDiv = document.createElement("div");
    const id = `sidebar-msg-${Date.now()}`;
    messageDiv.id = id;
    messageDiv.className = `chat-message ${type}`;
    if (type === "assistant") {
        messageDiv.innerHTML = text;
    } else {
        messageDiv.textContent = text;
    }

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return id;
}

export function openChatModal(nctId, title) {
    appState.currentChatStudy = nctId;
    byId("chatStudyTitle").textContent = `Chat about ${nctId}`;
    byId("chatModal").style.display = "block";
    byId("chatMessages").innerHTML = `
        <div style="text-align: center; color: #666; padding: 20px;">
            <strong>${title}</strong><br><br>
            Ask any question about this study.
        </div>
    `;
}

export function closeChatModal() {
    byId("chatModal").style.display = "none";
    appState.currentChatStudy = null;
}

export async function sendMessage() {
    const input = byId("chatInput");
    const question = input.value.trim();
    if (!question || !appState.currentChatStudy) {
        return;
    }

    addModalMessage(question, "user");
    input.value = "";
    const loadingId = addModalMessage("Analyzing study data...", "loading");

    try {
        const data = await requestJson("/api/chat", {
            nctId: appState.currentChatStudy,
            question,
        });
        removeElementById(loadingId);
        addModalMessage(data.answer || "No answer returned.", "assistant");
    } catch (error) {
        removeElementById(loadingId);
        addModalMessage(`Error: ${error.message}`, "assistant");
    }
}

export function setSidebarChatEnabled(enabled) {
    byId("sidebarChatInput").disabled = !enabled;
    byId("sidebarSendBtn").disabled = !enabled;
}

export function resetSidebarChat() {
    byId("sidebarChatMessages").innerHTML = `
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
    `;
    setSidebarChatEnabled(false);
}

export async function sendSidebarMessage() {
    const input = byId("sidebarChatInput");
    const question = input.value.trim();
    if (!question || !appState.currentFilters) {
        return;
    }

    addSidebarMessage(question, "user");
    input.value = "";
    setSidebarChatEnabled(false);

    const modeText = appState.isAdvancedMode ? "ALL complete study data" : "essential study data";
    const loadingId = addSidebarMessage(`Analyzing ${modeText} from filtered studies...`, "loading");

    try {
        const data = await requestJson("/api/chat-all", {
            filters: appState.currentFilters,
            question,
            advancedMode: appState.isAdvancedMode,
            model: byId("llmModelSelector").value,
        });

        removeElementById(loadingId);
        let answer = data.answer || "No answer returned.";
        if (data.info) {
            answer = `
                <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 10px; margin-bottom: 10px; font-size: 12px;">
                    <strong>ℹ️ Analysis Info:</strong> ${data.info}
                </div>
                ${answer}
            `;
        }
        addSidebarMessage(answer, "assistant");
    } catch (error) {
        removeElementById(loadingId);
        addSidebarMessage(`Error: ${error.message}`, "assistant");
    } finally {
        setSidebarChatEnabled(true);
    }
}

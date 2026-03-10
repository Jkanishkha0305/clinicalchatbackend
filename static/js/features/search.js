import { requestJson } from "../core/api.js";
import { byId } from "../core/dom.js";
import { appState } from "../core/state.js";
import { openChatModal, resetSidebarChat, setSidebarChatEnabled } from "./chat.js";

function getStatusClass(status) {
    if (status === "RECRUITING") {
        return "recruiting";
    }
    if (status === "COMPLETED") {
        return "completed";
    }
    return "";
}

function createMetaItem(label, value, className = "meta-value") {
    const item = document.createElement("div");
    item.className = "meta-item";
    item.innerHTML = `<span class="meta-label">${label}:</span><span class="${className}">${value}</span>`;
    return item;
}

function createStudyCard(study) {
    const protocol = study.protocolSection;
    const identification = protocol.identificationModule;
    const status = protocol.statusModule;
    const design = protocol.designModule;
    const sponsor = protocol.sponsorCollaboratorsModule;

    const card = document.createElement("div");
    card.className = "study-card";

    const heading = document.createElement("h3");
    heading.textContent = `${identification.nctId}: ${identification.briefTitle || "No title"}`;
    heading.addEventListener("click", () => {
        window.open(`https://clinicaltrials.gov/study/${identification.nctId}`, "_blank");
    });
    card.appendChild(heading);

    const meta = document.createElement("div");
    meta.className = "study-meta";
    meta.appendChild(
        createMetaItem(
            "Status",
            `<span class="badge ${getStatusClass(status.overallStatus)}">${status.overallStatus}</span>`,
            "",
        ),
    );
    meta.appendChild(createMetaItem("Type", design.studyType));
    meta.appendChild(createMetaItem("Phase", design.phases?.join(", ") || "N/A"));
    meta.appendChild(createMetaItem("Sponsor", sponsor?.leadSponsor?.name || "N/A"));
    meta.appendChild(
        createMetaItem(
            "Results",
            study.hasResults
                ? '<span class="badge has-results">Has Results</span>'
                : "No Results",
            study.hasResults ? "" : "meta-value",
        ),
    );
    card.appendChild(meta);

    const button = document.createElement("button");
    button.textContent = "💬 Ask About This Study";
    button.style.marginTop = "10px";
    button.style.padding = "8px 16px";
    button.style.background = "#28a745";
    button.style.color = "white";
    button.style.border = "none";
    button.style.borderRadius = "4px";
    button.style.cursor = "pointer";
    button.addEventListener("click", (event) => {
        event.stopPropagation();
        openChatModal(identification.nctId, identification.briefTitle || "");
    });
    card.appendChild(button);

    return card;
}

function displayResults(data) {
    const container = byId("resultsContainer");
    container.innerHTML = "";

    if (!data.results.length) {
        container.innerHTML = "<p>No studies found matching your criteria.</p>";
        return;
    }

    data.results.forEach((study) => {
        container.appendChild(createStudyCard(study));
    });
}

function displayPagination() {
    const container = byId("pagination");
    container.innerHTML = "";

    if (appState.totalPages <= 1) {
        return;
    }

    const prevButton = document.createElement("button");
    prevButton.textContent = "« Previous";
    prevButton.disabled = appState.currentPage === 1;
    prevButton.onclick = () => searchStudies(appState.currentPage - 1);
    container.appendChild(prevButton);

    const startPage = Math.max(1, appState.currentPage - 3);
    const endPage = Math.min(appState.totalPages, appState.currentPage + 3);
    for (let page = startPage; page <= endPage; page += 1) {
        const button = document.createElement("button");
        button.textContent = page;
        if (page === appState.currentPage) {
            button.className = "active";
        }
        button.onclick = () => searchStudies(page);
        container.appendChild(button);
    }

    const nextButton = document.createElement("button");
    nextButton.textContent = "Next »";
    nextButton.disabled = appState.currentPage === appState.totalPages;
    nextButton.onclick = () => searchStudies(appState.currentPage + 1);
    container.appendChild(nextButton);
}

function updateResultsCount(data) {
    const totalCount = byId("totalCount");
    if (data.searchType && data.searchType.startsWith("semantic")) {
        totalCount.innerHTML = `${data.total.toLocaleString()} <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 5px;">🚀 AI SEMANTIC</span>`;
        return;
    }
    totalCount.textContent = data.total.toLocaleString();
}

function updateAdvancedWarning() {
    const warning = byId("advancedWarning");
    if (appState.isAdvancedMode && appState.totalResults > 50) {
        warning.classList.add("show");
    } else {
        warning.classList.remove("show");
    }
}

export function getFilters() {
    const interventionText = byId("intervention").value.trim();
    const intervention = interventionText
        ? interventionText.split(",").map((item) => item.trim()).filter(Boolean)
        : [];

    return {
        condition: byId("condition").value,
        intervention,
        location: byId("location").value,
        status: Array.from(byId("status").selectedOptions).map((option) => option.value),
        studyType: Array.from(byId("studyType").selectedOptions).map((option) => option.value),
        phase: Array.from(byId("phase").selectedOptions).map((option) => option.value),
        sex: byId("sex").value,
        ageGroups: Array.from(document.querySelectorAll('input[name="ageGroups"]:checked')).map((checkbox) => checkbox.value),
        healthyVolunteers: byId("healthyVolunteers").checked,
        hasResults: byId("hasResults").value,
        hasProtocol: byId("hasProtocol").checked,
        hasSAP: byId("hasSAP").checked,
        hasICF: byId("hasICF").checked,
        funderType: Array.from(byId("funderType").selectedOptions).map((option) => option.value),
        studyStartFrom: byId("studyStartFrom").value,
        studyStartTo: byId("studyStartTo").value,
        primaryCompletionFrom: byId("primaryCompletionFrom").value,
        primaryCompletionTo: byId("primaryCompletionTo").value,
        title: byId("title").value,
        outcome: byId("outcome").value,
        sponsor: byId("sponsor").value,
        nctId: byId("nctId").value,
        fdaaa801Violation: byId("fdaaa801Violation").checked,
        useSemanticSearch: byId("useSemanticSearch").checked,
        page: appState.currentPage,
        per_page: Number.parseInt(byId("perPage").value, 10),
    };
}

export async function searchStudies(page = 1) {
    appState.currentPage = page;
    const filters = getFilters();
    appState.currentFilters = filters;

    try {
        const data = await requestJson("/api/search", filters);
        appState.totalPages = data.total_pages;
        appState.totalResults = data.total;
        updateResultsCount(data);
        displayResults(data);
        displayPagination();
        setSidebarChatEnabled(true);
        updateAdvancedWarning();
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

export function clearFilters() {
    document.querySelectorAll('input[type="text"], input[type="date"]').forEach((input) => {
        input.value = "";
    });
    document.querySelectorAll("select").forEach((select) => {
        if (select.id === "perPage") {
            return;
        }
        Array.from(select.options).forEach((option) => {
            option.selected = false;
        });
    });
    document.querySelectorAll('input[type="checkbox"]').forEach((checkbox) => {
        checkbox.checked = false;
    });

    appState.currentPage = 1;
    appState.currentFilters = null;
    appState.totalPages = 1;
    appState.totalResults = 0;

    byId("resultsContainer").innerHTML = "";
    byId("pagination").innerHTML = "";
    byId("totalCount").textContent = "0";
    resetSidebarChat();
    updateAdvancedWarning();
}

export function toggleAdvancedMode() {
    appState.isAdvancedMode = byId("advancedModeCheckbox").checked;
    const badge = byId("modeBadge");
    if (appState.isAdvancedMode) {
        badge.textContent = "ADVANCED";
        badge.className = "mode-badge advanced";
    } else {
        badge.textContent = "ESSENTIAL";
        badge.className = "mode-badge essential";
    }
    updateAdvancedWarning();
}

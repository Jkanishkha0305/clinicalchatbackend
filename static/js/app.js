import { resetSidebarChat, openChatModal, closeChatModal, sendMessage, sendSidebarMessage } from "./features/chat.js";
import { downloadReportAsPDF } from "./features/pdf.js";
import { generateProtocolReport } from "./features/protocol.js";
import { clearFilters, searchStudies, toggleAdvancedMode } from "./features/search.js";
import { switchTab } from "./features/tabs.js";
import {
    runAgenticSearch,
    runAmendmentRisk,
    runDesignPatterns,
    runMultiAgentAnalysis,
    runSoAComposer,
    runTrialComparison,
} from "./features/agents.js";

Object.assign(window, {
    clearFilters,
    closeChatModal,
    downloadReportAsPDF,
    generateProtocolReport,
    openChatModal,
    runAgenticSearch,
    runAmendmentRisk,
    runDesignPatterns,
    runMultiAgentAnalysis,
    runSoAComposer,
    runTrialComparison,
    searchStudies,
    sendMessage,
    sendSidebarMessage,
    switchTab,
    toggleAdvancedMode,
});

document.addEventListener("DOMContentLoaded", () => {
    resetSidebarChat();
});

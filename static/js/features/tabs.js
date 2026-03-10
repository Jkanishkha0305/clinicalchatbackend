import { byId } from "../core/dom.js";

const TAB_CONFIG = {
    search: {
        sectionId: "searchSection",
        tabId: "searchTab",
        background: "#4CAF50",
        color: "white",
    },
    protocol: {
        sectionId: "protocolDesigner",
        tabId: "protocolTab",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        color: "white",
    },
    agents: {
        sectionId: "agentsSection",
        tabId: "agentsTab",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        color: "white",
    },
};

export function switchTab(activeTab) {
    Object.entries(TAB_CONFIG).forEach(([name, config]) => {
        const section = byId(config.sectionId);
        const tab = byId(config.tabId);
        section.style.display = name === activeTab ? "block" : "none";
        tab.style.background = name === activeTab ? config.background : "#ddd";
        tab.style.color = name === activeTab ? config.color : "#333";
    });
}

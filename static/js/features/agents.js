import { requestJson } from "../core/api.js";
import { byId, withParagraphSpacing } from "../core/dom.js";

function showAgentLoading(resultId) {
    const resultDiv = byId(resultId);
    resultDiv.innerHTML = `
        <div style="text-align: center; padding: 40px; color: #667eea;">
            <div style="font-size: 48px; margin-bottom: 15px;">⏳</div>
            <h3>AI agents working...</h3>
            <p>Multiple specialized agents are analyzing your request</p>
        </div>
    `;
    resultDiv.style.display = "block";
}

function showAgentError(resultId, message) {
    const resultDiv = byId(resultId);
    resultDiv.innerHTML = `
        <div style="background: #fee; color: #c00; padding: 15px; border-radius: 5px; border-left: 4px solid #c00;">
            <strong>Error:</strong> ${message}
        </div>
    `;
    resultDiv.style.display = "block";
}

function renderBanner(message) {
    return `<div style="background: #efe; color: #060; padding: 15px; border-radius: 5px; border-left: 4px solid #060; margin-bottom: 15px;">✅ ${message}</div>`;
}

function renderPanel({ title, accent, content, subtitle = "", borderAccent = accent }) {
    const subtitleHtml = subtitle
        ? `<div style="color: #888; font-size: 12px; margin-bottom: 10px; font-style: italic;">${subtitle}</div>`
        : "";
    return `
        <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid ${borderAccent}; line-height: 1.8;">
            <h3 style="color: ${accent}; font-size: 16px; margin-bottom: 10px;">${title}</h3>
            ${subtitleHtml}
            ${content}
        </div>
    `;
}

function renderDownloadButton(resultId, filename, color) {
    return `
        <div style="text-align: right; margin-top: 15px;">
            <button onclick="downloadReportAsPDF('${resultId}', '${filename}')" style="padding: 10px 20px; background: ${color}; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
                📥 Download as PDF
            </button>
        </div>
    `;
}

async function runAgentAction({ buttonId, resultId, payload, endpoint, buildHtml }) {
    const button = byId(buttonId);
    button.disabled = true;
    showAgentLoading(resultId);

    try {
        const data = await requestJson(endpoint, payload);
        byId(resultId).innerHTML = buildHtml(data);
    } catch (error) {
        showAgentError(resultId, error.message);
    } finally {
        button.disabled = false;
    }
}

export function runAgenticSearch() {
    const query = byId("agenticSearchQuery").value.trim();
    if (!query) {
        alert("Please enter a search query");
        return;
    }

    runAgentAction({
        buttonId: "agenticSearchBtn",
        resultId: "agenticSearchResult",
        endpoint: "/api/agentic-search",
        payload: { query },
        buildHtml: (data) => {
            const terminology = data.terminology_expansion;
            const strategy = data.search_strategy;
            return `
                ${renderBanner("Search enhancement complete!")}
                ${renderPanel({
                    title: "📚 Medical Terminology Agent",
                    accent: "#667eea",
                    content: `
                        <p><strong>Synonyms:</strong> ${terminology.synonyms.join(", ")}</p>
                        <p><strong>Related Terms:</strong> ${terminology.related_terms.join(", ")}</p>
                        <p><strong>Abbreviations:</strong> ${terminology.abbreviations.join(", ")}</p>
                    `,
                })}
                ${renderPanel({
                    title: "🎯 Search Strategy Agent",
                    accent: "#667eea",
                    content: `
                        <p><strong>Strategy:</strong> ${strategy.boolean_strategy}</p>
                        <p><strong>Priority Terms:</strong> ${strategy.priority_terms.join(" > ")}</p>
                    `,
                })}
                ${renderPanel({
                    title: "✨ Enhanced Search Terms",
                    accent: "#667eea",
                    content: `<p style="margin-bottom: 1em;">${data.enhanced_search_terms.join(", ")}</p>`,
                })}
                ${renderDownloadButton("agenticSearchResult", "Agentic-Search-Report", "#667eea")}
            `;
        },
    });
}

export function runMultiAgentAnalysis() {
    const nctId = byId("analysisNctId").value.trim();
    if (!nctId) {
        alert("Please enter an NCT ID");
        return;
    }

    runAgentAction({
        buttonId: "analysisBtn",
        resultId: "analysisResult",
        endpoint: "/api/multi-agent-analysis",
        payload: { nctId },
        buildHtml: (data) => `
            ${renderBanner("Multi-agent analysis complete!")}
            ${renderPanel({
                title: `Trial: ${data.trial.nct_id}`,
                accent: "#667eea",
                content: `<p>${data.trial.title}</p>`,
            })}
            ${data.agent_analyses.map((agent) => renderPanel({
                title: agent.agent,
                accent: "#667eea",
                subtitle: `Focus: ${agent.focus_areas.join(", ")}`,
                content: withParagraphSpacing(agent.content),
            })).join("")}
            ${renderPanel({
                title: "👔 Executive Summary (Chief Strategist)",
                accent: "#d4af37",
                borderAccent: "#d4af37",
                content: withParagraphSpacing(data.executive_summary),
            })}
            ${renderDownloadButton("analysisResult", "Multi-Agent-Analysis-Report", "#667eea")}
        `,
    });
}

export function runTrialComparison() {
    const nctIds = [
        byId("compareNct1").value.trim(),
        byId("compareNct2").value.trim(),
        byId("compareNct3").value.trim(),
    ].filter(Boolean);

    if (nctIds.length < 2) {
        alert("Please enter at least 2 NCT IDs");
        return;
    }

    runAgentAction({
        buttonId: "compareBtn",
        resultId: "compareResult",
        endpoint: "/api/compare-trials",
        payload: { nctIds },
        buildHtml: (data) => `
            ${renderBanner("Multi-agent comparison complete!")}
            ${renderPanel({
                title: "Trials Compared",
                accent: "#667eea",
                content: `<ul>${data.trials.map((trial) => `<li><strong>${trial.nct_id}</strong>: ${trial.title}</li>`).join("")}</ul>`,
            })}
            ${renderPanel({ title: "👥 Eligibility Comparison", accent: "#667eea", content: withParagraphSpacing(data.comparisons.eligibility) })}
            ${renderPanel({ title: "📐 Design Comparison", accent: "#667eea", content: withParagraphSpacing(data.comparisons.design) })}
            ${renderPanel({ title: "🎯 Endpoints Comparison", accent: "#667eea", content: withParagraphSpacing(data.comparisons.endpoints) })}
            ${renderPanel({
                title: "🧠 Strategic Synthesis",
                accent: "#d4af37",
                borderAccent: "#d4af37",
                content: withParagraphSpacing(data.strategic_synthesis),
            })}
            ${renderDownloadButton("compareResult", "Trial-Comparison-Report", "#667eea")}
        `,
    });
}

export function runAmendmentRisk() {
    const nctId = byId("amendmentNctId").value.trim();
    if (!nctId) {
        alert("Please enter an NCT ID");
        return;
    }

    runAgentAction({
        buttonId: "amendmentBtn",
        resultId: "amendmentResult",
        endpoint: "/api/amendment-risk",
        payload: { nctId },
        buildHtml: (data) => `
            ${renderBanner("Amendment risk analysis complete!")}
            ${renderPanel({
                title: `Trial: ${data.trial.nct_id}`,
                accent: "#ff6b6b",
                borderAccent: "#ff6b6b",
                content: `<p>${data.trial.title}</p>`,
            })}
            ${data.agent_analyses.map((agent) => renderPanel({
                title: agent.agent,
                accent: "#ff6b6b",
                borderAccent: "#ff6b6b",
                subtitle: `Focus: ${agent.focus_areas.join(", ")}`,
                content: withParagraphSpacing(agent.content),
            })).join("")}
            ${renderPanel({
                title: "⚠️ Overall Risk Assessment",
                accent: "#d4af37",
                borderAccent: "#d4af37",
                content: withParagraphSpacing(data.risk_assessment),
            })}
            ${renderDownloadButton("amendmentResult", "Amendment-Risk-Report", "#ff6b6b")}
        `,
    });
}

export function runDesignPatterns() {
    const condition = byId("patternCondition").value.trim();
    const phase = byId("patternPhase").value;
    const interventionType = byId("patternIntervention").value.trim();
    if (!condition) {
        alert("Please enter a condition");
        return;
    }

    runAgentAction({
        buttonId: "patternBtn",
        resultId: "patternResult",
        endpoint: "/api/design-patterns",
        payload: { condition, phase: phase || null, interventionType: interventionType || null },
        buildHtml: (data) => `
            ${renderBanner("Design pattern analysis complete!")}
            ${renderPanel({
                title: "Query Summary",
                accent: "#4ecdc4",
                borderAccent: "#4ecdc4",
                content: `
                    <p><strong>Condition:</strong> ${data.query.condition}</p>
                    <p><strong>Phase:</strong> ${data.query.phase || "All phases"}</p>
                    <p><strong>Trials Analyzed:</strong> ${data.query.trials_analyzed}</p>
                    ${data.trials_by_phase ? `<p><strong>Distribution:</strong> ${Object.entries(data.trials_by_phase).map(([key, value]) => `${key}: ${value}`).join(", ")}</p>` : ""}
                `,
            })}
            ${data.agent_analyses.map((agent) => renderPanel({
                title: agent.agent,
                accent: "#4ecdc4",
                borderAccent: "#4ecdc4",
                subtitle: `Focus: ${agent.focus_areas.join(", ")}`,
                content: withParagraphSpacing(agent.content),
            })).join("")}
            ${renderPanel({
                title: "🧠 Strategic Design Blueprint",
                accent: "#d4af37",
                borderAccent: "#d4af37",
                content: withParagraphSpacing(data.strategic_insights),
            })}
            ${renderDownloadButton("patternResult", "Design-Pattern-Report", "#4ecdc4")}
        `,
    });
}

export function runSoAComposer() {
    const condition = byId("soaCondition").value.trim();
    const phase = byId("soaPhase").value;
    const interventionType = byId("soaIntervention").value.trim();
    if (!condition) {
        alert("Please enter a condition");
        return;
    }

    runAgentAction({
        buttonId: "soaBtn",
        resultId: "soaResult",
        endpoint: "/api/soa-composer",
        payload: { condition, phase: phase || null, interventionType: interventionType || null },
        buildHtml: (data) => `
            ${renderBanner("SoA composition complete!")}
            ${renderPanel({
                title: "Query Summary",
                accent: "#f7b731",
                borderAccent: "#f7b731",
                content: `
                    <p><strong>Condition:</strong> ${data.query.condition}</p>
                    <p><strong>Phase:</strong> ${data.query.phase || "All phases"}</p>
                    <p><strong>Trials Analyzed:</strong> ${data.query.trials_analyzed}</p>
                `,
            })}
            ${data.reference_trials?.length ? renderPanel({
                title: "📚 Reference Trials",
                accent: "#f7b731",
                borderAccent: "#f7b731",
                content: `<ul style="margin: 0; padding-left: 20px;">${data.reference_trials.slice(0, 5).map((trial) => `<li><strong>${trial.nct_id}</strong>: ${trial.primary_outcome} (${trial.primary_timeframe})</li>`).join("")}</ul>`,
            }) : ""}
            ${data.agent_analyses.map((agent) => renderPanel({
                title: agent.agent,
                accent: "#f7b731",
                borderAccent: "#f7b731",
                subtitle: `Focus: ${agent.focus_areas.join(", ")}`,
                content: withParagraphSpacing(agent.content),
            })).join("")}
            ${renderPanel({
                title: "📋 Complete Schedule of Assessments",
                accent: "#d4af37",
                borderAccent: "#d4af37",
                content: withParagraphSpacing(data.complete_soa),
            })}
            ${renderDownloadButton("soaResult", "SoA-Report", "#f7b731")}
        `,
    });
}

import { requestJson } from "../core/api.js";
import { byId, setDisplay, withParagraphSpacing } from "../core/dom.js";

export async function generateProtocolReport() {
    const condition = byId("protocolCondition").value.trim();
    const intervention = byId("protocolIntervention").value.trim();

    if (!condition) {
        alert("Please enter a condition or disease");
        return;
    }

    setDisplay("protocolLoading", "block");
    setDisplay("protocolReportContainer", "none");

    try {
        const data = await requestJson("/api/generate-protocol-report", {
            condition,
            intervention,
        });

        setDisplay("protocolLoading", "none");
        byId("protocolReportContent").innerHTML = withParagraphSpacing(data.report);
        setDisplay("protocolReportContainer", "block");
        byId("protocolReportContainer").scrollIntoView({
            behavior: "smooth",
            block: "start",
        });
    } catch (error) {
        setDisplay("protocolLoading", "none");
        alert(`Error generating report: ${error.message}`);
    }
}

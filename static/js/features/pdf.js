import { byId } from "../core/dom.js";

export function downloadReportAsPDF(elementId, filename) {
    const element = byId(elementId);
    if (!element) {
        return;
    }

    if (typeof html2pdf !== "undefined") {
        const options = {
            margin: 1,
            filename: `${filename}.pdf`,
            image: { type: "jpeg", quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: "in", format: "letter", orientation: "portrait" },
        };
        html2pdf().set(options).from(element).save();
        return;
    }

    const printWindow = window.open("", "_blank");
    if (!printWindow) {
        return;
    }

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
            <body>${element.innerHTML}</body>
        </html>
    `);
    printWindow.document.close();
    printWindow.print();
}

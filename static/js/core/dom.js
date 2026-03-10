export function byId(id) {
    return document.getElementById(id);
}

export function setDisplay(id, display) {
    const element = byId(id);
    if (element) {
        element.style.display = display;
    }
}

export function withParagraphSpacing(html) {
    return (html || "").replace(/<p>/g, '<p style="margin-bottom: 1em;">');
}

"""Shared helpers for markdown and clinical trial link rendering."""

from html import escape
import re

import markdown

BASE_MARKDOWN_EXTENSIONS = ["extra", "sane_lists"]
NCT_LINK_RE = re.compile(r"(NCT\d{8})")
NCT_LINK_REPLACE = (
    r'<a href="https://clinicaltrials.gov/study/\1" '
    r'target="_blank" style="color:#4f46e5;text-decoration:underline;">\1</a>'
)


def link_nct_ids(html: str) -> str:
    """Convert NCT IDs inside rendered HTML into clinicaltrials.gov links."""
    return NCT_LINK_RE.sub(NCT_LINK_REPLACE, html or "")


def render_markdown(
    text: str,
    *,
    tables: bool = False,
    link_trials: bool = False,
    line_breaks: bool = True,
) -> str:
    """Render markdown consistently across the app."""
    extensions = list(BASE_MARKDOWN_EXTENSIONS)
    if tables:
        extensions.append("tables")
    if line_breaks:
        extensions.append("nl2br")

    html = markdown.markdown(text or "", extensions=extensions)
    if link_trials:
        html = link_nct_ids(html)
    return html


def split_report_line(line: str) -> tuple[str, str]:
    """Split a metadata line into label/value for report headers."""
    if ":" not in line:
        return "Detail", line

    label, value = line.split(":", 1)
    return label.strip(), value.strip()


def escape_html(text: str | None) -> str:
    """Escape plain text before embedding it in HTML shells."""
    return escape(text or "")

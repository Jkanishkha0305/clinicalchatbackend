"""LLM report generation helpers used by the report routes."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Callable

from services.rendering import escape_html, render_markdown, split_report_line


def _truncate_text(value: str | None, max_chars: int) -> str:
    text = value or ""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _normalize_trial_summary(trial: dict, *, max_summary_chars: int = 500) -> dict:
    protocol = trial.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    status_module = protocol.get("statusModule") or {}
    conditions_module = protocol.get("conditionsModule") or {}
    interventions_module = protocol.get("armsInterventionsModule") or {}
    description_module = protocol.get("descriptionModule") or {}

    interventions = trial.get("interventions")
    if not interventions:
        interventions = [
            item.get("name")
            for item in interventions_module.get("interventions", [])
            if item.get("name")
        ]

    return {
        "nct_id": trial.get("nct_id") or identification.get("nctId"),
        "title": (
            trial.get("title")
            or identification.get("briefTitle")
            or identification.get("officialTitle")
        ),
        "status": trial.get("status") or status_module.get("overallStatus"),
        "conditions": trial.get("conditions") or conditions_module.get("conditions", []),
        "interventions": interventions or [],
        "summary": _truncate_text(
            trial.get("summary")
            or trial.get("description")
            or description_module.get("briefSummary")
            or description_module.get("detailedDescription"),
            max_summary_chars,
        ),
    }


def summarize_protocol_trials(trials: list[dict]) -> list[dict]:
    """Create compact trial summaries for protocol report prompting."""
    return [_normalize_trial_summary(trial) for trial in trials]


def summarize_study(trial: dict, *, max_summary_chars: int = 1000) -> dict:
    """Create a compact study summary for single-study report prompting."""
    return _normalize_trial_summary(trial, max_summary_chars=max_summary_chars)


def summarize_conversation(messages: list[dict], *, limit: int | None = None, max_chars: int = 500) -> str:
    """Convert chat history into a concise prompt-friendly transcript."""
    relevant_messages = messages[-limit:] if limit is not None else messages
    conversation_summary: list[str] = []
    for message in relevant_messages:
        role = message.get("role", "unknown").upper()
        content = _truncate_text(message.get("content", ""), max_chars)
        conversation_summary.append(f"{role}: {content}")
    return "\n\n".join(conversation_summary)


def _run_report_completion(
    openai_client,
    count_tokens: Callable[..., int],
    *,
    label: str,
    system_message: str,
    user_message: str,
    model: str = "gpt-4o",
    max_tokens: int = 2500,
) -> str:
    message_list = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    token_count = count_tokens(message_list, model=model)
    print(f"📊 {label} - Token count: {token_count:,} tokens")

    completion = openai_client.chat.completions.create(
        model=model,
        messages=message_list,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def _render_report_shell(
    raw_report: str,
    *,
    title: str,
    report_label: str,
    lines: list[str],
    format_type: str,
) -> str:
    report_html = render_markdown(raw_report, tables=True, link_trials=True, line_breaks=False)
    generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    meta_items = []
    for line in lines:
        label, value = split_report_line(line)
        meta_items.append(
            f"""
            <div class="generated-report__meta-item">
                <dt class="generated-report__meta-label">{escape_html(label)}</dt>
                <dd class="generated-report__meta-value">{escape_html(value)}</dd>
            </div>
            """
        )

    meta_markup = "".join(meta_items)

    return f"""
    <section class="generated-report generated-report--{escape_html(format_type)}">
        <header class="generated-report__header">
            <div class="generated-report__eyebrow">{escape_html(report_label)}</div>
            <h1 class="generated-report__title">{escape_html(title)}</h1>
            <dl class="generated-report__meta">
                {meta_markup}
            </dl>
            <p class="generated-report__timestamp">Generated {escape_html(generated_at)}</p>
        </header>
        <div class="generated-report__body report-content protocol-report-content">
            {report_html}
        </div>
    </section>
    """


def build_report_doc(
    *,
    report_html: str,
    metadata: dict,
    condition: str | None,
    intervention: str | None,
) -> dict:
    """Build the persisted report document shape used by sessions."""
    return {
        "condition": condition,
        "intervention": intervention,
        "report": report_html,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata,
    }


def generate_protocol_report(
    *,
    openai_client,
    count_tokens: Callable[..., int],
    condition: str,
    intervention: str,
    similar_trials: list[dict],
    total_count: int,
    format_type: str = "styled",
) -> dict:
    """Generate the protocol design report and return rendered output."""
    trials_summary = summarize_protocol_trials(similar_trials)
    system_message = f"""You are a clinical trial protocol design expert. Generate a comprehensive protocol research report with detailed statistics.

DATASET: {len(trials_summary)} similar clinical trials
Condition: {condition}
{f"Intervention: {intervention}" if intervention else ""}

TRIALS DATA:
{json.dumps(trials_summary, indent=1)}

Generate a detailed protocol research report with these sections. Include QUANTITATIVE STATISTICS in every section:

1. ELIGIBILITY CRITERIA RECOMMENDATIONS
   - Analyze the most common inclusion criteria across trials with percentages (e.g., "Age ≥18: 85% of trials")
   - Analyze the most common exclusion criteria with frequencies
   - Provide specific recommendations with statistical support
   - Include: prevalence (%), counts, and ranges where applicable

2. STUDY DESIGN PATTERNS
   - Identify common study designs with distribution (e.g., "Randomized: 60%, Single-arm: 25%")
   - Typical duration ranges with median and mean values
   - Common sample sizes: provide min, max, median, and quartiles
   - Phase distribution with percentages
   - Include specific counts and statistical breakdowns

3. KEY INTERVENTIONS ANALYSIS
   - Most common interventions with usage percentages
   - Typical dosing/treatment approaches with frequency data
   - Combination vs monotherapy statistics
   - Include prevalence data for each intervention type

4. SIMILAR TRIALS REFERENCE
   - List top 5-8 most relevant trial NCT IDs with brief descriptions
   - Include trial phase, status, and key characteristics
   - Format NCT IDs exactly as: NCT00000000 (they will be converted to links)

5. STUDY LIMITATIONS
   - Discuss data completeness and quality issues
   - Scope of analysis: date ranges, trial selection criteria
   - Potential biases in the dataset
   - Recommendations for interpreting these results
   - Statistical limitations and confidence considerations

IMPORTANT:
- Include specific numbers, percentages, and statistical measures in EVERY section
- Use quantitative evidence to support all recommendations
- Provide counts alongside percentages (e.g., "45% (18/40 trials)")
- Format the report professionally with clear sections and bullet points"""

    raw_report = _run_report_completion(
        openai_client,
        count_tokens,
        label="Protocol Report",
        system_message=system_message,
        user_message=(
            f"Generate a protocol research report for designing a new {condition} trial"
            f"{f' using {intervention}' if intervention else ''}."
        ),
    )

    report_html = _render_report_shell(
        raw_report,
        title="Clinical Trial Protocol Research Report",
        report_label="Protocol report",
        lines=[
            f"Indication: {condition}",
            *([f"Intervention: {intervention}"] if intervention else []),
            f"Analysis Based On: {len(trials_summary)} similar trials (out of {total_count} total)",
        ],
        format_type=format_type,
    )
    metadata = {
        "trials_analyzed": len(trials_summary),
        "total_matching": total_count,
        "condition": condition,
        "intervention": intervention or None,
    }
    return {
        "report": report_html,
        "metadata": metadata,
        "report_doc": build_report_doc(
            report_html=report_html,
            metadata=metadata,
            condition=condition,
            intervention=intervention or None,
        ),
    }


def generate_chat_report(
    *,
    openai_client,
    count_tokens: Callable[..., int],
    condition: str,
    intervention: str,
    messages: list[dict],
    studies: list[dict],
    total_count: int,
    format_type: str = "styled",
) -> dict:
    """Generate a report from a saved cross-study conversation."""
    conversation_text = summarize_conversation(messages, limit=10)
    system_message = f"""You are a clinical research analyst. Generate a comprehensive research report based on this chat conversation.

CONVERSATION SUMMARY:
{conversation_text}

PRIMARY FOCUS: {condition}
{f"INTERVENTION: {intervention}" if intervention else ""}
STUDIES AVAILABLE: {len(studies)} studies analyzed (out of {total_count} matching)

Generate a detailed research report with these sections:

1. CONVERSATION OVERVIEW
   - Summarize the key questions and topics discussed
   - Highlight main findings from the conversation

2. KEY INSIGHTS
   - Extract the most important insights from the conversation
   - Provide evidence-based conclusions

3. STUDY LANDSCAPE
   - Summarize the relevant clinical trials landscape
   - Provide statistics on study designs, phases, and interventions

4. RECOMMENDATIONS
   - Based on the conversation, provide actionable recommendations
   - Suggest areas for further investigation

5. REFERENCES
   - List relevant NCT IDs mentioned or related to the discussion
   - Format NCT IDs as: NCT00000000

Format the report professionally with clear sections and bullet points."""

    raw_report = _run_report_completion(
        openai_client,
        count_tokens,
        label="Chat Report",
        system_message=system_message,
        user_message=f"Generate a comprehensive research report based on this chat conversation about {condition}.",
    )

    report_html = _render_report_shell(
        raw_report,
        title="Chat Conversation Research Report",
        report_label="Conversation report",
        lines=[
            f"Primary Focus: {condition}",
            *([f"Intervention: {intervention}"] if intervention else []),
            f"Messages: {len(messages)} | Studies: {len(studies)}/{total_count}",
        ],
        format_type=format_type,
    )
    metadata = {
        "messages_count": len(messages),
        "studies_analyzed": len(studies),
        "total_matching": total_count,
        "condition": condition,
        "intervention": intervention or None,
        "report_type": "chat_report",
    }
    return {
        "report": report_html,
        "metadata": metadata,
        "report_doc": build_report_doc(
            report_html=report_html,
            metadata=metadata,
            condition=condition,
            intervention=intervention or None,
        ),
    }


def generate_study_chat_report(
    *,
    openai_client,
    count_tokens: Callable[..., int],
    study_id: str,
    study_title: str,
    study: dict,
    messages: list[dict],
    format_type: str = "styled",
) -> dict:
    """Generate a report from a saved single-study conversation."""
    conversation_text = summarize_conversation(messages)
    study_json = json.dumps(summarize_study(study), indent=2)
    system_message = f"""You are a clinical research analyst. Generate a comprehensive report based on the conversation about this specific clinical trial.

STUDY INFORMATION:
{study_json}

CONVERSATION SUMMARY:
{conversation_text}

Generate a detailed report with these sections:

1. STUDY OVERVIEW
   - Summarize the key aspects of this clinical trial
   - Highlight the study design, intervention, and objectives

2. CONVERSATION INSIGHTS
   - Summarize the key questions and topics discussed about this study
   - Extract important insights from the conversation

3. DETAILED ANALYSIS
   - Provide in-depth analysis based on the conversation
   - Address specific questions or concerns raised

4. KEY FINDINGS
   - List the most important findings discussed
   - Provide evidence-based conclusions

5. CLINICAL IMPLICATIONS
   - Discuss the clinical relevance of this study
   - Suggest practical applications or considerations

Format the report professionally with clear sections and bullet points.
Always reference the study by its NCT ID: {study_id}"""

    raw_report = _run_report_completion(
        openai_client,
        count_tokens,
        label="Study Chat Report",
        system_message=system_message,
        user_message=f"Generate a comprehensive report about study {study_id} based on our conversation.",
    )

    report_html = _render_report_shell(
        raw_report,
        title="Study Conversation Research Report",
        report_label="Study report",
        lines=[
            f"Study: {study_id}",
            f"Title: {study_title}",
            f"Messages: {len(messages)}",
        ],
        format_type=format_type,
    )
    metadata = {
        "messages_count": len(messages),
        "study_id": study_id,
        "study_title": study_title,
        "report_type": "study_chat_report",
    }
    return {
        "report": report_html,
        "metadata": metadata,
        "report_doc": build_report_doc(
            report_html=report_html,
            metadata=metadata,
            condition=None,
            intervention=None,
        ),
    }

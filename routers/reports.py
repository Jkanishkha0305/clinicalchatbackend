import json
import re
import markdown as md_lib
import secrets
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import dependencies as deps
from models import ProtocolReportRequest, ChatReportRequest, StudyChatReportRequest

router = APIRouter()

NCT_LINK = r'<a href="https://clinicaltrials.gov/study/\1" target="_blank" style="color: #4f46e5; text-decoration: underline;">\1</a>'


def _nct_links(html: str) -> str:
    return re.sub(r'(NCT\d{8})', NCT_LINK, html)


@router.post("/generate-protocol-report")
def generate_protocol_report(body: ProtocolReportRequest):
    condition = body.condition or ""
    intervention = body.intervention or ""
    session_id = body.sessionId
    format_type = body.format or "styled"

    if not condition:
        return JSONResponse({"success": False, "error": "Condition is required"}, status_code=400)

    query = {}
    if condition:
        query["conditions"] = {"$regex": condition, "$options": "i"}
    if intervention:
        query["interventions"] = {"$regex": intervention, "$options": "i"}

    total_count = deps.collection.count_documents(query)
    if total_count == 0:
        return JSONResponse({"success": False, "error": f"No trials found for {condition}"}, status_code=404)

    limit = min(total_count, 100)
    similar_trials = list(deps.collection.find(query).limit(limit))

    trials_summary = [
        {
            "nct_id": t.get("nct_id"),
            "title": t.get("title"),
            "status": t.get("status"),
            "conditions": t.get("conditions", []),
            "interventions": t.get("interventions", []),
            "summary": t.get("summary", "")[:500],
        }
        for t in similar_trials
    ]
    trials_json = json.dumps(trials_summary, indent=1)

    system_message = f"""You are a clinical trial protocol design expert. Generate a comprehensive protocol research report with detailed statistics.

DATASET: {len(trials_summary)} similar clinical trials
Condition: {condition}
{f"Intervention: {intervention}" if intervention else ""}

TRIALS DATA:
{trials_json}

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

    try:
        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate a protocol research report for designing a new {condition} trial{f' using {intervention}' if intervention else ''}."},
        ]

        token_count = deps.count_tokens(message_list, model="gpt-4o")
        print(f"📊 Protocol Report - Token count: {token_count:,} tokens")

        completion = deps.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_list,
            temperature=0.3,
            max_tokens=2500,
        )

        report = completion.choices[0].message.content
        report_html = _nct_links(md_lib.markdown(report, extensions=["extra", "nl2br", "tables"]))

        header = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">📋 Clinical Trial Protocol Research Report</h2>
            <p style="margin: 5px 0;"><strong>Indication:</strong> {condition}</p>
            {f'<p style="margin: 5px 0;"><strong>Intervention:</strong> {intervention}</p>' if intervention else ""}
            <p style="margin: 5px 0;"><strong>Analysis Based On:</strong> {len(trials_summary)} similar trials (out of {total_count} total)</p>
            <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        """

        full_report = header + report_html

        report_doc = {
            "condition": condition,
            "intervention": intervention if intervention else None,
            "report": full_report,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "trials_analyzed": len(trials_summary),
                "total_matching": total_count,
                "condition": condition,
                "intervention": intervention if intervention else None,
            },
        }

        session_info = None
        if session_id:
            session = deps.chat_sessions_collection.find_one({"_id": session_id})
            if session:
                reports = session.get("reports", [])
                reports.append(report_doc)
                deps.chat_sessions_collection.update_one(
                    {"_id": session_id},
                    {"$set": {
                        "reports": reports,
                        "last_report_filters": {"condition": condition, "intervention": intervention if intervention else None},
                        "updated_at": datetime.now().isoformat(),
                    }},
                )
                session_info = {
                    "id": session["_id"],
                    "title": session.get("title", "Protocol Report Session"),
                    "description": session.get("description", ""),
                }
        else:
            new_session_id = secrets.token_urlsafe(16)
            new_session = {
                "_id": new_session_id,
                "title": f"Protocol Report: {condition}",
                "description": f"Protocol research report for {condition}" + (f" with {intervention}" if intervention else ""),
                "last_filters": {"condition": condition, "intervention": intervention},
                "last_report_filters": {"condition": condition, "intervention": intervention if intervention else None},
                "messages": [],
                "reports": [report_doc],
                "custom_questions": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            deps.chat_sessions_collection.insert_one(new_session)
            session_info = {"id": new_session_id, "title": new_session["title"], "description": new_session["description"]}

        resp = {
            "success": True,
            "report": full_report,
            "metadata": {
                "trials_analyzed": len(trials_summary),
                "total_matching": total_count,
                "condition": condition,
                "intervention": intervention,
            },
        }
        if session_info:
            resp["sessionInfo"] = session_info
        return resp

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"success": False, "error": f"AI error: {e}"}, status_code=500)


@router.post("/generate-chat-report")
def generate_chat_report(body: ChatReportRequest):
    session_id = body.sessionId or ""
    format_type = body.format or "styled"

    if not session_id:
        return JSONResponse({"success": False, "error": "Session ID is required"}, status_code=400)

    try:
        session = deps.chat_sessions_collection.find_one({"_id": session_id})
        if not session:
            return JSONResponse({"success": False, "error": "Chat session not found"}, status_code=404)

        messages = session.get("messages", [])
        last_filters = session.get("last_filters", {})
        condition = last_filters.get("condition", "Unknown")
        intervention = last_filters.get("intervention", "")

        query = deps.build_query_from_filters(last_filters) if last_filters else {}
        total_count = deps.collection.count_documents(query)
        studies = list(deps.collection.find(query).limit(min(total_count, 50)))

        conversation_summary = []
        for msg in messages[-10:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            conversation_summary.append(f"{role.upper()}: {content}")
        conversation_text = "\n\n".join(conversation_summary)

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

        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate a comprehensive research report based on this chat conversation about {condition}."},
        ]

        token_count = deps.count_tokens(message_list, model="gpt-4o")
        print(f"📊 Chat Report - Token count: {token_count:,} tokens")

        completion = deps.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_list,
            temperature=0.3,
            max_tokens=2500,
        )

        report = completion.choices[0].message.content
        report_html = _nct_links(md_lib.markdown(report, extensions=["extra", "nl2br", "tables"]))

        if format_type == "styled":
            header = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0;">💬 Chat Conversation Research Report</h2>
                <p style="margin: 5px 0;"><strong>Primary Focus:</strong> {condition}</p>
                {f'<p style="margin: 5px 0;"><strong>Intervention:</strong> {intervention}</p>' if intervention else ""}
                <p style="margin: 5px 0;"><strong>Messages Analyzed:</strong> {len(messages)}</p>
                <p style="margin: 5px 0;"><strong>Studies Referenced:</strong> {len(studies)} studies (out of {total_count} matching)</p>
                <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        elif format_type == "professional":
            header = f"""
            <div style="border-bottom: 3px solid #667eea; padding-bottom: 15px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0; color: #333;">Chat Conversation Research Report</h2>
                <p style="margin: 5px 0; color: #666;"><strong>Primary Focus:</strong> {condition}</p>
                {f'<p style="margin: 5px 0; color: #666;"><strong>Intervention:</strong> {intervention}</p>' if intervention else ""}
                <p style="margin: 5px 0; color: #666;"><strong>Messages:</strong> {len(messages)} | <strong>Studies:</strong> {len(studies)}/{total_count}</p>
                <p style="margin: 5px 0; font-size: 12px; color: #999;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        else:
            header = f"""
            <div style="margin-bottom: 20px;">
                <h2>Chat Conversation Research Report</h2>
                <p><strong>Primary Focus:</strong> {condition}</p>
                {f'<p><strong>Intervention:</strong> {intervention}</p>' if intervention else ""}
                <p><strong>Messages:</strong> {len(messages)} | <strong>Studies:</strong> {len(studies)}/{total_count}</p>
                <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """

        full_report = header + report_html

        reports = session.get("reports", [])
        reports.append({
            "condition": condition,
            "intervention": intervention if intervention else None,
            "report": full_report,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "messages_count": len(messages),
                "studies_analyzed": len(studies),
                "total_matching": total_count,
                "condition": condition,
                "intervention": intervention if intervention else None,
                "report_type": "chat_report",
            },
        })
        deps.chat_sessions_collection.update_one({"_id": session_id}, {"$set": {"reports": reports}})

        return {
            "success": True,
            "report": full_report,
            "metadata": {
                "messages_count": len(messages),
                "studies_analyzed": len(studies),
                "total_matching": total_count,
                "condition": condition,
                "intervention": intervention if intervention else None,
            },
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"success": False, "error": f"AI error: {e}"}, status_code=500)


@router.post("/generate-study-chat-report")
def generate_study_chat_report(body: StudyChatReportRequest):
    study_id = body.studyId or ""
    chat_session_id = body.chatSessionId or ""
    format_type = body.format or "styled"

    if not study_id:
        return JSONResponse({"success": False, "error": "Study ID is required"}, status_code=400)

    try:
        study = deps.collection.find_one({"nct_id": study_id})
        if not study:
            return JSONResponse({"success": False, "error": "Study not found"}, status_code=404)

        study_title = study.get("title", "Unknown Study")

        query = {"study_id": study_id}
        if chat_session_id:
            query["chat_session_id"] = chat_session_id

        study_chat = deps.study_chats_collection.find_one(query)
        messages = study_chat.get("messages", []) if study_chat else []

        conversation_summary = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            conversation_summary.append(f"{role.upper()}: {content}")
        conversation_text = "\n\n".join(conversation_summary)

        study_summary = {
            "nct_id": study.get("nct_id"),
            "title": study.get("title"),
            "status": study.get("status"),
            "conditions": study.get("conditions", []),
            "interventions": study.get("interventions", []),
            "summary": study.get("summary", "")[:1000],
        }
        study_json = json.dumps(study_summary, indent=2)

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

        message_list = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate a comprehensive report about study {study_id} based on our conversation."},
        ]

        token_count = deps.count_tokens(message_list, model="gpt-4o")
        print(f"📊 Study Chat Report - Token count: {token_count:,} tokens")

        completion = deps.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_list,
            temperature=0.3,
            max_tokens=2500,
        )

        report = completion.choices[0].message.content
        report_html = _nct_links(md_lib.markdown(report, extensions=["extra", "nl2br", "tables"]))

        if format_type == "styled":
            header = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0;">🔬 Study Chat Research Report</h2>
                <p style="margin: 5px 0;"><strong>Study:</strong> {study_id}</p>
                <p style="margin: 5px 0;"><strong>Title:</strong> {study_title}</p>
                <p style="margin: 5px 0;"><strong>Messages Analyzed:</strong> {len(messages)}</p>
                <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        elif format_type == "professional":
            header = f"""
            <div style="border-bottom: 3px solid #667eea; padding-bottom: 15px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0; color: #333;">Study Chat Research Report</h2>
                <p style="margin: 5px 0; color: #666;"><strong>Study:</strong> {study_id}</p>
                <p style="margin: 5px 0; color: #666;"><strong>Title:</strong> {study_title}</p>
                <p style="margin: 5px 0; color: #666;"><strong>Messages:</strong> {len(messages)}</p>
                <p style="margin: 5px 0; font-size: 12px; color: #999;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """
        else:
            header = f"""
            <div style="margin-bottom: 20px;">
                <h2>Study Chat Research Report</h2>
                <p><strong>Study:</strong> {study_id}</p>
                <p><strong>Title:</strong> {study_title}</p>
                <p><strong>Messages:</strong> {len(messages)}</p>
                <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            """

        full_report = header + report_html

        if study_chat:
            reports = study_chat.get("reports", [])
            reports.append({
                "condition": None,
                "intervention": None,
                "report": full_report,
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "messages_count": len(messages),
                    "study_id": study_id,
                    "study_title": study_title,
                    "report_type": "study_chat_report",
                },
            })
            deps.study_chats_collection.update_one(query, {"$set": {"reports": reports}})

        return {
            "success": True,
            "report": full_report,
            "metadata": {
                "messages_count": len(messages),
                "study_id": study_id,
                "study_title": study_title,
                "report_type": "study_chat",
            },
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"success": False, "error": f"AI error: {e}"}, status_code=500)

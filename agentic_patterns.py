"""
Design Pattern Discovery - Multi-Agent System
Identifies recurring trial design patterns and trends across similar trials
"""

from openai import OpenAI
import os
import re
import markdown
from pymongo import MongoClient

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# MongoDB connection
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['clinical_trials']
collection = db['studies']

# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

AGENTS = {
    "pattern_recognition": {
        "name": "Dr. Elena Petrov - Design Pattern Recognition Specialist",
        "role": "Identifies common trial design archetypes and patterns",
        "focus": [
            "Trial design archetypes (RCT, adaptive, basket/umbrella)",
            "Randomization schemes",
            "Blinding strategies",
            "Control arm choices"
        ]
    },
    "temporal_trends": {
        "name": "Dr. David Zhang - Temporal Trends Analyst",
        "role": "Analyzes how trial designs evolved over time",
        "focus": [
            "Design evolution over years",
            "Endpoint duration trends",
            "Enrollment size trends",
            "Regulatory requirement changes"
        ]
    },
    "best_practices": {
        "name": "Dr. Sophia Williams - Best Practices Consultant",
        "role": "Recommends optimal design choices based on successful trials",
        "focus": [
            "Success factors in completed trials",
            "Common pitfalls to avoid",
            "Industry standards",
            "Evidence-based recommendations"
        ]
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fetch_similar_trials(condition, phase=None, intervention_type=None, limit=50):
    """Fetch similar trials from MongoDB - Works with simplified structure"""

    query = {}

    # Build conditions array for $and
    and_conditions = []

    # Condition search (case-insensitive, partial match)
    if condition:
        and_conditions.append({
            '$or': [
                {'conditions': {'$regex': condition, '$options': 'i'}},
                {'title': {'$regex': condition, '$options': 'i'}}
            ]
        })

    # Combine with $and if multiple conditions
    if and_conditions:
        if len(and_conditions) == 1:
            query = and_conditions[0]
        else:
            query = {'$and': and_conditions}

    trials = list(collection.find(query).limit(limit))
    return trials


def summarize_trials_for_analysis(trials):
    """Create a summary of trials for agent analysis"""

    summary = {
        "total_trials": len(trials),
        "trials_by_phase": {},
        "trials_by_status": {},
        "trials_by_year": {},
        "sample_trials": []
    }

    for trial in trials[:20]:  # Sample first 20 for detailed analysis
        try:
            protocol = trial.get('protocolSection', {})
            identification = protocol.get('identificationModule', {})
            design = protocol.get('designModule', {})
            eligibility = protocol.get('eligibilityModule', {})
            arms = protocol.get('armsInterventionsModule', {})
            outcomes = protocol.get('outcomesModule', {})

            trial_info = {
                "nct_id": identification.get('nctId', 'N/A'),
                "title": identification.get('briefTitle', 'N/A'),
                "phase": ', '.join(design.get('phases', ['N/A'])),
                "study_type": design.get('studyType', 'N/A'),
                "allocation": design.get('designInfo', {}).get('allocation', 'N/A'),
                "masking": design.get('designInfo', {}).get('maskingInfo', {}).get('masking', 'N/A'),
                "enrollment": design.get('enrollmentInfo', {}).get('count', 'N/A'),
                "primary_purpose": design.get('designInfo', {}).get('primaryPurpose', 'N/A'),
                "number_of_arms": len(arms.get('armGroups', [])),
                "intervention_types": list(set([
                    i.get('type', 'N/A') for i in arms.get('interventions', [])
                ])),
                "primary_outcome_timeframe": outcomes.get('primaryOutcomes', [{}])[0].get('timeFrame', 'N/A') if outcomes.get('primaryOutcomes') else 'N/A',
            }

            summary["sample_trials"].append(trial_info)

            # Aggregate statistics
            phase = trial_info['phase']
            summary["trials_by_phase"][phase] = summary["trials_by_phase"].get(phase, 0) + 1

        except Exception as e:
            continue

    return summary


def render_formats(raw_text: str):
    """Return html and plain text variants for a model response."""
    html = markdown.markdown(raw_text, extensions=["extra", "nl2br", "tables"])
    plain = re.sub(r"<[^>]+>", "", html)
    return html, plain


def call_agent(agent_key, condition, phase, intervention_type, trials_summary):
    """Call a specific agent to analyze trial patterns"""
    agent = AGENTS[agent_key]

    prompt = f"""You are {agent['name']}, a specialized agent focused on: {agent['role']}

Your specific focus areas are:
{chr(10).join(f"- {area}" for area in agent['focus'])}

Analyze trial design patterns for:
- Condition: {condition}
- Phase: {phase or 'All phases'}
- Intervention Type: {intervention_type or 'All types'}

Data Summary:
- Total trials analyzed: {trials_summary['total_trials']}
- Trials by phase: {trials_summary['trials_by_phase']}

Sample Trials:
{chr(10).join([f"• {t['nct_id']}: {t['title'][:80]}... | Phase: {t['phase']} | Design: {t['allocation']}, {t['masking']} | Arms: {t['number_of_arms']}" for t in trials_summary['sample_trials'][:10]])}

Based on your expertise, provide (plain text, no HTML):
1. Key patterns identified — include at least 3 numeric stats (e.g., % by phase, median enrollment, most common masking/allocation with counts)
2. Design archetypes — categorize top 2–3 patterns with prevalence (% or counts)
3. Success indicators — call out design choices that correlate with success, with quantitative evidence (e.g., completion rates, dropout ranges)
4. Recommendations — 3–5 actionable next design choices with expected impact (short rationale and a number: % improvement, weeks saved, or visits reduced)

Use short headings and bullet points. Include a mini “Key Metrics” bullet list summarizing the top numbers you cite."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a clinical trial design pattern expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1500
    )

    return response.choices[0].message.content


def synthesize_strategic_insights(condition, phase, intervention_type, trials_summary, agent_analyses):
    """Strategic advisor synthesizes all pattern analyses"""

    combined_analysis = "\n\n".join([
        f"**{AGENTS[key]['name']}**:\n{content}"
        for key, content in agent_analyses.items()
    ])

    prompt = f"""You are the Strategic Design Advisor synthesizing pattern discovery insights.

Query: {condition} | Phase: {phase or 'All'} | Intervention: {intervention_type or 'All'}
Trials Analyzed: {trials_summary['total_trials']}

Your team has identified these patterns:

{combined_analysis}

Provide a STRATEGIC DESIGN BLUEPRINT (plain text):
1. Recommended design archetype (with prevalence % from observed trials)
2. Key design decisions (randomization, blinding, control, duration) — include numbers: randomization ratio, median duration, visit count range
3. Competitive landscape — brief comparison with at least 2 numeric contrasts (e.g., median enrollment, dropout, duration)
4. Innovation opportunities — 2–3 ideas with estimated impact (%, weeks, or visits saved)

Be strategic, actionable, and forward-thinking. Use headings and bullet points, no HTML. Start with a 2–3 bullet “Key Metrics” snapshot (phase distribution %, median enrollment, common design type)."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a Strategic Design Advisor providing blueprint-level guidance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )

    return response.choices[0].message.content


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def design_pattern_discovery(condition, phase=None, intervention_type=None):
    """
    Run multi-agent design pattern discovery

    Args:
        condition (str): Medical condition
        phase (str): Trial phase (optional)
        intervention_type (str): Intervention type (optional)

    Returns:
        dict: Pattern analysis results
    """

    try:
        # Fetch similar trials
        print(f"🔍 Fetching trials for: {condition}...")
        trials = fetch_similar_trials(condition, phase, intervention_type, limit=100)

        if len(trials) == 0:
            return {
                "success": False,
                "error": f"No trials found for {condition}. Try a different search."
            }

        # Summarize trials
        trials_summary = summarize_trials_for_analysis(trials)

        # Run all agents
        agent_analyses = {}
        agent_results = []

        for agent_key in AGENTS.keys():
            print(f"🤖 Running {AGENTS[agent_key]['name']}...")
            analysis_raw = call_agent(agent_key, condition, phase, intervention_type, trials_summary)
            analysis_html, analysis_text = render_formats(analysis_raw)
            agent_analyses[agent_key] = analysis_raw

            agent_results.append({
                "agent": AGENTS[agent_key]['name'],
                "focus_areas": AGENTS[agent_key]['focus'],
                "content": analysis_html,  # backward-compatible
                "content_raw": analysis_raw,
                "content_html": analysis_html,
                "content_text": analysis_text
            })

        # Synthesize strategic insights
        print(f"🧠 Strategic Advisor synthesizing...")
        strategic_insights_raw = synthesize_strategic_insights(
            condition, phase, intervention_type, trials_summary, agent_analyses
        )
        strategic_insights_html, strategic_insights_text = render_formats(strategic_insights_raw)

        return {
            "success": True,
            "query": {
                "condition": condition,
                "phase": phase,
                "intervention_type": intervention_type,
                "trials_analyzed": trials_summary['total_trials']
            },
            "agent_analyses": agent_results,
            "strategic_insights": strategic_insights_html,  # backward-compatible
            "strategic_insights_raw": strategic_insights_raw,
            "strategic_insights_html": strategic_insights_html,
            "strategic_insights_text": strategic_insights_text,
            "trials_summary": trials_summary
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# =============================================================================
# TEST FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("Testing Design Pattern Discovery...")

    result = design_pattern_discovery(
        condition="diabetes",
        phase="PHASE3",
        intervention_type="DRUG"
    )

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"\nQuery: {result['query']}")
        print(f"\nTrials analyzed: {result['query']['trials_analyzed']}")
        print(f"\nNumber of agent analyses: {len(result['agent_analyses'])}")
        print(f"\nStrategic Insights Preview:")
        print(result['strategic_insights'][:500] + "...")
    else:
        print(f"\n❌ ERROR: {result['error']}")

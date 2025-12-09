"""
Amendment Risk Predictor - Multi-Agent System
Predicts likelihood of protocol amendments based on design complexity
"""

from openai import OpenAI
import os
import re
import markdown

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

AGENTS = {
    "eligibility_complexity": {
        "name": "Dr. Rachel Kim - Eligibility Criteria Complexity Analyst",
        "role": "Analyzes inclusion/exclusion criteria complexity and restrictiveness",
        "focus": [
            "Number of inclusion/exclusion criteria",
            "Overly restrictive criteria (rare biomarkers, narrow age ranges)",
            "Feasibility of recruiting target population",
            "Historical comparison with similar trials"
        ]
    },
    "endpoint_feasibility": {
        "name": "Dr. Michael Chen - Endpoint Feasibility Analyst",
        "role": "Evaluates endpoint structure and feasibility",
        "focus": [
            "Number of primary and secondary endpoints",
            "Endpoint complexity and measurability",
            "Timeline feasibility for endpoints",
            "Risk of endpoint changes"
        ]
    },
    "statistical_power": {
        "name": "Dr. Lisa Martinez - Statistical Power & Sample Size Analyst",
        "role": "Assesses statistical design and power calculations",
        "focus": [
            "Sample size adequacy",
            "Power calculation assumptions",
            "Recruitment feasibility",
            "Risk of under-enrollment"
        ]
    },
    "operational_burden": {
        "name": "Dr. James O'Connor - Operational Complexity Analyst",
        "role": "Evaluates operational burden and execution complexity",
        "focus": [
            "Visit schedule complexity",
            "Assessment burden on participants",
            "Site burden and feasibility",
            "Protocol procedural complexity"
        ]
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def call_agent(agent_key, trial_data):
    """Call a specific agent to analyze trial data"""
    agent = AGENTS[agent_key]

    prompt = f"""You are {agent['name']}, a specialized agent focused on: {agent['role']}

Your specific focus areas are:
{chr(10).join(f"- {area}" for area in agent['focus'])}

Analyze this clinical trial for amendment risk:

NCT ID: {trial_data.get('nct_id', 'N/A')}
Title: {trial_data.get('title', 'N/A')}
Phase: {trial_data.get('phase', 'N/A')}
Status: {trial_data.get('status', 'N/A')}
Enrollment: {trial_data.get('enrollment', 'N/A')}

Eligibility Criteria:
{trial_data.get('eligibility', 'Not specified')}

Outcome Measures:
{trial_data.get('outcomes', 'Not specified')}

Study Design:
{trial_data.get('design', 'Not specified')}

Based on your expertise, provide (plain text, no HTML):
1. Risk assessment: Low/Medium/High for amendments in your focus area AND a % likelihood
2. Key risk factors: 2–3 issues with probability ranges or numeric indicators (e.g., “60–70% screening failure”, “15 visits → high burden”)
3. Recommendations: 3 actions with expected impact (e.g., % reduction in risk, weeks saved, visit reduction)
4. Quantify burden: visits, hours per visit, enrollment duration, dropout % if relevant

Use short headings and bullet points. Include a “Key Metrics” bullet list with at least 3 numbers."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a clinical trial design expert analyzing amendment risk."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1500
    )

    return response.choices[0].message.content


def synthesize_risk_assessment(trial_data, agent_analyses):
    """Chief Risk Officer synthesizes all agent analyses"""

    combined_analysis = "\n\n".join([
        f"**{AGENTS[key]['name']}**:\n{content}"
        for key, content in agent_analyses.items()
    ])

    prompt = f"""You are the Chief Risk Officer synthesizing multiple expert analyses.

Trial: {trial_data.get('nct_id')} - {trial_data.get('title')}

Your team has analyzed this trial from 4 perspectives:

{combined_analysis}

Provide an EXECUTIVE RISK ASSESSMENT (plain text):
1. Overall amendment risk score: Low/Medium/High with confidence % AND numeric likelihood range
2. Top 3 risk factors: Each with likelihood % and brief rationale
3. Priority recommendations: Top 3 with expected impact (%, weeks, visit reduction, or cost/patient delta)
4. Comparative insight: How this compares to typical trials in this phase/indication (at least 2 numeric contrasts: e.g., visit count, duration, enrollment)

Be concise, actionable, and strategic. Use headings and bullet points, no HTML. Start with a “Key Metrics” mini-list (risk %, visit count, enrollment duration, dropout % if applicable)."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are the Chief Risk Officer providing executive-level risk synthesis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )

    return response.choices[0].message.content


def render_formats(raw_text: str):
    """Return html and plain text variants for a model response."""
    html = markdown.markdown(raw_text, extensions=["extra", "nl2br", "tables"])
    plain = re.sub(r"<[^>]+>", "", html)
    return html, plain


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def amendment_risk_analysis(trial_data):
    """
    Run multi-agent amendment risk analysis

    Args:
        trial_data (dict): Trial information with keys: nct_id, title, phase,
                          status, enrollment, eligibility, outcomes, design

    Returns:
        dict: {
            'success': bool,
            'trial': trial_data,
            'agent_analyses': list of agent analysis results,
            'risk_assessment': executive summary
        }
    """

    try:
        # Run all agents in parallel (conceptually - could use threading)
        agent_analyses = {}
        agent_results = []

        for agent_key in AGENTS.keys():
            print(f"🤖 Running {AGENTS[agent_key]['name']}...")
            analysis_raw = call_agent(agent_key, trial_data)
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

        # Synthesize with Chief Risk Officer
        print(f"👔 Chief Risk Officer synthesizing...")
        risk_assessment_raw = synthesize_risk_assessment(trial_data, agent_analyses)
        risk_assessment_html, risk_assessment_text = render_formats(risk_assessment_raw)

        return {
            "success": True,
            "trial": {
                "nct_id": trial_data.get('nct_id'),
                "title": trial_data.get('title')
            },
            "agent_analyses": agent_results,
            "risk_assessment": risk_assessment_html,  # backward-compatible
            "risk_assessment_raw": risk_assessment_raw,
            "risk_assessment_html": risk_assessment_html,
            "risk_assessment_text": risk_assessment_text
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
    # Test data
    test_trial = {
        "nct_id": "NCT12345678",
        "title": "Phase 3 Study of Novel GLP-1 Agonist in Type 2 Diabetes",
        "phase": "Phase 3",
        "status": "Not yet recruiting",
        "enrollment": "500",
        "eligibility": """
        Inclusion Criteria:
        - Age 18-65
        - Type 2 Diabetes with HbA1c 7.5-10%
        - BMI 25-40
        - Stable on metformin for 3+ months

        Exclusion Criteria:
        - Type 1 diabetes
        - Recent cardiovascular event (< 6 months)
        - eGFR < 60
        - History of pancreatitis
        - Pregnant or breastfeeding
        """,
        "outcomes": """
        Primary: Change in HbA1c from baseline to week 26
        Secondary: Weight loss, fasting glucose, lipid panel, adverse events
        """,
        "design": """
        Randomized, double-blind, placebo-controlled
        3 arms: Drug low dose, Drug high dose, Placebo
        26-week primary endpoint, 52-week safety follow-up
        15 visits total
        """
    }

    print("Testing Amendment Risk Analysis...")
    result = amendment_risk_analysis(test_trial)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"\nTrial: {result['trial']['nct_id']}")
        print(f"\nNumber of agent analyses: {len(result['agent_analyses'])}")
        print(f"\nRisk Assessment Preview:")
        print(result['risk_assessment'][:500] + "...")
    else:
        print(f"\n❌ ERROR: {result['error']}")

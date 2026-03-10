"""
Schedule of Assessments (SoA) Composer - Multi-Agent System
Generates draft SoA based on similar trial patterns
"""

from agentic.common import call_text_completion, render_formats
from agentic.data import fetch_similar_trials

# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

AGENTS = {
    "visit_schedule": {
        "name": "Dr. Amanda Foster - Visit Schedule Architect",
        "role": "Designs optimal visit schedule and timing",
        "focus": [
            "Visit windows (screening, baseline, follow-up)",
            "Visit frequency by trial phase",
            "Primary/secondary endpoint timing",
            "Industry standard schedules"
        ]
    },
    "assessment_frequency": {
        "name": "Dr. Robert Kim - Assessment Frequency Specialist",
        "role": "Determines assessment types and frequencies",
        "focus": [
            "Lab assessments (hematology, chemistry, biomarkers)",
            "Vital signs and physical exams",
            "Imaging and diagnostic tests",
            "PK/PD and immunogenicity sampling"
        ]
    },
    "safety_monitoring": {
        "name": "Dr. Patricia Lopez - Safety Monitoring Expert",
        "role": "Designs safety monitoring strategy",
        "focus": [
            "Adverse event monitoring frequency",
            "ECG and cardiac monitoring",
            "Pregnancy tests (if applicable)",
            "Concomitant medication review"
        ]
    },
    "patient_burden": {
        "name": "Dr. Kevin Wong - Patient Burden Optimizer",
        "role": "Balances scientific rigor with patient experience",
        "focus": [
            "Visit burden optimization",
            "Assessment consolidation opportunities",
            "Patient-friendly scheduling",
            "Retention risk mitigation"
        ]
    }
}

def extract_outcome_timeframes(trials):
    """Extract outcome measure timeframes from similar trials"""

    timeframes = {
        "primary": [],
        "secondary": []
    }

    for trial in trials:
        try:
            protocol = trial.get('protocolSection', {})
            outcomes = protocol.get('outcomesModule', {})

            # Primary outcomes
            for outcome in outcomes.get('primaryOutcomes', []):
                timeframe = outcome.get('timeFrame', '')
                if timeframe:
                    timeframes['primary'].append(timeframe)

            # Secondary outcomes
            for outcome in outcomes.get('secondaryOutcomes', []):
                timeframe = outcome.get('timeFrame', '')
                if timeframe:
                    timeframes['secondary'].append(timeframe)

        except Exception:
            continue

    return timeframes
def summarize_trials_for_soa(trials):
    """Summarize trials for SoA generation"""

    summary = {
        "total_trials": len(trials),
        "sample_trials": [],
        "common_timeframes": extract_outcome_timeframes(trials)
    }

    for trial in trials[:15]:
        try:
            protocol = trial.get('protocolSection', {})
            identification = protocol.get('identificationModule', {})
            design = protocol.get('designModule', {})
            outcomes = protocol.get('outcomesModule', {})

            trial_info = {
                "nct_id": identification.get('nctId', 'N/A'),
                "title": identification.get('briefTitle', 'N/A'),
                "phase": ', '.join(design.get('phases', ['N/A'])),
                "primary_outcome": outcomes.get('primaryOutcomes', [{}])[0].get('measure', 'N/A') if outcomes.get('primaryOutcomes') else 'N/A',
                "primary_timeframe": outcomes.get('primaryOutcomes', [{}])[0].get('timeFrame', 'N/A') if outcomes.get('primaryOutcomes') else 'N/A',
                "enrollment": design.get('enrollmentInfo', {}).get('count', 'N/A')
            }

            summary["sample_trials"].append(trial_info)

        except Exception:
            continue

    return summary


def call_agent(agent_key, condition, phase, intervention_type, trials_summary):
    """Call a specific agent to design SoA components"""
    agent = AGENTS[agent_key]

    prompt = f"""You are {agent['name']}, a specialized agent focused on: {agent['role']}

Your specific focus areas are:
{chr(10).join(f"- {area}" for area in agent['focus'])}

Design Schedule of Assessments for:
- Condition: {condition}
- Phase: {phase or 'Not specified'}
- Intervention Type: {intervention_type or 'Not specified'}

Reference Data from {trials_summary['total_trials']} Similar Trials:

Sample Primary Timeframes:
{chr(10).join([f"• {tf}" for tf in trials_summary['common_timeframes']['primary'][:10]])}

Sample Trials:
{chr(10).join([f"• {t['nct_id']}: {t['primary_outcome']} at {t['primary_timeframe']}" for t in trials_summary['sample_trials'][:8]])}

Based on your expertise, provide (plain text, no HTML):
1. Your SoA component: Specific recommendations for your focus area
2. Timing & frequency: When and how often assessments should occur (visit windows like “Day -14”, “Day 1”, “Week 4 ±3d”)
3. Rationale: Why these choices based on similar trials
4. Special considerations: Phase- or condition-specific notes
5. Quantify: total visits, key assessments per visit, estimated patient time/visit (hours) and total burden (hours)

Use headings and bullet points. Include a compact “Key Metrics” list (visit count, total hours, key assessment frequencies)."""

    return call_text_completion(
        "You are a Schedule of Assessments design expert.",
        prompt,
        max_tokens=1500,
        temperature=0.4,
    )


def synthesize_soa(condition, phase, intervention_type, trials_summary, agent_analyses):
    """SoA Architect synthesizes complete schedule"""

    combined_analysis = "\n\n".join([
        f"**{AGENTS[key]['name']}**:\n{content}"
        for key, content in agent_analyses.items()
    ])

    prompt = f"""You are the Chief SoA Architect synthesizing a complete Schedule of Assessments.

Trial Parameters:
- Condition: {condition}
- Phase: {phase or 'Not specified'}
- Intervention: {intervention_type or 'Not specified'}
- Based on {trials_summary['total_trials']} similar trials

Your team's recommendations:

{combined_analysis}

Create a COMPREHENSIVE DRAFT SOA (plain text):

1. Visit schedule overview: list at least 6–10 visits with timing windows
2. Assessment matrix: what gets done at each visit (bullets per visit); include counts where possible (e.g., “Labs: 5 draws total”, “Imaging: 2 scans”)
3. Special assessments: PK, biomarkers, imaging with timing
4. Implementation notes: key considerations for sites
5. Burden summary: total visits, total estimated patient hours, highest-burden visits

Provide clear headings and bullet points. No HTML or tables in the output. Start with a “Key Metrics” snapshot (visit count, total hours, primary assessment frequency)."""

    return call_text_completion(
        "You are the Chief SoA Architect creating comprehensive visit schedules.",
        prompt,
        model="gpt-4o",
        max_tokens=3000,
    )


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def soa_composer(condition, phase=None, intervention_type=None):
    """
    Run multi-agent SoA composition

    Args:
        condition (str): Medical condition
        phase (str): Trial phase (optional)
        intervention_type (str): Intervention type (optional)

    Returns:
        dict: SoA composition results
    """

    try:
        # Fetch similar trials
        print(f"🔍 Fetching trials for SoA composition: {condition}...")
        trials = fetch_similar_trials(condition, phase, intervention_type, limit=50)

        if len(trials) == 0:
            return {
                "success": False,
                "error": f"No trials found for {condition}. Try a different search."
            }

        # Summarize trials
        trials_summary = summarize_trials_for_soa(trials)

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

        # Synthesize complete SoA
        print("📋 Chief SoA Architect composing schedule...")
        complete_soa_raw = synthesize_soa(
            condition, phase, intervention_type, trials_summary, agent_analyses
        )
        complete_soa_html, complete_soa_text = render_formats(complete_soa_raw)

        return {
            "success": True,
            "query": {
                "condition": condition,
                "phase": phase,
                "intervention_type": intervention_type,
                "trials_analyzed": trials_summary['total_trials']
            },
            "agent_analyses": agent_results,
            "complete_soa": complete_soa_html,  # backward-compatible
            "complete_soa_raw": complete_soa_raw,
            "complete_soa_html": complete_soa_html,
            "complete_soa_text": complete_soa_text,
            "reference_trials": trials_summary['sample_trials']
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
    print("Testing SoA Composer...")

    result = soa_composer(
        condition="diabetes",
        phase="PHASE3",
        intervention_type="DRUG"
    )

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"\nQuery: {result['query']}")
        print(f"\nTrials analyzed: {result['query']['trials_analyzed']}")
        print(f"\nNumber of agent analyses: {len(result['agent_analyses'])}")
        print("\nComplete SoA Preview:")
        print(result['complete_soa'][:800] + "...")
    else:
        print(f"\n❌ ERROR: {result['error']}")

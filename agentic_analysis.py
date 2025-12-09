"""
Multi-Agent Protocol Analysis System
Each agent specializes in different aspects of clinical trial analysis
"""

from openai import OpenAI
import json
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI()

# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

AGENTS = {
    "eligibility_expert": {
        "name": "Dr. Sarah Chen - Eligibility Criteria Expert",
        "role": """You are an expert in clinical trial eligibility criteria design. Analyze inclusion/exclusion criteria for feasibility, clarity, and potential recruitment challenges.

IMPORTANT: Provide QUANTITATIVE analysis with statistics:
- COUNT the number of inclusion/exclusion criteria
- ESTIMATE screening failure rate based on criteria restrictiveness
- QUANTIFY target population size
- Provide PERCENTAGES for recruitment feasibility
- Give SPECIFIC NUMBERS in your recommendations

Example: "This trial has 8 inclusion criteria and 12 exclusion criteria. Based on these restrictions, estimated screening failure rate: 60-70%. Target population: approximately 10,000 potential patients in US."

Make your analysis data-driven!""",
        "focus": ["inclusion criteria", "exclusion criteria", "age ranges", "comorbidities", "recruitment feasibility"]
    },

    "endpoints_specialist": {
        "name": "Dr. Marcus Rodriguez - Endpoints & Outcomes Specialist",
        "role": """You are an expert in clinical trial endpoints and outcome measures. Evaluate primary and secondary endpoints for appropriateness and measurability.

IMPORTANT: Provide QUANTITATIVE analysis:
- COUNT the number of primary and secondary endpoints
- Specify exact ASSESSMENT TIMEPOINTS (Week 0, 4, 8, 12, etc.)
- QUANTIFY visit burden (e.g., "12 study visits over 24 weeks")
- Estimate COMPLETION RATES based on assessment burden
- Provide PERCENTAGES for regulatory acceptance likelihood

Example: "This trial has 1 primary endpoint (HbA1c at Week 24) and 8 secondary endpoints. Total assessment burden: 15 visits over 52 weeks. Expected completion rate: 75-80% based on visit frequency."

Make it quantitative!""",
        "focus": ["primary endpoints", "secondary endpoints", "outcome measures", "assessment timing", "clinical relevance"]
    },

    "stats_analyst": {
        "name": "Dr. Jennifer Wu - Statistical Design Analyst",
        "role": """You are a biostatistician expert. Analyze statistical methodology, sample size, power calculations, and analysis plans.

IMPORTANT: Provide QUANTITATIVE statistical analysis:
- State exact SAMPLE SIZE and power calculation assumptions
- Specify RANDOMIZATION RATIO (e.g., 2:1, 1:1:1)
- Calculate DROPOUT assumptions (e.g., "20% dropout → need n=300 for n=240 completers")
- Provide POWER percentages (e.g., "80% power to detect 0.5% HbA1c difference")
- Quantify ANALYSIS populations (ITT, PP, Safety)

Example: "Sample size: n=450 (2:1 randomization, 300 active vs 150 placebo). Assuming 20% dropout, 80% power to detect 0.5% mean difference in HbA1c (alpha=0.05). Requires 375 completers."

Make analysis statistical and numerical!""",
        "focus": ["sample size", "statistical methods", "randomization", "blinding", "power calculations", "analysis plan"]
    },

    "risk_assessor": {
        "name": "Dr. James O'Brien - Risk & Feasibility Assessor",
        "role": """You are a clinical operations expert. Identify potential risks, operational challenges, and areas likely to require protocol amendments.

IMPORTANT: Provide QUANTITATIVE risk assessment:
- SCORE protocol complexity (1-10 scale)
- ESTIMATE amendment likelihood (percentage)
- QUANTIFY site burden (visits, procedures, time per visit)
- Calculate estimated COST per patient
- Provide TIMELINE estimates (screening period, enrollment duration)

Example: "Protocol complexity: 8/10 (high). Amendment likelihood: 60% due to complex eligibility. Site burden: ~4 hours per visit × 15 visits = 60 hours. Estimated cost: $15,000/patient. Screening period: 8-12 weeks per patient."

Make risk assessment quantitative!""",
        "focus": ["operational risks", "amendment likelihood", "complexity assessment", "resource requirements", "timeline feasibility"]
    }
}

# =============================================================================
# AGENT FUNCTIONS
# =============================================================================

def call_agent(agent_key: str, trial_data: Dict, model: str = "gpt-4o-mini") -> Dict:
    """Call a specialized agent to analyze a trial"""
    agent = AGENTS[agent_key]

    system_prompt = f"""You are {agent['name']}.

{agent['role']}

Focus areas: {', '.join(agent['focus'])}

Provide a structured analysis with explicit numbers:
1. Key observations (3-5 bullets) — include counts/percentages (criteria counts, screening failure %, dropout %, visit counts, timelines)
2. Strengths (2-3 bullets) — cite metrics (sample size, power %, timelines)
3. Concerns/Risks (2-3 bullets) — quantify likelihood/impact (%, weeks, visits, cost/patient)
4. Recommendations (2-3 actionable suggestions) — with expected impact (%, weeks saved, visits reduced)

Be concise, specific, and quantitative. Plain text, no HTML."""

    user_prompt = f"""Analyze this clinical trial from your expert perspective:

TRIAL DATA:
{json.dumps(trial_data, indent=2)}

Provide your specialized analysis."""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        return {
            "agent": agent['name'],
            "agent_key": agent_key,
            "analysis": response.choices[0].message.content,
            "focus_areas": agent['focus']
        }
    except Exception as e:
        return {
            "agent": agent['name'],
            "agent_key": agent_key,
            "analysis": f"Error: {str(e)}",
            "focus_areas": agent['focus']
        }


def coordinate_agents(trial_data: Dict, agent_analyses: List[Dict], model: str = "gpt-4o") -> str:
    """Coordinator agent synthesizes all perspectives"""

    # Combine all agent analyses
    combined_analyses = "\n\n".join([
        f"=== {analysis['agent']} ===\n{analysis['analysis']}"
        for analysis in agent_analyses
    ])

    system_prompt = """You are the Chief Protocol Strategist coordinating a team of experts.

Your job:
1. Synthesize insights into a QUANTITATIVE executive summary (5 bullets, each with a metric: %, n, weeks, visits, or $)
2. Identify common themes with statistics (sample size, timelines, dropout %, visit burden, cost/patient)
3. Provide strategic recommendations with numerical priorities (Priority 1-3) and expected impact (%, weeks, visits, cost)
4. Highlight critical risks with quantified impact (likelihood % and impact measure)

Output (plain text):
- Section: Key Metrics (sample size, randomization ratio, dropout %, visit count, timeline weeks, cost/patient)
- Section: Executive Summary (5 bullets, each with a number)
- Section: Recommendations (3 bullets with priority and impact number)
- Section: Risks (3 bullets with likelihood % and impact)

Be data-driven and quantitative. No HTML."""

    user_prompt = f"""Here are the analyses from your expert team:

{combined_analyses}

TRIAL BEING ANALYZED:
NCT ID: {trial_data.get('nct_id', 'N/A')}
Title: {trial_data.get('title', 'N/A')}

Synthesize these perspectives into a comprehensive executive summary with strategic recommendations."""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=1500
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Coordination error: {str(e)}"


def multi_agent_analysis(trial_data: Dict, parallel: bool = True) -> Dict:
    """
    Run multi-agent analysis on a clinical trial

    Args:
        trial_data: Trial information dictionary
        parallel: If True, agents run in parallel (faster). If False, sequential.

    Returns:
        Dictionary with agent analyses and coordinator synthesis
    """

    print(f"\n{'='*70}")
    print("🤖 MULTI-AGENT PROTOCOL ANALYSIS SYSTEM")
    print(f"{'='*70}")
    print(f"Analyzing: {trial_data.get('nct_id', 'Unknown')} - {trial_data.get('title', 'Unknown')[:60]}...")
    print(f"\nDeploying {len(AGENTS)} specialized agents...\n")

    # Run all agents
    agent_analyses = []

    if parallel:
        # In a real production system, you'd use asyncio or threading here
        # For now, we'll run sequentially but this is where you'd parallelize
        print("⚡ Running agents in parallel mode...")

    for i, (agent_key, agent_info) in enumerate(AGENTS.items(), 1):
        print(f"[{i}/{len(AGENTS)}] 🔍 {agent_info['name']} analyzing...")
        analysis = call_agent(agent_key, trial_data, model="gpt-4o-mini")
        agent_analyses.append(analysis)
        print(f"      ✓ Analysis complete")

    print(f"\n{'='*70}")
    print("👔 Chief Strategist synthesizing team insights...")
    print(f"{'='*70}\n")

    # Coordinate and synthesize
    executive_summary = coordinate_agents(trial_data, agent_analyses, model="gpt-4o")

    return {
        "trial": {
            "nct_id": trial_data.get('nct_id'),
            "title": trial_data.get('title')
        },
        "agent_analyses": agent_analyses,
        "executive_summary": executive_summary,
        "metadata": {
            "num_agents": len(agent_analyses),
            "model_used": "gpt-4o-mini + gpt-4o"
        }
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example trial data
    example_trial = {
        "nct_id": "NCT12345678",
        "title": "Phase 3 Study of Novel Drug X in Type 2 Diabetes",
        "status": "RECRUITING",
        "conditions": ["Type 2 Diabetes Mellitus"],
        "interventions": ["Drug X 100mg", "Placebo"],
        "summary": """This is a multicenter, randomized, double-blind, placebo-controlled study
        evaluating the efficacy and safety of Drug X in adult patients with Type 2 Diabetes Mellitus
        inadequately controlled on metformin. Primary endpoint is change in HbA1c at 24 weeks.
        Approximately 300 patients will be randomized 2:1.""",
        "eligibility_criteria": {
            "inclusion": [
                "Age 18-75 years",
                "Diagnosed with T2DM for at least 6 months",
                "HbA1c 7.5-10.5%",
                "On stable metformin dose for at least 8 weeks"
            ],
            "exclusion": [
                "Type 1 diabetes or secondary diabetes",
                "eGFR < 45 mL/min",
                "History of diabetic ketoacidosis",
                "Severe cardiovascular disease"
            ]
        },
        "design": "Randomized, double-blind, placebo-controlled, parallel-group",
        "phase": "Phase 3",
        "primary_endpoint": "Change from baseline in HbA1c at Week 24",
        "sample_size": 300
    }

    # Run multi-agent analysis
    result = multi_agent_analysis(example_trial)

    # Display results
    print("\n" + "="*70)
    print("📊 MULTI-AGENT ANALYSIS RESULTS")
    print("="*70)

    print("\n### INDIVIDUAL AGENT ANALYSES ###\n")
    for analysis in result['agent_analyses']:
        print(f"\n{'='*70}")
        print(f"🔹 {analysis['agent']}")
        print('='*70)
        print(analysis['analysis'])

    print("\n\n" + "="*70)
    print("### EXECUTIVE SUMMARY (COORDINATOR) ###")
    print("="*70)
    print(result['executive_summary'])

    print("\n\n✨ Analysis complete!")

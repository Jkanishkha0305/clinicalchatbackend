"""
Multi-Agent Protocol Comparison System
Different agents compare protocols from specialized perspectives
"""

from openai import OpenAI
import json
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI()

# =============================================================================
# COMPARISON AGENTS
# =============================================================================

def eligibility_comparator_agent(trials: List[Dict]) -> str:
    """Compare eligibility criteria across trials"""

    system_prompt = """You are an eligibility criteria expert. Compare the eligibility criteria across these trials.

Provide (plain text, quantitative):
1. Common inclusion criteria patterns (with counts)
2. Key differences in exclusion criteria (counts; identify most restrictive vs. most permissive)
3. Recruitment implications (screening failure % range; estimated eligible pool size if possible)
4. Rank trials by restrictiveness (1 = most restrictive)

Include at least 3 numeric comparisons (criteria counts, screening failure %, eligible pool estimates)."""

    trials_summary = []
    for trial in trials:
        trials_summary.append({
            'nct_id': trial.get('nct_id'),
            'title': trial.get('title', '')[:100],
            'conditions': trial.get('conditions', []),
            'summary': trial.get('summary', '')[:500]
        })

    user_prompt = f"""Compare eligibility criteria across these {len(trials)} trials:

{json.dumps(trials_summary, indent=2)}

Provide detailed comparison of eligibility approaches."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def design_comparator_agent(trials: List[Dict]) -> str:
    """Compare study designs across trials"""

    system_prompt = """You are a clinical trial design expert. Compare the study designs across these trials.

Analyze (plain text, quantitative):
1. Study design types (counts per type)
2. Randomization and blinding approaches (ratios, counts)
3. Treatment duration and follow-up periods (ranges/medians in weeks)
4. Sample size considerations (n, arms)
5. Design strengths and limitations (include at least 3 numeric contrasts)

Provide clear comparative insights with numbers; rank trials for methodological rigor (1 = strongest)."""

    trials_summary = []
    for trial in trials:
        trials_summary.append({
            'nct_id': trial.get('nct_id'),
            'title': trial.get('title', '')[:100],
            'status': trial.get('status'),
            'interventions': trial.get('interventions', []),
            'summary': trial.get('summary', '')[:500]
        })

    user_prompt = f"""Compare study designs across these {len(trials)} trials:

{json.dumps(trials_summary, indent=2)}

Focus on design methodology and quality."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def endpoints_comparator_agent(trials: List[Dict]) -> str:
    """Compare endpoints and outcome measures"""

    system_prompt = """You are a clinical endpoints expert. Compare the outcome measures across these trials.

Analyze (plain text, quantitative):
1. Primary endpoint choices (counts) and rationale
2. Secondary endpoint patterns (counts)
3. Assessment timing and frequency (visit/timepoint counts; ranges)
4. Clinical relevance/regulatory acceptance (likelihood % if inferable)
5. Endpoint sensitivity/specificity (comparative notes)

Provide actionable insights with at least 3 numeric contrasts; rank endpoints set quality (1 = strongest)."""

    trials_summary = []
    for trial in trials:
        trials_summary.append({
            'nct_id': trial.get('nct_id'),
            'title': trial.get('title', '')[:100],
            'conditions': trial.get('conditions', []),
            'interventions': trial.get('interventions', []),
            'summary': trial.get('summary', '')[:500]
        })

    user_prompt = f"""Compare endpoints and outcomes across these {len(trials)} trials:

{json.dumps(trials_summary, indent=2)}

Focus on endpoint strategy and appropriateness."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def strategic_synthesis_agent(comparisons: Dict, trials: List[Dict]) -> str:
    """Synthesize all comparisons into strategic recommendations"""

    system_prompt = """You are the Chief Protocol Strategist. Synthesize the expert comparisons into actionable strategic recommendations.

Provide (plain text):
1. Executive summary (3–5 bullets, each with a number: n, %, weeks, visits)
2. Best practices identified (with frequency or %)
3. Innovation opportunities (2–3 with expected impact)
4. Risk mitigation strategies (2–3 with likelihood/impact %)
5. Top 3 design recommendations (ranked, each with expected impact number)

Be strategic and forward-thinking; start with a “Key Metrics” snapshot (sample size ranges, duration ranges, restrictiveness rank)."""

    all_comparisons = f"""
    ELIGIBILITY COMPARISON:
    {comparisons['eligibility']}

    DESIGN COMPARISON:
    {comparisons['design']}

    ENDPOINTS COMPARISON:
    {comparisons['endpoints']}
    """

    trials_list = [f"{t.get('nct_id')} - {t.get('title', '')[:80]}" for t in trials]

    user_prompt = f"""Synthesize these expert comparisons into strategic recommendations:

    TRIALS ANALYZED:
    {chr(10).join(trials_list)}

    EXPERT ANALYSES:
    {all_comparisons}

    Provide strategic synthesis with actionable recommendations."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def multi_agent_comparison(trials: List[Dict]) -> Dict:
    """
    Compare multiple trials using specialized agents

    Args:
        trials: List of trial dictionaries (2-5 trials recommended)

    Returns:
        Dictionary with comparative analyses and synthesis
    """

    if len(trials) < 2:
        return {'error': 'Need at least 2 trials to compare'}

    if len(trials) > 5:
        print("⚠️  Warning: Comparing more than 5 trials. Using first 5...")
        trials = trials[:5]

    print(f"\n{'='*70}")
    print("🔬 MULTI-AGENT PROTOCOL COMPARISON SYSTEM")
    print(f"{'='*70}")
    print(f"Comparing {len(trials)} clinical trials:")
    for i, trial in enumerate(trials, 1):
        print(f"  [{i}] {trial.get('nct_id', 'Unknown')} - {trial.get('title', 'Unknown')[:60]}...")

    print(f"\n{'='*70}")
    print("Deploying specialized comparison agents...")
    print(f"{'='*70}\n")

    comparisons = {}

    # Agent 1: Eligibility Comparison
    print("[1/3] 👥 Eligibility Comparator Agent analyzing...")
    comparisons['eligibility'] = eligibility_comparator_agent(trials)
    print("      ✓ Eligibility comparison complete")

    # Agent 2: Design Comparison
    print("[2/3] 📐 Design Comparator Agent analyzing...")
    comparisons['design'] = design_comparator_agent(trials)
    print("      ✓ Design comparison complete")

    # Agent 3: Endpoints Comparison
    print("[3/3] 🎯 Endpoints Comparator Agent analyzing...")
    comparisons['endpoints'] = endpoints_comparator_agent(trials)
    print("      ✓ Endpoints comparison complete")

    # Synthesis Agent
    print(f"\n{'='*70}")
    print("🧠 Strategic Synthesis Agent integrating insights...")
    print(f"{'='*70}\n")

    strategic_synthesis = strategic_synthesis_agent(comparisons, trials)

    print("✨ Multi-agent comparison complete!\n")

    return {
        'trials': [{'nct_id': t.get('nct_id'), 'title': t.get('title')} for t in trials],
        'comparisons': comparisons,
        'strategic_synthesis': strategic_synthesis,
        'metadata': {
            'num_trials': len(trials),
            'agents_used': 4
        }
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Compare 3 diabetes trials
    example_trials = [
        {
            'nct_id': 'NCT12345678',
            'title': 'Phase 3 Study of Drug A in Type 2 Diabetes',
            'status': 'RECRUITING',
            'conditions': ['Type 2 Diabetes Mellitus'],
            'interventions': ['Drug A 100mg', 'Placebo'],
            'summary': 'A randomized, double-blind study evaluating Drug A...'
        },
        {
            'nct_id': 'NCT98765432',
            'title': 'Phase 3 Study of Drug B in Type 2 Diabetes',
            'status': 'COMPLETED',
            'conditions': ['Type 2 Diabetes Mellitus'],
            'interventions': ['Drug B 200mg', 'Metformin'],
            'summary': 'An open-label comparative study of Drug B vs Metformin...'
        },
        {
            'nct_id': 'NCT11223344',
            'title': 'Phase 2/3 Study of Combination Therapy in Diabetes',
            'status': 'ACTIVE_NOT_RECRUITING',
            'conditions': ['Type 2 Diabetes Mellitus'],
            'interventions': ['Drug A + Drug B', 'Standard of Care'],
            'summary': 'A multicenter study evaluating combination therapy...'
        }
    ]

    result = multi_agent_comparison(example_trials)

    print("\n" + "="*70)
    print("📊 COMPARISON RESULTS")
    print("="*70)

    print("\n### ELIGIBILITY COMPARISON ###")
    print(result['comparisons']['eligibility'])

    print("\n### DESIGN COMPARISON ###")
    print(result['comparisons']['design'])

    print("\n### ENDPOINTS COMPARISON ###")
    print(result['comparisons']['endpoints'])

    print("\n### STRATEGIC SYNTHESIS ###")
    print(result['strategic_synthesis'])

    print("\n✅ Comparison complete!")

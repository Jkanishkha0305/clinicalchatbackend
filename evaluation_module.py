"""
Evaluation Module for AI Agent Outputs
Adapted from CTBench methodology for clinical trial analysis agents
"""

import re
import json
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Tuple
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

############################ AUTOMATED METRICS ##############################

def extract_numbers_from_text(text: str) -> List[str]:
    """Extract all numeric values from text including percentages, counts, ranges"""
    # Pattern matches: 60%, 15 visits, 60-70%, 2-3 issues, $500, etc.
    number_patterns = [
        r'\d+\.?\d*%',  # percentages: 60%, 15.5%
        r'\d+\.?\d*\s*-\s*\d+\.?\d*%',  # percentage ranges: 60-70%
        r'\d+\.?\d*\s*-\s*\d+\.?\d*',  # numeric ranges: 2-3, 15-20
        r'\$\d+[,\d]*\.?\d*[KMB]?',  # dollar amounts: $500, $1.5M
        r'\d+\.?\d*\s*(weeks?|months?|days?|hours?|visits?|patients?|sites?)',  # units
        r'\d+\.?\d*',  # standalone numbers
    ]

    all_numbers = []
    for pattern in number_patterns:
        all_numbers.extend(re.findall(pattern, text, re.IGNORECASE))

    return list(set(all_numbers))  # Remove duplicates

def count_quantitative_density(text: str) -> Dict:
    """Calculate numeric density metrics"""
    numbers = extract_numbers_from_text(text)
    words = text.split()
    sentences = text.split('.')

    return {
        'total_numbers': len(numbers),
        'numbers_per_100_words': (len(numbers) / len(words) * 100) if words else 0,
        'numbers_per_sentence': (len(numbers) / len(sentences)) if sentences else 0,
        'extracted_numbers': numbers[:10]  # Sample of numbers found
    }

def check_section_completeness(text: str, required_sections: List[str]) -> Dict:
    """Check if required sections are present in the output"""
    text_lower = text.lower()

    present_sections = []
    missing_sections = []

    for section in required_sections:
        # Fuzzy matching - check for key terms
        section_terms = section.lower().split()
        if any(term in text_lower for term in section_terms):
            present_sections.append(section)
        else:
            missing_sections.append(section)

    completeness_score = (len(present_sections) / len(required_sections) * 100) if required_sections else 100

    return {
        'completeness_score': completeness_score,
        'present_sections': present_sections,
        'missing_sections': missing_sections,
        'total_required': len(required_sections),
        'total_present': len(present_sections)
    }

def extract_key_metrics_section(text: str) -> Dict:
    """Extract and parse 'Key Metrics' section if present"""
    # Look for "Key Metrics" heading
    key_metrics_pattern = r'(?:Key Metrics|Summary Metrics)[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)'
    match = re.search(key_metrics_pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        metrics_text = match.group(1)
        numbers = extract_numbers_from_text(metrics_text)
        return {
            'has_key_metrics_section': True,
            'metrics_text': metrics_text.strip(),
            'metrics_count': len(numbers),
            'metrics_values': numbers
        }
    else:
        return {
            'has_key_metrics_section': False,
            'metrics_text': '',
            'metrics_count': 0,
            'metrics_values': []
        }

############################ GPT-4O AS JUDGE ##############################

def build_amendment_eval_prompt(agent_output: str, nct_id: str) -> Tuple[str, str]:
    """Build evaluation prompt for amendment risk analysis output"""
    system = """You are an expert evaluator for clinical trial risk assessment reports.
Your task is to score an AI-generated amendment risk analysis on specific criteria.

Provide scores (0-10) and brief justifications for each criterion:
1. **Quantitative Rigor**: Does the report include specific numbers, percentages, ranges, and numeric indicators?
2. **Risk Assessment Clarity**: Are risks clearly categorized (Low/Medium/High) with probability estimates?
3. **Actionability**: Are recommendations specific and quantified (e.g., "reduce visits by 3", "save 4 weeks")?
4. **Completeness**: Does it cover all key risk areas (eligibility, endpoints, visits, recruitment, retention)?
5. **Clinical Validity**: Are the risk factors and recommendations clinically sound and realistic?

Return ONLY a JSON object with this exact structure:
{
  "quantitative_rigor": {"score": 0-10, "justification": "brief explanation"},
  "risk_clarity": {"score": 0-10, "justification": "brief explanation"},
  "actionability": {"score": 0-10, "justification": "brief explanation"},
  "completeness": {"score": 0-10, "justification": "brief explanation"},
  "clinical_validity": {"score": 0-10, "justification": "brief explanation"},
  "overall_score": 0-10,
  "overall_assessment": "1-2 sentence summary"
}"""

    question = f"""Trial ID: {nct_id}

Amendment Risk Analysis Report:
{agent_output}

Please evaluate this report according to the criteria above."""

    return system, question

def build_design_patterns_eval_prompt(agent_output: str, nct_id: str) -> Tuple[str, str]:
    """Build evaluation prompt for design pattern discovery output"""
    system = """You are an expert evaluator for clinical trial design pattern analysis.
Your task is to score an AI-generated design pattern discovery report on specific criteria.

Provide scores (0-10) and brief justifications for each criterion:
1. **Pattern Identification**: Are specific design patterns identified with prevalence data (%, counts)?
2. **Statistical Evidence**: Are patterns supported by numeric evidence (completion rates, enrollment stats)?
3. **Comparative Analysis**: Does it compare design archetypes with quantitative metrics?
4. **Actionability**: Are design recommendations specific with expected impact (% improvement, time saved)?
5. **Insight Quality**: Are the insights meaningful and clinically relevant for trial design?

Return ONLY a JSON object with this exact structure:
{
  "pattern_identification": {"score": 0-10, "justification": "brief explanation"},
  "statistical_evidence": {"score": 0-10, "justification": "brief explanation"},
  "comparative_analysis": {"score": 0-10, "justification": "brief explanation"},
  "actionability": {"score": 0-10, "justification": "brief explanation"},
  "insight_quality": {"score": 0-10, "justification": "brief explanation"},
  "overall_score": 0-10,
  "overall_assessment": "1-2 sentence summary"
}"""

    question = f"""Trial ID: {nct_id}

Design Pattern Discovery Report:
{agent_output}

Please evaluate this report according to the criteria above."""

    return system, question

def build_soa_eval_prompt(agent_output: str, nct_id: str) -> Tuple[str, str]:
    """Build evaluation prompt for SoA composer output"""
    system = """You are an expert evaluator for clinical trial Schedule of Assessments (SoA).
Your task is to score an AI-generated SoA composition report on specific criteria.

Provide scores (0-10) and brief justifications for each criterion:
1. **Specificity**: Are visit timings specific with windows (e.g., "Day 1", "Week 4 ±3d")?
2. **Quantification**: Are visit counts, assessment frequencies, and patient burden hours specified?
3. **Rationale Quality**: Are timing/frequency choices justified with evidence from similar trials?
4. **Completeness**: Does it cover all necessary assessment domains with appropriate detail?
5. **Feasibility**: Is the proposed SoA realistic and implementable in a clinical trial setting?

Return ONLY a JSON object with this exact structure:
{
  "specificity": {"score": 0-10, "justification": "brief explanation"},
  "quantification": {"score": 0-10, "justification": "brief explanation"},
  "rationale_quality": {"score": 0-10, "justification": "brief explanation"},
  "completeness": {"score": 0-10, "justification": "brief explanation"},
  "feasibility": {"score": 0-10, "justification": "brief explanation"},
  "overall_score": 0-10,
  "overall_assessment": "1-2 sentence summary"
}"""

    question = f"""Trial ID: {nct_id}

Schedule of Assessments (SoA) Report:
{agent_output}

Please evaluate this report according to the criteria above."""

    return system, question

def run_gpt4o_evaluation(system_prompt: str, user_prompt: str) -> Dict:
    """Run GPT-4o evaluation and return scores"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            seed=42
        )

        result = json.loads(response.choices[0].message.content)
        return {
            'success': True,
            'evaluation': result
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

############################ MAIN EVALUATION FUNCTIONS ##############################

def evaluate_amendment_report(agent_output: str, nct_id: str = "Unknown") -> Dict:
    """Complete evaluation for amendment risk analysis"""

    # Required sections for amendment analysis
    required_sections = [
        "Risk Assessment", "Risk Factors", "Recommendations",
        "Key Metrics", "Probability", "Impact"
    ]

    # Automated metrics
    numeric_density = count_quantitative_density(agent_output)
    section_check = check_section_completeness(agent_output, required_sections)
    key_metrics = extract_key_metrics_section(agent_output)

    # GPT-4o evaluation
    system, question = build_amendment_eval_prompt(agent_output, nct_id)
    gpt_eval = run_gpt4o_evaluation(system, question)

    # Compile results
    automated_score = (
        (numeric_density['numbers_per_100_words'] / 10) * 0.3 +  # Max 10 numbers per 100 words = full score
        (section_check['completeness_score'] / 100) * 10 * 0.3 +
        (min(key_metrics['metrics_count'], 5) / 5) * 10 * 0.4  # At least 5 metrics = full score
    )

    return {
        'report_type': 'Amendment Risk Analysis',
        'nct_id': nct_id,
        'timestamp': datetime.now().isoformat(),
        'automated_metrics': {
            'numeric_density': numeric_density,
            'section_completeness': section_check,
            'key_metrics_section': key_metrics,
            'automated_score': round(automated_score, 2)
        },
        'gpt4o_evaluation': gpt_eval,
        'overall_grade': calculate_overall_grade(automated_score, gpt_eval)
    }

def evaluate_design_patterns_report(agent_output: str, nct_id: str = "Unknown") -> Dict:
    """Complete evaluation for design pattern discovery"""

    required_sections = [
        "Patterns Identified", "Design Archetypes", "Success Indicators",
        "Recommendations", "Key Metrics", "Statistics"
    ]

    numeric_density = count_quantitative_density(agent_output)
    section_check = check_section_completeness(agent_output, required_sections)
    key_metrics = extract_key_metrics_section(agent_output)

    system, question = build_design_patterns_eval_prompt(agent_output, nct_id)
    gpt_eval = run_gpt4o_evaluation(system, question)

    automated_score = (
        (numeric_density['numbers_per_100_words'] / 10) * 0.3 +
        (section_check['completeness_score'] / 100) * 10 * 0.3 +
        (min(key_metrics['metrics_count'], 5) / 5) * 10 * 0.4
    )

    return {
        'report_type': 'Design Pattern Discovery',
        'nct_id': nct_id,
        'timestamp': datetime.now().isoformat(),
        'automated_metrics': {
            'numeric_density': numeric_density,
            'section_completeness': section_check,
            'key_metrics_section': key_metrics,
            'automated_score': round(automated_score, 2)
        },
        'gpt4o_evaluation': gpt_eval,
        'overall_grade': calculate_overall_grade(automated_score, gpt_eval)
    }

def evaluate_soa_report(agent_output: str, nct_id: str = "Unknown") -> Dict:
    """Complete evaluation for SoA composer"""

    required_sections = [
        "Timing", "Frequency", "Visit Windows", "Rationale",
        "Special Considerations", "Key Metrics", "Patient Burden"
    ]

    numeric_density = count_quantitative_density(agent_output)
    section_check = check_section_completeness(agent_output, required_sections)
    key_metrics = extract_key_metrics_section(agent_output)

    system, question = build_soa_eval_prompt(agent_output, nct_id)
    gpt_eval = run_gpt4o_evaluation(system, question)

    automated_score = (
        (numeric_density['numbers_per_100_words'] / 10) * 0.3 +
        (section_check['completeness_score'] / 100) * 10 * 0.3 +
        (min(key_metrics['metrics_count'], 5) / 5) * 10 * 0.4
    )

    return {
        'report_type': 'SoA Composition',
        'nct_id': nct_id,
        'timestamp': datetime.now().isoformat(),
        'automated_metrics': {
            'numeric_density': numeric_density,
            'section_completeness': section_check,
            'key_metrics_section': key_metrics,
            'automated_score': round(automated_score, 2)
        },
        'gpt4o_evaluation': gpt_eval,
        'overall_grade': calculate_overall_grade(automated_score, gpt_eval)
    }

def calculate_overall_grade(automated_score: float, gpt_eval: Dict) -> Dict:
    """Calculate final grade combining automated and GPT-4o scores"""
    if not gpt_eval.get('success'):
        return {
            'final_score': automated_score,
            'grade': score_to_letter_grade(automated_score),
            'note': 'Based on automated metrics only (GPT-4o eval failed)'
        }

    gpt_overall = gpt_eval['evaluation'].get('overall_score', 5)

    # Weighted average: 40% automated, 60% GPT-4o
    final_score = (automated_score * 0.4) + (gpt_overall * 0.6)

    return {
        'final_score': round(final_score, 2),
        'automated_score': round(automated_score, 2),
        'gpt4o_score': gpt_overall,
        'grade': score_to_letter_grade(final_score),
        'status': interpret_score(final_score)
    }

def score_to_letter_grade(score: float) -> str:
    """Convert 0-10 score to letter grade"""
    if score >= 9.0:
        return 'A+'
    elif score >= 8.5:
        return 'A'
    elif score >= 8.0:
        return 'A-'
    elif score >= 7.5:
        return 'B+'
    elif score >= 7.0:
        return 'B'
    elif score >= 6.5:
        return 'B-'
    elif score >= 6.0:
        return 'C+'
    elif score >= 5.5:
        return 'C'
    elif score >= 5.0:
        return 'C-'
    elif score >= 4.0:
        return 'D'
    else:
        return 'F'

def interpret_score(score: float) -> str:
    """Interpret final score"""
    if score >= 8.5:
        return 'Excellent - Production Ready'
    elif score >= 7.5:
        return 'Good - Minor improvements recommended'
    elif score >= 6.5:
        return 'Acceptable - Some improvements needed'
    elif score >= 5.5:
        return 'Fair - Significant improvements needed'
    else:
        return 'Poor - Major revisions required'

############################ BATCH EVALUATION ##############################

def evaluate_all_reports(amendment_output: str = None,
                        design_output: str = None,
                        soa_output: str = None,
                        nct_id: str = "Unknown") -> Dict:
    """Evaluate all three agent outputs at once"""

    results = {
        'nct_id': nct_id,
        'timestamp': datetime.now().isoformat(),
        'evaluations': {}
    }

    if amendment_output:
        results['evaluations']['amendment'] = evaluate_amendment_report(amendment_output, nct_id)

    if design_output:
        results['evaluations']['design_patterns'] = evaluate_design_patterns_report(design_output, nct_id)

    if soa_output:
        results['evaluations']['soa'] = evaluate_soa_report(soa_output, nct_id)

    # Calculate aggregate score
    scores = []
    for eval_result in results['evaluations'].values():
        if eval_result['overall_grade'].get('final_score'):
            scores.append(eval_result['overall_grade']['final_score'])

    if scores:
        results['aggregate'] = {
            'average_score': round(sum(scores) / len(scores), 2),
            'grade': score_to_letter_grade(sum(scores) / len(scores)),
            'reports_evaluated': len(scores)
        }

    return results

############################ SIMPLE BENCHMARK COMPARISON ##############################

def compare_to_baseline(current_score: float, baseline_scores: List[float]) -> Dict:
    """Compare current performance to baseline"""
    if not baseline_scores:
        return {'status': 'No baseline available'}

    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    improvement = current_score - avg_baseline
    improvement_pct = (improvement / avg_baseline * 100) if avg_baseline > 0 else 0

    return {
        'current_score': current_score,
        'baseline_average': round(avg_baseline, 2),
        'improvement': round(improvement, 2),
        'improvement_percentage': round(improvement_pct, 1),
        'status': 'Improved' if improvement > 0 else 'Declined' if improvement < 0 else 'Same'
    }

"""
Quick Test Script for Agent Evaluation
Run this to test evaluation on your agent outputs
"""

from evaluation_module import (
    evaluate_amendment_report,
    evaluate_design_patterns_report,
    evaluate_soa_report,
    evaluate_all_reports
)
import json

# Sample outputs for testing (replace with actual outputs from your agents)
SAMPLE_AMENDMENT_OUTPUT = """
**Amendment Risk Analysis**

Risk Assessment: MEDIUM (65% likelihood of protocol amendments)

Key Risk Factors:
1. Complex eligibility criteria (18 inclusion, 12 exclusion criteria) - 60-70% screening failure rate expected
2. High visit burden (15 visits over 12 months) - dropout risk 25-30%
3. Rare disease population - enrollment may take 18-24 months vs planned 12 months

Recommendations:
1. Simplify exclusion criteria - reduce from 12 to 8 criteria - Expected impact: 15% reduction in screening failure
2. Consolidate visits - reduce from 15 to 12 visits - Expected impact: 10% reduction in dropout, save 8 weeks
3. Add more sites - increase from 20 to 30 sites - Expected impact: reduce enrollment time by 6 months

Key Metrics:
- Total visits: 15
- Hours per visit: 3-4 hours
- Enrollment duration: 18-24 months (50% over target)
- Dropout risk: 25-30%
- Screening failure: 60-70%
"""

SAMPLE_DESIGN_PATTERN_OUTPUT = """
**Design Pattern Analysis**

Key Patterns Identified:
1. Phase Distribution: 45% Phase 3, 30% Phase 2, 15% Phase 1, 10% Phase 4
2. Median enrollment: 250 patients (range 50-500)
3. Most common masking: Double-blind (65%), Open-label (25%), Single-blind (10%)
4. Allocation: Randomized (80%), Non-randomized (20%)

Design Archetypes:
1. Large Phase 3 RCTs (40% prevalence) - 300+ patients, double-blind, parallel assignment
2. Small Phase 2 trials (35% prevalence) - 100-200 patients, often open-label
3. Phase 1 dose-finding (25% prevalence) - 20-50 patients, non-randomized

Success Indicators:
- Trials with ≤10 visits: 75% completion rate vs 55% for >10 visits
- Dropout range: 15-20% for simple protocols vs 25-35% for complex

Recommendations:
1. Limit to 10 visits - Expected impact: 20% improvement in completion rate
2. Use double-blind design if feasible - Expected impact: increases credibility, reduces bias
3. Target 200-250 patients for Phase 2 - Expected impact: adequate power, manageable recruitment (saves 12 weeks)

Key Metrics:
- Average visit count: 12 visits
- Average enrollment duration: 18 months
- Success rate by design: Double-blind 70%, Open-label 60%
"""

SAMPLE_SOA_OUTPUT = """
**Schedule of Assessments**

Safety & Tolerability Component:

Timing & Frequency:
- Screening: Day -14 to Day -1
- Baseline: Day 1
- Treatment visits: Week 2, Week 4 ±3d, Week 8 ±3d, Week 12 ±7d
- Follow-up: Week 16 ±7d
- Early termination visit if needed

Assessment Details:
1. Vital Signs: All visits (7 total assessments)
2. Adverse Event Monitoring: All visits from Day 1 onward (6 assessments)
3. Laboratory Safety Panel: Screening, Week 4, Week 12, Follow-up (4 assessments)
4. ECG: Screening, Week 8, Week 12 (3 assessments)

Rationale:
Based on 25 similar oncology trials, Week 4 ±3d timing captures early toxicity events (observed in 80% of cases).
Week 12 assessment aligns with treatment cycle completion in 90% of comparable studies.

Special Considerations:
- Phase 2 oncology requires closer early monitoring
- Narrow visit windows for safety (±3d for first 2 months)
- Wider windows acceptable later (±7d) as safety profile established

Quantification:
- Total visits: 7
- Key assessments per visit: 4-6
- Estimated patient time per visit: 2-3 hours
- Total patient burden: 14-21 hours over 16 weeks

Key Metrics:
- Visit count: 7
- Total burden: 14-21 hours
- Assessment frequency: Vitals 7x, Labs 4x, ECG 3x, AE monitoring continuous
"""

def test_individual_evaluations():
    """Test each evaluation function separately"""
    print("="*80)
    print("TESTING INDIVIDUAL EVALUATIONS")
    print("="*80)

    # Test Amendment
    print("\n1. Amendment Risk Analysis Evaluation")
    print("-"*80)
    amendment_eval = evaluate_amendment_report(SAMPLE_AMENDMENT_OUTPUT, "NCT00000001")
    print(f"Automated Score: {amendment_eval['automated_metrics']['automated_score']}/10")
    print(f"Numeric Density: {amendment_eval['automated_metrics']['numeric_density']['total_numbers']} numbers found")
    print(f"Section Completeness: {amendment_eval['automated_metrics']['section_completeness']['completeness_score']:.1f}%")

    if amendment_eval['gpt4o_evaluation']['success']:
        gpt_eval = amendment_eval['gpt4o_evaluation']['evaluation']
        print(f"\nGPT-4o Overall Score: {gpt_eval.get('overall_score', 'N/A')}/10")
        print(f"Assessment: {gpt_eval.get('overall_assessment', 'N/A')}")

    print(f"\n>>> FINAL GRADE: {amendment_eval['overall_grade']['grade']} ({amendment_eval['overall_grade']['final_score']}/10)")
    print(f">>> STATUS: {amendment_eval['overall_grade']['status']}")

    # Test Design Patterns
    print("\n\n2. Design Pattern Discovery Evaluation")
    print("-"*80)
    design_eval = evaluate_design_patterns_report(SAMPLE_DESIGN_PATTERN_OUTPUT, "NCT00000001")
    print(f"Automated Score: {design_eval['automated_metrics']['automated_score']}/10")
    print(f"Numeric Density: {design_eval['automated_metrics']['numeric_density']['total_numbers']} numbers found")
    print(f"Section Completeness: {design_eval['automated_metrics']['section_completeness']['completeness_score']:.1f}%")

    if design_eval['gpt4o_evaluation']['success']:
        gpt_eval = design_eval['gpt4o_evaluation']['evaluation']
        print(f"\nGPT-4o Overall Score: {gpt_eval.get('overall_score', 'N/A')}/10")
        print(f"Assessment: {gpt_eval.get('overall_assessment', 'N/A')}")

    print(f"\n>>> FINAL GRADE: {design_eval['overall_grade']['grade']} ({design_eval['overall_grade']['final_score']}/10)")
    print(f">>> STATUS: {design_eval['overall_grade']['status']}")

    # Test SoA
    print("\n\n3. SoA Composer Evaluation")
    print("-"*80)
    soa_eval = evaluate_soa_report(SAMPLE_SOA_OUTPUT, "NCT00000001")
    print(f"Automated Score: {soa_eval['automated_metrics']['automated_score']}/10")
    print(f"Numeric Density: {soa_eval['automated_metrics']['numeric_density']['total_numbers']} numbers found")
    print(f"Section Completeness: {soa_eval['automated_metrics']['section_completeness']['completeness_score']:.1f}%")

    if soa_eval['gpt4o_evaluation']['success']:
        gpt_eval = soa_eval['gpt4o_evaluation']['evaluation']
        print(f"\nGPT-4o Overall Score: {gpt_eval.get('overall_score', 'N/A')}/10")
        print(f"Assessment: {gpt_eval.get('overall_assessment', 'N/A')}")

    print(f"\n>>> FINAL GRADE: {soa_eval['overall_grade']['grade']} ({soa_eval['overall_grade']['final_score']}/10)")
    print(f">>> STATUS: {soa_eval['overall_grade']['status']}")

def test_batch_evaluation():
    """Test evaluating all reports at once"""
    print("\n\n" + "="*80)
    print("TESTING BATCH EVALUATION (ALL THREE AGENTS)")
    print("="*80)

    all_evals = evaluate_all_reports(
        amendment_output=SAMPLE_AMENDMENT_OUTPUT,
        design_output=SAMPLE_DESIGN_PATTERN_OUTPUT,
        soa_output=SAMPLE_SOA_OUTPUT,
        nct_id="NCT00000001"
    )

    print(f"\nTrial ID: {all_evals['nct_id']}")
    print(f"Reports Evaluated: {all_evals['aggregate']['reports_evaluated']}")
    print(f"\n>>> AGGREGATE SCORE: {all_evals['aggregate']['average_score']}/10")
    print(f">>> AGGREGATE GRADE: {all_evals['aggregate']['grade']}")

    print("\nBreakdown by Agent:")
    for agent_name, eval_data in all_evals['evaluations'].items():
        grade = eval_data['overall_grade']
        print(f"  - {eval_data['report_type']}: {grade['grade']} ({grade['final_score']}/10) - {grade['status']}")

    # Save to file for inspection
    with open('/Users/j_kanishkha/ClinicalChat/evaluation_results.json', 'w') as f:
        json.dump(all_evals, f, indent=2)
    print("\n>>> Full results saved to: evaluation_results.json")

def main():
    """Run all tests"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  CLINICAL TRIAL AI AGENT EVALUATION SYSTEM".center(78) + "█")
    print("█" + "  Adapted from CTBench Methodology".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    test_individual_evaluations()
    test_batch_evaluation()

    print("\n\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review evaluation_results.json for detailed scores")
    print("2. Replace sample outputs with your actual agent outputs")
    print("3. Integrate evaluation endpoints into app_simple.py (see integration example)")
    print("4. Track scores over time to monitor improvements")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

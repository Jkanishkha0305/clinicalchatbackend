"""
Flask Integration for Evaluation Module
Add these endpoints to app_simple.py to enable evaluation from your web interface
"""

from flask import jsonify, request
from evaluation_module import (
    evaluate_amendment_report,
    evaluate_design_patterns_report,
    evaluate_soa_report,
    evaluate_all_reports
)
import json

# ============================================================================
# COPY THESE ENDPOINTS TO app_simple.py
# ============================================================================

"""
# Add this to your app_simple.py imports:
from evaluation_module import (
    evaluate_amendment_report,
    evaluate_design_patterns_report,
    evaluate_soa_report,
    evaluate_all_reports
)

# Then add these endpoints:
"""

# ENDPOINT 1: Evaluate Amendment Risk Report
@app.route('/api/evaluate/amendment', methods=['POST'])
def evaluate_amendment():
    """Evaluate amendment risk analysis output"""
    try:
        data = request.get_json()
        agent_output = data.get('output')
        nct_id = data.get('nct_id', 'Unknown')

        if not agent_output:
            return jsonify({'error': 'No output provided'}), 400

        # Run evaluation
        evaluation = evaluate_amendment_report(agent_output, nct_id)

        return jsonify({
            'success': True,
            'evaluation': evaluation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ENDPOINT 2: Evaluate Design Patterns Report
@app.route('/api/evaluate/design-patterns', methods=['POST'])
def evaluate_design():
    """Evaluate design pattern discovery output"""
    try:
        data = request.get_json()
        agent_output = data.get('output')
        nct_id = data.get('nct_id', 'Unknown')

        if not agent_output:
            return jsonify({'error': 'No output provided'}), 400

        evaluation = evaluate_design_patterns_report(agent_output, nct_id)

        return jsonify({
            'success': True,
            'evaluation': evaluation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ENDPOINT 3: Evaluate SoA Report
@app.route('/api/evaluate/soa', methods=['POST'])
def evaluate_soa():
    """Evaluate SoA composer output"""
    try:
        data = request.get_json()
        agent_output = data.get('output')
        nct_id = data.get('nct_id', 'Unknown')

        if not agent_output:
            return jsonify({'error': 'No output provided'}), 400

        evaluation = evaluate_soa_report(agent_output, nct_id)

        return jsonify({
            'success': True,
            'evaluation': evaluation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ENDPOINT 4: Evaluate All Reports (Batch)
@app.route('/api/evaluate/all', methods=['POST'])
def evaluate_all():
    """Evaluate all agent outputs at once"""
    try:
        data = request.get_json()
        nct_id = data.get('nct_id', 'Unknown')

        amendment_output = data.get('amendment_output')
        design_output = data.get('design_output')
        soa_output = data.get('soa_output')

        if not any([amendment_output, design_output, soa_output]):
            return jsonify({'error': 'No outputs provided'}), 400

        evaluation = evaluate_all_reports(
            amendment_output=amendment_output,
            design_output=design_output,
            soa_output=soa_output,
            nct_id=nct_id
        )

        return jsonify({
            'success': True,
            'evaluation': evaluation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ENDPOINT 5: Quick Metrics Only (No GPT-4o - instant results)
@app.route('/api/evaluate/quick', methods=['POST'])
def evaluate_quick():
    """Quick evaluation with automated metrics only (no GPT-4o call)"""
    try:
        from evaluation_module import (
            count_quantitative_density,
            check_section_completeness,
            extract_key_metrics_section
        )

        data = request.get_json()
        agent_output = data.get('output')
        report_type = data.get('report_type', 'amendment')  # amendment, design, soa

        if not agent_output:
            return jsonify({'error': 'No output provided'}), 400

        # Define required sections based on report type
        sections_map = {
            'amendment': ["Risk Assessment", "Risk Factors", "Recommendations", "Key Metrics"],
            'design': ["Patterns Identified", "Design Archetypes", "Success Indicators", "Recommendations"],
            'soa': ["Timing", "Frequency", "Visit Windows", "Rationale", "Key Metrics"]
        }

        required_sections = sections_map.get(report_type, [])

        # Run automated metrics only
        numeric_density = count_quantitative_density(agent_output)
        section_check = check_section_completeness(agent_output, required_sections)
        key_metrics = extract_key_metrics_section(agent_output)

        automated_score = (
            (numeric_density['numbers_per_100_words'] / 10) * 0.3 +
            (section_check['completeness_score'] / 100) * 10 * 0.3 +
            (min(key_metrics['metrics_count'], 5) / 5) * 10 * 0.4
        )

        return jsonify({
            'success': True,
            'quick_evaluation': {
                'automated_score': round(automated_score, 2),
                'numeric_density': numeric_density,
                'section_completeness': section_check,
                'key_metrics_section': key_metrics,
                'note': 'Quick metrics only - run full evaluation for GPT-4o assessment'
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# USAGE EXAMPLES (for testing with curl or Postman)
# ============================================================================

"""
# Test Amendment Evaluation:
curl -X POST http://localhost:5033/api/evaluate/amendment \\
  -H "Content-Type: application/json" \\
  -d '{
    "nct_id": "NCT00000001",
    "output": "Your amendment risk analysis output here..."
  }'

# Test Design Patterns Evaluation:
curl -X POST http://localhost:5033/api/evaluate/design-patterns \\
  -H "Content-Type: application/json" \\
  -d '{
    "nct_id": "NCT00000001",
    "output": "Your design patterns output here..."
  }'

# Test SoA Evaluation:
curl -X POST http://localhost:5033/api/evaluate/soa \\
  -H "Content-Type: application/json" \\
  -d '{
    "nct_id": "NCT00000001",
    "output": "Your SoA output here..."
  }'

# Test Batch Evaluation:
curl -X POST http://localhost:5033/api/evaluate/all \\
  -H "Content-Type: application/json" \\
  -d '{
    "nct_id": "NCT00000001",
    "amendment_output": "...",
    "design_output": "...",
    "soa_output": "..."
  }'

# Test Quick Evaluation (instant, no GPT-4o):
curl -X POST http://localhost:5033/api/evaluate/quick \\
  -H "Content-Type: application/json" \\
  -d '{
    "report_type": "amendment",
    "output": "Your output here..."
  }'
"""

# ============================================================================
# FRONTEND INTEGRATION (JavaScript/React Example)
# ============================================================================

"""
// Example: Call evaluation after generating a report

async function evaluateReport(reportType, output, nctId) {
  const response = await fetch(`/api/evaluate/${reportType}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ output, nct_id: nctId })
  });

  const result = await response.json();

  if (result.success) {
    const evaluation = result.evaluation;
    console.log('Overall Grade:', evaluation.overall_grade.grade);
    console.log('Final Score:', evaluation.overall_grade.final_score);
    console.log('Status:', evaluation.overall_grade.status);

    // Display metrics
    console.log('Numbers found:', evaluation.automated_metrics.numeric_density.total_numbers);
    console.log('Section completeness:', evaluation.automated_metrics.section_completeness.completeness_score);

    // Display GPT-4o evaluation
    if (evaluation.gpt4o_evaluation.success) {
      console.log('GPT-4o Assessment:', evaluation.gpt4o_evaluation.evaluation.overall_assessment);
    }
  }
}

// Usage:
evaluateReport('amendment', amendmentOutput, 'NCT00000001');
evaluateReport('design-patterns', designOutput, 'NCT00000001');
evaluateReport('soa', soaOutput, 'NCT00000001');

// Or evaluate all at once:
async function evaluateAllReports(nctId, outputs) {
  const response = await fetch('/api/evaluate/all', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      nct_id: nctId,
      amendment_output: outputs.amendment,
      design_output: outputs.design,
      soa_output: outputs.soa
    })
  });

  const result = await response.json();

  if (result.success) {
    console.log('Aggregate Score:', result.evaluation.aggregate.average_score);
    console.log('Aggregate Grade:', result.evaluation.aggregate.grade);

    // Show breakdown by agent
    for (const [agentName, evalData] of Object.entries(result.evaluation.evaluations)) {
      console.log(`${evalData.report_type}: ${evalData.overall_grade.grade}`);
    }
  }
}
"""

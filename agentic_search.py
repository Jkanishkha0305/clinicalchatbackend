"""
Agentic Search Enhancement System
Multiple agents collaborate to refine and improve search queries
"""

from openai import OpenAI
import json
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI()

# =============================================================================
# SEARCH AGENT DEFINITIONS
# =============================================================================

def medical_terminology_agent(query: str) -> Dict:
    """Agent that expands medical terminology and synonyms"""

    system_prompt = """You are a medical terminology expert. Your job is to:
1. Identify medical terms in the query
2. Provide clinical synonyms and related terms
3. Suggest both lay terms and technical terms
4. Include common abbreviations

Output JSON format:
{
    "original_terms": ["term1", "term2"],
    "synonyms": ["synonym1", "synonym2"],
    "related_terms": ["related1", "related2"],
    "abbreviations": ["abbr1", "abbr2"],
    "suggested_expansions": ["expansion1", "expansion2"]
}"""

    user_prompt = f"""Analyze this clinical trial search query and provide terminology expansions:

Query: "{query}"

Return ONLY valid JSON, no other text."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        result['agent'] = 'Medical Terminology Agent'
        return result

    except Exception as e:
        return {
            'agent': 'Medical Terminology Agent',
            'error': str(e),
            'original_terms': [],
            'synonyms': [],
            'related_terms': []
        }


def search_strategy_agent(query: str, terminology_expansion: Dict) -> Dict:
    """Agent that suggests optimal search strategies"""

    system_prompt = """You are a clinical trials search expert. Based on the query and terminology expansions, suggest:
1. Boolean search strategies
2. Priority terms (most important to least important)
3. Filters to apply (phase, status, intervention type)
4. Search refinement suggestions

Output JSON format:
{
    "boolean_strategy": "description of boolean approach",
    "priority_terms": ["highest priority", "medium", "low"],
    "suggested_filters": {
        "status": ["RECRUITING", "ACTIVE_NOT_RECRUITING"],
        "phase": ["PHASE3"]
    },
    "refinement_tips": ["tip1", "tip2"]
}"""

    user_prompt = f"""Given this query and terminology expansions, suggest optimal search strategy:

Original Query: "{query}"

Terminology Expansions:
{json.dumps(terminology_expansion, indent=2)}

Return ONLY valid JSON, no other text."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        result['agent'] = 'Search Strategy Agent'
        return result

    except Exception as e:
        return {
            'agent': 'Search Strategy Agent',
            'error': str(e),
            'boolean_strategy': '',
            'priority_terms': []
        }


def relevance_scoring_agent(query: str, results: List[Dict]) -> List[Dict]:
    """Agent that scores and reranks search results based on relevance"""

    # Take top 10 results to rerank
    top_results = results[:10] if len(results) > 10 else results

    # Create summaries of trials for the agent
    trial_summaries = []
    for i, trial in enumerate(top_results):
        summary = {
            'index': i,
            'nct_id': trial.get('nct_id', 'N/A'),
            'title': trial.get('title', 'N/A')[:200],
            'conditions': trial.get('conditions', []),
            'interventions': trial.get('interventions', []),
            'status': trial.get('status', 'UNKNOWN')
        }
        trial_summaries.append(summary)

    system_prompt = f"""You are a clinical trials relevance expert. Given a search query and trial results, score each trial's relevance on a scale of 1-10 and provide confidence %.

Consider:
- Direct condition match (highest weight)
- Intervention relevance
- Study phase and status
- Title clarity and specificity

Output JSON format:
{{
    "scored_results": [
        {{"index": 0, "relevance_score": 9, "confidence": 0.9, "reason": "Direct match for condition"}},
        {{"index": 1, "relevance_score": 7, "confidence": 0.7, "reason": "Related intervention"}}
    ]
}}"""

    user_prompt = f"""Score these trials for relevance to the query:

Query: "{query}"

Trials:
{json.dumps(trial_summaries, indent=2)}

Return ONLY valid JSON with scored_results, no other text."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"}
        )

        scoring_result = json.loads(response.choices[0].message.content)

        # Apply scores to original results
        scored_results = top_results.copy()
        for score_data in scoring_result.get('scored_results', []):
            idx = score_data['index']
            if idx < len(scored_results):
                scored_results[idx]['_relevance_score'] = score_data.get('relevance_score', 5)
                scored_results[idx]['_relevance_reason'] = score_data.get('reason', '')

        # Sort by relevance score
        scored_results.sort(key=lambda x: x.get('_relevance_score', 5), reverse=True)

        return scored_results

    except Exception as e:
        print(f"Relevance scoring error: {str(e)}")
        # Return original results if scoring fails
        return top_results


# =============================================================================
# MAIN AGENTIC SEARCH FUNCTION
# =============================================================================

def agentic_search_enhancement(query: str, initial_results: List[Dict] = None) -> Dict:
    """
    Multi-agent system to enhance search queries and results

    Returns enriched query suggestions and optionally reranked results
    """

    print(f"\n{'='*70}")
    print("🔍 AGENTIC SEARCH ENHANCEMENT SYSTEM")
    print(f"{'='*70}")
    print(f"Original Query: '{query}'")
    print(f"\nActivating search enhancement agents...\n")

    # Agent 1: Terminology Expansion
    print("[1/2] 📚 Medical Terminology Agent analyzing...")
    terminology_result = medical_terminology_agent(query)
    print(f"      ✓ Found {len(terminology_result.get('synonyms', []))} synonyms and related terms")

    # Agent 2: Search Strategy
    print("[2/2] 🎯 Search Strategy Agent optimizing...")
    strategy_result = search_strategy_agent(query, terminology_result)
    print(f"      ✓ Strategy recommendations generated")

    # Optional: Agent 3: Relevance Scoring (if results provided)
    reranked_results = None
    if initial_results and len(initial_results) > 0:
        print(f"\n[3/3] ⭐ Relevance Scoring Agent reranking {len(initial_results)} results...")
        reranked_results = relevance_scoring_agent(query, initial_results)
        print(f"      ✓ Results reranked by relevance")

    print(f"\n{'='*70}")
    print("✨ Search enhancement complete!")
    print(f"{'='*70}\n")

    return {
        "original_query": query,
        "terminology_expansion": terminology_result,
        "search_strategy": strategy_result,
        "reranked_results": reranked_results,
        "enhanced_search_terms": (
            terminology_result.get('synonyms', []) +
            terminology_result.get('related_terms', [])
        )
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Test the agentic search
    test_query = "diabetes treatment"

    result = agentic_search_enhancement(test_query)

    print("\n" + "="*70)
    print("📊 AGENTIC SEARCH RESULTS")
    print("="*70)

    print("\n### TERMINOLOGY EXPANSION ###")
    print(json.dumps(result['terminology_expansion'], indent=2))

    print("\n### SEARCH STRATEGY ###")
    print(json.dumps(result['search_strategy'], indent=2))

    print("\n### ENHANCED SEARCH TERMS ###")
    print(result['enhanced_search_terms'])

    print("\n✅ Test complete!")

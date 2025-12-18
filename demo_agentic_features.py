"""
Quick Demo Script for Agentic Features
Run this to test all new agentic AI capabilities
"""

import os
import requests
import json

from db_utils import get_mongo_client

BASE_URL = "http://localhost:5033"
DB_NAME = os.getenv("MONGO_DB_NAME", "clinical_trials")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "studies")


def print_section(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")

def demo_agentic_search():
    """Demo: Agentic Search Enhancement"""
    print_section("🔍 DEMO 1: AGENTIC SEARCH ENHANCEMENT")

    query = "diabetes treatment"
    print(f"Original Query: '{query}'\n")

    response = requests.post(f"{BASE_URL}/api/agentic-search", json={
        "query": query
    })

    if response.status_code == 200:
        result = response.json()
        print("✅ Agentic Search Enhancement Complete!\n")

        print("📚 Terminology Expansion:")
        terms = result['terminology_expansion']
        print(f"  Synonyms: {', '.join(terms.get('synonyms', [])[:5])}")
        print(f"  Related Terms: {', '.join(terms.get('related_terms', [])[:5])}")
        print(f"  Abbreviations: {', '.join(terms.get('abbreviations', [])[:3])}")

        print("\n🎯 Search Strategy:")
        strategy = result['search_strategy']
        print(f"  Strategy: {strategy.get('boolean_strategy', 'N/A')[:150]}...")
        print(f"  Priority Terms: {', '.join(strategy.get('priority_terms', [])[:3])}")

        print(f"\n✨ Enhanced Search Terms ({len(result['enhanced_search_terms'])} total):")
        print(f"  {', '.join(result['enhanced_search_terms'][:8])}...")

    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def demo_multi_agent_analysis():
    """Demo: Multi-Agent Protocol Analysis"""
    print_section("🤖 DEMO 2: MULTI-AGENT PROTOCOL ANALYSIS")

    # Get a sample trial from database
    try:
        client = get_mongo_client(serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    except Exception as e:
        print(f"❌ MongoDB connection failed: {str(e)}")
        return
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Find a trial with good data
    trial = collection.find_one({'status': 'RECRUITING'})

    if not trial:
        trial = collection.find_one({})

    if not trial:
        print("❌ No trials found in database")
        return

    nct_id = trial['nct_id']
    print(f"Analyzing Trial: {nct_id}")
    print(f"Title: {trial.get('title', 'N/A')[:80]}...\n")

    print("Deploying 4 specialized agents + coordinator...\n")

    response = requests.post(f"{BASE_URL}/api/multi-agent-analysis", json={
        "nctId": nct_id
    })

    if response.status_code == 200:
        result = response.json()
        print("✅ Multi-Agent Analysis Complete!\n")

        print(f"Agents Deployed: {result['metadata']['num_agents']}")
        print(f"Model Used: {result['metadata']['model_used']}\n")

        print("Agent Perspectives:")
        for i, analysis in enumerate(result['agent_analyses'], 1):
            print(f"\n  [{i}] {analysis['agent']}")
            print(f"      Focus: {', '.join(analysis['focus_areas'][:3])}")
            # Print first 200 chars of analysis
            content = analysis['content'].replace('<p>', '').replace('</p>', '')[:200]
            print(f"      Analysis: {content}...")

        print("\n\n📊 Executive Summary Generated:")
        summary_preview = result['executive_summary'].replace('<p>', '').replace('</p>', '')[:300]
        print(f"  {summary_preview}...")

    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def demo_trial_comparison():
    """Demo: Multi-Agent Trial Comparison"""
    print_section("🔬 DEMO 3: MULTI-AGENT TRIAL COMPARISON")

    # Get 3 trials from the same condition
    try:
        client = get_mongo_client(serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    except Exception as e:
        print(f"❌ MongoDB connection failed: {str(e)}")
        return
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Try to find trials with similar conditions
    trials = list(collection.find({}).limit(3))

    if len(trials) < 2:
        print("❌ Not enough trials in database")
        return

    nct_ids = [t['nct_id'] for t in trials]

    print(f"Comparing {len(nct_ids)} trials:")
    for i, trial in enumerate(trials, 1):
        print(f"  [{i}] {trial['nct_id']} - {trial.get('title', 'N/A')[:60]}...")

    print("\nDeploying comparison agents...\n")

    response = requests.post(f"{BASE_URL}/api/compare-trials", json={
        "nctIds": nct_ids
    })

    if response.status_code == 200:
        result = response.json()
        print("✅ Multi-Agent Comparison Complete!\n")

        print(f"Agents Used: {result['metadata']['agents_used']}")
        print(f"Trials Compared: {result['metadata']['num_trials']}\n")

        print("Comparison Perspectives Generated:")
        for key in result['comparisons'].keys():
            print(f"  ✓ {key.replace('_', ' ').title()} Comparison")

        print("\n📊 Strategic Synthesis Generated:")
        synthesis_preview = result['strategic_synthesis'].replace('<p>', '').replace('</p>', '')[:300]
        print(f"  {synthesis_preview}...")

    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("  🚀 AGENTIC AI FEATURES DEMO")
    print("  Clinical Trials Analysis with Multi-Agent Systems")
    print("="*70)

    print("\n⚠️  Make sure the Flask app is running on http://localhost:5033")
    print("⚠️  Make sure MongoDB is running with clinical trials data\n")

    input("Press Enter to start demos...")

    try:
        # Demo 1: Agentic Search
        demo_agentic_search()

        input("\n\nPress Enter for next demo...")

        # Demo 2: Multi-Agent Analysis
        demo_multi_agent_analysis()

        input("\n\nPress Enter for next demo...")

        # Demo 3: Trial Comparison
        demo_trial_comparison()

        print_section("✨ ALL DEMOS COMPLETE!")
        print("These agentic features demonstrate:")
        print("  • Multi-agent collaboration")
        print("  • Specialized domain expertise")
        print("  • Synthesis and coordination")
        print("  • Advanced AI capabilities for clinical trials\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to Flask app")
        print("Make sure app_simple.py is running on http://localhost:5033")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()

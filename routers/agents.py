import markdown as md_lib

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import dependencies as deps
from models import (
    CompareTrialsRequest,
    AgentSearchRequest,
    AgentNctRequest,
    AgentConditionRequest,
    AddDocumentsRequest,
)

router = APIRouter()


@router.post("/compare-trials")
def compare_trials(body: CompareTrialsRequest):
    from agentic_comparison import multi_agent_comparison

    nct_ids = body.nctIds or []
    if not nct_ids or len(nct_ids) < 2:
        return JSONResponse({"error": "At least 2 NCT IDs required"}, status_code=400)

    trials = []
    for nct_id in nct_ids[:5]:
        trial = deps.collection.find_one({"nct_id": nct_id})
        if trial:
            trials.append({k: v for k, v in trial.items() if k != "_id"})

    if len(trials) < 2:
        return JSONResponse({"error": "Not enough valid trials found"}, status_code=404)

    try:
        print(f"\n🔬 Starting multi-agent comparison of {len(trials)} trials...")
        result = multi_agent_comparison(trials)

        comparisons_html = {
            key: md_lib.markdown(content, extensions=["extra", "nl2br"])
            for key, content in result["comparisons"].items()
        }
        synthesis_html = md_lib.markdown(result["strategic_synthesis"], extensions=["extra", "nl2br", "tables"])

        return {
            "success": True,
            "trials": result["trials"],
            "comparisons": comparisons_html,
            "strategic_synthesis": synthesis_html,
            "metadata": result["metadata"],
        }
    except Exception as e:
        print(f"Error in trial comparison: {e}")
        return JSONResponse({"error": f"Comparison failed: {e}"}, status_code=500)


@router.post("/agentic-search")
def agentic_search(body: AgentSearchRequest):
    from agentic_search import agentic_search_enhancement

    if not body.query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    try:
        result = agentic_search_enhancement(body.query)
        return {
            "success": True,
            "original_query": result["original_query"],
            "terminology_expansion": result["terminology_expansion"],
            "search_strategy": result["search_strategy"],
            "enhanced_search_terms": result["enhanced_search_terms"],
        }
    except Exception as e:
        print(f"Error in agentic search: {e}")
        return JSONResponse({"error": f"Agentic search failed: {e}"}, status_code=500)


@router.post("/multi-agent-analysis")
def multi_agent_protocol_analysis(body: AgentNctRequest):
    from agentic_analysis import multi_agent_analysis

    if not body.nctId:
        return JSONResponse({"error": "NCT ID is required"}, status_code=400)

    trial = deps.collection.find_one({"nct_id": body.nctId})
    if not trial:
        return JSONResponse({"error": "Trial not found"}, status_code=404)

    trial_copy = {k: v for k, v in trial.items() if k != "_id"}

    try:
        print(f"\n🤖 Starting multi-agent analysis for {body.nctId}...")
        result = multi_agent_analysis(trial_copy)

        analyses_html = [
            {
                "agent": a["agent"],
                "focus_areas": a["focus_areas"],
                "content": md_lib.markdown(a["analysis"], extensions=["extra", "nl2br"]),
            }
            for a in result["agent_analyses"]
        ]
        executive_html = md_lib.markdown(result["executive_summary"], extensions=["extra", "nl2br", "tables"])

        return {
            "success": True,
            "trial": result["trial"],
            "agent_analyses": analyses_html,
            "executive_summary": executive_html,
            "metadata": result["metadata"],
        }
    except Exception as e:
        print(f"Error in multi-agent analysis: {e}")
        return JSONResponse({"error": f"Multi-agent analysis failed: {e}"}, status_code=500)


@router.post("/amendment-risk")
def amendment_risk_prediction(body: AgentNctRequest):
    from agentic_amendment import amendment_risk_analysis

    if not body.nctId:
        return JSONResponse({"error": "NCT ID is required"}, status_code=400)

    trial = deps.collection.find_one({"nct_id": body.nctId})
    if not trial:
        return JSONResponse({"error": "Trial not found"}, status_code=404)

    try:
        print(f"\n⚠️ Starting amendment risk analysis for {body.nctId}...")
        protocol = trial.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        design = protocol.get("designModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        outcomes = protocol.get("outcomesModule", {})

        trial_data = {
            "nct_id": trial.get("nct_id", "N/A"),
            "title": identification.get("briefTitle", "N/A"),
            "phase": ", ".join(design.get("phases", ["N/A"])),
            "status": trial.get("status", "N/A"),
            "enrollment": design.get("enrollmentInfo", {}).get("count", "N/A"),
            "eligibility": eligibility.get("eligibilityCriteria", "Not specified"),
            "outcomes": (
                f"Primary: {outcomes.get('primaryOutcomes', [{}])[0].get('measure', 'N/A')}\n"
                f"Secondary: {', '.join([o.get('measure', 'N/A') for o in outcomes.get('secondaryOutcomes', [])[:3]])}"
            ),
            "design": (
                f"{design.get('studyType', 'N/A')} | "
                f"{design.get('designInfo', {}).get('allocation', 'N/A')} | "
                f"{design.get('designInfo', {}).get('maskingInfo', {}).get('masking', 'N/A')}"
            ),
        }

        result = amendment_risk_analysis(trial_data)
        if not result["success"]:
            return JSONResponse({"error": result.get("error", "Analysis failed")}, status_code=500)

        return {
            "success": True,
            "trial": result["trial"],
            "agent_analyses": result["agent_analyses"],
            "risk_assessment": result.get("risk_assessment"),
            "risk_assessment_raw": result.get("risk_assessment_raw"),
            "risk_assessment_html": result.get("risk_assessment_html"),
            "risk_assessment_text": result.get("risk_assessment_text"),
        }
    except Exception as e:
        print(f"Error in amendment risk analysis: {e}")
        return JSONResponse({"error": f"Amendment risk analysis failed: {e}"}, status_code=500)


@router.post("/design-patterns")
def design_pattern_discovery_endpoint(body: AgentConditionRequest):
    from agentic_patterns import design_pattern_discovery

    if not body.condition:
        return JSONResponse({"error": "Condition is required"}, status_code=400)

    try:
        print(f"\n🔍 Starting design pattern discovery for {body.condition}...")
        result = design_pattern_discovery(body.condition, body.phase, body.interventionType)

        if not result["success"]:
            return JSONResponse({"error": result.get("error", "Analysis failed")}, status_code=500)

        return {
            "success": True,
            "query": result["query"],
            "agent_analyses": result["agent_analyses"],
            "strategic_insights": result.get("strategic_insights"),
            "strategic_insights_raw": result.get("strategic_insights_raw"),
            "strategic_insights_html": result.get("strategic_insights_html"),
            "strategic_insights_text": result.get("strategic_insights_text"),
            "trials_by_phase": result["trials_summary"]["trials_by_phase"],
        }
    except Exception as e:
        print(f"Error in design pattern discovery: {e}")
        return JSONResponse({"error": f"Design pattern discovery failed: {e}"}, status_code=500)


@router.post("/soa-composer")
def soa_composer_endpoint(body: AgentConditionRequest):
    from agentic_soa import soa_composer

    if not body.condition:
        return JSONResponse({"error": "Condition is required"}, status_code=400)

    try:
        print(f"\n📋 Starting SoA composition for {body.condition}...")
        result = soa_composer(body.condition, body.phase, body.interventionType)

        if not result["success"]:
            return JSONResponse({"error": result.get("error", "Analysis failed")}, status_code=500)

        return {
            "success": True,
            "query": result["query"],
            "agent_analyses": result["agent_analyses"],
            "complete_soa": result.get("complete_soa"),
            "complete_soa_raw": result.get("complete_soa_raw"),
            "complete_soa_html": result.get("complete_soa_html"),
            "complete_soa_text": result.get("complete_soa_text"),
            "reference_trials": result["reference_trials"],
        }
    except Exception as e:
        print(f"Error in SoA composition: {e}")
        return JSONResponse({"error": f"SoA composition failed: {e}"}, status_code=500)


@router.post("/documents")
def add_documents(body: AddDocumentsRequest):
    if not body.ids or not body.documents:
        return JSONResponse({"error": "ids and documents are required"}, status_code=400)
    if len(body.ids) != len(body.documents):
        return JSONResponse({"error": "ids and documents must have the same length"}, status_code=400)
    if deps.chroma_collection is None:
        return JSONResponse(
            {"error": "ChromaDB not available. Please configure CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE"},
            status_code=503,
        )

    try:
        add_params = {"ids": body.ids, "documents": body.documents}
        if body.metadatas and len(body.metadatas) == len(body.ids):
            add_params["metadatas"] = body.metadatas
        if body.embeddings and len(body.embeddings) == len(body.ids):
            add_params["embeddings"] = body.embeddings

        deps.chroma_collection.add(**add_params)
        return {"message": "Documents added successfully", "ids": body.ids, "count": len(body.ids)}
    except Exception as e:
        print(f"Error adding documents: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

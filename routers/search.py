from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import dependencies as deps
from models import SearchFilters, StatisticsRequest
from services.embeddings import EMBEDDING_MODEL
from services.trials import (
    build_query_from_filters,
    build_ranked_trial_query,
    format_study,
    get_semantic_query_text,
    normalize_intervention_filter,
    order_trials_by_rank,
)

router = APIRouter()


@router.get("/interventions")
def get_interventions():
    try:
        pipeline = [
            {
                "$project": {
                    "intervention_values": {
                        "$ifNull": [
                            "$interventions",
                            "$protocolSection.armsInterventionsModule.interventions.name",
                        ]
                    }
                }
            },
            {"$unwind": "$intervention_values"},
            {"$group": {"_id": "$intervention_values"}},
            {"$sort": {"_id": 1}},
            {"$limit": 500},
        ]
        result = deps.collection.aggregate(pipeline)
        return {"interventions": [doc["_id"] for doc in result if doc["_id"]]}
    except Exception:
        return {"interventions": []}


@router.post("/search")
def search_studies(body: SearchFilters):
    filters = body.dict()
    session_id = filters.pop("sessionId", None)
    filters["intervention"] = normalize_intervention_filter(filters.get("intervention"))

    use_semantic = filters.get("useSemanticSearch", False)

    if use_semantic and (deps.chroma_collection is not None or deps.qdrant_client is not None):
        if deps.qdrant_client is not None:
            return _qdrant_search(filters)
        return _chroma_search(filters)

    # Keyword search
    query = build_query_from_filters(filters)
    page = filters.get("page", 1) or 1
    per_page = min(filters.get("per_page", 20) or 20, 100)
    skip = (page - 1) * per_page

    total = deps.collection.count_documents(query)
    results = list(deps.collection.find(query).skip(skip).limit(per_page))
    simplified = [format_study(result) for result in results]

    session_info = None
    if session_id:
        deps.chat_sessions_collection.update_one(
            {"_id": session_id},
            {"$set": {"last_filters": filters, "updated_at": datetime.now().isoformat()}},
        )
        session = deps.chat_sessions_collection.find_one({"_id": session_id})
        if session:
            session_info = {
                "id": session["_id"],
                "title": session.get("title", "Search Session"),
                "description": session.get("description", ""),
            }

    resp = {
        "success": True,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "results": simplified,
        "searchType": "keyword",
    }
    if session_info:
        resp["sessionInfo"] = session_info
    return resp


@router.post("/statistics")
def get_statistics(body: StatisticsRequest):
    try:
        filters = body.dict(exclude_none=True)
        query = build_query_from_filters(filters)
        total = deps.collection.count_documents(query)

        status_dist = list(deps.collection.aggregate([
            {"$match": query},
            {
                "$project": {
                    "normalized_status": {
                        "$ifNull": ["$status", "$protocolSection.statusModule.overallStatus"]
                    }
                }
            },
            {"$group": {"_id": "$normalized_status", "count": {"$sum": 1}}},
        ]))

        phase_dist = list(deps.collection.aggregate([
            {"$match": query},
            {
                "$project": {
                    "phase": {
                        "$ifNull": ["$phase", "$protocolSection.designModule.phases"]
                    }
                }
            },
            {"$project": {"phase": {"$cond": {"if": {"$isArray": "$phase"}, "then": "$phase", "else": ["$phase"]}}}},
            {"$unwind": "$phase"},
            {"$group": {"_id": "$phase", "count": {"$sum": 1}}},
        ]))

        return {
            "success": True,
            "total": total,
            "status_distribution": status_dist,
            "phase_distribution": phase_dist,
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.get("/study/{nct_id}")
def get_study(nct_id: str):
    study = deps.collection.find_one(
        {
            "$or": [
                {"nct_id": nct_id.upper()},
                {"protocolSection.identificationModule.nctId": nct_id.upper()},
            ]
        }
    )
    if not study:
        return JSONResponse({"success": False, "error": "Study not found"}, status_code=404)
    study["_id"] = str(study["_id"])
    return {"success": True, "data": study}


# ── Internal semantic search helpers ──────────────────────────────────────────

def _chroma_search(filters: dict):
    semantic_query = get_semantic_query_text(filters)
    if not semantic_query:
        return JSONResponse({"error": "A condition or free-text query is required for semantic search"}, status_code=400)
    try:
        resp = deps.openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[semantic_query])
        embedding = resp.data[0].embedding
        page = filters.get("page", 1) or 1
        per_page = min(filters.get("per_page", 20) or 20, 100)
        n = min(page * per_page + 100, 1000)

        chroma_res = deps.chroma_collection.query(query_embeddings=[embedding], n_results=n)
        nct_ids = chroma_res["ids"][0] if chroma_res["ids"] else []
        if not nct_ids:
            return {
                "success": True,
                "total": 0,
                "page": page,
                "per_page": per_page,
                "total_pages": 0,
                "results": [],
                "searchType": "semantic-chromadb",
            }

        supplemental_filters = dict(filters)
        supplemental_filters.pop("condition", None)
        supplemental_filters.pop("query", None)
        supplemental_filters.pop("useSemanticSearch", None)
        supplemental_filters.pop("page", None)
        supplemental_filters.pop("per_page", None)
        base_query = build_query_from_filters(supplemental_filters)
        ranked_query = build_ranked_trial_query(base_query, nct_ids)
        matching_trials = list(deps.collection.find(ranked_query))
        ordered_trials = order_trials_by_rank(matching_trials, nct_ids)
        total = len(ordered_trials)
        page_results = ordered_trials[(page - 1) * per_page: page * per_page]
        return {
            "success": True,
            "total": total, "page": page, "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "results": [format_study(result) for result in page_results], "searchType": "semantic-chromadb",
        }
    except Exception as e:
        return JSONResponse({"error": f"Semantic search failed: {e}"}, status_code=500)


def _qdrant_search(filters: dict):
    semantic_query = get_semantic_query_text(filters)
    if not semantic_query:
        return JSONResponse({"error": "A condition or free-text query is required for semantic search"}, status_code=400)
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        resp = deps.openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[semantic_query])
        embedding = resp.data[0].embedding
        page = filters.get("page", 1) or 1
        per_page = min(filters.get("per_page", 20) or 20, 100)

        must = []
        if filters.get("status"):
            must.append(FieldCondition(key="status", match=MatchValue(any=filters["status"])))
        if filters.get("intervention"):
            for iv in filters["intervention"]:
                must.append(FieldCondition(key="interventions", match=MatchValue(value=iv)))

        search_filter = Filter(must=must) if must else None
        qr = deps.qdrant_client.query_points(
            collection_name=deps.QDRANT_COLLECTION_NAME,
            query=embedding,
            query_filter=search_filter,
            limit=min(page * per_page + 100, 1000),
            with_payload=True,
        )
        search_results = qr.points
        if not search_results:
            return {
                "success": True,
                "total": 0,
                "page": page,
                "per_page": per_page,
                "total_pages": 0,
                "results": [],
                "searchType": "semantic-qdrant",
            }

        nct_ids = [r.payload.get("nct_id") for r in search_results if r.payload.get("nct_id")]
        supplemental_filters = dict(filters)
        supplemental_filters.pop("condition", None)
        supplemental_filters.pop("query", None)
        supplemental_filters.pop("useSemanticSearch", None)
        supplemental_filters.pop("page", None)
        supplemental_filters.pop("per_page", None)
        base_query = build_query_from_filters(supplemental_filters)
        ranked_query = build_ranked_trial_query(base_query, nct_ids)
        matching_trials = list(deps.collection.find(ranked_query))
        ordered_trials = order_trials_by_rank(matching_trials, nct_ids)
        total = len(ordered_trials)
        page_results = ordered_trials[(page - 1) * per_page: page * per_page]
        return {
            "success": True,
            "total": total, "page": page, "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "results": [format_study(result) for result in page_results], "searchType": "semantic-qdrant",
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": f"Qdrant search failed: {e}"}, status_code=500)

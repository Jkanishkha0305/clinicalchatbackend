import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import dependencies as deps
from models import SearchFilters, StatisticsRequest

router = APIRouter()


def _format_study(result: dict) -> dict:
    result["_id"] = str(result["_id"])
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": result.get("nct_id", "N/A"),
                "briefTitle": result.get("title", "No title"),
            },
            "statusModule": {"overallStatus": result.get("status", "UNKNOWN")},
            "designModule": {"studyType": "INTERVENTIONAL", "phases": []},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "N/A"}},
        },
        "hasResults": False,
        "_original": result,
    }


@router.get("/interventions")
def get_interventions():
    try:
        pipeline = [
            {"$unwind": "$interventions"},
            {"$group": {"_id": "$interventions"}},
            {"$sort": {"_id": 1}},
            {"$limit": 500},
        ]
        result = deps.collection.aggregate(pipeline)
        return {"interventions": [doc["_id"] for doc in result if doc["_id"]]}
    except Exception as e:
        return {"interventions": []}


@router.post("/search")
def search_studies(body: SearchFilters):
    filters = body.dict()
    session_id = filters.pop("sessionId", None)

    # Normalise intervention
    intervention = filters.get("intervention")
    if isinstance(intervention, str):
        filters["intervention"] = [i.strip() for i in intervention.split(",") if i.strip()]
    elif not isinstance(intervention, list):
        filters["intervention"] = []

    use_semantic = filters.get("useSemanticSearch", False)

    if use_semantic and (deps.chroma_collection is not None or deps.qdrant_client is not None):
        if deps.qdrant_client is not None:
            return _qdrant_search(filters)
        return _chroma_search(filters)

    # Keyword search
    query = deps.build_query_from_filters(filters)
    page = filters.get("page", 1) or 1
    per_page = min(filters.get("per_page", 20) or 20, 100)
    skip = (page - 1) * per_page

    total = deps.collection.count_documents(query)
    results = list(deps.collection.find(query).skip(skip).limit(per_page))
    simplified = [_format_study(r) for r in results]

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
        query = deps.build_query_from_filters(filters)
        total = deps.collection.count_documents(query)

        status_dist = list(deps.collection.aggregate([
            {"$match": query},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]))

        phase_dist = list(deps.collection.aggregate([
            {"$match": query},
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
    study = deps.collection.find_one({"nct_id": nct_id.upper()})
    if not study:
        study = deps.collection.find_one({"nct_id": nct_id})
    if not study:
        return JSONResponse({"success": False, "error": "Study not found"}, status_code=404)
    study["_id"] = str(study["_id"])
    return {"success": True, "data": study}


# ── Internal semantic search helpers ──────────────────────────────────────────

def _chroma_search(filters: dict):
    condition = filters.get("condition", "")
    if not condition:
        return JSONResponse({"error": "Condition required for semantic search"}, status_code=400)
    try:
        resp = deps.openai_client.embeddings.create(model="text-embedding-ada-002", input=[condition])
        embedding = resp.data[0].embedding
        page = filters.get("page", 1) or 1
        per_page = min(filters.get("per_page", 20) or 20, 100)
        n = min(page * per_page + 100, 1000)

        chroma_res = deps.chroma_collection.query(query_embeddings=[embedding], n_results=n)
        nct_ids = chroma_res["ids"][0] if chroma_res["ids"] else []
        if not nct_ids:
            return {"total": 0, "page": page, "per_page": per_page, "total_pages": 0, "results": [], "searchType": "semantic-chromadb"}

        q = {"nct_id": {"$in": nct_ids}}
        if filters.get("status"):
            q["status"] = {"$in": filters["status"]}
        total = deps.collection.count_documents(q)
        results = list(deps.collection.find(q).skip((page - 1) * per_page).limit(per_page))
        return {
            "total": total, "page": page, "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "results": [_format_study(r) for r in results], "searchType": "semantic-chromadb",
        }
    except Exception as e:
        return JSONResponse({"error": f"Semantic search failed: {e}"}, status_code=500)


def _qdrant_search(filters: dict):
    condition = filters.get("condition", "")
    if not condition:
        return JSONResponse({"error": "Condition required for semantic search"}, status_code=400)
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        resp = deps.openai_client.embeddings.create(model="text-embedding-ada-002", input=[condition])
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
        search_results = deps.qdrant_client.search(
            collection_name=deps.QDRANT_COLLECTION_NAME,
            query_vector=embedding,
            query_filter=search_filter,
            limit=min(page * per_page + 100, 1000),
            with_payload=True,
        )
        if not search_results:
            return {"total": 0, "page": page, "per_page": per_page, "total_pages": 0, "results": [], "searchType": "semantic-qdrant"}

        nct_ids = [r.payload.get("nct_id") for r in search_results if r.payload.get("nct_id")]
        q = {"nct_id": {"$in": nct_ids}}
        total = deps.collection.count_documents(q)
        results = list(deps.collection.find(q).skip((page - 1) * per_page).limit(per_page))
        return {
            "total": total, "page": page, "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "results": [_format_study(r) for r in results], "searchType": "semantic-qdrant",
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": f"Qdrant search failed: {e}"}, status_code=500)

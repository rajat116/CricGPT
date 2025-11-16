from fastapi import APIRouter
from app.schemas.query import QueryRequest, QueryResponse
from cricket_tools.runner import run_query

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Main CricGPT inference endpoint."""
    resp = run_query(
        message=req.query,
        backend=req.backend,
        plot=req.plot,
        fallback=req.fallback,
        session_id=req.session_id,
    )
    return resp


@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "CricGPT FastAPI server running"}
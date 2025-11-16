from pydantic import BaseModel
from typing import Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    backend: str = "llm_reasoning"
    plot: bool = False
    fallback: bool = True
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    reply: str
    act: Optional[str]
    result: Dict[str, Any]
    plot_path: Optional[str]
    meta: Dict[str, Any]
    trace: Any
    session_id: str
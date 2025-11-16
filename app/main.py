from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.query import router as query_router

app = FastAPI(
    title="CricGPT API",
    description="FastAPI backend for CricGPT natural-language cricket analytics",
    version="1.0.0"
)

# Optional CORS for Streamlit UI (Phase-4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(query_router)

@app.get("/")
async def root():
    return {"message": "Welcome to CricGPT API"}
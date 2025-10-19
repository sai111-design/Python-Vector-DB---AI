print("main.py loaded")
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

# Ensure 'database.py' exists in the same directory or update the import path accordingly
from database import get_database
from endpoints import router as api_router
from models import HealthResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Vector DB CRUD API...")
    db = get_database()
    print(f"ChromaDB initialized with collection count: {len(db.client.list_collections())}")
    yield
    # Shutdown
    print("Shutting down Vector DB CRUD API...")

app = FastAPI(
    title="Vector Database CRUD API",
    description="FastAPI application for CRUD operations with ChromaDB",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="Vector DB API is running")

@app.get("/")
async def root():
    return {
        "message": "Vector Database CRUD API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
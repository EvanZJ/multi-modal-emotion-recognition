from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import ping, infer

app = FastAPI(
    title="Multi-Modal Emotion Recognition API",
    description="Backend API for the project; minimal FastAPI skeleton",
    version="0.1.0",
)

app.include_router(ping.router)
app.include_router(infer.router)

# Allow frontend origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

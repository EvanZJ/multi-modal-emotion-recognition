from fastapi import FastAPI
from .routers import ping, infer

app = FastAPI(
    title="Multi-Modal Emotion Recognition API",
    description="Backend API for the project; minimal FastAPI skeleton",
    version="0.1.0",
)

app.include_router(ping.router)
app.include_router(infer.router)

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

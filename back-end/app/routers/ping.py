from fastapi import APIRouter

router = APIRouter(prefix="", tags=["ping"])

@router.get("/ping")
async def ping():
    return {"message": "pong"}

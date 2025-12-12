from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ..libs.inference import infer_upload_file
import traceback

router = APIRouter(prefix="/infer", tags=["infer"])

@router.post("/", status_code=200)
async def infer(file: UploadFile = File(...), sample_frames: int = 32):
    # Basic validation of file type is left to ffmpeg/processing stage
    try:
        result = infer_upload_file(file, sample_frames=sample_frames)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)

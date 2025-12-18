from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ..libs.inference import infer_upload_file
import traceback

router = APIRouter(prefix="/infer", tags=["infer"])

@router.post("/", status_code=200)
async def infer(file: UploadFile = File(...), subchunk_size: int = 32, window_size: int = 5, explain: bool = False):
    # Basic validation of file type is left to ffmpeg/processing stage
    try:
        print(f"Received /infer request for file: {file.filename}")
        results = infer_upload_file(file, chunk_size=subchunk_size, window_size=window_size, explain=explain)
        print(f"/infer finished; bounding_box={len(results.get('bounding_box',[]))}, inference={len(results.get('inference',[]))}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=results)

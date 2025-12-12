# Back-end (FastAPI)

Minimal FastAPI skeleton for the multi-modal-emotion-recognition project.

To run locally:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

To build and run with Docker:

```bash
# Build the container image from within the `back-end/` directory
docker build -t mm-emotion-backend:latest .

# Run the image
docker run --rm -p 8000:8000 mm-emotion-backend:latest

# Alternatively, use docker-compose (recommended for local development):
docker-compose up --build
```

If you prefer to use a shell inside the container (helpful for debugging):

```bash
docker run --rm -it --entrypoint /bin/bash mm-emotion-backend:latest
```

Endpoints:
- `GET /` — nothing (not defined)
- `GET /ping` — returns `{"message": "pong"}`
- `GET /health` — returns status
 - `POST /infer` — accepts `multipart/form-data` with a `file` field (video). Returns predicted emotion and probabilities. Optional `sample_frames` query param controls how many frames are sampled from the video for faster inference (default 32).
 - `POST /infer/predict` — accepts multipart/form-data with key `file` (video) and returns predicted emotion scores

Troubleshooting and tips:
- If you run into package installation errors when building Docker, try clearing Docker caches or updating base image tags.
- For production, consider switching to gunicorn with Uvicorn workers and tuning worker count based on CPU.

Example curl call (replace sample.mp4 with your file):

```bash
curl -F "file=@sample.mp4" http://localhost:8000/infer/predict
```

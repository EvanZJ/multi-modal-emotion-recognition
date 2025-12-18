# Multi-Modal Emotion Recognition

A comprehensive emotion recognition system that combines video and audio modalities using deep learning. This project includes data preprocessing tools, model training pipelines, a FastAPI backend for inference, and a React frontend for visualization.

## Table of Contents

- [Project Overview](#project-overview)
- [Python Utilities](#python-utilities)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)

## Project Overview

This system performs multi-modal emotion recognition by:
1. **Extracting video features** using Vision Transformer (ViViT) embeddings from video frames
2. **Extracting audio features** using Wav2Vec2-based emotion embeddings from audio tracks
3. **Fusing both modalities** through attention-based cross-modal fusion
4. **Classifying emotions** into 6 categories: Neutral, Happy, Sad, Angry, Fearful, and Disgusted

The model is trained on datasets like RAVDESS and CREMA-D with careful preprocessing and data augmentation strategies.

### Emotion Classes

The system recognizes **6 emotion categories**:

| Class ID | Abbreviation | Full Name | Description |
|----------|-------------|-----------|-------------|
| 0 | NEU | Neutral | Calm, neutral emotional state |
| 1 | HAP | Happy | Joyful, positive emotional state |
| 2 | SAD | Sad | Unhappy, sorrowful emotional state |
| 3 | ANG | Angry | Irritated, frustrated emotional state |
| 4 | FEA | Fearful | Anxious, scared emotional state |
| 5 | DIS | Disgusted | Repulsed, disgusted emotional state |

**Note**: The model outputs probability scores for all 6 classes, allowing for multi-label interpretation and confidence analysis.

#### Dataset Emotion Mappings

**RAVDESS Dataset** (originally 8 emotions):
- **Original emotions**: calm, happy, sad, angry, fearful, surprise, disgust, neutral
- **Mapped to 6 classes**: calm → NEU, happy → HAP, sad → SAD, angry → ANG, fearful → FEA, surprise → removed, disgust → DIS, neutral → NEU

**CREMA-D Dataset** (originally 6 emotions):
- **Original emotions**: happy, sad, anger, fear, disgust, neutral
- **Mapped to 6 classes**: happy → HAP, sad → SAD, anger → ANG, fear → FEA, disgust → DIS, neutral → NEU

Both datasets are standardized to the same 6 emotion classes for unified model training and inference.

## Python Utilities

### Core Python Scripts

#### **`video_extractor.py`**
Extracts video frame embeddings using ViViT (Vision Video Transformer) model.
- **Functionality**: Converts video frames into 768-dimensional embeddings by tokenizing spatial-temporal patches (tubelets)
- **Key Classes**: 
  - `TubeletEmbedder`: Converts video frames into patch embeddings
  - `PreNorm`: Normalization wrapper for attention layers
  - `Attention`: Multi-head self-attention mechanism
- **Output**: `.npy` files with video features of shape `(sequence_length, 768)`

#### **`voice_extractor.py`**
Extracts audio emotion embeddings using pre-trained Wav2Vec2 model.
- **Functionality**: Processes audio files and outputs 1024-dimensional emotion-aware embeddings
- **Model**: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (HuggingFace)
- **Features**:
  - Batched processing with optimal GPU memory management
  - L2-normalization for consistency
  - Float16 precision for efficiency
- **Output**: `.npy` files with audio features of shape `(1024,)`

#### **`train.py` & `train2.py`**
Train the multi-modal emotion classifier.
- **Functionality**: Implements a CrossModalFusion module that combines video and audio features
- **Key Components**:
  - `CrossModalFusion`: Attention-based fusion of video sequences and audio embeddings
  - `FocalLoss`: Handles class imbalance in emotion datasets
  - `MultimodalEmotionModel`: Final classifier combining both modalities
- **Features**:
  - Class weight computation for imbalanced datasets
  - Learning rate scheduling with `ReduceLROnPlateau`
  - Precision, recall, F1-score metrics
  - Interpretability via attention weights (train2.py)
- **Output**: Trained model weights saved as `.pth` files in `training_runs/` or `training_runs_2/`

#### **`test.py`**
Validates extracted features and analyzes dataset statistics.
- **Functionality**: Loads video and audio feature files and prints their shapes
- **Use**: Check feature extraction quality and dataset consistency
- **Output**: Console output showing feature dimensions and counts

#### **`check.py`**
Verifies GPU/CUDA availability and setup.
- **Functionality**: Tests PyTorch and CUDA configuration
- **Use**: Ensure GPU is properly configured before running intensive jobs

#### **CREMA-D Dataset Processing**

**`cremad_extract_bboxes.py`**
- **Functionality**: Detects and extracts face bounding boxes from CREMA-D video files using YOLOv11n-face
- **Input**: Video files (`.flv` format from CREMA-D)
- **Output**: Bounding box coordinates saved in `extracted_bboxes/` directory
- **Requirements**: YOLO model auto-downloaded from HuggingFace (`AdamCodd/YOLOv11n-face-detection`)

**`cremad_video_to_audio_converter.py`**
- **Functionality**: Extracts audio tracks from CREMA-D videos using FFmpeg
- **Input**: CREMA-D video files
- **Output**: MP3 audio files in `extracted_audio/` directory
- **Features**: Configurable bitrate (default 320k), error handling

**`cremad_bbox_converter.py`**
- **Functionality**: Converts extracted bounding boxes into standardized format for model training

#### **RAVDESS Dataset Processing**

**`ravdess_extract_bboxes.py`**
- **Functionality**: Extracts face bounding boxes from RAVDESS video files using YOLOv11n-face
- **Input**: RAVDESS video files (organized in subdirectories)
- **Output**: Bounding box coordinates in `extracted_bboxes/` directory
- **GPU Support**: Automatically uses CUDA if available

**`ravdess_video_to_audio_converter.py`**
- **Functionality**: Extracts audio from RAVDESS videos using FFmpeg
- **Output**: Audio files in standardized format in `extracted_audio/`

**`ravdess_bbox_converter.py`**
- **Functionality**: Converts RAVDESS bounding boxes to training format

#### **`main.py`**
Simple entry point script for the project.

---

## Backend Setup

### Prerequisites

- Python 3.9+
- CUDA 11.8 or 12.x (for GPU inference)
- FFmpeg

### Local Development

1. **Install Dependencies**
   ```bash
   cd back-end
   pip install -r requirements.txt
   ```

2. **Run the FastAPI Server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **API will be available at**: `http://localhost:8000`

### Endpoints

- **`GET /`** — Info (not defined)
- **`GET /ping`** — Health check, returns `{"message": "pong"}`
- **`GET /health`** — Returns `{"status": "ok"}`
- **`POST /infer`** — Upload a video file for emotion inference
  - **Parameters**: 
    - `file` (multipart/form-data): Video file to analyze
    - `sample_frames` (optional query param): Number of frames to sample (default: 32)
  - **Returns**: Predicted emotion and confidence scores
- **`POST /infer/predict`** — Upload video and get emotion predictions
  - **Parameters**: `file` (multipart/form-data)
  - **Returns**: Emotion scores for all 6 classes

### Example API Call

```bash
curl -F "file=@sample.mp4" http://localhost:8000/infer/predict
```

### Docker

Build and run the backend in Docker:

```bash
# Build the container from the back-end directory
docker build -t mm-emotion-backend:latest .

# Run the container
docker run --rm -p 8000:8000 mm-emotion-backend:latest

# Interactive debugging shell
docker run --rm -it --entrypoint /bin/bash mm-emotion-backend:latest
```

### Backend Troubleshooting

- **Package installation errors**: Clear Docker cache with `docker builder prune` or update base image tags
- **GPU access**: Ensure Docker has GPU access by installing nvidia-docker and using appropriate runtime flags
- **Memory issues**: Adjust `sample_frames` query parameter to reduce memory usage
- **Production**: Use gunicorn with Uvicorn workers: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app`

---

## Frontend Setup

### Prerequisites

- Node.js 18+
- npm or yarn

### Local Development

1. **Navigate to frontend directory and install dependencies**
   ```bash
   cd front-end
   npm ci
   ```

2. **Start the development server**
   ```bash
   npm run dev
   ```

3. **App will be available at**: `http://localhost:5173`

### Build for Production

```bash
npm run build
```

Artifacts will be in the `dist/` directory.

### Docker

Build and run the frontend in Docker:

```bash
# Build the image
docker build -t mmer-frontend:latest .

# Run the container
docker run --rm -p 8080:80 mmer-frontend:latest
```

Frontend will be available at: `http://localhost:8080`

### Using shadcn-ui Components

To add additional UI components from shadcn:

```bash
cd front-end
npm install

# Initialize shadcn-ui
npx shadcn-ui@latest init

# Add specific components
npx shadcn-ui@latest add button
npx shadcn-ui@latest add card
```

### Frontend Features

- **Video Upload**: Users can upload videos for emotion analysis
- **Real-time Inference**: Integration with backend API for live predictions
- **Results Display**: Visualize emotion predictions with confidence scores
- **Responsive Design**: Built with Tailwind CSS for mobile and desktop

---

## Docker Deployment

Use Docker Compose to run both services together.

### Production Build

```bash
# Build images and start containers
docker-compose up --build -d

# Stop services
docker-compose down
```

**Access the application**:
- Frontend: `http://localhost:80`
- API: `http://localhost:80/api`
- Direct backend: `http://localhost:8000` (if backend exposed)

### Development with Hot Reload

```bash
# Build and run with live code changes
docker-compose -f docker-compose.dev.yml up --build

# Or in detached mode
docker-compose -f docker-compose.dev.yml up --build -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f
```

### Docker Compose Notes

- **`docker-compose.yml`**: Production setup with nginx reverse proxy on port 80
  - Frontend served on `/`
  - Backend API routed to `/api`
  
- **`docker-compose.dev.yml`**: Development setup with:
  - Source code volumes for live reload
  - `npm run dev` for frontend hot reloading
  - `start.sh --reload` for backend hot reloading
  
- **`docker-compose.override.yml`**: Optional custom configurations for local overrides

---

## Project Structure

```
multi-modal-emotion-recognition/
├── back-end/                      # FastAPI backend service
│   ├── app/
│   │   ├── main.py               # FastAPI app initialization
│   │   ├── inference.py          # Inference engine for emotion prediction
│   │   ├── routers/              # API endpoints
│   │   ├── libs/                 # Shared utilities
│   │   └── api/                  # API-specific modules
│   ├── requirements.txt
│   ├── Dockerfile
│   └── start.sh
│
├── front-end/                     # React + Vite frontend
│   ├── src/
│   │   ├── App.tsx               # Main React component
│   │   ├── components/           # Reusable UI components
│   │   ├── lib/                  # Utility functions
│   │   └── styles/               # CSS/Tailwind styles
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.cjs
│   ├── Dockerfile
│   └── nginx.conf
│
├── Python Utilities (Root)
│   ├── video_extractor.py        # ViViT-based video feature extraction
│   ├── voice_extractor.py        # Wav2Vec2-based audio feature extraction
│   ├── train.py                  # Model training pipeline (v1)
│   ├── train2.py                 # Advanced training with interpretability
│   ├── test.py                   # Feature validation
│   ├── check.py                  # CUDA/GPU verification
│   ├── cremad_extract_bboxes.py  # CREMA-D face detection
│   ├── cremad_video_to_audio_converter.py
│   ├── cremad_bbox_converter.py
│   ├── ravdess_extract_bboxes.py # RAVDESS face detection
│   ├── ravdess_video_to_audio_converter.py
│   ├── ravdess_bbox_converter.py
│   └── main.py
│
├── Datasets
│   ├── audio_features/           # Extracted audio embeddings (.npy)
│   ├── video_features/           # Extracted video embeddings (.npy)
│   ├── extracted_audio/          # Raw extracted audio files
│   ├── extracted_bboxes/         # Face bounding box data
│   └── extracted_faces_videos/   # Cropped face videos
│
├── Models
│   ├── models/                   # Pre-trained model weights
│   ├── training_runs/            # Training output (v1)
│   └── training_runs_2/          # Training output with interpretability
│
└── Configuration
    ├── docker-compose.yml        # Production compose
    ├── docker-compose.dev.yml    # Development compose
    ├── nginx.conf                # Reverse proxy configuration
    ├── requirements.txt          # Python dependencies
    └── pyproject.toml            # Project metadata

```

---

## Workflow Example

### 1. Prepare Dataset

```bash
# Extract audio from RAVDESS videos
python ravdess_video_to_audio_converter.py

# Extract faces (bounding boxes)
python ravdess_extract_bboxes.py
```

### 2. Extract Features

```bash
# Extract video embeddings (ViViT)
python video_extractor.py

# Extract audio embeddings (Wav2Vec2)
python voice_extractor.py
```

### 3. Train Model

```bash
# Train multi-modal emotion classifier
python train2.py --epochs 50 --batch-size 32 --learning-rate 1e-4
```

### 4. Run Backend & Frontend

```bash
# Production deployment
docker-compose up --build

# Or development with hot reload
docker-compose -f docker-compose.dev.yml up --build
```

### 5. Use the Application

- Open `http://localhost` in your browser
- Upload a video file
- View emotion predictions from the model

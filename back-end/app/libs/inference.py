import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import UploadFile

from .video_extractor import ViViTFeatureExtractor
from .voice_extractor import extract_embeddings_batch, get_audio_model_and_extractor, extract_embedding_from_file
from .model import MultimodalEmotionModel

# Lazy globals
_VIDEO_MODEL = None
_FUSION_MODEL = None
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_LABELS = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]

MODEL_PATH = Path(__file__).resolve().parents[3] / "training_runs_2" / "best_model_bs64_ep1000_lr1e-05_20251212_083304.pth"


# Efficient video frame sampler: instead of loading all frames, sample up to 'num_frames' frames uniformly
# and produce a single chunk sized (1, 3, chunk_size, H, W). This reduces heavy conv3D computation.

def _sample_video_frames(video_path: str, num_frames: int = 32, size=(224, 224)) -> Optional[torch.Tensor]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None

    # If total frames <= num_frames, we'll just read them all and pad at the end
    if total <= num_frames:
        indices = list(range(total))
    else:
        # Uniformly spaced sample indices as integers
        indices = np.linspace(0, total - 1, num=num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        frames.append(frame)

    cap.release()

    if not frames:
        return None

    # Ensure we have exactly num_frames by padding the last frame
    if len(frames) < num_frames:
        last = frames[-1]
        for _ in range(num_frames - len(frames)):
            frames.append(last.copy())

    # Convert to tensor (t, c, h, w) then to (1, c, t, h, w)
    video = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames])
    chunk = video.view(1, 3, num_frames, size[1], size[0])
    return chunk


def get_video_model(device=None):
    global _VIDEO_MODEL
    if _VIDEO_MODEL is None:
        device = device or _DEVICE
        # Match the model used when features were extracted/training
        _VIDEO_MODEL = ViViTFeatureExtractor(image_size=(224, 224), patch_size=(16, 16), num_frames=32, tubelet_size=4, dim=768, depth=12, heads=12, pool='cls')
        _VIDEO_MODEL.to(device)
        _VIDEO_MODEL.eval()
    return _VIDEO_MODEL


def get_fusion_model(device=None):
    global _FUSION_MODEL
    if _FUSION_MODEL is None:
        device = device or _DEVICE
        # Instantiate a MultimodalEmotionModel with the same dimensions used for training
        _FUSION_MODEL = MultimodalEmotionModel(
            video_dim=768,
            audio_dim=1024,
            fused_dim=512,
            num_classes=6,
            max_seq_len=57,
            fusion_num_layers=2,
            fusion_num_heads=8,
            fusion_dropout=0.1,
            classifier_hidden_dim=512,
            classifier_dropout=0.1,
        )
        if MODEL_PATH.exists():
            try:
                state = torch.load(str(MODEL_PATH), map_location=device)
                # The saved checkpoint might be a dict with several keys; attempt to find 'state_dict'
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                # Some saved files are state_dict directly
                _FUSION_MODEL.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"Warning: failed to load model from {MODEL_PATH}: {e}")
        _FUSION_MODEL.to(device)
        _FUSION_MODEL.eval()
    return _FUSION_MODEL


# Main pipeline inference function
def infer_video_file(video_file_path: str, sample_frames: int = 32):
    # 1) Sample video frames and compute video features
    device = _DEVICE
    video_model = get_video_model(device)
    fusion_model = get_fusion_model(device)

    sampled_chunk = _sample_video_frames(video_file_path, num_frames=sample_frames, size=(224, 224))
    if sampled_chunk is None:
        raise ValueError("Could not sample frames from the provided video file")

    sampled_chunk = sampled_chunk.to(device)

    with torch.no_grad():
        # Our ViViT model expects input shape (B, C, T, H, W)
        video_feat = video_model(sampled_chunk)  # -> (B, D_v)

    # Convert to sequence shape (B, T_seq=1, D_v) since we only sample one chunk
    video_feats = video_feat.unsqueeze(1).to(device)  # shape (1, 1, D_v)

    # 2) Extract audio embedding from the video file (using ffmpeg to convert to wav 16k)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        # Extract audio to wav 16k single channel
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_file_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            wav_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio_embedding = extract_embedding_from_file(wav_path)  # returns np.ndarray (D_a,)
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    audio_feats = torch.from_numpy(audio_embedding).unsqueeze(0).to(device).float()  # shape (1, D_a)

    # 3) Create mask for video sequence (no padding since T=1), use dtype bool (True = masked/pad)
    mask = torch.zeros((1, video_feats.shape[1]), dtype=torch.bool, device=device)

    # 4) Run through the fusion model
    with torch.no_grad():
        probs, logits, attn = fusion_model(video_feats, audio_feats, mask=mask, return_attn=True)

    probs = probs.detach().cpu().numpy()[0]  # shape (num_classes,)
    logits = logits.detach().cpu().numpy()[0]

    # Return top prediction + full probs
    pred_idx = int(np.argmax(probs))
    result = {
        "predicted_label": _LABELS[pred_idx] if pred_idx < len(_LABELS) else str(pred_idx),
        "predicted_index": pred_idx,
        "probabilities": probs.tolist(),
        "logits": logits.tolist(),
    }
    return result


# Convenience wrapper for FastAPI file upload processing
def infer_upload_file(upload_file: UploadFile, sample_frames: int = 32):
    # Save to a temp file, run inference, and clean up
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, Path(upload_file.filename).name)
        with open(tmp_path, "wb") as f:
            f.write(upload_file.file.read())
        result = infer_video_file(tmp_path, sample_frames=sample_frames)
        return result

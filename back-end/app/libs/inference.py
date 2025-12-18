import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import UploadFile
from captum.attr import IntegratedGradients

from .video_extractor import ViViTFeatureExtractor
from .voice_extractor import extract_embeddings_batch, get_audio_model_and_extractor, extract_embedding_from_file
from .model import MultimodalEmotionModel

# Lazy globals
_VIDEO_MODEL = None
_FUSION_MODEL = None
_YOLO_MODEL = None
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_LABELS = ["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]

MODEL_PATH = Path(__file__).resolve().parents[3] / "training_runs_2" / "best_model_bs64_ep1000_lr1e-05_20251212_083304.pth"


class ModelWrapper(torch.nn.Module):
    """
    Wrapper to make the model compatible with Captum (returns only logits).
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor, mask: Optional[torch.Tensor] = None):
        _, logits, _ = self.model(video_feats, audio_feats, mask=mask, return_attn=False)
        return logits


def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        import os
        os.makedirs("models", exist_ok=True)
        model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt", local_dir="models")
        _YOLO_MODEL = YOLO(model_path)
    return _YOLO_MODEL


# Get face sequences with delay tolerance
def get_face_sequences(video_path, yolo_model, max_delay=10, max_frames_per_sequence=32):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_frames = []  # list of (frame_number, bboxes)

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        results = yolo_model(frame, conf=0.5)
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                bboxes.append((int(x1), int(y1), int(x2), int(y2), confidence))
        if bboxes:
            face_frames.append((frame_number, bboxes))
    cap.release()

    # Group into sequences
    sequences = []
    if not face_frames:
        return sequences, fps

    current_sequence = [face_frames[0]]
    last_face_frame = face_frames[0][0]
    for fn, bbs in face_frames[1:]:
        if fn - last_face_frame <= max_delay:
            current_sequence.append((fn, bbs))
            last_face_frame = fn
        else:
            sequences.append(current_sequence)
            current_sequence = [(fn, bbs)]
            last_face_frame = fn
    if current_sequence:
        sequences.append(current_sequence)

    # Limit to max_frames_per_sequence
    processed_sequences = []
    for seq in sequences:
        if len(seq) > max_frames_per_sequence:
            seq = seq[:max_frames_per_sequence]
        processed_sequences.append(seq)

    return processed_sequences, fps


# Create subchunks from face sequence
def create_subchunks_from_sequence(video_path, sequence, subchunk_size=32, size=(224, 224)):
    import cv2
    cap = cv2.VideoCapture(video_path)
    subchunks = []
    frame_data = []  # list of (frame_number, bboxes)
    for fn, bboxes in sequence:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn - 1)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_data.append((fn, bboxes, frame_rgb))
    cap.release()

    # Group into subchunks of subchunk_size frames
    for i in range(0, len(frame_data), subchunk_size):
        sub_seq = frame_data[i:i+subchunk_size]
        frames = []
        bboxes_sub = []
        for fn, bboxes, frame_rgb in sub_seq:
            if bboxes:
                x1, y1, x2, y2, _ = bboxes[0]
                face = frame_rgb[y1:y2, x1:x2]
                if face.size > 0:
                    face_resized = cv2.resize(face, size)
                    frames.append(face_resized)
                else:
                    frame_resized = cv2.resize(frame_rgb, size)
                    frames.append(frame_resized)
            else:
                frame_resized = cv2.resize(frame_rgb, size)
                frames.append(frame_resized)
            bboxes_sub.append(bboxes)
        if frames:
            # Pad to subchunk_size
            while len(frames) < subchunk_size:
                frames.append(frames[-1].copy())
                bboxes_sub.append([])
            video = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames])
            chunk = video.view(1, 3, subchunk_size, size[1], size[0])
            subchunks.append((chunk, bboxes_sub))
    return subchunks


# Efficient video chunker: split video into 32-frame chunks
def _chunk_video_frames(video_path: str, chunk_size: int = 32, size=(224, 224)) -> Optional[list[tuple[torch.Tensor, int]]]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None

    chunks = []
    for start_frame in range(0, total, chunk_size):
        frames = []
        for i in range(chunk_size):
            frame_idx = start_frame + i
            if frame_idx >= total:
                # Pad with last frame if needed
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, size)
                    frames.append(frame)
                else:
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        break

        if len(frames) == chunk_size:
            # Convert to tensor (c, t, h, w) -> (1, c, t, h, w)
            video = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames])
            chunk = video.view(1, 3, chunk_size, size[1], size[0])
            chunks.append((chunk, start_frame))

    return chunks if chunks else None


# Face detection and cropping with YOLO
def _detect_and_crop_faces_in_chunk(chunk_tensor, yolo_model, size=(224, 224)):
    import cv2
    # chunk_tensor: (1, 3, 32, 224, 224)
    # Take the middle frame for detection
    frame = chunk_tensor[0, :, 16, :, :].permute(1, 2, 0).numpy() * 255  # (224, 224, 3)
    frame = frame.astype(np.uint8)
    
    results = yolo_model(frame)
    bboxes = []
    cropped_faces = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            if cls == 0:  # Assuming class 0 is person, but for face, may need face model
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Crop face
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    face_resized = cv2.resize(face, size)
                    face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
                    cropped_faces.append(face_tensor)
                    bboxes.append([x1, y1, x2-x1, y2-y1])
    if cropped_faces:
        # Use the largest face or first
        face_tensor = cropped_faces[0]
        # Create a chunk with the face repeated for all frames? No, for simplicity, use the face as the frame
        # But since it's video, perhaps repeat the face for all 32 frames
        face_chunk = face_tensor.unsqueeze(1).repeat(1, 32, 1, 1).unsqueeze(0)  # (1, 3, 32, 224, 224)
        return face_chunk, bboxes[0]
    else:
        return None, []


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


def compute_attributions(
    model: torch.nn.Module,
    video_feats: torch.Tensor,
    audio_feats: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    baseline: str = "zeros",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Integrated Gradients attributions for video and audio features.

    Args:
        model: Trained MultimodalEmotionModel.
        video_feats: (B, T, 768) tensor.
        audio_feats: (B, 1024) tensor.
        mask: (B, T) padding mask.
        target: (B,) class indices to attribute to (if None, uses predicted class).
        n_steps: Riemann approximation steps.
        baseline: Input baseline ("zeros" or "mean").

    Returns:
        attr_video: (B, T, 768) attributions.
        attr_audio: (B, 1024) attributions.
    """
    model.eval()
    video_feats = video_feats.to(device)
    audio_feats = audio_feats.to(device)
    if mask is not None:
        mask = mask.to(device)

    wrapper = ModelWrapper(model).to(device)
    ig = IntegratedGradients(wrapper)

    # Determine baseline
    if baseline == "zeros":
        video_baseline = torch.zeros_like(video_feats)
        audio_baseline = torch.zeros_like(audio_feats)
    else:
        raise ValueError("Only 'zeros' baseline supported for now")

    # If target is None, predict it
    if target is None:
        with torch.no_grad():
            _, logits, _ = model(video_feats, audio_feats, mask=mask)
            target = logits.argmax(dim=-1)

    # Compute attributions
    attr_video, attr_audio = ig.attribute(
        inputs=(video_feats, audio_feats),
        baselines=(video_baseline, audio_baseline),
        additional_forward_args=mask,
        target=target,
        n_steps=n_steps,
    )

    return attr_video, attr_audio


def aggregate_importances(
    attr_video: torch.Tensor,
    attr_audio: torch.Tensor,
    abs_sum: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate attributions to get per-feature importance.

    - Video: Sum over time (T) -> (B, 768)
    - Audio: Already (B, 1024)

    Returns:
        video_importance: (B, 768)
        audio_importance: (B, 1024)
    """
    if abs_sum:
        attr_video = attr_video.abs()
        attr_audio = attr_audio.abs()

    # Sum over time for video
    video_importance = attr_video.sum(dim=1)  # (B, 768)

    audio_importance = attr_audio  # (B, 1024)

    return video_importance, audio_importance


# Main pipeline inference function with sliding window
def infer_video_file(video_file_path: str, subchunk_size: int = 32, window_size: int = 5, explain: bool = False):
    # 1) Get face sequences
    yolo_model = get_yolo_model()
    sequences, fps = get_face_sequences(video_file_path, yolo_model, max_delay=10, max_frames_per_sequence=10000)  # Allow long sequences
    if not sequences:
        return {"bounding_box": [], "inference": []}

    # Collect all bounding boxes
    bounding_box = []
    for seq in sequences:
        for fn, bboxes in seq:
            for bbox in bboxes:
                bounding_box.append({
                    "frame": fn,
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                    "confidence": bbox[4]
                })

    device = _DEVICE
    video_model = get_video_model(device)
    fusion_model = get_fusion_model(device)

    inference = []

    # 2) Process each sequence
    for seq_idx, seq in enumerate(sequences):
        # Get subchunks
        subchunks = create_subchunks_from_sequence(video_file_path, seq, subchunk_size=subchunk_size, size=(224, 224))
        if not subchunks:
            continue

        num_subchunks = len(subchunks)

        # Sliding window over subchunks
        for start in range(num_subchunks):
            window_subchunks = []
            window_bboxes = []
            for i in range(window_size):
                idx = start + i
                if idx < num_subchunks:
                    window_subchunks.append(subchunks[idx][0].to(device))
                    window_bboxes.extend(subchunks[idx][1])
                else:
                    break

            if not window_subchunks:
                continue

            num_in_window = len(window_subchunks)

            # Calculate start frame
            start_frame_idx = start * subchunk_size
            start_frame = seq[start_frame_idx][0] if start_frame_idx < len(seq) else 0

            # Audio: concat from subchunks in window
            audio_segments = []
            for i in range(num_in_window):
                sub_idx = start + i
                if sub_idx < num_subchunks:
                    # Audio for this subchunk: from frame start to end
                    min_frame = min(fn for fn, _ in seq[sub_idx*subchunk_size:(sub_idx+1)*subchunk_size])
                    max_frame = max(fn for fn, _ in seq[sub_idx*subchunk_size:(sub_idx+1)*subchunk_size])
                    start_time = min_frame / fps
                    duration = (max_frame - min_frame + 1) / fps
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_tmp:
                        seg_path = seg_tmp.name
                    cmd_cut = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_file_path,
                        "-ss",
                        str(start_time),
                        "-t",
                        str(duration),
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        seg_path,
                    ]
                    subprocess.run(cmd_cut, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    audio_segments.append(seg_path)

            if audio_segments:
                concat_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                for seg in audio_segments:
                    concat_list.write(f"file '{seg}'\n")
                concat_list.close()

                window_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                cmd_concat = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_list.name,
                    "-c",
                    "copy",
                    window_audio_path,
                ]
                subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.unlink(concat_list.name)

                audio_embedding = extract_embedding_from_file(window_audio_path)
                os.unlink(window_audio_path)
                for seg in audio_segments:
                    os.unlink(seg)
            else:
                audio_embedding = np.zeros(1024)

            audio_feats = torch.from_numpy(audio_embedding).unsqueeze(0).to(device).float()

            # Stack video
            video_window = torch.cat(window_subchunks, dim=0).to(device)

            # Extract features
            video_feats_list = []
            with torch.no_grad():
                for chunk in window_subchunks:
                    feat = video_model(chunk)
                    video_feats_list.append(feat.squeeze(0))

            video_feats_window = torch.stack(video_feats_list).unsqueeze(0).to(device)

            # Mask should match the actual number of tokens (num_in_window)
            mask = torch.zeros((1, num_in_window), dtype=torch.bool, device=device)

            # Classify
            with torch.no_grad():
                probs, logits, attn = fusion_model(video_feats_window, audio_feats, mask=mask, return_attn=True)

            probs = probs.detach().cpu().numpy()[0]
            logits = logits.detach().cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

            inference_item = {
                "class": _LABELS[pred_idx] if pred_idx < len(_LABELS) else str(pred_idx),
                "frame": start_frame
            }

            if explain:
                # Compute attributions
                attr_video, attr_audio = compute_attributions(
                    fusion_model, video_feats_window, audio_feats, mask=mask, target=None, device=device
                )
                video_imp, audio_imp = aggregate_importances(attr_video, attr_audio)
                video_imp_list = video_imp.detach().cpu().numpy()[0]
                audio_imp_list = audio_imp.detach().cpu().numpy()[0]
                top_k = 10
                topk_video = sorted(enumerate(video_imp_list), key=lambda x: abs(x[1]), reverse=True)[:top_k]
                topk_audio = sorted(enumerate(audio_imp_list), key=lambda x: abs(x[1]), reverse=True)[:top_k]
                inference_item["feature_importance"] = {
                    "video": [{"dimension": idx, "importance": float(score)} for idx, score in topk_video],
                    "audio": [{"dimension": idx, "importance": float(score)} for idx, score in topk_audio]
                }

            inference.append(inference_item)

    return {"bounding_box": bounding_box, "inference": inference}


# Convenience wrapper for FastAPI file upload processing
def infer_upload_file(upload_file: UploadFile, chunk_size: int = 32, window_size: int = 5, explain: bool = False):
    # Save to a temp file, run inference, and clean up
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, Path(upload_file.filename).name)
        with open(tmp_path, "wb") as f:
            f.write(upload_file.file.read())
        results = infer_video_file(tmp_path, subchunk_size=chunk_size, window_size=window_size, explain=explain)
        return results

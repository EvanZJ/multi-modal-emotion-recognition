import os
import tempfile
import subprocess
from typing import Optional, Tuple, List

import numpy as np
import torch
import torchaudio
from pathlib import Path

# Import local extractors and model classes
from .libs.video_extractor import ViViTFeatureExtractor, load_video, extract_features
from .libs.voice_extractor import get_audio_model_and_extractor, extract_embedding_from_file
from .libs.train2_model import MultimodalEmotionModel

# Mapping labels to emotion names: final mapping used in train2.py
EMOTION_MAP = {
    0: 'NEU',
    1: 'HAP',
    2: 'SAD',
    3: 'ANG',
    4: 'FEA',
    5: 'DIS',
}


class InferenceEngine:
    def __init__(self, device: Optional[str] = None, model_path: Optional[str] = None):
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.video_extractor_model = ViViTFeatureExtractor()
        self.video_extractor_model.to(self.device)
        self.video_extractor_model.eval()

        # Audio model and feature extractor will be loaded lazily on demand to avoid heavy imports at startup.
        self.audio_model = None
        self.audio_feature_extractor = None

        # Load fusion classifier (train2.MultimodalEmotionModel)
        self.fusion_model = MultimodalEmotionModel()
        self.fusion_model.to(self.device)
        self.fusion_model.eval()
        # Determine model path: explicit model_path -> repo training_runs_2 -> None
        self.model_path = model_path
        if not self.model_path:
            # Look for latest model in repo's training_runs_2
            repo_root = Path(__file__).resolve().parents[2]
            training_dir = repo_root / "training_runs_2"
            if training_dir.exists():
                pths = sorted(training_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
                if pths:
                    self.model_path = str(pths[0])
        if self.model_path and os.path.exists(self.model_path):
            sd = torch.load(self.model_path, map_location=self.device)
            try:
                self.fusion_model.load_state_dict(sd)
            except Exception:
                # If direct load failed, try a more robust load (non-strict) with nested key discovery
                sd_inner = sd.get('model_state_dict', sd)
                try:
                    self.fusion_model.load_state_dict(sd_inner, strict=False)
                except Exception:
                    # Give up if it still fails; model will be uninitialized
                    pass

    def _extract_audio_embedding(self, video_path: str) -> Optional[np.ndarray]:
        # Use ffmpeg to extract audio to temporary wav (16 kHz mono)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
            out_wav = tf.name
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-ac', '1', '-ar', '16000',
            '-vn', out_wav
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wav, sr = torchaudio.load(out_wav)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)
            # Use the imported audio_feature_extractor and audio_model
            # lazy load audio model/extractor
            if self.audio_model is None or self.audio_feature_extractor is None:
                self.audio_model, self.audio_feature_extractor = get_audio_model_and_extractor()
                self.audio_model.to(self.device)
                self.audio_model.eval()
            inputs = self.audio_feature_extractor(wav.cpu().numpy(), sampling_rate=16000, return_tensors='pt', padding=True)
            input_values = inputs.input_values.to(self.device)
            with torch.no_grad():
                hidden_states = self.audio_model(input_values).last_hidden_state
            emb = hidden_states.mean(dim=1)
            emb = emb / emb.norm(dim=1, keepdim=True)
            emb_np = emb.cpu().numpy()[0]
            return emb_np
        except Exception:
            return None
        finally:
            try:
                os.unlink(out_wav)
            except Exception:
                pass

    def _extract_video_features(self, video_path: str, chunk_size: int = 32) -> Optional[np.ndarray]:
        chunks = load_video(video_path, chunk_size=chunk_size)
        if chunks is None:
            return None
        chunks = chunks.to(self.device)
        features_list = []
        with torch.no_grad():
            for chunk in chunks:
                chunk = chunk.unsqueeze(0)  # (1,3,32,224,224)
                feats = self.video_extractor_model(chunk)  # (1, 768)
                features_list.append(feats.cpu())
        features = torch.cat(features_list, dim=0)
        return features.numpy()

    def predict_from_file(self, video_path: str, top_k: int = 3) -> Optional[dict]:
        audio_emb = self._extract_audio_embedding(video_path)
        video_feats = self._extract_video_features(video_path)

        if audio_emb is None or video_feats is None:
            return None

        # Convert to tensors
        video_feats = torch.tensor(video_feats, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T, D_v)
        audio_feats = torch.tensor(audio_emb, dtype=torch.float32).unsqueeze(0).to(self.device)    # (1, D_a)

        # Build mask: False for real frames, True for padded frames
        T = video_feats.shape[1]
        mask = torch.zeros((1, T), dtype=torch.bool).to(self.device)

        # Clip or pad to fusion max seq len (extract from model)
        max_seq_len = self.fusion_model.fusion.pos_embed.size(1)
        if T > max_seq_len - 1:  # -1 because audio token is appended
            # Clip to first (max_seq_len - 1) tokens
            video_feats = video_feats[:, :max_seq_len - 1, :]
            mask = mask[:, :max_seq_len - 1]
            T = video_feats.shape[1]
        elif T < max_seq_len - 1:
            # Pad with zeros
            pad_len = (max_seq_len - 1) - T
            pad = torch.zeros((1, pad_len, video_feats.shape[2]), dtype=video_feats.dtype).to(self.device)
            video_feats = torch.cat([video_feats, pad], dim=1)
            pad_mask = torch.ones((1, pad_len), dtype=torch.bool).to(self.device)
            mask = torch.cat([mask, pad_mask], dim=1)

        # Inference
        with torch.no_grad():
            probs, logits, _ = self.fusion_model(video_feats, audio_feats, mask=mask)
            probs_np = probs.cpu().numpy().squeeze(0)

        # Top-k
        idxs = probs_np.argsort()[::-1][:top_k]
        results = [
            {"label": EMOTION_MAP[int(i)], "probability": float(probs_np[int(i)])}
            for i in idxs
        ]

        pred_idx = int(np.argmax(probs_np))
        return {
            "predicted_label": EMOTION_MAP[pred_idx],
            "predicted_index": pred_idx,
            "scores": results,
        }


# Instantiate a singleton engine for the API (lazy initialization)
_engine: Optional[InferenceEngine] = None


def get_engine(model_path: Optional[str] = None):
    global _engine
    if _engine is None:
        _engine = InferenceEngine(model_path=model_path)
    return _engine

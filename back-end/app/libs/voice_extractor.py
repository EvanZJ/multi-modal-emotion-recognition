import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import soundfile as sf
try:
    import torchaudio
except Exception:
    torchaudio = None
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_audio_model = None
_audio_feature_extractor = None

def get_audio_model_and_extractor():
    global _audio_model, _audio_feature_extractor
    if _audio_model is None or _audio_feature_extractor is None:
        _audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        _audio_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        _audio_model.to(DEVICE)
        _audio_model.eval()
    return _audio_model, _audio_feature_extractor


@torch.no_grad()
def extract_embeddings_batch(waveforms: List[torch.Tensor]) -> np.ndarray:
    # Lazy load model and extractor on first call
    model, feature_extractor = get_audio_model_and_extractor()
    inputs = feature_extractor(
        [w.cpu().numpy() for w in waveforms],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(DEVICE)
    hidden_states = model(input_values).last_hidden_state
    embeddings = hidden_states.mean(dim=1)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings.cpu().numpy()


def extract_embedding_from_file(wave_path: str) -> np.ndarray:
    if torchaudio is not None:
        try:
            wav, sr = torchaudio.load(wave_path)
        except Exception:
            # torchaudio may not support some formats without torchcodec; fallback
            data, sr = sf.read(wave_path, dtype='float32')
            wav = torch.from_numpy(np.array(data)).unsqueeze(0)
    else:
        data, sr = sf.read(wave_path, dtype='float32')
        wav = torch.from_numpy(np.array(data)).unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    emb = extract_embeddings_batch([wav])
    return emb[0]

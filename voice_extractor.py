# =============================================================================
# Audio Emotion Feature Extractor — FINAL FIXED VERSION (Nov 2025)
# Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim → 1024-dim embedding
# Saves one .npy per video → L2-normalized + float16
# =============================================================================

import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from pathlib import Path

# --------------------------- CONFIG ---------------------------
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8                    # ← 8–16 is optimal on GPU. Set to 1 only if you want ultra-safe
OUTPUT_DIR = Path("audio_features")
CHUNK_DURATION = 10               # Max duration per chunk in seconds to avoid OOM
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --------------------------- LOAD MODEL ---------------------------
print(f"Loading {MODEL_NAME} ...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()
model.to(DEVICE)
print("Model loaded successfully!")

# --------------------------- EXTRACTION FUNCTION ---------------------------
@torch.no_grad()
def extract_embeddings_batch(waveforms: list[torch.Tensor]) -> np.ndarray:
    # waveforms: list of 1D tensors (T,)
    inputs = feature_extractor(
        [w.cpu().numpy() for w in waveforms],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(DEVICE)

    hidden_states = model(input_values).last_hidden_state          # (B, seq_len, 1024)
    embeddings = hidden_states.mean(dim=1)                          # (B, 1024)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # L2 normalize
    return embeddings.cpu().numpy()                                 # (B, 1024)

# --------------------------- MAIN PROCESSING ---------------------------
def process_folder(input_folder: str):
    audio_paths = []
    for ext in ["*.mp3", "*.wav", "*.flac", "*.aac", "*.ogg"]:
        audio_paths.extend(Path(input_folder).rglob(ext))

    print(f"Found {len(audio_paths)} audio files. Starting audio feature extraction...")

    batch_waveforms = []
    batch_paths = []

    for path in tqdm(audio_paths, desc="Processing audio files"):
        try:
            # 1. Load audio directly
            wav, sr = torchaudio.load(str(path))
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)   # stereo → mono
            wav = wav.squeeze(0)  # (T,)

            duration = len(wav) / sr
            if duration > CHUNK_DURATION:
                # Process in chunks
                chunk_size = int(CHUNK_DURATION * sr)
                chunks = [wav[i:i+chunk_size] for i in range(0, len(wav), chunk_size) if len(wav[i:i+chunk_size]) > 0]
                embeddings = []
                for chunk in chunks:
                    emb = extract_embeddings_batch([chunk])
                    embeddings.append(emb[0])
                if embeddings:
                    final_emb = np.mean(embeddings, axis=0)
                    final_emb = final_emb / np.linalg.norm(final_emb)
                    # Parse filename
                    stem = path.stem
                    if '-' in stem:
                        parts = stem.split('-')
                        actor = parts[-1]
                        emotion = '-'.join(parts)
                        out_path = OUTPUT_DIR / f"Video_Speech_Actor_{actor}_{emotion}_voice_mp4_features.npy"
                    else:
                        parts = stem.split('_')
                        actor = parts[0]
                        emotion = '_'.join(parts[1:])
                        out_path = OUTPUT_DIR / f"{actor}_{emotion}_voice_mp4_features.npy"
                    np.save(out_path, final_emb.astype(np.float16))
            else:
                # Add to batch for short audios
                batch_waveforms.append(wav)
                batch_paths.append(path)

                # 2. Process full batch
                if len(batch_waveforms) >= BATCH_SIZE:
                    embeddings = extract_embeddings_batch(batch_waveforms)

                    for emb, p in zip(embeddings, batch_paths):
                        # Parse filename
                        stem = p.stem
                        if '-' in stem:
                            parts = stem.split('-')
                            actor = parts[-1]
                            emotion = '-'.join(parts)
                            out_path = OUTPUT_DIR / f"Video_Speech_Actor_{actor}_{emotion}_voice_mp4_features.npy"
                        else:
                            parts = stem.split('_')
                            actor = parts[0]
                            emotion = '_'.join(parts[1:])
                            out_path = OUTPUT_DIR / f"{actor}_{emotion}_voice_mp4_features.npy"
                        np.save(out_path, emb.astype(np.float16))
                        # print(f"Saved → {out_path.name}")

                    batch_waveforms.clear()
                    batch_paths.clear()

        except Exception as e:
            print(f"Error on {path.name}: {e}")

    if batch_waveforms:
        print(f"Processing final batch of {len(batch_waveforms)} clips...")
        embeddings = extract_embeddings_batch(batch_waveforms)
        for emb, p in zip(embeddings, batch_paths):
            stem = p.stem
            if '-' in stem:
                parts = stem.split('-')
                actor = parts[-1]
                emotion = '-'.join(parts)
                out_path = OUTPUT_DIR / f"Video_Speech_Actor_{actor}_{emotion}_voice_mp4_features.npy"
            else:
                parts = stem.split('_')
                actor = parts[0]
                emotion = '_'.join(parts[1:])
                out_path = OUTPUT_DIR / f"{actor}_{emotion}_voice_mp4_features.npy"
            np.save(out_path, emb.astype(np.float16))
            print(f"Saved → {out_path.name}")

    print(f"\nAll done! {len(list(OUTPUT_DIR.glob('*.npy')))} audio features saved in:")
    print(f"    {OUTPUT_DIR.resolve()}")

# --------------------------- RUN ---------------------------
if __name__ == "__main__":
    process_folder("/home/sionna/evan/multi-modal-emotion-recognition/extracted_audio")
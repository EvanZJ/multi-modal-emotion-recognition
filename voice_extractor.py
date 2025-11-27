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
import subprocess
import tempfile

# --------------------------- CONFIG ---------------------------
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8                    # ← 8–16 is optimal on GPU. Set to 1 only if you want ultra-safe
OUTPUT_DIR = Path("audio_features")
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
    video_paths = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.wmv", "*.MP4"]:
        video_paths.extend(Path(input_folder).rglob(ext))

    print(f"Found {len(video_paths)} video files. Starting audio extraction...")

    batch_waveforms = []
    batch_paths = []

    for path in tqdm(video_paths, desc="Processing videos"):
        temp_wav = None
        try:
            # 1. Extract audio with ffmpeg
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()

            cmd = [
                "ffmpeg", "-y", "-i", str(path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                temp_wav.name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg failed on {path.name}: {result.stderr}")
                continue

            # 2. Load audio
            wav, sr = torchaudio.load(temp_wav.name)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)   # stereo → mono
            wav = wav.squeeze(0)  # (T,)

            batch_waveforms.append(wav)
            batch_paths.append(path)

            # 3. Process full batch
            if len(batch_waveforms) >= BATCH_SIZE:
                embeddings = extract_embeddings_batch(batch_waveforms)

                for emb, p in zip(embeddings, batch_paths):
                    out_path = OUTPUT_DIR / f"{p.stem}.npy"
                    np.save(out_path, emb.astype(np.float16))
                    # print(f"Saved → {out_path.name}")

                batch_waveforms.clear()
                batch_paths.clear()

        except Exception as e:
            print(f"Error on {path.name}: {e}")
        finally:
            if temp_wav and os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)

    # --------------------------- FINAL BATCH (always runs) ---------------------------
    if batch_waveforms:
        print(f"Processing final batch of {len(batch_waveforms)} clips...")
        embeddings = extract_embeddings_batch(batch_waveforms)
        for emb, p in zip(embeddings, batch_paths):
            out_path = OUTPUT_DIR / f"{p.stem}.npy"
            np.save(out_path, emb.astype(np.float16))
            print(f"Saved → {out_path.name}")

    print(f"\nAll done! {len(list(OUTPUT_DIR.glob('*.npy')))} audio features saved in:")
    print(f"    {OUTPUT_DIR.resolve()}")

# --------------------------- RUN ---------------------------
if __name__ == "__main__":
    process_folder("/home/sionna/Downloads/1188976")
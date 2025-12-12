import os
import glob
import numpy as np
import torch

def check_shapes(video_feat_dir: str, audio_feat_dir: str):
    """
    Load video and audio features and print their shapes.
    """
    video_files = sorted(glob.glob(os.path.join(video_feat_dir, "*.npy")))
    audio_files = sorted(glob.glob(os.path.join(audio_feat_dir, "*.npy")))

    print(f"Found {len(video_files)} video files and {len(audio_files)} audio files.")

    video_shapes = []
    audio_shapes = []

    for v_file, a_file in zip(video_files[:10], audio_files[:10]):  # Check first 10 for brevity
        v_feat = np.load(v_file).astype(np.float32)
        a_feat = np.load(a_file).astype(np.float32)

        video_shapes.append(v_feat.shape)
        audio_shapes.append(a_feat.shape)

        print(f"Video: {os.path.basename(v_file)} -> Shape: {v_feat.shape}")
        print(f"Audio: {os.path.basename(a_file)} -> Shape: {a_feat.shape}")
        print("---")

    # Compute max T for videos
    all_video_shapes = [np.load(f).shape[0] for f in video_files]
    max_t = max(all_video_shapes) if all_video_shapes else 0
    print(f"Max frames (T) across all videos: {max_t}")

if __name__ == "__main__":
    base_dir = "/home/sionna/evan/multi-modal-emotion-recognition"
    check_shapes(
        video_feat_dir=os.path.join(base_dir, "video_features"),
        audio_feat_dir=os.path.join(base_dir, "audio_features"),
    )
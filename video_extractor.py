import torch
import torch.nn as nn
import os
import cv2
import numpy as np

class TubeletEmbedder(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, tubelet_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size  # (h, w)
        self.patch_size = patch_size  # (p_h, p_w)
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size  # t
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (num_frames // tubelet_size)
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=(tubelet_size, patch_size[0], patch_size[1]), stride=(tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x):
        # x: (b, c, t, h, w)
        x = self.proj(x)  # (b, d, t', h', w')
        x = x.flatten(2)  # (b, d, n)
        x = x.transpose(1, 2)  # (b, n, d)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(b, n, self.heads, -1).permute(0, 2, 1, 3) for t in qkv]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViViTFeatureExtractor(nn.Module):
    def __init__(self, image_size=(224, 224), patch_size=(16, 16), num_frames=32, tubelet_size=4, dim=768, depth=12, heads=12, pool='cls', in_channels=3, dim_head=64, mlp_dim=3072):
        super().__init__()
        self.pool = pool
        self.embedder = TubeletEmbedder(image_size, patch_size, num_frames, tubelet_size, in_channels, dim)
        num_patches = self.embedder.num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1 if pool == 'cls' else num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if pool == 'cls' else None
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, video):
        x = self.embedder(video)
        b, n, d = x.shape
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.transformer(x)
        if self.pool == 'cls' and self.cls_token is not None:
            features = x[:, 0]
        else:
            features = x.mean(dim=1)
        return features

# Example usage:
# model = ViViTFeatureExtractor()
# video = torch.randn(1, 3, 32, 224, 224)  # Dummy video input
# features = model(video)
# print(features.shape)  # Should be torch.Size([1, 768])

def load_video(video_path, num_frames=32, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    if total_frames < num_frames:
        # Sample all frames and pad by duplicating the last frame
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, size)
                frames.append(frame)
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((size[1], size[0], 3), dtype=np.uint8))
    else:
        step = total_frames // num_frames
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, size)
                frames.append(frame)
            if len(frames) == num_frames:
                break
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    # Convert to tensor: (t, h, w, c) -> (t, c, h, w) -> (1, c, t, h, w)
    video = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames])
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, 3, 32, 224, 224)
    return video

def extract_features(video_path, model, device):
    video_tensor = load_video(video_path)
    if video_tensor is None:
        print(f"Failed to load video: {video_path}")
        return None
    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        features = model(video_tensor)
    return features.cpu().numpy()

def process_folder(folder_path, model, device):
    extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
    count = 0
    output_dir = "video_features"
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                video_path = os.path.join(root, file)
                print(f"Processing: {video_path}")
                features = extract_features(video_path, model, device)
                if features is not None:
                    # Create a unique name based on relative path
                    rel_path = os.path.relpath(video_path, folder_path)
                    safe_name = rel_path.replace(os.sep, '_').replace('.', '_') + '_features.npy'
                    output_path = os.path.join(output_dir, safe_name)
                    np.save(output_path, features)
                    count += 1
                    print(f"Saved features to: {output_path}")
    print(f"Finished! Processed {count} video files.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = ViViTFeatureExtractor()
    model.to(device)
    model.eval()
    directory = "/home/sionna/evan/multi-modal-emotion-recognition/extracted_faces_videos"
    process_folder(directory, model, device)
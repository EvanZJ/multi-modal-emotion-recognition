import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):
    def __init__(
        self,
        video_dim: int = 768,
        audio_dim: int = 1024,
        fused_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 101,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)
        self.norm_video = nn.LayerNorm(fused_dim) if use_layernorm else nn.Identity()
        self.norm_audio = nn.LayerNorm(fused_dim) if use_layernorm else nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, fused_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=num_heads,
            dim_feedforward=4 * fused_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(fused_dim) if use_layernorm else nn.Identity()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor, mask: torch.Tensor = None, return_attn: bool = False):
        b, t, _ = video_feats.shape
        video = self.video_proj(video_feats)
        video = self.norm_video(video)
        audio = self.audio_proj(audio_feats)
        audio = self.norm_audio(audio).unsqueeze(1)
        combined = torch.cat([video, audio], dim=1)
        combined = combined + self.pos_embed[:, :t + 1, :]
        combined = self.dropout_layer(combined)
        if mask is not None:
            audio_mask = torch.zeros(b, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([mask, audio_mask], dim=1)
        else:
            full_mask = None
        combined = combined.permute(1, 0, 2)
        transformer_out = self.transformer(
            combined,
            src_key_padding_mask=full_mask
        )
        attn_weights = None
        fused = transformer_out.permute(1, 0, 2)
        if full_mask is not None:
            attend_mask = (~full_mask).float().unsqueeze(-1)
            fused_embedding = (fused * attend_mask).sum(dim=1) / attend_mask.sum(dim=1).clamp(min=1e-6)
        else:
            fused_embedding = fused.mean(dim=1)
        fused_embedding = self.out_norm(fused_embedding)
        return fused_embedding, attn_weights

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim: int = 512, num_classes: int = 6, hidden_dim: int = None, dropout: float = 0.2, use_layernorm: bool = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        Norm = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Norm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            Norm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.dropout = dropout
        self.hidden_dim = hidden_dim

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        logits = self.net(fused_embedding)
        return logits

class MultimodalEmotionModel(nn.Module):
    def __init__(self, video_dim: int = 768, audio_dim: int = 1024, fused_dim: int = 512, num_classes: int = 6, max_seq_len: int = 101, fusion_num_layers: int = 2, fusion_num_heads: int = 8, fusion_dropout: float = 0.1, classifier_hidden_dim: int = None, classifier_dropout: float = 0.2):
        super().__init__()
        self.fusion = CrossModalFusion(video_dim=video_dim, audio_dim=audio_dim, fused_dim=fused_dim, num_layers=fusion_num_layers, num_heads=fusion_num_heads, dropout=fusion_dropout, max_seq_len=max_seq_len)
        self.classifier = EmotionClassifier(input_dim=fused_dim, num_classes=num_classes, hidden_dim=classifier_hidden_dim, dropout=classifier_dropout)

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor, mask: torch.Tensor = None, return_attn: bool = False):
        fused, attn_weights = self.fusion(video_feats, audio_feats, mask=mask, return_attn=return_attn)
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        return probs, logits, attn_weights

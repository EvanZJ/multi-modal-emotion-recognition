import os
import glob
import json
import argparse
from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from captum.attr import IntegratedGradients
import pandas as pd  # For saving importances as CSV


# ============================================================
# Loss functions
# ============================================================

class ModelWrapper(nn.Module):
    """
    Wrapper to make the model compatible with Captum (returns only logits).
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor, mask: Optional[torch.Tensor] = None):
        _, logits, _ = self.model(video_feats, audio_feats, mask=mask, return_attn=False)
        return logits

class FocalLoss(nn.Module):
    """
    Standard Focal Loss for multi-class classification.

    Args:
        gamma: focusing parameter (default: 2.0).
        alpha: per-class weighting tensor of shape (C,) or None.
        reduction: 'none', 'mean', or 'sum'.
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C) logits
        # targets: (B,) class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# Model components: Fusion + Classifier
# ============================================================

class CrossModalFusion(nn.Module):
    """
    Self-attention fusion for video (sequence) and audio (global) features.

    - Video: (B, T, D_v)
    - Audio: (B, D_a)
    - We project both modalities to a shared fused_dim.
    - We treat [video tokens + 1 audio token] as a sequence for a TransformerEncoder.
    - We use masked mean pooling to produce a fused embedding (B, fused_dim).
    """
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

        # Per-token normalization (LayerNorm is more stable than BatchNorm for sequences)
        self.norm_video = nn.LayerNorm(fused_dim) if use_layernorm else nn.Identity()
        self.norm_audio = nn.LayerNorm(fused_dim) if use_layernorm else nn.Identity()

        # Positional embedding for max_seq_len tokens (T_max + 1 audio token)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, fused_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=num_heads,
            dim_feedforward=4 * fused_dim,
            dropout=dropout,
            batch_first=False,  # Transformer expects (S, B, F)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout_layer = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(fused_dim) if use_layernorm else nn.Identity()

        # Store hyperparameters (used in logging)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout  # float, not a module

    def forward(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """
        Args:
            video_feats: (B, T, D_v)
            audio_feats: (B, D_a)
            mask: (B, T), True = padded position (ignore in attention)
            return_attn: currently not used (PyTorch TransformerEncoder
                         does not return attention weights directly).

        Returns:
            fused_embedding: (B, fused_dim)
            attn_weights: None (placeholder)
        """
        b, t, _ = video_feats.shape

        # Project and normalize
        video = self.video_proj(video_feats)          # (B, T, F)
        video = self.norm_video(video)

        audio = self.audio_proj(audio_feats)          # (B, F)
        audio = self.norm_audio(audio).unsqueeze(1)   # (B, 1, F)

        # Concatenate video tokens with audio token
        combined = torch.cat([video, audio], dim=1)   # (B, T+1, F)

        # Add positional embeddings (T+1 <= max_seq_len)
        combined = combined + self.pos_embed[:, :t + 1, :]
        combined = self.dropout_layer(combined)

        # Build padding mask for transformer: (B, T+1)
        if mask is not None:
            # audio token should never be masked
            audio_mask = torch.zeros(b, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([mask, audio_mask], dim=1)  # (B, T+1)
        else:
            full_mask = None

        # TransformerEncoder uses (S, B, F)
        combined = combined.permute(1, 0, 2)          # (T+1, B, F)
        transformer_out = self.transformer(
            combined,
            src_key_padding_mask=full_mask
        )                                             # (T+1, B, F)

        # Attention weights are not extracted here
        attn_weights = None

        fused = transformer_out.permute(1, 0, 2)      # (B, T+1, F)

        # Masked mean pooling over the sequence length (ignore padded positions)
        if full_mask is not None:
            attend_mask = (~full_mask).float().unsqueeze(-1)   # (B, T+1, 1)
            fused_embedding = (fused * attend_mask).sum(dim=1) / \
                              attend_mask.sum(dim=1).clamp(min=1e-6)
        else:
            fused_embedding = fused.mean(dim=1)                # (B, F)

        fused_embedding = self.out_norm(fused_embedding)       # (B, F)

        return fused_embedding, attn_weights


class EmotionClassifier(nn.Module):
    """
    Deep classifier head on top of the fused embedding.

    Returns logits only; softmax is applied outside.
    """
    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 6,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        use_layernorm: bool = True,
    ):
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

        # Store hyperparameters for logging
        self.dropout = dropout
        self.hidden_dim = hidden_dim

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        # fused_embedding: (B, input_dim)
        logits = self.net(fused_embedding)  # (B, num_classes)
        return logits


class MultimodalEmotionModel(nn.Module):
    """
    Full multimodal model: fusion module + classifier head.

    External API:
        forward(video_feats, audio_feats, mask=None, return_attn=False)
        -> probs, logits, attn_weights
    """
    def __init__(
        self,
        video_dim: int = 768,
        audio_dim: int = 1024,
        fused_dim: int = 512,
        num_classes: int = 6,
        max_seq_len: int = 101,
        fusion_num_layers: int = 2,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
        classifier_hidden_dim: Optional[int] = None,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()

        self.fusion = CrossModalFusion(
            video_dim=video_dim,
            audio_dim=audio_dim,
            fused_dim=fused_dim,
            num_layers=fusion_num_layers,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
            max_seq_len=max_seq_len,
        )

        self.classifier = EmotionClassifier(
            input_dim=fused_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
        )

    def forward(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        fused, attn_weights = self.fusion(video_feats, audio_feats, mask=mask, return_attn=return_attn)
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        # Training loop uses probs for accuracy and logits for the loss.
        return probs, logits, attn_weights


# ============================================================
# Data loading
# ============================================================

def load_data(
    video_feat_dir: str,
    audio_feat_dir: str,
    batch_size: int = 32,
):
    """
    Load pre-extracted video/audio features, map filenames to labels,
    apply global normalization, and create train/val/test loaders.

    Returns:
        train_loader, val_loader, test_loader, max_chunks, class_weights
    """
    video_features: list[torch.Tensor] = []
    audio_features: list[torch.Tensor] = []
    labels: list[int] = []

    video_files = sorted(glob.glob(os.path.join(video_feat_dir, "*.npy")))
    audio_files = sorted(glob.glob(os.path.join(audio_feat_dir, "*.npy")))

    print("Example video/audio file pairs:")
    for v_file, a_file in list(zip(video_files, audio_files))[:10]:
        print(os.path.basename(v_file), "<--->", os.path.basename(a_file))

    # ------------------------------------------------------------
    # 1) Load raw features (no normalization yet)
    # ------------------------------------------------------------
    for v_file, a_file in zip(video_files, audio_files):
        basename = os.path.basename(v_file)

        # RAVDESS naming: e.g., 03-01-05-01-02-01-12.mp4
        if "-" in basename:
            parts = basename.split("-")
            label_num = int(parts[2])
            # Skip non-used classes
            if label_num in [2, 8]:
                continue
            # Map to final labels:
            # 01->0 (NEU), 03->1 (HAP), 04->2 (SAD),
            # 05->3 (ANG), 06->4 (FEA), 07->5 (DIS)
            ravdess_map = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
            label = ravdess_map[label_num]
        else:
            # CREMA-D naming: e.g., 1001_DFA_ANG_XX.mp4
            emotion = basename.split("_")[2]
            cremad_map = {"ANG": 5, "DIS": 7, "FEA": 6, "HAP": 3, "NEU": 1, "SAD": 4}
            label_num = cremad_map[emotion]
            # Map to final labels:
            # 1->0 (NEU), 3->1 (HAP), 4->2 (SAD),
            # 5->3 (ANG), 6->4 (FEA), 7->5 (DIS)
            cremad_to_final = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
            label = cremad_to_final[label_num]

        # Load as float32, keep raw for now
        v_feat = np.load(v_file).astype(np.float32)  # (T, D_v)
        a_feat = np.load(a_file).astype(np.float32)  # (D_a,) or (T_a, D_a) depending on preprocessing

        video_features.append(torch.from_numpy(v_feat))
        audio_features.append(torch.from_numpy(a_feat))
        labels.append(label)

    # ------------------------------------------------------------
    # 2) Compute global mean / std over entire dataset
    # ------------------------------------------------------------
    if len(video_features) > 0:
        # Concatenate along time axis -> (total_frames, D_v)
        all_video = torch.cat(video_features, dim=0)
        video_mean = all_video.mean(dim=0)          # (D_v,)
        video_std = all_video.std(dim=0) + 1e-6     # (D_v,)

        # Stack audio per sample -> (N, D_a)
        all_audio = torch.stack(audio_features, dim=0)  # (N, D_a) or (N,) if 1D
        audio_mean = all_audio.mean(dim=0)          # (D_a,) or scalar
        audio_std = all_audio.std(dim=0) + 1e-6

        # --------------------------------------------------------
        # 3) Apply global normalization per sample
        # --------------------------------------------------------
        for i in range(len(video_features)):
            video_features[i] = (video_features[i] - video_mean) / video_std
            audio_features[i] = (audio_features[i] - audio_mean) / audio_std
    else:
        video_mean = video_std = audio_mean = audio_std = None

    # ------------------------------------------------------------
    # 4) Compute max_chunks (max sequence length in frames)
    # ------------------------------------------------------------
    if video_features:
        max_chunks = max(v.shape[0] for v in video_features)
    else:
        max_chunks = 0
    print(f"Maximum number of video chunks: {max_chunks}")

    # Dataset as list of tuples: (video_sequence, audio_vector, label)
    dataset = list(zip(video_features, audio_features, labels))

    # ------------------------------------------------------------
    # Train/val/test split with stratification on labels
    # ------------------------------------------------------------
    indices = list(range(len(dataset)))
    all_labels = [item[2] for item in dataset]

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=all_labels,
    )
    temp_labels = [all_labels[i] for i in temp_indices]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )

    # ------------------------------------------------------------
    # Custom collate function: pad video sequence + create mask
    # ------------------------------------------------------------
    def collate_fn(batch):
        """
        batch: list of (video_seq, audio_feat, label)

        Returns:
            videos_padded: (B, T_max, D_v)
            audios_stacked: (B, D_a)
            labels_tensor: (B,)
            masks_padded: (B, T_max), True for padded positions
        """
        videos, audios, labels_batch = zip(*batch)

        # Pad variable-length video sequences (pad with zeros)
        videos_padded = pad_sequence(videos, batch_first=True, padding_value=0.0)

        # Stack fixed-size audio features
        audios_stacked = torch.stack(audios)

        # Labels tensor
        labels_tensor = torch.tensor(labels_batch, dtype=torch.long)

        # Build mask: False for real frames, True for padded frames
        masks = [torch.zeros(len(v), dtype=torch.bool) for v in videos]
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=True)

        return videos_padded, audios_stacked, labels_tensor, masks_padded

    # DataLoaders
    train_loader = DataLoader(
        [dataset[i] for i in train_indices],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        [dataset[i] for i in val_indices],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        [dataset[i] for i in test_indices],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Print label distributions
    train_labels = [dataset[i][2] for i in train_indices]
    val_labels = [dataset[i][2] for i in val_indices]
    test_labels = [dataset[i][2] for i in test_indices]

    print("Train label distribution:", Counter(train_labels))
    print("Val label distribution:", Counter(val_labels))
    print("Test label distribution:", Counter(test_labels))

    # Compute class weights for imbalanced classes
    classes = np.unique(train_labels)
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    # Optional mild boost for difficult classes (Fear=4, Disgust=5)
    boost_factor = 1.2
    class_weights[4] = class_weights[4] * boost_factor  # FEA
    class_weights[5] = class_weights[5] * boost_factor  # DIS

    return train_loader, val_loader, test_loader, max_chunks, class_weights


# ============================================================
# Training loop
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    num_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    batch_size: int = 128,
    device: str = "cuda",
):
    """
    Full training loop with:
    - class-weighted CrossEntropy loss
    - ReduceLROnPlateau scheduler on validation loss
    - early stopping on validation accuracy
    - logging of metrics and saving best/final models
    """
    device = device if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Move class weights to the same device
    class_weights = class_weights.to(device)

    # Main loss function
    # (You can switch to FocalLoss if needed: FocalLoss(gamma=2.0, alpha=class_weights))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=20)

    # Collect hyperparameters for logging
    hyperparameters = {
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "batch_size": batch_size,
        "device": device,
        "video_dim": model.fusion.video_proj.in_features,
        "audio_dim": model.fusion.audio_proj.in_features,
        "fused_dim": model.fusion.video_proj.out_features,
        "num_classes": model.classifier.net[-1].out_features,
        "max_seq_len": model.fusion.pos_embed.size(1),
        "fusion_dropout": model.fusion.dropout,
        "classifier_dropout": model.classifier.dropout,
        "num_layers": model.fusion.num_layers,
        "num_heads": model.fusion.num_heads,
        "scheduler_factor": 0.3,
        "scheduler_patience": 5,
        "focal_gamma": 2.0,  # for reference if you switch to FocalLoss
    }

    results = []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    previous_val_loss = float('inf')

    for epoch in range(num_epochs):
        # ----------------------------
        # Training phase
        # ----------------------------
        model.train()
        total_train_loss = 0.0

        for videos, audios, labels, masks in train_loader:
            videos = videos.to(device)
            audios = audios.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            probs, logits, _ = model(videos, audios, mask=masks)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ----------------------------
        # Validation phase
        # ----------------------------
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels_list = []

        with torch.no_grad():
            for videos, audios, labels, masks in val_loader:
                videos = videos.to(device)
                audios = audios.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                probs, logits, _ = model(videos, audios, mask=masks)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(probs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        scheduler.step(avg_val_loss)

        # Save best model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1

        # Early stopping: check if improvement from previous epoch is significant
        epsilon = 1e-4
        if previous_val_loss - avg_val_loss < epsilon:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            epochs_without_improvement = 0

        # Update previous loss for next epoch
        previous_val_loss = avg_val_loss

        # Validation metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels_list,
            all_preds,
            average="macro",
            zero_division=0,
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            all_labels_list,
            all_preds,
            average="micro",
            zero_division=0,
        )

        # ----------------------------
        # Test evaluation (no loss)
        # ----------------------------
        test_preds = []
        test_labels_list = []

        with torch.no_grad():
            for videos, audios, labels, masks in test_loader:
                videos = videos.to(device)
                audios = audios.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                probs, _, _ = model(videos, audios, mask=masks)
                _, predicted = torch.max(probs, dim=1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())

        test_acc = 100.0 * sum(int(p == l) for p, l in zip(test_preds, test_labels_list)) / len(test_labels_list)
        test_macro_precision, test_macro_recall, test_macro_f1, _ = precision_recall_fscore_support(
            test_labels_list,
            test_preds,
            average="macro",
        )
        test_micro_precision, test_micro_recall, test_micro_f1, _ = precision_recall_fscore_support(
            test_labels_list,
            test_preds,
            average="micro",
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )
        print(
            f"Val Macro P/R/F1: {macro_precision:.4f}/{macro_recall:.4f}/{macro_f1:.4f}, "
            f"Micro P/R/F1: {micro_precision:.4f}/{micro_recall:.4f}/{micro_f1:.4f}"
        )
        print(
            f"Test Acc: {test_acc:.2f}%, "
            f"Test Macro P/R/F1: {test_macro_precision:.4f}/{test_macro_recall:.4f}/{test_macro_f1:.4f}, "
            f"Micro P/R/F1: {test_micro_precision:.4f}/{test_micro_recall:.4f}/{test_micro_f1:.4f}"
        )

        # Store results for this epoch
        results.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_macro_precision": macro_precision,
            "val_macro_recall": macro_recall,
            "val_macro_f1": macro_f1,
            "val_micro_precision": micro_precision,
            "val_micro_recall": micro_recall,
            "val_micro_f1": micro_f1,
            "test_acc": test_acc,
            "test_macro_precision": test_macro_precision,
            "test_macro_recall": test_macro_recall,
            "test_macro_f1": test_macro_f1,
            "test_micro_precision": test_micro_precision,
            "test_micro_recall": test_micro_recall,
            "test_micro_f1": test_micro_f1,
        })

    # ------------------------------------------------------------
    # Evaluate best model on test set + confusion matrix
    # ------------------------------------------------------------
    if best_model_state is not None:
        print("\nEvaluating BEST model on test set for confusion matrix ...")
        model.load_state_dict(best_model_state)
        model.to(device)
        model.eval()

        all_test_labels = []
        all_test_preds = []

        with torch.no_grad():
            for videos, audios, labels, masks in test_loader:
                videos = videos.to(device)
                audios = audios.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                probs, _, _ = model(videos, audios, mask=masks)
                _, predicted = torch.max(probs, dim=1)

                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(predicted.cpu().numpy())

        cm = confusion_matrix(all_test_labels, all_test_preds, labels=[0, 1, 2, 3, 4, 5])
        print("Confusion matrix (rows = true, cols = pred):")
        print(cm)

    # ------------------------------------------------------------
    # Save results and models
    # ------------------------------------------------------------
    os.makedirs("training_runs_2", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save training log
    results_name = f"results_bs{batch_size}_ep{num_epochs}_lr{lr}_{timestamp}.json"
    with open(os.path.join("training_runs_2", results_name), "w") as f:
        json.dump(
            {
                "training_progress": results,
                "best_model": {"epoch": best_epoch},
                "hyperparameters": hyperparameters,
            },
            f,
            indent=4,
        )
    print(f"Training results saved to training_runs_2/{results_name}")

    # Save best model (by validation accuracy)
    best_model_name = f"best_model_bs{batch_size}_ep{num_epochs}_lr{lr}_{timestamp}.pth"
    torch.save(best_model_state, os.path.join("training_runs_2", best_model_name))
    print(f"Best model (epoch {best_epoch}, val_loss {best_val_loss:.4f}) saved to training_runs_2/{best_model_name}")

    # Save final model (last epoch)
    final_model_name = f"final_model_bs{batch_size}_ep{num_epochs}_lr{lr}_{timestamp}.pth"
    torch.save(model.state_dict(), os.path.join("training_runs_2", final_model_name))
    print(f"Final model saved to training_runs_2/{final_model_name}")

def compute_attributions(
    model: nn.Module,
    video_feats: torch.Tensor,
    audio_feats: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    n_steps: int = 50,  # Number of integration steps (trade-off: accuracy vs. speed)
    baseline: str = "zeros",  # "zeros" or "mean" (dataset mean, but requires precomputed means)
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
        baseline: Input baseline for integration ("zeros" for black-box, or "mean" if you have global means).
    
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
    elif baseline == "mean":
        # Assuming you have precomputed global means from load_data (video_mean, audio_mean)
        # Modify load_data to return them if needed. For now, placeholder.
        video_baseline = torch.ones_like(video_feats) * video_mean.unsqueeze(0).unsqueeze(0)  # Broadcast
        audio_baseline = torch.ones_like(audio_feats) * audio_mean.unsqueeze(0)
    else:
        raise ValueError("Invalid baseline")

    # If target is None, predict it
    if target is None:
        with torch.no_grad():
            _, logits, _ = model(video_feats, audio_feats, mask=mask)
            target = logits.argmax(dim=-1)

    # Compute attributions (handles batches)
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
    abs_sum: bool = True,  # Use absolute values for importance magnitude
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

    # Sum over time for video (ignore padded via mask if needed, but here we assume post-attribution)
    video_importance = attr_video.sum(dim=1)  # (B, 768)

    audio_importance = attr_audio  # (B, 1024)

    return video_importance, audio_importance

def interpret_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    output_dir: str = "training_runs_2",
    top_k: int = 10,  # Report top-K most important features per modality
    global_average: bool = True,  # Also compute global (averaged over test set)
):
    """
    Compute and save feature importances for the test set.
    Saves per-sample and global importances as CSV/JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_video_importances = []
    all_audio_importances = []
    all_labels = []

    for videos, audios, labels, masks in test_loader:
        attr_video, attr_audio = compute_attributions(
            model, videos, audios, mask=masks, target=None, device=device
        )
        video_imp, audio_imp = aggregate_importances(attr_video, attr_audio)
        
        all_video_importances.append(video_imp.cpu())
        all_audio_importances.append(audio_imp.cpu())
        all_labels.append(labels.cpu())

    # Concatenate
    video_importances = torch.cat(all_video_importances, dim=0)  # (N_test, 768)
    audio_importances = torch.cat(all_audio_importances, dim=0)  # (N_test, 1024)
    labels = torch.cat(all_labels, dim=0)  # (N_test,)

    # Per-sample: Save as CSV (rows = samples, columns = features + label)
    video_df = pd.DataFrame(video_importances.numpy(), columns=[f"video_dim_{i}" for i in range(768)])
    video_df["label"] = labels.numpy()
    video_df.to_csv(os.path.join(output_dir, f"video_importances_{timestamp}.csv"), index=False)

    audio_df = pd.DataFrame(audio_importances.numpy(), columns=[f"audio_dim_{i}" for i in range(1024)])
    audio_df["label"] = labels.numpy()
    audio_df.to_csv(os.path.join(output_dir, f"audio_importances_{timestamp}.csv"), index=False)

    print(f"Per-sample importances saved to {output_dir}")

    # Global average (mean importance per feature across all samples)
    if global_average:
        global_video_imp = video_importances.mean(dim=0)  # (768,)
        global_audio_imp = audio_importances.mean(dim=0)  # (1024,)

        # Report top-K
        top_video_indices = global_video_imp.argsort(descending=True)[:top_k]
        top_audio_indices = global_audio_imp.argsort(descending=True)[:top_k]

        print("\nGlobal Top-K Video Feature Importances (dim index: score):")
        for idx in top_video_indices:
            print(f"Dim {idx}: {global_video_imp[idx]:.4f}")

        print("\nGlobal Top-K Audio Feature Importances (dim index: score):")
        for idx in top_audio_indices:
            print(f"Dim {idx}: {global_audio_imp[idx]:.4f}")

        # Save global as JSON
        global_results = {
            "global_video": {f"dim_{i}": float(global_video_imp[i]) for i in range(768)},
            "global_audio": {f"dim_{i}": float(global_audio_imp[i]) for i in range(1024)},
        }
        with open(os.path.join(output_dir, f"global_importances_{timestamp}.json"), "w") as f:
            json.dump(global_results, f, indent=4)
        print(f"Global importances saved to {output_dir}/global_importances_{timestamp}.json")

# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train multimodal emotion recognition model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    base_dir = "/home/sionna/evan/multi-modal-emotion-recognition"

    train_loader, val_loader, test_loader, max_chunks, class_weights = load_data(
        video_feat_dir=os.path.join(base_dir, "video_features"),
        audio_feat_dir=os.path.join(base_dir, "audio_features"),
        batch_size=args.batch_size,
    )

    print(
        f"Train samples: {len(train_loader.dataset)}, "
        f"Val samples: {len(val_loader.dataset)}, "
        f"Test samples: {len(test_loader.dataset)}"
    )

    # max_seq_len = max_chunks + 1 (extra token for audio)
    max_seq_len = max_chunks + 1 if max_chunks > 0 else 2  # minimum 2 if no data

    model = MultimodalEmotionModel(
        video_dim=768,
        audio_dim=1024,
        fused_dim=512,
        num_classes=6,
        max_seq_len=max_seq_len,
        fusion_num_layers=2,
        fusion_num_heads=8,
        fusion_dropout=0.1,
        classifier_hidden_dim=512,
        classifier_dropout=0.1,
    )

    train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        class_weights,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device="cuda",
    )
    
    interpret_test_set(model, test_loader, device="cuda")


if __name__ == "__main__":
    main()

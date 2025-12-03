import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import captum.attr as captum_attr  # For interpretability, install if needed, but assuming available or use native grads
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import glob
from sklearn.model_selection import train_test_split
import json
import argparse
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

class CrossModalFusion(nn.Module):
    """
    Attention-based fusion module for video (sequence) and audio (global) features.
    - Projects both to a common dimension.
    - Treats video as a sequence of tokens, audio as an additional token.
    - Uses transformer layers to fuse them via self-attention.
    - Outputs fused embedding (pooled), and attention weights for interpretability.
    """
    def __init__(self, video_dim=768, audio_dim=1024, fused_dim=512, num_layers=4, num_heads=8, dropout=0.6, max_seq_len=101):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)
        self.bn_video = nn.BatchNorm1d(fused_dim)
        self.bn_audio = nn.BatchNorm1d(fused_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, fused_dim))  # For max_chunks + audio token
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fused_dim, nhead=num_heads, dim_feedforward=2048, dropout=dropout),
            num_layers=num_layers
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # For pooling the sequence to single vector
        # Store for hyperparameters
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, video_feats, audio_feats, mask=None, return_attn=False):
        b, t, _ = video_feats.shape
        video = self.video_proj(video_feats)  # (b, t, fused_dim)
        video = video.permute(0, 2, 1)  # (b, fused_dim, t) for BatchNorm
        video = self.bn_video(video)
        video = video.permute(0, 2, 1)  # Back to (b, t, fused_dim)
        
        audio = self.audio_proj(audio_feats.unsqueeze(1))  # (b, 1, fused_dim)
        audio = audio.permute(0, 2, 1)  # (b, fused_dim, 1)
        audio = self.bn_audio(audio)
        audio = audio.permute(0, 2, 1)  # Back to (b, 1, fused_dim)
        
        combined = torch.cat([video, audio], dim=1)  # (b, t+1, fused_dim)
        combined = combined + self.pos_embed[:, :t+1, :]  # Slice from expanded pos_embed
        
        # Padding mask for transformer: (b, t+1), True=ignore
        if mask is not None:
            audio_mask = torch.zeros(b, 1, dtype=torch.bool, device=mask.device)  # False for audio (attend)
            full_mask = torch.cat([mask, audio_mask], dim=1)  # (b, t+1)
        else:
            full_mask = None
        
        combined = combined.permute(1, 0, 2)  # (t+1, b, fused_dim)
        
        # Transformer with mask
        if return_attn:
            # To get attn, need to extract from last layer; for simplicity, assume no return_attn support with mask or adjust
            transformer_out = self.transformer(combined, src_key_padding_mask=full_mask)
            attn_weights = None  # TODO: Extract if needed
        else:
            transformer_out = self.transformer(combined, src_key_padding_mask=full_mask)
            attn_weights = None
        
        fused = transformer_out.permute(1, 0, 2)  # (b, t+1, fused_dim)
        
        # Masked mean pooling (ignore padded)
        if full_mask is not None:
            attend_mask = (~full_mask).float().unsqueeze(-1)  # (b, t+1, 1), 1.0 for attend
            fused_embedding = (fused * attend_mask).sum(dim=1) / attend_mask.sum(dim=1).clamp(min=1e-6)
        else:
            fused_embedding = self.pool(fused.permute(0, 2, 1)).squeeze(-1)
        
        return fused_embedding, attn_weights

class EmotionClassifier(nn.Module):
    """
    Deeper classifier on top of fused embedding to add capacity.
    - Multiple FC layers with ReLU and dropout.
    """
    def __init__(self, input_dim=512, num_classes=6, dropout=0.6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.bn_fc1 = nn.BatchNorm1d(input_dim // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim // 2, num_classes)
        self.dropout2 = nn.Dropout(dropout)
        # Store for hyperparameters
        self.dropout = dropout

    def forward(self, fused_embedding):
        x = self.fc1(fused_embedding)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return probs, logits

# Example full model
class MultimodalEmotionModel(nn.Module):
    def __init__(self, video_dim=768, audio_dim=1024, fused_dim=512, num_classes=6, max_seq_len=101):
        super().__init__()
        self.fusion = CrossModalFusion(video_dim, audio_dim, fused_dim, dropout=0.6, max_seq_len=max_seq_len)
        self.classifier = EmotionClassifier(fused_dim, num_classes, dropout=0.6)

    def forward(self, video_feats, audio_feats, mask=None, return_attn=False):
        fused, attn_weights = self.fusion(video_feats, audio_feats, mask=mask, return_attn=return_attn)
        probs, logits = self.classifier(fused)
        return probs, logits, attn_weights
    
def load_data(video_feat_dir, audio_feat_dir, batch_size=32):
    video_features = []
    audio_features = []
    labels = []
    
    video_files = sorted(glob.glob(os.path.join(video_feat_dir, '*.npy')))
    audio_files = sorted(glob.glob(os.path.join(audio_feat_dir, '*.npy')))

    for v_file, a_file in zip(video_files, audio_files):
        basename = os.path.basename(v_file)
        if '-' in basename:
            # RAVDESS
            parts = basename.split('-')
            label_num = int(parts[2])
            if label_num in [2, 8]:
                continue
            # Map to final labels: 01->0 (NEU), 03->1 (HAP), 04->2 (SAD), 05->3 (ANG), 06->4 (FEA), 07->5 (DIS)
            ravdess_map = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
            label = ravdess_map[label_num]
        else:
            # CREMA-D
            emotion = basename.split('_')[2]
            cremad_map = {'ANG': 5, 'DIS': 7, 'FEA': 6, 'HAP': 3, 'NEU': 1, 'SAD': 4}
            label_num = cremad_map[emotion]
            # Map to final: 1->0 (NEU), 3->1 (HAP), 4->2 (SAD), 5->3 (ANG), 6->4 (FEA), 7->5 (DIS)
            cremad_to_final = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
            label = cremad_to_final[label_num]
        
        # Load features
        v_feat = np.load(v_file)
        a_feat = np.load(a_file)
        video_features.append(torch.tensor(v_feat, dtype=torch.float32))
        audio_features.append(torch.tensor(a_feat, dtype=torch.float32))
        labels.append(label)
    
    # Compute max_chunks
    if video_features:
        max_chunks = max(v.shape[0] for v in video_features)
    else:
        max_chunks = 0
    print(f"Maximum number of video chunks: {max_chunks}")
            
    # Dataset as list of tuples (variable-length videos ok)
    dataset = list(zip(video_features, audio_features, labels))
    
    # Indices for splits
    indices = list(range(len(dataset)))
    all_labels = [item[2] for item in dataset]
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=all_labels)
    temp_labels = [all_labels[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42, stratify=temp_labels)
    
    # Oversample label 0 in training set
    minority_indices = [i for i in train_indices if dataset[i][2] == 0]
    majority_count = 1170  # From your counter
    minority_count = len(minority_indices)
    oversample_factor = majority_count // minority_count
    additional_minority = minority_indices * (oversample_factor - 1)
    # Randomly sample to make exact match
    remaining = majority_count - (minority_count * oversample_factor)
    additional_minority += np.random.choice(minority_indices, remaining, replace=False).tolist()
    train_indices += additional_minority
    
    # Shuffle after oversampling
    np.random.shuffle(train_indices)
    
    # Custom collate function for padding and masking
    def collate_fn(batch):
        videos, audios, labels_batch = zip(*batch)
        # Pad videos (batch_first=True, pad with 0.0)
        videos_padded = pad_sequence(videos, batch_first=True, padding_value=0.0)
        # Stack audios (fixed size)
        audios_stacked = torch.stack(audios)
        # Tensor for labels
        labels_tensor = torch.tensor(labels_batch, dtype=torch.long)
        # Mask: False for real positions (attend), True for padded (ignore)
        masks = [torch.zeros(len(v), dtype=torch.bool) for v in videos]
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=True)
        return videos_padded, audios_stacked, labels_tensor, masks_padded
    
    # DataLoaders with collate_fn
    train_loader = DataLoader([dataset[i] for i in train_indices], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader([dataset[i] for i in val_indices], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader([dataset[i] for i in test_indices], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Print label distributions (updated for oversampled train)
    train_labels = [dataset[i][2] for i in train_indices]
    val_labels = [dataset[i][2] for i in val_indices]
    test_labels = [dataset[i][2] for i in test_indices]
    
    print("Train label distribution (after oversampling):", Counter(train_labels))
    print("Val label distribution:", Counter(val_labels))
    print("Test label distribution:", Counter(test_labels))
    
    # Compute class weights for imbalanced classes (optional, but can combine with oversampling)
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    return train_loader, val_loader, test_loader, max_chunks, class_weights
    

def train_model(model, train_loader, val_loader, test_loader, class_weights, num_epochs=10, lr=1e-4, weight_decay=2e-4, patience=100, batch_size=16, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # Add class weights
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
    
    # Collect all hyperparameters
    hyperparameters = {
        'num_epochs': num_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'patience': patience,
        'batch_size': batch_size,
        'device': device,
        'video_dim': model.fusion.video_proj.in_features,
        'audio_dim': model.fusion.audio_proj.in_features,
        'fused_dim': model.fusion.video_proj.out_features,
        'num_classes': model.classifier.fc2.out_features,
        'max_seq_len': model.fusion.pos_embed.size(1),
        'fusion_dropout': model.fusion.dropout,
        'classifier_dropout': model.classifier.dropout,
        'num_layers': model.fusion.num_layers,
        'num_heads': model.fusion.num_heads,
        'scheduler_factor': 0.1,
        'scheduler_patience': 20
    }
    
    results = []
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for videos, audios, labels, masks in train_loader:
            videos = videos.to(device)
            audios = audios.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            probs, logits, _ = model(videos, audios, mask=masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
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
                
                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        scheduler.step(avg_val_loss)
        
        # Check if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Compute additional metrics for validation
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels_list, all_preds, average='macro')
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels_list, all_preds, average='micro')
        
        # Test evaluation (no loss calculation)
        test_preds = []
        test_labels_list = []
        with torch.no_grad():
            for videos, audios, labels, masks in test_loader:
                videos = videos.to(device)
                audios = audios.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                probs, _, _ = model(videos, audios, mask=masks)
                
                _, predicted = torch.max(probs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())
        
        test_acc = 100 * sum(1 for p, l in zip(test_preds, test_labels_list) if p == l) / len(test_labels_list)
        test_macro_precision, test_macro_recall, test_macro_f1, _ = precision_recall_fscore_support(test_labels_list, test_preds, average='macro')
        test_micro_precision, test_micro_recall, test_micro_f1, _ = precision_recall_fscore_support(test_labels_list, test_preds, average='micro')
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Macro P/R/F1: {macro_precision:.4f}/{macro_recall:.4f}/{macro_f1:.4f}, Micro P/R/F1: {micro_precision:.4f}/{micro_recall:.4f}/{micro_f1:.4f}")
        print(f"Test Acc: {test_acc:.2f}%, Test Macro P/R/F1: {test_macro_precision:.4f}/{test_macro_recall:.4f}/{test_macro_f1:.4f}, Micro P/R/F1: {test_micro_precision:.4f}/{test_micro_recall:.4f}/{test_micro_f1:.4f}")
        
        # Save results
        results.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'val_macro_precision': macro_precision,
            'val_macro_recall': macro_recall,
            'val_macro_f1': macro_f1,
            'val_micro_precision': micro_precision,
            'val_micro_recall': micro_recall,
            'val_micro_f1': micro_f1,
            'test_acc': test_acc,
            'test_macro_precision': test_macro_precision,
            'test_macro_recall': test_macro_recall,
            'test_macro_f1': test_macro_f1,
            'test_micro_precision': test_micro_precision,
            'test_micro_recall': test_micro_recall,
            'test_micro_f1': test_micro_f1
        })
    
    # Create training_runs folder
    os.makedirs('training_runs', exist_ok=True)
    
    # Save to JSON with hyperparameters in name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_name = f'results_bs{batch_size}_ep{num_epochs}_lr{lr}_{timestamp}.json'
    with open(os.path.join('training_runs', results_name), 'w') as f:
        json.dump({
            "training_progress": results,
            "best_model": {
                "epoch": best_epoch
            },
            "hyperparameters": hyperparameters
        }, f, indent=4)
    print(f"Training results saved to training_runs/{results_name}")
    
    # Save the best model (based on lowest validation loss)
    best_model_name = f'best_model_bs{batch_size}_ep{num_epochs}_lr{lr}_{timestamp}.pth'
    torch.save(best_model_state, os.path.join('training_runs', best_model_name))
    print(f"Best model (epoch {best_epoch}, val_loss {best_val_loss:.4f}) saved to training_runs/{best_model_name}")
    
    # Save the final model
    final_model_name = f'final_model_bs{batch_size}_ep{num_epochs}_lr{lr}_{timestamp}.pth'
    torch.save(model.state_dict(), os.path.join('training_runs', final_model_name))
    print(f"Final model saved to training_runs/{final_model_name}")

def main():
    parser = argparse.ArgumentParser(description='Train multimodal emotion recognition model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader, max_chunks, class_weights = load_data(
        video_feat_dir="/home/sionna/evan/multi-modal-emotion-recognition/video_features",
        audio_feat_dir="/home/sionna/evan/multi-modal-emotion-recognition/audio_features",
        batch_size=args.batch_size
    )
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Initialize model with max_seq_len = max_chunks + 1 (for audio token)
    max_seq_len = max_chunks + 1 if max_chunks > 0 else 2  # Minimum 2 if no data
    model = MultimodalEmotionModel(video_dim=768, audio_dim=1024, fused_dim=512, num_classes=6, max_seq_len=max_seq_len)
    
    # Train the model
    train_model(model, train_loader, val_loader, test_loader, class_weights, num_epochs=args.num_epochs, lr=args.lr, batch_size=args.batch_size, device='cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    main()
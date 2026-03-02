# Part 2: Implementation Guide — Step-by-Step
## Addiction Risk Scoring via Dual-Head Temporal Attention Network (DT-AttNet)

> **Stack:** Python 3.10+ | PyTorch | Google Colab (T4 GPU, free tier) or Kaggle GPU
>
> **Deliverables:** GitHub repo (code + README) + PDF report (3,000 words, 60% of module mark)

---

## Project Repository Structure

```
addiction-risk-dtattnet/
├── README.md                    # Setup + reproduction instructions
├── requirements.txt             # Dependencies
├── config.yaml                  # All hyperparameters in one place
├── data/
│   ├── download_data.py         # Script to download datasets from Kaggle API
│   └── README.md                # Data provenance & licensing info
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_Training.ipynb        # Full training pipeline (run top-to-bottom)
│   └── 03_Evaluation.ipynb      # Metrics, plots, ablation results
├── src/
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset & DataLoader
│   ├── preprocessing.py         # Feature engineering, normalization, sequencing
│   ├── model.py                 # DT-AttNet architecture
│   ├── losses.py                # Combined CE + SupCon loss
│   ├── train.py                 # Training loop with early stopping
│   ├── evaluate.py              # Metrics computation
│   └── utils.py                 # Seed setting, logging, device management
├── experiments/
│   ├── baseline_mlp/            # Baseline comparison experiment
│   └── ablation/                # Ablation study configs and results
├── results/
│   ├── figures/                 # Saved plots (learning curves, confusion matrices)
│   ├── attention_maps/          # Attention weight visualizations
│   └── metrics.json             # Final evaluation metrics
└── LICENSE
```

---

## Step 1: Dataset Selection & Acquisition

### Primary Dataset
**Social Media Addiction & Usage Patterns Dataset** (Kaggle)
- URL: `https://www.kaggle.com/datasets/` (search: "Social Media Addiction Usage Patterns")
- ~1000+ records, CSV format, CC0 Public Domain
- Features: daily_usage_duration, preferred_platform, night_usage, long_term_exposure, mental_health_score, productivity_impact, addiction_level

### Secondary Dataset (for robustness)
**Smartphone Usage & Addiction Prediction Dataset** (Kaggle)
- ~7,500 records, CSV format, CC0 Public Domain
- Features: screen_time, social_media_usage, gaming_activity, sleep_patterns, stress_levels, productivity_impact, addiction_severity_label

### Download Script (`data/download_data.py`)
```python
import os
import subprocess

def download_datasets():
    """Download datasets using Kaggle API. Requires ~/.kaggle/kaggle.json"""
    datasets = [
        "your-dataset-slug/social-media-addiction-usage",   # Replace with actual slug
        "your-dataset-slug/smartphone-addiction-prediction"  # Replace with actual slug
    ]
    os.makedirs("data/raw", exist_ok=True)
    for ds in datasets:
        subprocess.run(["kaggle", "datasets", "download", "-d", ds, "-p", "data/raw", "--unzip"])

if __name__ == "__main__":
    download_datasets()
```

---

## Step 2: Exploratory Data Analysis (EDA)

Perform in `notebooks/01_EDA.ipynb`:

### 2.1 Basic Statistics
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/social_media_addiction.csv")
print(df.shape)
print(df.describe())
print(df['addiction_level'].value_counts())  # Class distribution
```

### 2.2 Key Analyses to Perform
1. **Class distribution** of addiction levels → check for imbalance (expect it; justify contrastive loss)
2. **Correlation heatmap** of all features → identify multicollinearity
3. **Temporal/usage pattern distributions** by addiction level → boxplots
4. **Feature importance** via Random Forest → sanity check which features matter before DL

### 2.3 Visualizations to Save (for the report)
- Class distribution bar chart
- Correlation heatmap
- Pairplot of top-5 features colored by addiction level
- Usage duration distribution across addiction classes

---

## Step 3: Data Preprocessing (`src/preprocessing.py`)

### 3.1 Feature Engineering
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from raw data."""
    # 1. Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if col != 'addiction_level':
            df[col] = le.fit_transform(df[col].astype(str))

    # 2. Create derived temporal features
    if 'daily_usage_hours' in df.columns:
        df['usage_intensity'] = df['daily_usage_hours'] * df.get('sessions_per_day', 1)
        df['avg_session_length'] = df['daily_usage_hours'] / df.get('sessions_per_day', 1).replace(0, 1)

    # 3. Normalize numerical features
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('addiction_level', errors='ignore')
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
```

### 3.2 Sequence Construction (for temporal model)
```python
def create_sequences(df: pd.DataFrame, seq_len: int = 10) -> tuple:
    """
    Convert flat records into sliding-window sequences.
    Since dataset is cross-sectional, we simulate temporal sequences
    by grouping users and creating synthetic time-steps with augmentation.
    """
    # For cross-sectional data: use feature-level windowing
    # Each sample becomes a (seq_len, n_features) tensor
    features = df.drop(columns=['addiction_level']).values
    labels = df['addiction_level'].values

    X_seq, y_seq = [], []
    for i in range(len(features)):
        # Create pseudo-sequence via feature perturbation (data augmentation)
        base = features[i]
        seq = [base + np.random.normal(0, 0.05, base.shape) for _ in range(seq_len)]
        X_seq.append(np.array(seq))
        y_seq.append(labels[i])

    return np.array(X_seq), np.array(y_seq)
```

### 3.3 Train/Val/Test Split
```python
from sklearn.model_selection import train_test_split

# Stratified split: 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```

---

## Step 4: PyTorch Dataset & DataLoader (`src/dataset.py`)

```python
import torch
from torch.utils.data import Dataset, DataLoader

class AddictionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)   # Shape: (N, seq_len, n_features)
        self.y = torch.LongTensor(y)    # Shape: (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    train_ds = AddictionDataset(X_train, y_train)
    val_ds   = AddictionDataset(X_val, y_val)
    test_ds  = AddictionDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
```

---

## Step 5: Model Architecture (`src/model.py`)

### 5.1 DT-AttNet — Dual-Head Temporal Attention Network
```python
import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """1D-CNN + BiLSTM for temporal pattern extraction."""
    def __init__(self, input_dim, cnn_filters=64, lstm_hidden=128, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, cnn_filters, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)          # → (batch, features, seq_len) for Conv1d
        x = self.relu(self.bn(self.conv1d(x)))
        x = x.permute(0, 2, 1)          # → (batch, seq_len, cnn_filters) for LSTM
        output, _ = self.bilstm(x)      # → (batch, seq_len, 2*lstm_hidden)
        return output


class AttentionRiskHead(nn.Module):
    """Multi-head self-attention for risk-relevant temporal weighting."""
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(x + attn_out)     # Residual connection
        return x, attn_weights


class DTAttNet(nn.Module):
    """Dual-Head Temporal Attention Network for Addiction Risk Scoring."""
    def __init__(self, input_dim, num_classes=3, cnn_filters=64,
                 lstm_hidden=128, n_heads=4, dropout=0.3):
        super().__init__()
        d_model = lstm_hidden * 2   # BiLSTM output dim

        self.temporal_encoder = TemporalEncoder(input_dim, cnn_filters, lstm_hidden)
        self.attention_head = AttentionRiskHead(d_model, n_heads)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        # Projection head for contrastive loss (during training only)
        self.projector = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x, return_attention=False):
        temporal_out = self.temporal_encoder(x)               # (B, T, 2H)
        attn_out, attn_weights = self.attention_head(temporal_out)  # (B, T, 2H)

        # Global average pooling over temporal dimension
        pooled = attn_out.mean(dim=1)   # (B, 2H)

        logits = self.classifier(pooled)       # (B, num_classes)
        projections = self.projector(pooled)   # (B, 64) — for contrastive loss

        if return_attention:
            return logits, projections, attn_weights
        return logits, projections
```

### 5.2 Baseline MLP (for comparison)
```python
class BaselineMLP(nn.Module):
    """Simple MLP baseline for ablation comparison."""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)     # Flatten sequence
        return self.net(x), None
```

---

## Step 6: Loss Functions (`src/losses.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, projection_dim), labels: (B,)
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask: 1 where labels match, 0 otherwise
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()
        mask.fill_diagonal_(0)  # Exclude self-similarity

        # Log-sum-exp trick for numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob.mean()
        return loss


class CombinedLoss(nn.Module):
    """CrossEntropy + λ * SupConLoss."""
    def __init__(self, lambda_con=0.5, temperature=0.07, class_weights=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.con_loss = SupConLoss(temperature)
        self.lambda_con = lambda_con

    def forward(self, logits, projections, labels):
        ce = self.ce_loss(logits, labels)
        con = self.con_loss(projections, labels)
        return ce + self.lambda_con * con, ce.item(), con.item()
```

---

## Step 7: Training Loop (`src/train.py`)

```python
import torch
import json
from pathlib import Path

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_ce, total_con = 0, 0, 0
    correct, total = 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits, projections = model(X_batch)
        loss, ce, con = criterion(logits, projections, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce
        total_con += con
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    n = len(loader)
    return total_loss/n, total_ce/n, total_con/n, correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits, projections = model(X_batch)
        loss, _, _ = criterion(logits, projections, y_batch)

        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())

    return total_loss/len(loader), correct/total, all_preds, all_labels


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs=100, patience=15, save_dir="results"):
    """Full training loop with early stopping and checkpointing."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, ce, con, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | CE: {ce:.4f} | Con: {con:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    with open(f"{save_dir}/history.json", "w") as f:
        json.dump(history, f)

    return history
```

---

## Step 8: Evaluation & Metrics (`src/evaluate.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
import json

def full_evaluation(model, test_loader, criterion, device, class_names, save_dir="results/figures"):
    """Comprehensive evaluation with all metrics and visualizations."""
    import os; os.makedirs(save_dir, exist_ok=True)

    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    # 1. Classification Report
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    print(classification_report(labels, preds, target_names=class_names))

    # 2. Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — DT-AttNet')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    # 3. Metrics summary
    metrics = {
        'accuracy': test_acc,
        'macro_f1': f1_score(labels, preds, average='macro'),
        'weighted_f1': f1_score(labels, preds, average='weighted'),
        'macro_precision': precision_score(labels, preds, average='macro'),
        'macro_recall': recall_score(labels, preds, average='macro'),
        'per_class_report': report
    }

    with open(f"{save_dir}/../metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def plot_training_curves(history, save_dir="results/figures"):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_title('Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.close()


def visualize_attention(model, test_loader, device, save_dir="results/attention_maps"):
    """Extract and visualize attention weights for interpretability."""
    import os; os.makedirs(save_dir, exist_ok=True)
    model.eval()

    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch.to(device)
    _, _, attn_weights = model(X_batch, return_attention=True)

    # Average attention across heads: (B, T, T) → take first 5 samples
    for i in range(min(5, attn_weights.size(0))):
        attn = attn_weights[i].cpu().detach().numpy()
        plt.figure(figsize=(8, 6))
        sns.heatmap(attn, cmap='viridis')
        plt.title(f'Attention Map — Sample {i} (Label: {y_batch[i].item()})')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/attention_sample_{i}.png", dpi=150)
        plt.close()
```

---

## Step 9: Ablation Studies & Experiments

### 9.1 Experiments to Run

| Experiment | Model | Loss | Purpose |
|---|---|---|---|
| E1: Baseline MLP | BaselineMLP | CrossEntropy | Baseline comparison |
| E2: DT-AttNet (CE only) | DTAttNet | CrossEntropy only | Ablation: is contrastive loss needed? |
| E3: DT-AttNet (CE + SupCon) | DTAttNet | CombinedLoss | Full proposed model |
| E4: DT-AttNet (no attention) | DTAttNet minus attention | CombinedLoss | Ablation: is attention needed? |
| E5: DT-AttNet (λ sweep) | DTAttNet | CombinedLoss (λ=0.1,0.5,1.0) | Hyperparam sensitivity |

### 9.2 Hyperparameter Configuration (`config.yaml`)
```yaml
data:
  seq_len: 10
  batch_size: 32
  test_size: 0.3
  val_size: 0.5   # of remaining after test split

model:
  cnn_filters: 64
  lstm_hidden: 128
  n_heads: 4
  dropout: 0.3
  num_classes: 3

training:
  epochs: 100
  patience: 15
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: ReduceLROnPlateau
  scheduler_patience: 5
  grad_clip: 1.0

loss:
  lambda_con: 0.5
  temperature: 0.07

seed: 42
```

### 9.3 Results Table (template for your report)
```markdown
| Model | Accuracy | Macro-F1 | Precision | Recall | Notes |
|-------|----------|----------|-----------|--------|-------|
| Baseline MLP | -- | -- | -- | -- | No temporal modeling |
| DT-AttNet (CE) | -- | -- | -- | -- | Ablation |
| DT-AttNet (CE+SupCon) | -- | -- | -- | -- | **Full proposed** |
| DT-AttNet (no attn) | -- | -- | -- | -- | Ablation |
```

---

## Step 10: Main Execution Script

### For Colab/Kaggle Notebook (`notebooks/02_Training.ipynb`)
```python
import torch
import yaml
import sys
sys.path.append('..')

from src.preprocessing import preprocess, create_sequences
from src.dataset import get_dataloaders
from src.model import DTAttNet, BaselineMLP
from src.losses import CombinedLoss
from src.train import train
from src.evaluate import full_evaluation, plot_training_curves, visualize_attention

# 1. Load config
with open('../config.yaml') as f:
    cfg = yaml.safe_load(f)

# 2. Set seed for reproducibility
torch.manual_seed(cfg['seed'])

# 3. Load and preprocess data
import pandas as pd
df = pd.read_csv('../data/raw/social_media_addiction.csv')
df = preprocess(df)
X, y = create_sequences(df, seq_len=cfg['data']['seq_len'])

# 4. Split and create loaders
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
train_loader, val_loader, test_loader = get_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=cfg['data']['batch_size'])

# 5. Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train.shape[2]
model = DTAttNet(input_dim=input_dim, **{k: cfg['model'][k] for k in ['cnn_filters','lstm_hidden','n_heads','dropout','num_classes']}).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 6. Loss, optimizer, scheduler
criterion = CombinedLoss(lambda_con=cfg['loss']['lambda_con'], temperature=cfg['loss']['temperature'])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'],
                              weight_decay=cfg['training']['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg['training']['scheduler_patience'])

# 7. Train
history = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs=cfg['training']['epochs'], patience=cfg['training']['patience'])

# 8. Evaluate
model.load_state_dict(torch.load('results/best_model.pt'))
metrics = full_evaluation(model, test_loader, criterion, device,
                           class_names=['Low', 'Moderate', 'High'])
plot_training_curves(history)
visualize_attention(model, test_loader, device)

print(f"\n{'='*50}")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1:      {metrics['macro_f1']:.4f}")
```

---

## Step 11: Report Structure for Part 2 (3,000 words)

### Section Breakdown

| Section | Words | Content |
|---------|-------|---------|
| **Title Page** | — | Title, name, student number, module, date |
| **Abstract** | ~150 | Problem, approach (DT-AttNet), key result (accuracy/F1) |
| **1. Introduction** | ~300 | Link to Part 1 thesis, state aims: build a DL model to quantify addiction risk |
| **2. Background** | ~400 | Review: social media addiction detection ML work, time-series classification, contrastive learning |
| **3. Methodology** | ~700 | Dataset, preprocessing, DT-AttNet architecture, loss function, training strategy |
| **4. Experiments & Results** | ~600 | Baseline vs. DT-AttNet, ablation table, training curves, confusion matrix, attention maps |
| **5. Discussion** | ~500 | Why it worked/didn't, challenges (data limitations, synthetic sequences), hypothesis outcomes |
| **6. Ethics & Scalability** | ~200 | Privacy of behavioral data, surveillance risk, GDPR, computational cost analysis |
| **7. Conclusion** | ~150 | Summary of contributions, limitations, future work |
| **References** | — | IEEE format, ~15-20 sources |

---

## Step 12: Benchmarking & Comparison Standards

### Metrics to Report
- **Primary:** Accuracy, Macro-F1, Weighted-F1
- **Per-class:** Precision, Recall, F1 for each addiction level
- **Advanced (distinction-level):** ROC-AUC (one-vs-rest), attention weight statistics

### Baseline Comparisons
1. **Random baseline** — expected ~33% accuracy (3-class)
2. **Logistic Regression** — classical ML sanity check
3. **BaselineMLP** — deep learning baseline without temporal modeling
4. **DT-AttNet (CE only)** — ablation of contrastive loss

---

## Step 13: README.md Template

```markdown
# Addiction Risk Scoring via DT-AttNet

Dual-Head Temporal Attention Network for predicting digital addiction risk from
user behavioral patterns. Built as part of MSc AI coursework on Deep Learning
and Generative AI (CMP030L043).

## Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.x
- CUDA (optional, but recommended)

### Installation
```bash
pip install -r requirements.txt
```

### Data Setup
1. Place Kaggle API key at `~/.kaggle/kaggle.json`
2. Run: `python data/download_data.py`

### Training
```bash
# Full training pipeline
python -m src.train
# Or use the notebook: notebooks/02_Training.ipynb
```

### Evaluation
```bash
python -m src.evaluate
```

## Architecture
DT-AttNet: 1D-CNN → BiLSTM → Multi-Head Self-Attention → Classification + Contrastive Head

## Results
| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Baseline MLP | -- | -- |
| DT-AttNet (Full) | -- | -- |

## License
MIT
```

---

## Colab/Kaggle Tips (Free GPU)

> [!TIP]
> - **Kaggle:** 30 hours/week of free T4 GPU. Use `!pip install` at notebook start
> - **Colab:** T4 GPU free tier. Mount Google Drive for persistence: `drive.mount('/content/drive')`
> - **Save checkpoints to Drive/Kaggle output** to avoid losing progress on session timeout
> - **Keep models small:** DT-AttNet with default config is ~500K parameters — trains in minutes on T4
> - **Run all cells top-to-bottom** before submission so outputs are visible to markers
> - **Pin exact library versions** in `requirements.txt` for reproducibility

---

## Critical Distinctions for High Marks

> [!IMPORTANT]
> - **Ablation studies** are what separate Distinction from Merit — show that each component matters
> - **Attention visualizations** demonstrate interpretability — link weights back to the "shortened loop" thesis
> - **Error analysis** — don't just report metrics; discuss *which samples* the model misclassifies and *why*
> - **Acknowledge the synthetic sequence limitation** openly in Discussion — shows maturity
> - **Ethics section** must be specific to YOUR project, not generic AI ethics boilerplate

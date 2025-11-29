"""
Description
-----------
This script trains and evaluates a 14-class BioMedCLIP (ViT-based) classifier
on a balanced MIMIC-CXR subset. It builds PNG-based datasets from merged label
and DICOM metadata CSVs, filters for frontal AP/PA views, applies OpenCLIP’s
standard preprocessing pipeline, and optimizes a weighted Binary Cross-Entropy
loss with per-class pos_weight.

Inputs
------
- mimic_cxr_balanced_subset_v2.csv
  Columns include:
    - subject_id
    - study_id
    - 14 CheXpert-style label columns (0/1/NaN)

- dcm_info.csv
  Columns include:
    - subject_id
    - study_id
    - image_path (rooted under "images/...")
    - ViewPosition (filtered to ["AP", "PA"])

- Image directory
  - All PNG files reachable under:
    dataset downloading/Dataset/images/

Outputs
-------
- biomedclip_14class_weightedbce.pth
  Fine-tuned BioMedCLIP classifier weights (14-class multilabel).

- evaluation_csvs/biomedclip_14class_weightedbce_perclass.csv
  Per-label ROC AUC, PR AUC, best F1, and best threshold.

- evaluation_csvs/biomedclip_14class_weightedbce_summary.csv
  Global metrics: mean ROC/PR AUC, macro/micro F1, precision, recall,
  micro ROC AUC, and ECE.

- evaluation_csvs/biomedclip_14class_weightedbce_top5_pr.png
  Precision–recall curves for the top-5 labels by PR AUC.

Notes
-----
- Uses BioMedCLIP (“microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224”)
  via OpenCLIP as the visual backbone.
- Pos-weights are estimated from the *training* labels only, clipped and
  normalized for numerical stability.
- All labels are sanitized: NaNs → 0, values clipped to [0, 1].
- Augmentations are not applied manually; OpenCLIP provides preprocessing
  pipelines for both train and eval splits.
"""

# ===============================================================
# Imports and Device
# ===============================================================
import gc
import os
import re  # kept for symmetry, even if not used heavily

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
# Configuration & Paths
# ===============================================================
LABEL_CSV = "dataset downloading/Dataset/mimic_cxr_balanced_subset_v2.csv"
DCM_CSV = "dataset downloading/Dataset/dcm_info.csv"
IMAGE_BASE = "dataset downloading/Dataset/images"

# 14 classification labels
label_cols = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
    "Pleural Other",
    "Support Devices",
    "No Finding",
]

# Checkpoint / evaluation paths
CKPT_PATH = "biomedclip_14class_weightedbce.pth"
EVAL_DIR = "evaluation_csvs"
os.makedirs(EVAL_DIR, exist_ok=True)


# ===============================================================
# Dataset Definition
# ===============================================================
class CXRDataset(Dataset):
    """
    Simple image + multi-label dataset.

    Each row in df must contain:
        - "image_path": absolute or base-resolved path to the CXR image
        - label_cols : float/int columns in [0, 1] for 14 labels

    transform: torchvision-like callable applied to PIL.Image.
    """

    def __init__(self, df: pd.DataFrame, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return img, label


# ===============================================================
# Load CSVs, Merge Metadata, and Build Splits
# ===============================================================
labels = pd.read_csv(LABEL_CSV)
dcm_info = pd.read_csv(DCM_CSV)

# keep AP/PA views only
dcm_info = dcm_info[dcm_info["ViewPosition"].isin(["AP", "PA"])]

# inner join on subject + study
merged = pd.merge(labels, dcm_info, on=["subject_id", "study_id"], how="inner")

# keep only path + labels
merged = merged[["image_path"] + label_cols]

# fix image base path
merged["image_path"] = merged["image_path"].str.replace(
    r"^images", IMAGE_BASE, regex=True
)

# train / val / test split (70 / 15 / 15)
train_df, temp_df = train_test_split(merged, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# ===============================================================
# Label Cleaning (NaNs, clipping)
# ===============================================================
for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    bad = df[label_cols].isna().sum().sum()
    print(f"{df_name}: NaNs in labels ->", bad)
    if bad > 0:
        df[label_cols] = df[label_cols].fillna(0)
    df[label_cols] = df[label_cols].clip(0, 1)

print("Val NaNs after cleaning:", val_df[label_cols].isna().sum().sum())


# ===============================================================
# OpenCLIP Transforms & DataLoaders
# ===============================================================
# Use OpenCLIP's built-in transforms for BioMedCLIP
_, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

train_loader = DataLoader(
    CXRDataset(train_df, label_cols, preprocess_train),
    batch_size=32,
    shuffle=True,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
)

val_loader = DataLoader(
    CXRDataset(val_df, label_cols, preprocess_val),
    batch_size=32,
    shuffle=False,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
)

test_loader = DataLoader(
    CXRDataset(test_df, label_cols, preprocess_val),
    batch_size=32,
    shuffle=False,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
)

print("✅ DataLoaders ready.")


# ===============================================================
# BioMedCLIP-based Classifier
# ===============================================================
class BioMedCLIP_CXR(nn.Module):
    """
    14-class classifier on top of BioMedCLIP's visual tower.

    Uses open_clip.create_model_and_transforms(...) internally to load the
    pretrained visual encoder, then attaches a linear head.
    """

    def __init__(self, num_classes=14):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.visual = self.model.visual

        # infer output dim automatically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.visual(dummy)
            in_features = out.shape[-1]

        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.visual(x)  # [B, D]
        return self.head(x)


num_labels = len(label_cols)
model = BioMedCLIP_CXR(num_classes=num_labels).to(DEVICE)

# quick sanity check
x = torch.randn(2, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    y = model(x)
print("✅ BioMedCLIP_CXR output shape:", y.shape)


# ===============================================================
# Weighted BCE Loss & pos_weight Computation
# ===============================================================
class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy loss (no focal term).
    Supports per-class positive weights (pos_weight) for imbalance correction.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        base_loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        if self.pos_weight is not None:
            # weight positive samples by pos_weight
            weight = 1.0 + (self.pos_weight - 1.0) * targets
            base_loss = base_loss * weight

        return base_loss.mean()


def compute_stable_pos_weight(train_df, label_cols, clip_lo=0.5, clip_hi=3.0, eps=1e-6):
    """
    Computes a numerically stable per-label pos_weight vector from the training set only.

    Steps:
    - pos_weight = (negatives / positives)
    - Avoid div-by-zero with eps
    - Clip to [clip_lo, clip_hi] to prevent overweighting rare labels
    - Normalize by mean to keep overall loss scale stable
    """
    pos = (train_df[label_cols] == 1).sum()
    neg = (train_df[label_cols] == 0).sum()

    pw = (neg / (pos + eps)).astype(float)
    pw = pw.clip(lower=clip_lo, upper=clip_hi)

    mean_pw = pw.mean()
    pw = (pw / (mean_pw + eps)).round(3)

    print("=== pos_weight (train only; clipped & normalized) ===")
    print(pw)

    return tf.constant(pw.values, dtype=tf.float32)


# Compute pos_weight (TensorFlow → PyTorch)
pos_weight_tf = compute_stable_pos_weight(train_df, label_cols)
pos_weight_torch = torch.tensor(pos_weight_tf.numpy(), dtype=torch.float32).to(DEVICE)

criterion = WeightedBCELoss(pos_weight=pos_weight_torch).to(DEVICE)


# ===============================================================
# Optimizer & Scheduler
# ===============================================================
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2
)


# ===============================================================
# Training & Validation Loops
# ===============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        labels = labels.float().squeeze()
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        labels = labels.float().squeeze()
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        outputs = torch.sigmoid(model(imgs))
        preds.append(outputs.cpu())
        trues.append(labels.cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    aucs = []
    for i in range(trues.shape[1]):
        try:
            aucs.append(roc_auc_score(trues[:, i], preds[:, i]))
        except Exception:
            aucs.append(float("nan"))
    return np.nanmean(aucs)


# ===============================================================
# Training Loop
# ===============================================================
EPOCHS = 20
best_auc = 0.0

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_auc = evaluate(model, val_loader)

    scheduler.step(epoch + val_auc * 0.1)  # mild modulation by val AUC

    print(
        f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), CKPT_PATH)
        print("✅ Model improved and saved.")
    else:
        print("⏸ No improvement.")

print(f"🏁 Training complete. Best Val AUC: {best_auc:.4f}")


# ===============================================================
# Evaluation on Test Set (Reload Best Checkpoint)
# ===============================================================
# Recreate the same BioMedCLIP_CXR architecture
eval_model = BioMedCLIP_CXR(num_classes=num_labels).to(DEVICE)

state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
eval_model.load_state_dict(state_dict)
eval_model.eval()

# run inference on test set
all_probs, all_labels = [], []
print("Running inference on test set...")
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = eval_model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

# stack predictions and ground truths
preds = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(all_labels, axis=0)

# ===============================================================
# Per-class Metrics
# ===============================================================
results = []
for i, c in enumerate(label_cols):
    try:
        y_pred = preds[:, i]
        roc = roc_auc_score(y_true[:, i], y_pred)
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred)
        pr_auc = auc(recall, precision)

        # compute best F1 threshold
        f1_vals = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_vals)
        best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_vals[best_idx]

        results.append(
            {
                "Label": c,
                "ROC_AUC": roc,
                "PR_AUC": pr_auc,
                "Best_F1": best_f1,
                "Best_Threshold": best_thr,
            }
        )
    except ValueError:
        print("Skipping", c, ": only one class present")
        continue

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
print("\nPer-label metrics:")
print(results_df.round(3))


# ===============================================================
# Global Metrics & Calibration
# ===============================================================
# global metrics using 0.5 threshold
y_pred_bin = (preds > 0.5).astype(int)
macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
macro_prec = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)
macro_rec = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)
micro_roc = roc_auc_score(y_true.ravel(), preds.ravel())

# calibration (ECE)
prob_true, prob_pred = calibration_curve(y_true.ravel(), preds.ravel(), n_bins=10)
ece = np.abs(prob_true - prob_pred).mean()

summary_metrics = {
    "Mean ROC_AUC": results_df["ROC_AUC"].mean(),
    "Mean PR_AUC": results_df["PR_AUC"].mean(),
    "Mean F1": results_df["Best_F1"].mean(),
    "Macro F1@0.5": macro_f1,
    "Micro F1@0.5": micro_f1,
    "Macro Precision@0.5": macro_prec,
    "Macro Recall@0.5": macro_rec,
    "Micro ROC_AUC": micro_roc,
    "ECE": ece,
}
summary_df = pd.DataFrame([summary_metrics])
print("\nGlobal summary:")
print(summary_df.round(4))


# ===============================================================
# Export CSVs & Top-5 PR Curves
# ===============================================================
model_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]

results_path = os.path.join(EVAL_DIR, f"{model_name}_perclass.csv")
summary_path = os.path.join(EVAL_DIR, f"{model_name}_summary.csv")
prcurve_path = os.path.join(EVAL_DIR, f"{model_name}_top5_pr.png")

results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)
print("Saved metrics:", results_path)
print("Saved summary:", summary_path)

# Plot and export Top-5 PR curves
plt.figure(figsize=(6, 5))
for _, row in results_df.head(5).iterrows():
    i = label_cols.index(row["Label"])
    precision, recall, _ = precision_recall_curve(y_true[:, i], preds[:, i])
    plt.plot(recall, precision, label=f'{row["Label"]} (AUC={row["PR_AUC"]:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Top-5 PR Curves — BioMedCLIP")
plt.legend()
plt.tight_layout()
plt.savefig(prcurve_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved PR figure:", prcurve_path)

# free GPU memory
torch.cuda.empty_cache()
gc.collect()

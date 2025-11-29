"""
Description
-----------
This script trains and evaluates a 14-class SwinV2-Large classifier on a
balanced MIMIC-CXR subset. It builds PNG-based datasets from merged label
and DICOM metadata CSVs, applies weighted Binary Cross-Entropy loss with
per-class pos_weight, and reports per-class and global metrics.

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
    IMAGE_BASE = "dataset downloading/Dataset/images"

Outputs
-------
- swinv2_large_14class_weightedbce.pth
  Fine-tuned SwinV2-Large weights (14-class multilabel classifier).

- evaluation_csvs/swinv2_large_14class_weightedbce_perclass.csv
  Per-label ROC AUC, PR AUC, best F1, and best threshold.

- evaluation_csvs/swinv2_large_14class_weightedbce_summary.csv
  Global metrics: mean ROC/PR AUC, macro/micro F1, precision, recall,
  micro ROC AUC, and ECE.

- evaluation_csvs/swinv2_large_14class_weightedbce_top5_pr.png
  Precision–recall curves for the top-5 labels by PR AUC.

Notes
-----
- Uses SwinV2 Large (timm: "swinv2_large_window12to16_192to256") with a
  14-class classifier head and global average pooling.
- Pos-weights are estimated from the *training* labels only, clipped and
  normalized for numerical stability.
- Augmentations are applied only to the training split; val/test use
  plain resizing + normalization.
"""

# ===============================================================
# Imports
# ===============================================================
import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

import timm
import tensorflow as tf


# ===============================================================
# Device
# ===============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# Dataset Definition
# ===============================================================
class CXRDataset(Dataset):
    """
    Simple image + multilabel dataset for PNG CXRs.

    Expects dataframe with:
      - "image_path" column for PNG path
      - label_cols as float columns in [0, 1]
    """

    def __init__(self, df, label_cols, transform=None):
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
# Paths, CSV Loading, and Merging
# ===============================================================
LABEL_CSV = "dataset downloading/Dataset/mimic_cxr_balanced_subset_v2.csv"
DCM_CSV = "dataset downloading/Dataset/dcm_info.csv"
IMAGE_BASE = "dataset downloading/Dataset/images"

labels = pd.read_csv(LABEL_CSV)
dcm_info = pd.read_csv(DCM_CSV)

# Restrict to frontal views compatible with classifier
dcm_info = dcm_info[dcm_info["ViewPosition"].isin(["AP", "PA"])]

# Merge labels with DICOM metadata on subject_id + study_id
merged = pd.merge(labels, dcm_info, on=["subject_id", "study_id"], how="inner")

# CheXpert-style label columns (14 classes)
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

# Keep only image_path + labels, remap image root
merged = merged[["image_path"] + label_cols]
merged["image_path"] = merged["image_path"].str.replace(
    r"^images", IMAGE_BASE, regex=True
)

# Train / val / test split (70 / 15 / 15)
train_df, temp_df = train_test_split(merged, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# ===============================================================
# Label Cleaning (NaNs → 0, clip to [0, 1])
# ===============================================================
for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    bad = df[label_cols].isna().sum().sum()
    print(f"{df_name}: NaNs in labels ->", bad)
    if bad > 0:
        df[label_cols] = df[label_cols].fillna(0)
    df[label_cols] = df[label_cols].clip(0, 1)

print("NaNs in val_df after cleaning:", val_df[label_cols].isna().sum().sum())


# ===============================================================
# Image Transforms & DataLoaders
# ===============================================================
transform_train = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(contrast=0.1, brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

num_workers = os.cpu_count() // 2 if os.cpu_count() is not None else 2

train_loader = DataLoader(
    CXRDataset(train_df, label_cols, transform_train),
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    CXRDataset(val_df, label_cols, transform_val),
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

test_loader = DataLoader(
    CXRDataset(test_df, label_cols, transform_val),
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


# ===============================================================
# Model: SwinV2-Large (14-class multilabel head)
# ===============================================================
NUM_CLASSES = 14

model = timm.create_model(
    "swinv2_large_window12to16_192to256",
    pretrained=True,
    num_classes=NUM_CLASSES,  # 14-class head
    global_pool="avg",  # GAP → classifier
).to(DEVICE)


# ===============================================================
# Loss: Weighted BCE (no focal term)
# ===============================================================
class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy loss (no focal term).

    Supports per-class positive weights (pos_weight) for imbalance correction:
      - pos_weight[c] > 1 up-weights positives of class c.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor(pos_weight, dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        base_loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        if self.pos_weight is not None:
            # weight positive samples by pos_weight per label
            weight = 1.0 + (self.pos_weight - 1.0) * targets
            base_loss = base_loss * weight

        return base_loss.mean()


# ===============================================================
# pos_weight Computation (train-only)
# ===============================================================
def compute_stable_pos_weight(train_df, label_cols, clip_lo=0.5, clip_hi=3.0, eps=1e-6):
    """
    Computes a numerically stable per-label pos_weight vector from the training set only.

    Steps
    -----
    - pos_weight = negatives / positives
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


# Compute pos_weight in TF, then convert to Torch
pos_weight_tf = compute_stable_pos_weight(train_df, label_cols)
pos_weight_torch = torch.tensor(pos_weight_tf.numpy(), dtype=torch.float32).to(DEVICE)

criterion = WeightedBCELoss(pos_weight=pos_weight_torch).to(DEVICE)


# ===============================================================
# Optimizer & LR Scheduler
# ===============================================================
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2
)


# ===============================================================
# Training & Evaluation Loops
# ===============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    """
    One supervised epoch over the training loader.
    """
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        labels = labels.float().squeeze()  # ensure [B, num_classes]
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
    """
    Returns mean per-label ROC AUC over the given loader.
    """
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
        except ValueError:
            aucs.append(float("nan"))
    return np.nanmean(aucs)


# ===============================================================
# Supervised Training Loop
# ===============================================================
EPOCHS = 20
best_auc = 0.0
CKPT_PATH = "swinv2_large_14class_weightedbce.pth"

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_auc = evaluate(model, val_loader)

    # Optional modulation by val AUC
    scheduler.step(epoch + val_auc * 0.1)

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
# Evaluation on Test Set + Metrics Export
# ===============================================================
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)

# Recreate exact architecture and load best checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_eval = timm.create_model(
    "swinv2_large_window12to16_192to256",
    pretrained=False,  # do NOT reload pretrained weights here
    num_classes=NUM_CLASSES,
    global_pool="avg",
).to(DEVICE)

state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
model_eval.load_state_dict(state_dict)
model_eval.eval()

# Run inference on test set
all_probs, all_labels = [], []
print("Running inference on test set...")

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model_eval(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

preds = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(all_labels, axis=0)

# Per-class metrics
results = []
for i, c in enumerate(label_cols):
    try:
        y_pred = preds[:, i]
        roc = roc_auc_score(y_true[:, i], y_pred)
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred)
        pr_auc = auc(recall, precision)

        # Best F1 threshold
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

# Global metrics @ threshold 0.5
y_pred_bin = (preds > 0.5).astype(int)
macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
macro_prec = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)
macro_rec = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)
micro_roc = roc_auc_score(y_true.ravel(), preds.ravel())

# Calibration (ECE)
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
# Save CSVs and PR Curves
# ===============================================================
os.makedirs("evaluation_csvs", exist_ok=True)
model_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]

results_path = f"evaluation_csvs/{model_name}_perclass.csv"
summary_path = f"evaluation_csvs/{model_name}_summary.csv"
prcurve_path = f"evaluation_csvs/{model_name}_top5_pr.png"

results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)
print("Saved metrics:", results_path)
print("Saved summary:", summary_path)

# Plot top-5 labels by PR AUC
plt.figure(figsize=(6, 5))
for _, row in results_df.head(5).iterrows():
    i = label_cols.index(row["Label"])
    precision, recall, _ = precision_recall_curve(y_true[:, i], preds[:, i])
    plt.plot(recall, precision, label=f'{row["Label"]} (AUC={row["PR_AUC"]:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Top-5 PR Curves — Swin Transformer V2 Large")
plt.legend()
plt.tight_layout()
plt.savefig(prcurve_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved PR figure:", prcurve_path)

# Free GPU / CPU memory
torch.cuda.empty_cache()
gc.collect()

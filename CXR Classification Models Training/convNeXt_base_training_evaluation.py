"""
Description
-----------
This script trains and evaluates a 14-class ConvNeXt-Base classifier on a
balanced MIMIC-CXR subset. It merges CheXpert-style labels with DICOM-derived
PNG metadata, filters to frontal (AP/PA) views, constructs train/val/test
splits, applies standard image augmentations, and optimizes a weighted
Binary Cross-Entropy loss using per-label pos_weight.

Inputs
------
- mimic_cxr_balanced_subset_v2.csv
    Columns include:
      - subject_id, study_id
      - 14 CheXpert-style label columns (0/1/NaN)
- dcm_info.csv
    Columns include:
      - subject_id, study_id
      - image_path (rooted under "images/...")
      - ViewPosition (filtered to ["AP", "PA"])
- Image directory
    IMAGE_BASE = "dataset downloading/Dataset/images"

Outputs
-------
- convnext_base_best_14class_bce.pth
    Fine-tuned ConvNeXt-Base checkpoint (14-class multilabel classifier).

- evaluation_csvs/convnext_base_best_14class_bce_perclass.csv
    Per-label ROC AUC, PR AUC, best F1, and best threshold.

- evaluation_csvs/convnext_base_best_14class_bce_summary.csv
    Global metrics: mean ROC/PR AUC, macro/micro F1, precision, recall,
    micro ROC AUC, and ECE.

- evaluation_csvs/convnext_base_best_14class_bce_top5_pr.png
    Precision–recall curves for the top-5 labels by PR AUC.

Notes
-----
- Uses ConvNeXt-Base (timm: "convnext_base") with a lightweight 14-class
  head and explicit global average pooling.
- pos_weight is computed *only* from the training split, clipped and
  normalized for stability, and passed to a custom WeightedBCELoss.
- Train split uses augmentations; val/test use simple resize + normalization.
- All metrics follow the same evaluation protocol as the SwinV2 notebook
  for consistency across model families.
"""

# ===============================================================
# Imports
# ===============================================================
import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
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
# Dataset definition
# ===============================================================
class CXRDataset(Dataset):
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
# Load CSVs, merge, and split
# ===============================================================
LABEL_CSV = "dataset downloading/Dataset/mimic_cxr_balanced_subset_v2.csv"
DCM_CSV = "dataset downloading/Dataset/dcm_info.csv"
IMAGE_BASE = "dataset downloading/Dataset/images"

labels = pd.read_csv(LABEL_CSV)
dcm_info = pd.read_csv(DCM_CSV)
dcm_info = dcm_info[dcm_info["ViewPosition"].isin(["AP", "PA"])]
merged = pd.merge(labels, dcm_info, on=["subject_id", "study_id"], how="inner")

# Choose labels compatible with pretrained models
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

merged = merged[["image_path"] + label_cols]
merged["image_path"] = merged["image_path"].str.replace(
    r"^images", IMAGE_BASE, regex=True
)

train_df, temp_df = train_test_split(merged, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# ===============================================================
# Clean NaNs and clip labels
# ===============================================================
# Clean NaNs and clip to [0, 1]
for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    bad = df[label_cols].isna().sum().sum()
    print(f"{df_name}: NaNs in labels ->", bad)
    if bad > 0:
        df[label_cols] = df[label_cols].fillna(0)
    df[label_cols] = df[label_cols].clip(0, 1)

print(val_df[label_cols].isna().sum().sum())


# ===============================================================
# Transforms and DataLoaders
# ===============================================================
transform_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

train_loader = DataLoader(
    CXRDataset(train_df, label_cols, transform_train),
    batch_size=32,
    shuffle=True,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
)
val_loader = DataLoader(
    CXRDataset(val_df, label_cols, transform_val),
    batch_size=32,
    shuffle=False,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
)
test_loader = DataLoader(
    CXRDataset(test_df, label_cols, transform_val),
    batch_size=32,
    shuffle=False,
    num_workers=os.cpu_count() // 2,
    pin_memory=True,
)


# ===============================================================
# ConvNeXt-Base model
# ===============================================================
class ConvNeXtCXR(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        # Load pretrained ConvNeXt backbone
        self.backbone = timm.create_model(
            "convnext_base", pretrained=True, features_only=False
        )
        in_features = self.backbone.num_features  # should be 768

        # Add an explicit global pooling + new head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # [B, 768, 7, 7]
        x = self.pool(x)  # [B, 768, 1, 1]
        x = torch.flatten(x, 1)  # [B, 768]
        x = self.head(x)  # [B, num_classes]
        return x


# 🔧 Instantiate
num_labels = len(label_cols)
model = ConvNeXtCXR(num_classes=num_labels).to(DEVICE)

# ✅ Test shape
x = torch.randn(2, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    y = model(x)
print("✅ Output shape:", y.shape)


# ===============================================================
# Loss: Weighted BCE
# ===============================================================
class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy loss (no focal term).
    Supports per-class positive weights (pos_weight) for imbalance correction.
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
        base_loss = -(
            targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs)
        )

        if self.pos_weight is not None:
            # weight positive samples by pos_weight
            weight = 1.0 + (self.pos_weight - 1.0) * targets
            base_loss = base_loss * weight

        return base_loss.mean()


# ===============================================================
# Compute pos_weight from training set
# ===============================================================
def compute_stable_pos_weight(
    train_df, label_cols, clip_lo=0.5, clip_hi=3.0, eps=1e-6
):
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


# Compute with your helper (still returns tf.Tensor)
pos_weight_tf = compute_stable_pos_weight(train_df, label_cols)

# Convert to PyTorch tensor
pos_weight_torch = torch.tensor(
    pos_weight_tf.numpy(), dtype=torch.float32
).to(DEVICE)

# Pass to your loss
criterion = WeightedBCELoss(pos_weight=pos_weight_torch).to(DEVICE)


# ===============================================================
# Optimizer & LR scheduler
# ===============================================================
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2
)


# ===============================================================
# Training & validation loops
# ===============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
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
# Main training loop
# ===============================================================
EPOCHS = 20
best_auc = 0.0

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_auc = evaluate(model, val_loader)
    scheduler.step(epoch + val_auc)  # fine-tune schedule step

    print(
        f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "convnext_base_best_14class_bce.pth")
        print("✅ Model improved and saved.")
    else:
        print("⏸ No improvement.")

print(f"🏁 Training complete. Best Val AUC: {best_auc:.4f}")


# ===============================================================
# Evaluation on test set + metrics export
# ===============================================================
CKPT_PATH = "convnext_base_best_14class_bce.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ Load trained model weights
model = model.__class__()  # re-instantiate same architecture
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# run inference on test set
all_probs, all_labels = [], []
print("Running inference...")
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

# stack predictions and ground truths
preds = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(all_labels, axis=0)

# compute per-class metrics
results = []
for i, c in enumerate(label_cols):
    try:
        y_pred = preds[:, i]
        roc = roc_auc_score(y_true[:, i], y_pred)
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, i], y_pred
        )
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

# compute global metrics using 0.5 threshold
y_pred_bin = (preds > 0.5).astype(int)
macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
micro_f1 = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
macro_prec = precision_score(
    y_true, y_pred_bin, average="macro", zero_division=0
)
macro_rec = recall_score(
    y_true, y_pred_bin, average="macro", zero_division=0
)
micro_roc = roc_auc_score(y_true.ravel(), preds.ravel())

# compute calibration (ECE)
prob_true, prob_pred = calibration_curve(
    y_true.ravel(), preds.ravel(), n_bins=10
)
ece = np.abs(prob_true - prob_pred).mean()

# store summary metrics
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

# create output folder if needed
os.makedirs("evaluation_csvs", exist_ok=True)
model_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]

# define export paths
results_path = f"evaluation_csvs/{model_name}_perclass.csv"
summary_path = f"evaluation_csvs/{model_name}_summary.csv"
prcurve_path = f"evaluation_csvs/{model_name}_top5_pr.png"

# export CSVs
results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)
print("Saved metrics:", results_path)
print("Saved summary:", summary_path)

# plot and export top-5 PR curves
plt.figure(figsize=(6, 5))
for _, row in results_df.head(5).iterrows():
    i = label_cols.index(row["Label"])
    precision, recall, _ = precision_recall_curve(
        y_true[:, i], preds[:, i]
    )
    plt.plot(
        recall,
        precision,
        label=f'{row["Label"]} (AUC={row["PR_AUC"]:.2f})',
    )
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Top-5 PR Curves — ConvNeXt Base")
plt.legend()
plt.tight_layout()
plt.savefig(prcurve_path, dpi=300, bbox_inches="tight")
plt.show()
print("Saved PR figure:", prcurve_path)

# free GPU memory
torch.cuda.empty_cache()
gc.collect()

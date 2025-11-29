"""
Description
-----------
This script trains and evaluates an attention-based SwinV2-Large localization
model on the ChestX-Det dataset, supervised on 8 localization classes.

It:
- Loads ChestX-Det JSON annotations and PNG images.
- Maps ChestX-Det labels into 8 localization classes aligned with your MIMIC
  14-label ordering.
- Builds per-image:
    * 8-class multi-hot classification labels.
    * 8-channel Gaussian heatmaps derived from bounding boxes.
- Uses a SwinV2-Large backbone (pretrained on MIMIC 14-class classification),
  adds a spatial attention block and a UNet-lite decoder to predict 8
  localization maps at 256×256 resolution.
- Trains with a joint objective:
    * Weighted BCE on 8-class labels (from 14-class logits).
    * MSE loss on 8-channel heatmaps.
- Saves the best checkpoint by validation loss and evaluates localization
  metrics on the ChestX-Det test set (IoU, Dice, Corr, Box IoU).

Inputs
------
- ChestX-Det JSON annotations:
    CHESTX_TRAIN_JSON = "ChestX-Det Dataset/ChestX_Det_train.json"
    CHESTX_TEST_JSON  = "ChestX-Det Dataset/ChestX_Det_test.json"

- ChestX-Det image folders:
    TRAIN_IMAGE_ROOT = "ChestX-Det Dataset/train_data/train"
    TEST_IMAGE_ROOT  = "ChestX-Det Dataset/test_data/test"

- MIMIC SwinV2-Large checkpoint (14-class classifier):
    MIMIC_CKPT = "swinv2_large_14class_weightedbce.pth"

Outputs
-------
- Attention-based localization checkpoint:
    swinv2_loc_attention_best.pth

- Printed localization metrics on the test set:
    For each of the 8 localization classes:
      - Mean IoU (mask)
      - Mean Dice
      - Mean Pearson correlation (heatmap-level)
      - Mean Box IoU (CAM-derived box vs GT boxes)

Notes
-----
- Supervised localization classes (8) are:
    ["Atelectasis", "Cardiomegaly", "Consolidation", "Pleural Effusion",
     "Fracture", "Pneumothorax", "Lung Lesion", "Pleural Other"]
- These 8 classes are mapped into your MIMIC 14-label ordering via MAP_8_TO_14.
- pos_weight for the 8 classes is computed directly from the ChestX-Det JSON.
- SwinV2 features are refined via a spatial attention block, then decoded
  with a UNet-lite segmentation decoder to 8×256×256 localization maps.
"""

# ===============================================================
# Imports & Global Settings
# ===============================================================
import os
import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
import timm

# Device & image size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

print("✅ Device:", DEVICE)


# ===============================================================
# Paths (ChestX-Det & MIMIC Checkpoint)
# ===============================================================
CHESTX_TRAIN_JSON = "ChestX-Det Dataset/ChestX_Det_train.json"
CHESTX_TEST_JSON = "ChestX-Det Dataset/ChestX_Det_test.json"

TRAIN_IMAGE_ROOT = "ChestX-Det Dataset/train_data/train"
TEST_IMAGE_ROOT = "ChestX-Det Dataset/test_data/test"

# MIMIC-trained SwinV2 checkpoint (14-class classifier)
MIMIC_CKPT = "swinv2_large_14class_weightedbce.pth"


# ===============================================================
# Class Mapping to 8 Localization Classes
# ---------------------------------------------------------------
# We supervise only 8 classes that have ChestX-Det boxes.
# Ordering here is used everywhere in training/eval.
# ===============================================================
LOC_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Fracture",
    "Pneumothorax",
    "Lung Lesion",
    "Pleural Other",
]

# ChestX-Det labels → mapped to LOC_CLASSES
CHESTX_TO_LOC = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Effusion": "Pleural Effusion",
    "Fracture": "Fracture",
    "Pneumothorax": "Pneumothorax",
    "Nodule": "Lung Lesion",
    "Mass": "Lung Lesion",
    "Pleural Thickening": "Pleural Other",
    "Fibrosis": "Pleural Other",
}

LOC_TO_IDX = {c: i for i, c in enumerate(LOC_CLASSES)}

# Your MIMIC ordering (14 labels) used in Swin training
MIMIC_14 = [
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

# Indices of LOC_CLASSES inside MIMIC_14
MAP_8_TO_14 = [MIMIC_14.index(c) for c in LOC_CLASSES]
print("✅ MAP_8_TO_14:", MAP_8_TO_14)


# ===============================================================
# ChestXDetDataset (PNG + JSON boxes → labels + heatmaps)
# ===============================================================
class ChestXDetDataset(Dataset):
    """
    Loads ChestX-Det samples from JSON + PNG folders.

    Returns per sample
    ------------------
    img_t : torch.FloatTensor [3, H, W]
        Image tensor normalized to ImageNet stats.
    y_cls : torch.FloatTensor [8]
        Multi-hot vector over LOC_CLASSES.
    y_hm  : torch.FloatTensor [8, H, W]
        Gaussian heatmaps per class (continuous supervision).
    meta  : dict
        Contains "file_name", "boxes_scaled", and "mapped_syms".
    """

    def __init__(self, json_path, image_root, transform=None, img_size=256):
        super().__init__()
        self.image_root = image_root
        self.transform = transform
        self.img_size = img_size

        # Load JSON annotations (list of dicts)
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _gaussian_heatmap(H, W, x1, y1, x2, y2, sigma_scale=0.25):
        """
        Build a soft 2D Gaussian blob for a bounding box.

        Parameters
        ----------
        H, W : int
            Output height and width.
        x1, y1, x2, y2 : float
            Box corners in [0, W) × [0, H).
        sigma_scale : float
            Relative spread of the Gaussian w.r.t box size.

        Returns
        -------
        g : np.ndarray [H, W]
            Normalized Gaussian in [0, 1].
        """
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(1.0, (x2 - x1))
        bh = max(1.0, (y2 - y1))

        sigma_x = bw * sigma_scale
        sigma_y = bh * sigma_scale

        xs = np.arange(W)
        ys = np.arange(H)
        xv, yv = np.meshgrid(xs, ys)

        g = np.exp(
            -(((xv - cx) ** 2) / (2 * sigma_x**2 + 1e-6)
              + ((yv - cy) ** 2) / (2 * sigma_y**2 + 1e-6))
        )
        g = g / (g.max() + 1e-6)
        return g

    def __getitem__(self, idx):
        entry = self.data[idx]
        file_name = entry["file_name"]
        syms = entry["syms"]
        boxes = entry["boxes"]

        img_path = os.path.join(self.image_root, file_name)

        # If image missing/corrupt → return None (safe_collate will skip)
        if not os.path.exists(img_path):
            return None

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        # Apply transformations (resize + normalize)
        if self.transform is not None:
            img_t = self.transform(img)
        else:
            img_t = T.ToTensor()(img)

        H = W = self.img_size

        # ---- 8-class multi-hot label ----
        y_cls = torch.zeros(len(LOC_CLASSES), dtype=torch.float32)

        # ---- 8-class Gaussian heatmap ----
        y_hm = np.zeros((len(LOC_CLASSES), H, W), dtype=np.float32)

        # Resize-scale factor (original PNG → IMG_SIZE)
        orig_w, orig_h = img.size
        sx = W / orig_w
        sy = H / orig_h

        boxes_scaled = []
        mapped_syms = []

        # Build labels/heatmaps from all boxes
        for sym, box in zip(syms, boxes):

            if sym not in CHESTX_TO_LOC:
                # Skip classes that are not in the 8 supervised labels
                continue

            loc_name = CHESTX_TO_LOC[sym]
            loc_idx = LOC_TO_IDX[loc_name]

            # Mark presence in classification label
            y_cls[loc_idx] = 1.0

            # Scale box into IMG_SIZE coordinate system
            x1, y1, x2, y2 = box
            x1s, x2s = int(x1 * sx), int(x2 * sx)
            y1s, y2s = int(y1 * sy), int(y2 * sy)

            boxes_scaled.append([x1s, y1s, x2s, y2s])
            mapped_syms.append(loc_name)

            # Add a Gaussian blob to the GT heatmap for this class
            g = self._gaussian_heatmap(H, W, x1s, y1s, x2s, y2s)
            y_hm[loc_idx] = np.maximum(y_hm[loc_idx], g)

        y_hm = torch.tensor(y_hm, dtype=torch.float32)

        meta = {
            "file_name": file_name,
            "boxes_scaled": boxes_scaled,
            "mapped_syms": mapped_syms,
        }

        return img_t, y_cls, y_hm, meta


# ===============================================================
# safe_collate
# ---------------------------------------------------------------
# - Skips None samples
# - Stacks tensors
# - Keeps metadata as list
# ===============================================================
def safe_collate(batch):
    """
    Collate function that drops None or malformed samples.

    Returns
    -------
    batch_out : tuple or None
        (imgs, y_cls, y_hm, metas) with stacked tensors, or None if
        no valid samples exist in the batch.
    """
    imgs, ycls, yhms, metas = [], [], [], []

    for item in batch:
        if item is None:
            continue

        img_t, y_cls, y_hm, meta = item

        # Shape guards to avoid collate crashes
        if img_t.ndim != 3:
            continue
        if y_cls.numel() != len(LOC_CLASSES):
            continue
        if y_hm.ndim != 3 or y_hm.shape[0] != len(LOC_CLASSES):
            continue

        imgs.append(img_t)
        ycls.append(y_cls)
        yhms.append(y_hm)
        metas.append(meta)

    if len(imgs) == 0:
        return None

    return (
        torch.stack(imgs, dim=0),
        torch.stack(ycls, dim=0),
        torch.stack(yhms, dim=0),
        metas,
    )


# ===============================================================
# Transforms + DataLoaders
# ===============================================================
transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

full_train_ds = ChestXDetDataset(
    CHESTX_TRAIN_JSON, TRAIN_IMAGE_ROOT, transform=transform, img_size=IMG_SIZE
)

test_ds = ChestXDetDataset(
    CHESTX_TEST_JSON, TEST_IMAGE_ROOT, transform=transform, img_size=IMG_SIZE
)

# Split train → train/val (80/20)
val_ratio = 0.2
val_size = int(val_ratio * len(full_train_ds))
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

num_workers = os.cpu_count() // 2 if os.cpu_count() is not None else 2

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=safe_collate,
)

val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=safe_collate,
)

test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=safe_collate,
)

print(f"Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")


# ===============================================================
# pos_weight_8 + Weighted BCE Loss
# ===============================================================
def compute_pos_weight_8_fast(json_path):
    """
    Compute per-class pos_weight for LOC_CLASSES directly from JSON.

    pos_weight[c] = negatives[c] / positives[c], clipped and normalized.

    Parameters
    ----------
    json_path : str
        Path to ChestX-Det JSON file.

    Returns
    -------
    pos_weight : torch.FloatTensor [8]
        Clipped and mean-normalized pos_weight vector.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    N = len(LOC_CLASSES)
    pos = np.zeros(N, dtype=float)
    neg = np.zeros(N, dtype=float)

    for entry in data:
        present = np.zeros(N, dtype=float)

        for s in entry["syms"]:
            if s in CHESTX_TO_LOC:
                mapped = CHESTX_TO_LOC[s]
                present[LOC_TO_IDX[mapped]] = 1.0

        pos += present
        neg += (1.0 - present)

    pw = neg / (pos + 1e-6)
    pw = np.clip(pw, 0.5, 3.0)
    pw = pw / pw.mean()

    return torch.tensor(pw, dtype=torch.float32)


pos_weight_8 = compute_pos_weight_8_fast(CHESTX_TRAIN_JSON).to(DEVICE)
print("pos_weight_8:", pos_weight_8.cpu().numpy().round(3))


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy for 8-class multi-label classification.

    Uses a per-class pos_weight vector to correct class imbalance.
    """

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.float())

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


# ===============================================================
# SwinV2 Backbone + Spatial Attention + Seg Decoder
# ---------------------------------------------------------------
# Outputs:
#   logits_14 : [B, 14]
#   loc_maps  : [B, 8, 256, 256]
# ===============================================================
class SpatialAttention(nn.Module):
    """Spatial attention to sharpen class-relevant regions."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, 1)  # 1×1 conv → attention mask
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.conv(x))  # [B,1,H,W]
        return x * att  # reweight features


class SegDecoder(nn.Module):
    """UNet-lite decoder to refine deep features into 256×256 masks."""

    def __init__(self, in_ch=1536, out_ch=8):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8→16
            nn.Conv2d(in_ch, 512, 3, padding=1),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16→32
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32→64
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(True),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64→128
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 128→256
            nn.Conv2d(64, out_ch, 3, padding=1),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x


class SwinV2_AttentionSeg(nn.Module):
    """
    SwinV2 backbone (MIMIC-trained) + spatial attention + segmentation decoder.

    Forward
    -------
    x : torch.FloatTensor [B, 3, 256, 256]

    Returns
    -------
    logits_14 : torch.FloatTensor [B, 14]
        Global 14-class logits (MIMIC ordering).
    loc_maps  : torch.FloatTensor [B, 8, 256, 256]
        Localization heatmaps for LOC_CLASSES.
    """

    def __init__(self, backbone_name="swinv2_large_window12to16_192to256"):
        super().__init__()

        # SwinV2 as features-only backbone (keeps spatial tokens)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=[3],  # deep stage → semantic features
        )

        self.att = SpatialAttention(1536)
        self.decoder = SegDecoder(in_ch=1536, out_ch=len(LOC_CLASSES))

        # Custom classifier head to keep 14-class predictions
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(1536)
        self.classifier = nn.Linear(1536, 14)

    def forward(self, x):
        # 1) Extract Swin features
        feat = self.backbone(x)[0]

        # 2) timm SwinV2 features may come as NHWC: [B,H,W,C]
        #    Conv2d expects NCHW, so we convert if needed.
        if feat.ndim == 4 and feat.shape[-1] == 1536:
            feat = feat.permute(0, 3, 1, 2).contiguous()  # → [B,1536,H,W]

        # 3) Attention refinement
        feat_att = self.att(feat)  # [B,1536,H,W]

        # 4) Segmentation refinement maps (8 classes)
        loc_maps = self.decoder(feat_att)  # [B,8,256,256]

        # 5) Global pooled classifier (14 classes)
        pooled = self.pool(feat_att).flatten(1)  # [B,1536]
        pooled = self.norm(pooled)
        logits_14 = self.classifier(pooled)  # [B,14]

        return logits_14, loc_maps


# ===============================================================
# Init Model + Load MIMIC Weights into Backbone & Classifier
# ===============================================================
backbone_name = "swinv2_large_window12to16_192to256"
model = SwinV2_AttentionSeg(backbone_name).to(DEVICE)

# Load MIMIC classifier checkpoint (14-class SwinV2)
mimic_state = torch.load(MIMIC_CKPT, map_location=DEVICE)

# 1) Load backbone weights (patch_embed, stages, norm)
backbone_state = {
    k.replace("backbone.", ""): v
    for k, v in mimic_state.items()
    if k.startswith("patch_embed") or k.startswith("stages") or k.startswith("norm")
}
model.backbone.load_state_dict(backbone_state, strict=False)

# 2) Load classifier head weights from MIMIC Swin
if "head.fc.weight" in mimic_state:
    model.classifier.weight.data.copy_(mimic_state["head.fc.weight"])
    model.classifier.bias.data.copy_(mimic_state["head.fc.bias"])
elif "head.weight" in mimic_state:
    model.classifier.weight.data.copy_(mimic_state["head.weight"])
    model.classifier.bias.data.copy_(mimic_state["head.bias"])

# Freeze early backbone stages (train last stage + new heads)
for name, p in model.backbone.named_parameters():
    if "stages.3" not in name:
        p.requires_grad = False

print("✅ Loaded MIMIC weights and froze early backbone stages.")


# ===============================================================
# Losses + Optimizer
# ===============================================================
bce_cls = WeightedBCELoss(pos_weight_8).to(DEVICE)  # 8-class BCE
mse_loc = nn.MSELoss()  # heatmap regression

LAMBDA_LOC = 0.15  # localization supervision weight

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=1e-5)

print("✅ Trainable params:", sum(p.numel() for p in params))


# ===============================================================
# Train / Val for One Epoch
# ===============================================================
def train_one_epoch(model, loader):
    """
    One training epoch.

    Objective
    ---------
    loss = BCE_8cls + LAMBDA_LOC * MSE_heatmaps
    """
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc="Training", leave=False)
    for batch in loop:
        if batch is None:
            continue

        imgs, y_cls, y_hm, _ = batch
        imgs = imgs.to(DEVICE)
        y_cls = y_cls.to(DEVICE)
        y_hm = y_hm.to(DEVICE)

        optimizer.zero_grad()

        logits_14, pred_maps = model(imgs)
        logits_8 = logits_14[:, MAP_8_TO_14]  # extract supervised classes

        loss_cls = bce_cls(logits_8, y_cls)
        loss_loc = mse_loc(pred_maps, y_hm)

        loss = loss_cls + LAMBDA_LOC * loss_loc
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        loop.set_postfix(total=loss.item(), cls=loss_cls.item(), loc=loss_loc.item())

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    """
    Validation loss (same objective, no gradient).
    """
    model.eval()
    total_loss = 0.0

    for batch in loader:
        if batch is None:
            continue

        imgs, y_cls, y_hm, _ = batch
        imgs = imgs.to(DEVICE)
        y_cls = y_cls.to(DEVICE)
        y_hm = y_hm.to(DEVICE)

        logits_14, pred_maps = model(imgs)
        logits_8 = logits_14[:, MAP_8_TO_14]

        loss_cls = bce_cls(logits_8, y_cls)
        loss_loc = mse_loc(pred_maps, y_hm)

        loss = loss_cls + LAMBDA_LOC * loss_loc
        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


# ===============================================================
# Training Loop (saves swinv2_loc_attention_best.pth)
# ===============================================================
EPOCHS = 20
best_val = float("inf")
CKPT_PATH = "swinv2_loc_attention_best.pth"

for ep in range(EPOCHS):
    print(f"\n===== Epoch {ep + 1}/{EPOCHS} =====")

    tr_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    print(f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), CKPT_PATH)
        print("✅ Saved best model.")
    else:
        print("⏸ No improvement.")

print("🏁 Training done. Best Val Loss =", best_val)


# ===============================================================
# Load Best Model for Test Evaluation
# ===============================================================
model = SwinV2_AttentionSeg().to(DEVICE)

state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
model.eval()

print("✅ Loaded attention-based localization model for evaluation.")


# ===============================================================
# Localization Evaluation Metrics (IoU, Dice, Corr, Box IoU)
# ===============================================================
def compute_iou(pred_mask, gt_mask):
    """Compute IoU between binary masks."""
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return np.nan
    return inter / union


def compute_dice(pred_mask, gt_mask):
    """Dice coefficient for binary masks."""
    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return np.nan
    return 2 * inter / denom


def mask_from_box(box, H=256, W=256):
    """Convert box [x1,y1,x2,y2] to binary mask."""
    x1, y1, x2, y2 = box
    m = np.zeros((H, W), dtype=np.uint8)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    m[y1:y2, x1:x2] = 1
    return m


def compute_best_box_iou(pred_cam, gt_boxes, thresh=0.4):
    """
    Find predicted connected-component box from CAM,
    match to GT boxes → return best IoU.
    """
    H, W = pred_cam.shape
    if len(gt_boxes) == 0:
        return np.nan

    cam_norm = (pred_cam - pred_cam.min()) / (
        pred_cam.max() - pred_cam.min() + 1e-6
    )
    mask = (cam_norm > thresh).astype(np.uint8)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return 0.0  # no predicted box

    # Get largest component as predicted lesion
    idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, area = stats[idx]
    pred_box = [x, y, x + w, y + h]

    # Evaluate IoU w.r.t each GT
    best = 0.0
    for gt in gt_boxes:
        gt_mask = mask_from_box(gt, H, W)
        pred_mask = mask_from_box(pred_box, H, W)
        iou = compute_iou(pred_mask, gt_mask)
        best = max(best, iou)

    return best


@torch.no_grad()
def evaluate_localization_metrics(model, loader):
    """
    Compute localization metrics for LOC_CLASSES.

    Metrics
    -------
    For each class:
      - IoU between binary predicted/GT heatmaps (thresholded @ 0.4).
      - Dice coefficient.
      - Pearson correlation between continuous maps.
      - Best Box IoU from CAM-derived box vs GT boxes.
    """
    model.eval()

    # Store metrics for 8 classes
    metrics = {
        cls: {"count": 0, "iou": [], "dice": [], "corr": [], "box_iou": []}
        for cls in LOC_CLASSES
    }

    print("Evaluating localization metrics...")
    for batch in tqdm(loader):
        if batch is None:
            continue

        imgs, y_cls, y_hm, metas = batch
        imgs = imgs.to(DEVICE)

        # Forward pass
        logits_14, pred_maps_t = model(imgs)
        pred_maps = pred_maps_t.cpu().numpy()  # [B,8,256,256]
        y_hm = y_hm.numpy()

        for b in range(pred_maps.shape[0]):
            meta = metas[b]
            for cid, cls_name in enumerate(LOC_CLASSES):
                gt_hm = y_hm[b, cid]
                pred = pred_maps[b, cid]

                # Skip if class not present
                if gt_hm.max() == 0:
                    continue

                # Normalize CAM
                pred_n = (pred - pred.min()) / (
                    pred.max() - pred.min() + 1e-6
                )

                # IoU / Dice on continuous maps (threshold @ 0.4)
                pred_mask = pred_n > 0.4
                gt_mask = gt_hm > 0.4

                iou = compute_iou(pred_mask, gt_mask)
                dice = compute_dice(pred_mask, gt_mask)

                # Pearson correlation
                corr = np.corrcoef(pred_n.flatten(), gt_hm.flatten())[0, 1]
                if np.isnan(corr):
                    corr = 0.0

                # Resolve GT boxes (only boxes belonging to this class)
                gt_boxes = [
                    b for b, s in zip(meta["boxes_scaled"], meta["mapped_syms"]) if s == cls_name
                ]

                box_iou = compute_best_box_iou(pred, gt_boxes, thresh=0.4)

                metrics[cls_name]["count"] += 1
                metrics[cls_name]["iou"].append(iou)
                metrics[cls_name]["dice"].append(dice)
                metrics[cls_name]["corr"].append(corr)
                metrics[cls_name]["box_iou"].append(box_iou)

    # Print results
    print("================ Localization Metrics ================")
    for cls in LOC_CLASSES:
        m = metrics[cls]
        if m["count"] == 0:
            print(
                f"\n📌 {cls}\n   No positive GT samples in test set.\n--------------------------------------------------"
            )
            continue

        print(f"\n📌 {cls}")
        print(f"   Samples        : {m['count']}")
        print(f"   IoU (mask)     : {np.nanmean(m['iou']):.4f}")
        print(f"   Dice           : {np.nanmean(m['dice']):.4f}")
        print(f"   Corr           : {np.nanmean(m['corr']):.4f}")
        print(f"   Box IoU        : {np.nanmean(m['box_iou']):.4f}")
        print("--------------------------------------------------")

    return metrics


# ---------------------------------------------------------------
# Run localization evaluation on test set
# ---------------------------------------------------------------
metrics = evaluate_localization_metrics(model, test_loader)
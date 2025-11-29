"""
Description
-----------
This script fine-tunes a SwinV2-Large backbone for **8-class localization**
on the ChestX-Det dataset, using Gaussian heatmaps derived from bounding boxes.

It:
- Loads ChestX-Det JSON annotations for train/test.
- Maps original ChestX-Det symbols into 8 unified localization classes.
- Builds (image, 8-label vector, 8-heatmap stack) samples.
- Initializes a SwinV2-Large model with:
    - A 14-class classification head (copied from a MIMIC-CXR checkpoint).
    - An 8-class localization head on stage-2 features, upsampled to 256×256.
- Freezes early layers, trains deeper Swin stages + localization head using:
    - Weighted BCE for the 8-class labels.
    - MSE for the 8-channel heatmaps (with a configurable λ for localization loss).
- Saves the best localization checkpoint by validation loss.
- Reloads the trained model and computes localization metrics on the test set:
    - Mask IoU and Dice
    - Pearson correlation with GT heatmaps
    - Box IoU via connected components (largest CAM region)

Inputs
------
- ChestX_Det_train.json
- ChestX_Det_test.json
    Each JSON entry is expected to contain:
      - "file_name": relative image filename
      - "syms": list of original ChestX-Det pathology symbols
      - "boxes": list of [x1, y1, x2, y2] bounding boxes

- Image folders:
    CHESTX_TRAIN_IMG_DIR = "ChestX-Det Dataset/train_data/train"
    CHESTX_TEST_IMG_DIR  = "ChestX-Det Dataset/test_data/test"

Outputs
-------
- swinv2_loc_best.pth
    Best SwinV2 localization model (14-class classifier + 8-class loc head).

- Console metrics for each of the 8 localization classes:
    - IoU (heatmap mask)
    - Dice
    - Corr (Pearson)
    - Box IoU

Notes
-----
- Uses a pre-trained SwinV2-Large 14-class classifier trained on MIMIC-CXR:
    "swinv2_large_14class_weightedbce.pth"
  to initialize the backbone and 14-class head.

- 8 localization classes (LOC_CLASSES) are a subset/mapping of ChestX-Det labels:
    ["Atelectasis", "Cardiomegaly", "Consolidation", "Pleural Effusion",
     "Fracture", "Pneumothorax", "Lung Lesion", "Pleural Other"]

- Gaussian heatmaps are generated for each GT box, merged per class by max.
- Class-wise pos_weight for the 8 localization labels is computed directly
  from the train JSON and used in a custom WeightedBCELoss.
"""

# ===============================================================
# Imports
# ===============================================================
import os
import json
import math

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import timm
import matplotlib.pyplot as plt  # (only used if you later add plots)


# ===============================================================
# Device
# ===============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# Paths and Class Mapping
# ===============================================================
# JSON paths
CHESTX_TRAIN_JSON = "ChestX-Det Dataset/ChestX_Det_train.json"
CHESTX_TEST_JSON = "ChestX-Det Dataset/ChestX_Det_test.json"

# Image folders
CHESTX_TRAIN_IMG_DIR = "ChestX-Det Dataset/train_data/train"
CHESTX_TEST_IMG_DIR = "ChestX-Det Dataset/test_data/test"

# 8 localization classes used in your pipeline
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
LOC_CLASS_TO_ID = {c: i for i, c in enumerate(LOC_CLASSES)}

# ChestX-Det → 8-class mapping
CHESTX_TO_LOC = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Effusion": "Pleural Effusion",
    "Fracture": "Fracture",
    "Pneumothorax": "Pneumothorax",
    # Lesion bucket
    "Mass": "Lung Lesion",
    "Nodule": "Lung Lesion",
    "Diffuse Nodule": "Lung Lesion",
    # Pleural bucket
    "Pleural Thickening": "Pleural Other",
    "Pleural Thickening ": "Pleural Other",  # guard for whitespace variants
    "Pleural Other": "Pleural Other",
}


# ===============================================================
# JSON Loading Utilities
# ===============================================================
def load_chestx_json(json_path):
    """
    Load ChestX-Det JSON.

    Returns
    -------
    records : list[dict]
        Each dict typically has:
        {
          "file_name": str,
          "syms": list[str],
          "boxes": list[list[float]],
          ...
        }
    """
    with open(json_path, "r") as f:
        records = json.load(f)
    return records


train_records = load_chestx_json(CHESTX_TRAIN_JSON)
test_records = load_chestx_json(CHESTX_TEST_JSON)

print("Train records:", len(train_records))
print("Test records :", len(test_records))
print("Example record keys:", train_records[0].keys())


# ===============================================================
# ChestX-Det Dataset → (image, 8-label vector, 8-heatmaps)
# ===============================================================
class ChestXDetDataset(Dataset):
    """
    Dataset for (PNG chest X-ray, multi-label pathology, GT Gaussian heatmaps).

    Output per sample
    -----------------
    img_t : torch.FloatTensor [3, 256, 256]
    y_cls : torch.FloatTensor [8]
        8-class multi-label vector aligned with LOC_CLASSES
    y_hm  : torch.FloatTensor [8, 256, 256]
        One Gaussian heatmap per class derived from GT boxes
    meta  : dict
        Useful info for debugging/visualization
    """

    def __init__(self, records, img_dir, transform=None, out_size=256, sigma_frac=0.08):
        self.records = records
        self.img_dir = img_dir
        self.transform = transform
        self.out_size = out_size
        self.sigma_frac = sigma_frac  # Gaussian spread relative to bbox size

    def __len__(self):
        return len(self.records)

    def _get_image_path(self, file_name):
        """Resolve full image path from file_name."""
        return os.path.join(self.img_dir, file_name)

    def _map_syms(self, syms):
        """
        Map ChestX-Det symbols to the 8 localization classes.

        Returns
        -------
        mapped_syms : list[str]
        """
        mapped = []
        for s in syms:
            if s in CHESTX_TO_LOC:
                mapped.append(CHESTX_TO_LOC[s])
        return mapped

    def _scale_boxes(self, boxes, orig_w, orig_h):
        """
        Scale boxes from original image coords to out_size×out_size coords.

        boxes in JSON are [x1, y1, x2, y2] (pixel coords).
        """
        sx = self.out_size / orig_w
        sy = self.out_size / orig_h

        scaled = []
        for (x1, y1, x2, y2) in boxes:
            scaled.append(
                [
                    int(x1 * sx),
                    int(y1 * sy),
                    int(x2 * sx),
                    int(y2 * sy),
                ]
            )
        return scaled

    def _gaussian_heatmap(self, box, H, W):
        """
        Create a 2D Gaussian heatmap inside the bbox region.

        Parameters
        ----------
        box : [x1, y1, x2, y2] scaled to H,W
        """
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        # Degenerate box → zeros
        if x2 <= x1 or y2 <= y1:
            return np.zeros((H, W), dtype=np.float32)

        # Gaussian center
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        bw = max(1.0, (x2 - x1))
        bh = max(1.0, (y2 - y1))

        # Sigma proportional to bbox size
        sigma_x = self.sigma_frac * bw
        sigma_y = self.sigma_frac * bh

        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        g = np.exp(
            -(
                ((X - cx) ** 2) / (2 * sigma_x**2 + 1e-6)
                + ((Y - cy) ** 2) / (2 * sigma_y**2 + 1e-6)
            )
        )

        # Normalize to [0,1]
        g = g / (g.max() + 1e-6)
        return g.astype(np.float32)

    def __getitem__(self, idx):
        rec = self.records[idx]
        file_name = rec["file_name"]
        syms = rec["syms"]
        boxes = rec["boxes"]

        img_path = self._get_image_path(file_name)
        if not os.path.exists(img_path):
            # Safe skip for missing images
            return None

        # Load image and get original size
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Apply transforms → out_size × out_size tensor
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        # Map ChestX-Det symbols to 8-class names
        mapped_syms = self._map_syms(syms)

        # Scale all boxes to out_size × out_size
        boxes_scaled = self._scale_boxes(boxes, orig_w, orig_h)

        # Initialize label vector and heatmaps
        y_cls = np.zeros((len(LOC_CLASSES),), dtype=np.float32)
        y_hm = np.zeros(
            (len(LOC_CLASSES), self.out_size, self.out_size),
            dtype=np.float32,
        )

        # Fill labels + heatmaps
        for sym, box_s in zip(mapped_syms, boxes_scaled):
            cid = LOC_CLASS_TO_ID[sym]
            y_cls[cid] = 1.0

            # Merge multiple boxes of same class by max
            hm = self._gaussian_heatmap(box_s, self.out_size, self.out_size)
            y_hm[cid] = np.maximum(y_hm[cid], hm)

        meta = {
            "file_name": file_name,
            "orig_size": (orig_h, orig_w),
            "mapped_syms": mapped_syms,
            "boxes_scaled": boxes_scaled,
        }

        return img_t, torch.tensor(y_cls), torch.tensor(y_hm), meta


# ===============================================================
# Transforms + Dataset + Train/Val/Test DataLoaders
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

transform_eval = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

full_train_ds = ChestXDetDataset(
    train_records,
    img_dir=CHESTX_TRAIN_IMG_DIR,
    transform=transform_train,
)

test_ds = ChestXDetDataset(
    test_records,
    img_dir=CHESTX_TEST_IMG_DIR,
    transform=transform_eval,
)

# Small validation split from train JSON
val_ratio = 0.1
val_size = int(len(full_train_ds) * val_ratio)
train_size = len(full_train_ds) - val_size

gen = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=gen)

print("Train size:", len(train_ds))
print("Val size  :", len(val_ds))
print("Test size :", len(test_ds))


def safe_collate(batch):
    """
    Collate function that drops None samples.

    Returns
    -------
    (imgs, y_cls, y_hm, metas) or None if batch empty.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, y_cls, y_hm, metas = zip(*batch)
    return torch.stack(imgs), torch.stack(y_cls), torch.stack(y_hm), metas


num_workers = os.cpu_count() // 2 if os.cpu_count() is not None else 2

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=safe_collate,
)

val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=safe_collate,
)

test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=safe_collate,
)


# ===============================================================
# pos_weight for 8-class ChestXDet labels
# ===============================================================
def compute_pos_weight_from_records(records):
    """
    Compute pos_weight vector for LOC_CLASSES directly from JSON.

    Definition
    ----------
    pos_weight[c] = negatives / positives

    Returns
    -------
    pos_weight_8 : torch.FloatTensor [8]
    """
    pos_counts = np.zeros((len(LOC_CLASSES),), dtype=np.float32)
    total = 0

    for rec in records:
        mapped_syms = []
        for s in rec["syms"]:
            if s in CHESTX_TO_LOC:
                mapped_syms.append(CHESTX_TO_LOC[s])

        # Unique labels per image
        mapped_syms = set(mapped_syms)

        # Count positives if label appears
        for sym in mapped_syms:
            pos_counts[LOC_CLASS_TO_ID[sym]] += 1

        total += 1

    neg_counts = total - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)

    # Clip for stability
    pos_weight = np.clip(pos_weight, 0.5, 3.0)

    print("pos_weight_8:", pos_weight.round(3).tolist())
    return torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)


pos_weight_8 = compute_pos_weight_from_records(train_records)


# ===============================================================
# Weighted BCE Loss (8-class)
# ===============================================================
class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy loss.

    Supports per-class positive weights to correct imbalance.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        base_loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        if self.pos_weight is not None:
            weight = 1.0 + (self.pos_weight - 1.0) * targets
            base_loss = base_loss * weight

        return base_loss.mean()


# ===============================================================
# SwinV2 with 14-class classifier + 8-class localization head
# -----------------------------------------------------------
# - features_only returns NHWC tensors:
#     stage2_feats: [B, 32, 32, 384]
#     stage4_feats: [B,  8,  8, 1536]
# - We permute to NCHW before GAP / Conv2d
# ===============================================================
class SwinV2WithLocalization(nn.Module):
    """
    SwinV2 backbone with:
      1) 14-class classification head (uses stage-4 GAP features)
      2) 8-class localization head attached to stage-2 features

    Forward Returns
    ---------------
    logits_14 : torch.FloatTensor [B, 14]
    loc_maps  : torch.FloatTensor [B, 8, 256, 256]
    """

    def __init__(self, backbone_name, num_cls=14, num_loc=8):
        super().__init__()

        # Extract stage-2 and stage-4 feature maps
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(1, 3),
        )

        # Channel dims for selected stages
        chs = self.backbone.feature_info.channels()
        stage2_dim = chs[0]  # 384
        stage4_dim = chs[1]  # 1536

        # 14-class classifier on stage-4 GAP
        self.classifier_head = nn.Linear(stage4_dim, num_cls)

        # 8-class CAM head on stage-2 spatial map
        self.loc_head = nn.Conv2d(stage2_dim, num_loc, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)  # list of NHWC features
        stage2_feats = feats[0]  # [B, 32, 32, 384]
        stage4_feats = feats[1]  # [B,  8,  8, 1536]

        # NHWC → NCHW
        stage2_feats = stage2_feats.permute(0, 3, 1, 2)  # [B, 384, 32, 32]
        stage4_feats = stage4_feats.permute(0, 3, 1, 2)  # [B,1536, 8,  8]

        # Classification branch
        gap = stage4_feats.mean(dim=(2, 3))  # [B, 1536]
        logits_14 = self.classifier_head(gap)  # [B, 14]

        # Localization branch
        loc_maps = self.loc_head(stage2_feats)  # [B, 8, 32, 32]
        loc_maps = nn.functional.interpolate(
            loc_maps,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        return logits_14, loc_maps


# ===============================================================
# Load MIMIC-trained SwinV2 and initialize localization model
# ===============================================================
backbone_name = "swinv2_large_window12to16_192to256"
model = SwinV2WithLocalization(backbone_name).to(DEVICE)

# Full SwinV2 classifier (reference for weights)
mimic_model = timm.create_model(
    backbone_name,
    pretrained=False,
    num_classes=14,
    global_pool="avg",
).to(DEVICE)

mimic_ckpt = torch.load("swinv2_large_14class_weightedbce.pth", map_location=DEVICE)
mimic_model.load_state_dict(mimic_ckpt)

# Copy backbone weights (exclude head)
back_state = {k: v for k, v in mimic_model.state_dict().items() if "head" not in k}

# features_only wraps real model under .model
target_backbone = model.backbone.model if hasattr(model.backbone, "model") else model.backbone
target_backbone.load_state_dict(back_state, strict=False)

# Copy 14-class classifier weights
model.classifier_head.weight.data.copy_(mimic_model.head.fc.weight.data)
model.classifier_head.bias.data.copy_(mimic_model.head.fc.bias.data)

print("✅ Backbone + 14-class classifier loaded from MIMIC checkpoint.")


# ===============================================================
# Freeze Early Layers, Keep Deeper Stages Trainable
# ===============================================================
backbone_core = model.backbone.model if hasattr(model.backbone, "model") else model.backbone

for name, p in backbone_core.named_parameters():
    # Freeze patch embed + stage-1
    if "patch_embed" in name or "stages.0" in name:
        p.requires_grad = False
    else:
        p.requires_grad = True

trainable = sum(p.requires_grad for p in model.parameters())
frozen = sum(not p.requires_grad for p in model.parameters())
print(f"Trainable params: {trainable} | Frozen params: {frozen}")


# ===============================================================
# Losses and Optimizer
# ===============================================================
bce_cls = WeightedBCELoss(pos_weight=pos_weight_8).to(DEVICE)
mse_loc = nn.MSELoss()
LAMBDA_LOC = 0.3  # weight for localization loss

params = [p for p in model.parameters() if p.requires_grad]

optimizer = optim.AdamW(
    params,
    lr=3e-5,
    weight_decay=1e-5,
)

print("✅ Losses and optimizer ready.")
print("LAMBDA_LOC =", LAMBDA_LOC)


# ===============================================================
# (Optional) Inspect SwinV2 features_only outputs
# ===============================================================
temp = timm.create_model(
    backbone_name,
    pretrained=False,
    features_only=True,
    out_indices=(1, 3),
).to(DEVICE)

print("Feature Info:")
print(temp.feature_info)

x_tmp = torch.randn(1, 3, 256, 256).to(DEVICE)
feats_tmp = temp(x_tmp)
for i, f in enumerate(feats_tmp):
    print(f"Stage {i} → shape: {f.shape}")


# ===============================================================
# Training & Validation Loops
# ===============================================================
# Indices of the 8 localization classes inside your 14-class MIMIC ordering
MAP_8_TO_14 = [
    0,   # Atelectasis
    1,   # Cardiomegaly
    2,   # Consolidation
    4,   # Pleural Effusion
    7,   # Fracture
    6,   # Pneumothorax
    8,   # Lung Lesion
    11,  # Pleural Other
]


def train_one_epoch(model, loader):
    """
    One epoch of training over ChestX-Det.

    Returns
    -------
    avg_loss : float
    """
    model.train()
    total = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        if batch is None:
            continue

        imgs, y_cls, y_hm, _ = batch
        imgs = imgs.to(DEVICE)
        y_cls = y_cls.to(DEVICE)
        y_hm = y_hm.to(DEVICE)

        optimizer.zero_grad()

        logits_14, loc_maps = model(imgs)
        logits_8 = logits_14[:, MAP_8_TO_14]

        loss_cls = bce_cls(logits_8, y_cls)
        loss_loc = mse_loc(loc_maps, y_hm)
        loss = loss_cls + LAMBDA_LOC * loss_loc

        loss.backward()
        optimizer.step()

        total += loss.item() * imgs.size(0)
        loop.set_postfix(total=loss.item(), cls=loss_cls.item(), loc=loss_loc.item())

    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    """
    Validation loss (same objective as training).
    """
    model.eval()
    total = 0.0

    for batch in loader:
        if batch is None:
            continue

        imgs, y_cls, y_hm, _ = batch
        imgs = imgs.to(DEVICE)
        y_cls = y_cls.to(DEVICE)
        y_hm = y_hm.to(DEVICE)

        logits_14, loc_maps = model(imgs)
        logits_8 = logits_14[:, MAP_8_TO_14]

        loss_cls = bce_cls(logits_8, y_cls)
        loss_loc = mse_loc(loc_maps, y_hm)
        loss = loss_cls + LAMBDA_LOC * loss_loc

        total += loss.item() * imgs.size(0)

    return total / len(loader.dataset)


EPOCHS = 20
best_val = float("inf")
CKPT_LOC = "swinv2_loc_best.pth"

for ep in range(EPOCHS):
    print(f"\n===== Epoch {ep + 1}/{EPOCHS} =====")

    tr_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    print(f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), CKPT_LOC)
        print("✅ Saved best model.")
    else:
        print("⏸ No improvement.")

print("🏁 Training done. Best Val Loss =", best_val)


# ===============================================================
# Reload Trained Localization Model for Evaluation
# ===============================================================
model = SwinV2WithLocalization(backbone_name).to(DEVICE)

# Reload MIMIC backbone + classifier
mimic_model = timm.create_model(
    backbone_name,
    pretrained=False,
    num_classes=14,
    global_pool="avg",
).to(DEVICE)

mimic_ckpt = torch.load("swinv2_large_14class_weightedbce.pth", map_location=DEVICE)
mimic_model.load_state_dict(mimic_ckpt)

back_state = {k: v for k, v in mimic_model.state_dict().items() if "head" not in k}
model.backbone.load_state_dict(back_state, strict=False)

model.classifier_head.weight.data.copy_(mimic_model.head.fc.weight.data)
model.classifier_head.bias.data.copy_(mimic_model.head.fc.bias.data)

# Load localization checkpoint
loc_ckpt = torch.load(CKPT_LOC, map_location=DEVICE)
model.load_state_dict(loc_ckpt, strict=False)
model.eval()

print("✅ Localization model successfully loaded and ready for evaluation.")


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
    From a CAM heatmap, extract the largest connected-component box,
    then compute IoU vs all GT boxes; return best IoU.
    """
    H, W = pred_cam.shape
    if len(gt_boxes) == 0:
        return np.nan

    cam_norm = (pred_cam - pred_cam.min()) / (pred_cam.max() - pred_cam.min() + 1e-6)
    mask = (cam_norm > thresh).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return 0.0  # no predicted component

    # Largest component
    idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, area = stats[idx]
    pred_box = [x, y, x + w, y + h]

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
    Compute localization metrics for the 8 supervised classes:
      - IoU between predicted CAM and GT CAM masks
      - Dice overlap
      - Pearson correlation (per heatmap)
      - Best Box IoU (largest CAM component vs GT boxes)
    """
    model.eval()

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

        logits_14, pred_maps_t = model(imgs)
        pred_maps = pred_maps_t.cpu().numpy()  # [B,8,256,256]
        y_hm_np = y_hm.numpy()

        for b in range(pred_maps.shape[0]):
            meta = metas[b]
            for cid, cls_name in enumerate(LOC_CLASSES):
                gt_hm = y_hm_np[b, cid]
                pred = pred_maps[b, cid]

                # Skip if class not present in GT
                if gt_hm.max() == 0:
                    continue

                # Normalize CAM
                pred_n = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)

                # IoU / Dice using thresholded masks
                pred_mask = pred_n > 0.4
                gt_mask = gt_hm > 0.4

                iou = compute_iou(pred_mask, gt_mask)
                dice = compute_dice(pred_mask, gt_mask)

                # Pearson correlation
                corr = np.corrcoef(pred_n.flatten(), gt_hm.flatten())[0, 1]
                if np.isnan(corr):
                    corr = 0.0

                # GT boxes for this class
                gt_boxes = [
                    b for b, s in zip(meta["boxes_scaled"], meta["mapped_syms"]) if s == cls_name
                ]
                box_iou = compute_best_box_iou(pred, gt_boxes, thresh=0.4)

                metrics[cls_name]["count"] += 1
                metrics[cls_name]["iou"].append(iou)
                metrics[cls_name]["dice"].append(dice)
                metrics[cls_name]["corr"].append(corr)
                metrics[cls_name]["box_iou"].append(box_iou)

    print("================ Localization Metrics ================")
    for cls in LOC_CLASSES:
        m = metrics[cls]
        if m["count"] == 0:
            print(f"\n📌 {cls}\n   No positive GT samples in test set.\n--------------------------------------------------")
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
# Run localization evaluation on the test set
# ---------------------------------------------------------------
metrics = evaluate_localization_metrics(model, test_loader)

"""
Description
-----------
This script trains and evaluates a multi-scale SwinV2-Large localization model
with spatial attention and a compact multi-scale decoder on the ChestX-Det
dataset, supervised on 8 localization classes.

It:
- Loads ChestX-Det JSON annotations and PNG images.
- Maps ChestX-Det labels into 8 localization classes aligned with your MIMIC
  14-label ordering.
- Builds per-image:
    * 8-class multi-hot classification labels.
    * 8-channel Gaussian heatmaps derived from bounding boxes.
- Uses a SwinV2-Large backbone (pretrained on MIMIC 14-class classification),
  adds a spatial attention block on the 8×8 feature map and a multi-scale
  decoder that:
    * Upsamples 8×8 → 16×16 with an auxiliary 16×16 supervision head.
    * Fuses the 16×16 feature with the Swin 16×16 feature map.
    * Upsamples 16×16 → 256×256 to produce the final 8-channel heatmaps.

- Trains with a joint objective:
    * Weighted BCE on 8-class diagnosis logits (subset of 14-class Swin output).
    * Localization loss with deep supervision:
        - Main 256×256 scale:
            BCE + Dice loss.
        - Aux 16×16 scale:
            BCE + Dice loss (deep supervision).

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
- Multi-scale attention-based localization checkpoint:
    swinv2_loc_multiscale_attention_deepSupervision16_locLoss_best.pth

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
- pos_weight for the 8 classes is computed directly from the ChestX-Det JSON.
- The decoder uses a single auxiliary 16×16 head in addition to the main
  256×256 heatmaps.
"""

# ===============================================================
# Imports & Global Settings
# ===============================================================
import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt  # kept for potential visualization

# Paths (ChestX-Det & MIMIC Checkpoint)
CHESTX_TRAIN_JSON = "ChestX-Det Dataset/ChestX_Det_train.json"
CHESTX_TEST_JSON = "ChestX-Det Dataset/ChestX_Det_test.json"

TRAIN_IMAGE_ROOT = "ChestX-Det Dataset/train_data/train"
TEST_IMAGE_ROOT = "ChestX-Det Dataset/test_data/test"

# MIMIC-trained SwinV2 checkpoint (14-class classifier)
MIMIC_CKPT = "swinv2_large_14class_weightedbce.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

print("✅ Device:", DEVICE)


# ===============================================================
# Class Mapping to 8 Localization Classes
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

# MIMIC ordering (14 labels) used in Swin training
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

# Optional mapping for 8 CheXlocalize-style classes → CheXpert 14 indices
MAP_LOC_TO_14 = [
    10,  # Enlarged Cardiomediastinum  → idx 10
    1,   # Cardiomegaly                → idx 1
    8,   # Lung Lesion                 → idx 8
    9,   # Airspace Opacity → Lung Opacity (idx 9)
    3,   # Edema                       → idx 3
    2,   # Consolidation               → idx 2
    0,   # Atelectasis                 → idx 0
    6,   # Pneumothorax                → idx 6
]


# ===============================================================
# ChestXDetDataset (PNG + JSON boxes → cls + heatmap)
# ===============================================================
class ChestXDetDataset(Dataset):
    """
    Loads ChestX-Det samples from JSON + PNG folders.

    Returns per sample
    ------------------
    img_t : torch.FloatTensor [3, H, W]
        Normalized image tensor.
    y_cls : torch.FloatTensor [8]
        Multi-hot labels for LOC_CLASSES.
    y_hm  : torch.FloatTensor [8, H, W]
        Gaussian heatmaps per class (continuous supervision).
    meta  : dict
        Contains "file_name", "boxes_scaled", "mapped_syms".
    """

    def __init__(self, json_path, image_root, transform=None, img_size=256):
        super().__init__()
        self.image_root = image_root
        self.transform = transform
        self.img_size = img_size

        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _gaussian_heatmap(H, W, x1, y1, x2, y2, sigma_scale=0.25):
        """
        Builds soft Gaussian blob for a bounding box.
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

        if self.transform is not None:
            img_t = self.transform(img)
        else:
            img_t = T.ToTensor()(img)

        H = W = self.img_size

        # 8-class multi-hot labels
        y_cls = torch.zeros(len(LOC_CLASSES), dtype=torch.float32)

        # 8-class Gaussian heatmaps
        y_hm = np.zeros((len(LOC_CLASSES), H, W), dtype=np.float32)

        # scale boxes from original size to IMG_SIZE
        orig_w, orig_h = img.size
        sx = W / orig_w
        sy = H / orig_h

        boxes_scaled = []
        mapped_syms = []

        for sym, box in zip(syms, boxes):
            if sym not in CHESTX_TO_LOC:
                continue

            loc_name = CHESTX_TO_LOC[sym]
            loc_idx = LOC_TO_IDX[loc_name]

            y_cls[loc_idx] = 1.0

            x1, y1, x2, y2 = box
            x1s, x2s = int(x1 * sx), int(x2 * sx)
            y1s, y2s = int(y1 * sy), int(y2 * sy)

            boxes_scaled.append([x1s, y1s, x2s, y2s])
            mapped_syms.append(loc_name)

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
# safe_collate — skips None samples
# ===============================================================
def safe_collate(batch):
    imgs, ycls, yhms, metas = [], [], [], []

    for item in batch:
        if item is None:
            continue

        img_t, y_cls, y_hm, meta = item

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
    Computes pos_weight for LOC_CLASSES directly from JSON metadata.
    No image loading is required.

    pos_weight[c] = neg_count[c] / pos_count[c]
    (clipped to [0.5, 3.0] then normalized).
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
    Weighted BCE for 8-class multi-label classification using pos_weight.
    """

    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.float())

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


# ===============================================================
# Localization Loss (BCE + Dice, Deep Supervision @16×16)
# ===============================================================
def dice_loss_with_logits(logits, targets, eps=1e-6):
    """
    Dice loss for multi-label heatmaps.

    logits  : [B,8,H,W]
    targets : [B,8,H,W]
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), probs.size(1), -1)
    targets = targets.view(targets.size(0), targets.size(1), -1)

    inter = (probs * targets).sum(-1)
    union = probs.sum(-1) + targets.sum(-1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def localization_loss_deep_supervision(loc_256, aux16, gt_256):
    """
    Localization loss with deep supervision at 16×16.

    Inputs
    ------
    loc_256 : [B,8,256,256]
        Main 256×256 logits.
    aux16   : [B,8,16,16]
        Auxiliary 16×16 logits.
    gt_256  : [B,8,256,256]
        Target heatmaps.
    """
    # main scale
    loss256 = F.binary_cross_entropy_with_logits(loc_256, gt_256)
    loss256 += dice_loss_with_logits(loc_256, gt_256)

    # aux scale
    gt16 = F.interpolate(gt_256, size=(16, 16), mode="bilinear", align_corners=False)
    loss16 = F.binary_cross_entropy_with_logits(aux16, gt16)
    loss16 += dice_loss_with_logits(aux16, gt16)

    return loss256 + loss16


# ===============================================================
# SwinV2 Backbone + Spatial Attention + Multi-Scale Decoder
# ===============================================================
class SpatialAttention(nn.Module):
    """
    Spatial attention gate.

    Input
    -----
    x : [B, C, H, W]

    Output
    ------
    Spatially reweighted feature map with same shape.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.conv(x))  # [B,1,H,W]
        return x * att


class MultiScaleSegDecoder(nn.Module):
    """
    Multi-scale decoder for SwinV2.

    Expected feature sizes for 256×256 input:
      feat8  : [B,1536, 8, 8]   (stage-3)
      feat16 : [B, 768,16,16]   (stage-2)

    Decoder:
      8×8 → 16×16 (aux16 head).
      Fuse 16×16 maps from upsampled feat8 and feat16.
      16×16 → 256×256 final 8-channel heatmaps.

    Outputs
    -------
    seg_256 : [B,8,256,256]
    aux16   : [B,8,16,16]
    """

    def __init__(self, ch8=1536, ch16=768, out_ch=8):
        super().__init__()

        # 8×8 → 16×16
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch8, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.aux16 = nn.Conv2d(512, out_ch, 1)

        # Fuse with feat16
        self.fuse16 = nn.Sequential(
            nn.Conv2d(512 + ch16, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 16×16 → 256×256
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, 1),
        )

    def forward(self, feat8, feat16):
        up16 = self.up1(feat8)
        aux16 = self.aux16(up16)

        fused = torch.cat([up16, feat16], dim=1)
        fused = self.fuse16(fused)

        seg_256 = self.final_up(fused)
        return seg_256, aux16


class SwinV2_AttentionSeg(nn.Module):
    """
    SwinV2 backbone (MIMIC-trained) + spatial attention + multi-scale decoder.

    Forward
    -------
    x : torch.FloatTensor [B,3,256,256]

    Returns
    -------
    logits_14 : [B,14]
        Global 14-class logits (MIMIC ordering).
    seg_256   : [B,8,256,256]
        Main localization maps for LOC_CLASSES.
    aux16     : [B,8,16,16]
        Deep-supervision maps at 16×16.
    """

    def __init__(self, backbone_name="swinv2_large_window12to16_192to256",
                 num_cls=14, num_loc=8):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(2, 3),  # 16×16, 8×8
        )

        self.att8 = SpatialAttention(1536)
        self.decoder = MultiScaleSegDecoder(1536, 768, num_loc)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(1536)
        self.classifier = nn.Linear(1536, num_cls)

    def _to_nchw(self, feat):
        """
        Convert NHWC → NCHW if needed.
        """
        if feat.ndim == 4 and feat.shape[-1] in (384, 768, 1536):
            if feat.shape[1] not in (384, 768, 1536):
                feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x):
        feats = self.backbone(x)
        feat16 = self._to_nchw(feats[0])  # [B,768,16,16]
        feat8 = self._to_nchw(feats[1])   # [B,1536,8,8]

        feat8_att = self.att8(feat8)
        seg_256, aux16 = self.decoder(feat8_att, feat16)

        pooled = self.pool(feat8_att).flatten(1)
        pooled = self.norm(pooled)
        logits_14 = self.classifier(pooled)

        return logits_14, seg_256, aux16


# ===============================================================
# Init Model + Load MIMIC Weights + Freeze Layers
# ===============================================================
backbone_name = "swinv2_large_window12to16_192to256"
model = SwinV2_AttentionSeg(backbone_name).to(DEVICE)

mimic_state = torch.load(MIMIC_CKPT, map_location=DEVICE)

backbone_state = {}
for k, v in mimic_state.items():
    if k.startswith(("patch_embed", "layers", "norm")):
        backbone_state[k] = v

model.backbone.load_state_dict(backbone_state, strict=False)

# classifier head
if "head.fc.weight" in mimic_state:
    model.classifier.weight.data.copy_(mimic_state["head.fc.weight"])
    model.classifier.bias.data.copy_(mimic_state["head.fc.bias"])
elif "head.weight" in mimic_state:
    model.classifier.weight.data.copy_(mimic_state["head.weight"])
    model.classifier.bias.data.copy_(mimic_state["head.bias"])

# freeze all but last Swin stage
for name, p in model.backbone.named_parameters():
    if not name.startswith("layers.3"):
        p.requires_grad = False
    else:
        p.requires_grad = True

print("\n🔍 Trainable backbone layers:")
for name, p in model.backbone.named_parameters():
    if p.requires_grad:
        print("  →", name)

print("✅ MIMIC weights loaded and correct layers unfrozen.")


# ===============================================================
# Loss Weights, Optimizer
# ===============================================================
LAMBDA_LOC = 3.0  # weight for localization loss

criterion_cls = WeightedBCELoss(pos_weight_8).to(DEVICE)

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"🔧 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=1e-4,
    weight_decay=1e-4,
)

scheduler = None


# ===============================================================
# Train One Epoch
# ===============================================================
def train_one_epoch(model, loader):
    """
    One training epoch.

    Objective
    ---------
    total_loss = cls_loss + LAMBDA_LOC * loc_loss

    Where:
      cls_loss → Weighted BCE over 8 mapped labels.
      loc_loss → BCE + Dice @256
                 + deep supervision at 16×16.
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

        logits_14, seg_256, aux16 = model(imgs)

        logits_8 = logits_14[:, MAP_8_TO_14]
        loss_cls = criterion_cls(logits_8, y_cls)

        loss_loc = localization_loss_deep_supervision(seg_256, aux16, y_hm)

        loss = loss_cls + LAMBDA_LOC * loss_loc
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        loop.set_postfix(
            {
                "total": loss.item(),
                "cls": loss_cls.item(),
                "loc": loss_loc.item(),
            }
        )

    return total_loss / len(loader.dataset)


# ===============================================================
# Validation / Evaluation
# ===============================================================
@torch.no_grad()
def evaluate(model, loader):
    """
    Validation loss (same objective, no gradients).

    total_loss = cls_loss + LAMBDA_LOC * loc_loss
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

        logits_14, seg_256, aux16 = model(imgs)

        logits_8 = logits_14[:, MAP_8_TO_14]
        loss_cls = criterion_cls(logits_8, y_cls)

        loss_loc = localization_loss_deep_supervision(seg_256, aux16, y_hm)

        total_loss += (loss_cls + LAMBDA_LOC * loss_loc).item() * imgs.size(0)

    return total_loss / len(loader.dataset)


# ===============================================================
# Training Loop
# ===============================================================
EPOCHS = 20
best_val = float("inf")
CKPT_PATH = "swinv2_loc_multiscale_attention_deepSupervision16_locLoss_best.pth"

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

print(f"✅ Loaded localization model from: {CKPT_PATH}")


# ===============================================================
# Metric Helpers (IoU, Dice, Box IoU)
# ===============================================================
def compute_mask_iou(pred, gt):
    """
    Computes IoU between two binary masks.
    pred, gt: tensors [H,W] with {0,1}
    """
    inter = (pred * gt).sum().item()
    union = (pred + gt - pred * gt).sum().item()
    if union == 0:
        return 0.0
    return inter / union


def compute_mask_dice(pred, gt):
    """
    Computes Dice coefficient between two binary masks.
    pred, gt: tensors [H,W] with {0,1}
    """
    inter = (pred * gt).sum().item()
    total = pred.sum().item() + gt.sum().item()
    if total == 0:
        return 0.0
    return (2 * inter) / total


def cam_to_box(cam_np, thresh=0.3):
    """
    Converts CAM (normalized 2D numpy array) into a predicted bounding box.
    Returns (x1,y1,x2,y2) or None.
    """
    mask = cam_np > thresh
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)


def box_iou(boxA, boxB):
    """
    IoU between two bounding boxes.
    Each box = (x1,y1,x2,y2). Returns 0 if box doesn't exist.
    """
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    if union <= 0:
        return 0.0

    return inter / union


# ===============================================================
# Localization Metrics on Test Set
# ===============================================================
metrics = {
    cls: {"samples": 0, "iou": [], "dice": [], "corr": [], "box_iou": []}
    for cls in LOC_CLASSES
}

print("\nEvaluating localization metrics...")

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Eval", leave=False):
        if batch is None:
            continue

        imgs, y_cls, y_hm, meta = batch

        imgs = imgs.to(DEVICE)
        y_hm = y_hm.to(DEVICE)

        logits_14, seg_256, aux16 = model(imgs)

        pred_maps = seg_256.sigmoid().cpu()  # [B,8,256,256]
        y_hm = y_hm.cpu()

        for b in range(imgs.size(0)):
            gt_hm = y_hm[b]    # [8,H,W]
            cam = pred_maps[b]  # [8,H,W]
            info = meta[b]

            for ci, cls in enumerate(LOC_CLASSES):
                if gt_hm[ci].max() == 0:
                    continue

                metrics[cls]["samples"] += 1

                gt_mask = gt_hm[ci]
                pred_mask = cam[ci]

                pm = (pred_mask - pred_mask.min()) / (
                    pred_mask.max() - pred_mask.min() + 1e-6
                )

                pred_bin = (pm > 0.3).float()
                gt_bin = (gt_mask > 0.3).float()

                metrics[cls]["iou"].append(compute_mask_iou(pred_bin, gt_bin))
                metrics[cls]["dice"].append(compute_mask_dice(pred_bin, gt_bin))

                pred_vec = pm.flatten().numpy()
                gt_vec = gt_mask.flatten().numpy()
                corr = np.corrcoef(pred_vec, gt_vec)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                metrics[cls]["corr"].append(float(corr))

                pred_box = cam_to_box(pm.numpy(), thresh=0.3)

                if len(info["boxes_scaled"]) > 0:
                    gx1, gy1, gx2, gy2 = info["boxes_scaled"][0]
                    gt_box = (int(gx1), int(gy1), int(gx2), int(gy2))
                else:
                    gt_box = None

                metrics[cls]["box_iou"].append(box_iou(pred_box, gt_box))

print("\n================ Localization Metrics ================\n")
for cls in LOC_CLASSES:
    s = metrics[cls]["samples"]
    if s == 0:
        continue
    print(f"📌 {cls}")
    print(f"   Samples        : {s}")
    print(f"   IoU (mask)     : {np.mean(metrics[cls]['iou']):.4f}")
    print(f"   Dice           : {np.mean(metrics[cls]['dice']):.4f}")
    print(f"   Corr           : {np.mean(metrics[cls]['corr']):.4f}")
    print(f"   Box IoU        : {np.mean(metrics[cls]['box_iou']):.4f}")
    print("--------------------------------------------------")

"""
Description
-----------
This script trains and evaluates a multi-scale SwinV2-Large localization model
with spatial attention and a progressive UNet-lite decoder on the ChestX-Det
dataset, supervised on 8 localization classes.

It:
- Loads ChestX-Det JSON annotations and PNG images.
- Maps ChestX-Det labels into 8 localization classes aligned with your MIMIC
  14-label ordering.
- Builds per-image:
    * 8-class multi-hot classification labels.
    * 8-channel Gaussian heatmaps derived from bounding boxes.
- Uses a SwinV2-Large backbone (pretrained on MIMIC 14-class classification),
  adds a spatial attention block and a progressive multi-scale UNet-lite
  decoder to predict 8 localization maps at 256×256, with deep supervision
  at 32×32 and 16×16.
- Trains with a joint objective:
    * Weighted BCE on 8-class diagnosis logits (from 14-class Swin output).
    * Localization loss with deep supervision:
        - Main 256×256 scale:
            Focal BCE + Soft Dice + Lovasz hinge (IoU surrogate).
        - Aux 32×32 and 16×16 scales:
            Focal BCE + Soft Dice (weighted in total loss).

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
    swinv2_loc_multiscale_unetlite_attention_deepDS_locLoss_best.pth

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
- SwinV2 features from 32×32, 16×16, and 8×8 stages are fused by a progressive
  UNet-lite decoder for sharper 256×256 heatmaps.
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
import matplotlib.pyplot as plt  

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
# Localization Loss (Focal BCE + Soft Dice + Lovasz + Deep Supervision)
# ===============================================================
class FocalBCELoss(nn.Module):
    """
    Focal BCE for heatmap regression.

    Helps focus learning on hard pixels (small lesions)
    and suppresses easy background dominance.
    """

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def soft_dice_loss(logits, targets, eps=1e-6):
    """
    Soft Dice loss for multi-label heatmaps.

    logits  : [B,8,H,W]
    targets : [B,8,H,W]
    """
    probs = torch.sigmoid(logits)
    dims = (2, 3)

    inter = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)

    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    gt_sorted: [P] sorted ground-truth labels (0/1).
    """
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss for a single class (flattened).

    logits: [P]
    labels: [P] {0,1}
    """
    if labels.numel() == 0:
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    labels_sorted = labels[perm]
    grad = lovasz_grad(labels_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels):
    """
    Multi-label Lovasz hinge over channels.
    logits: [B,C,H,W]
    labels: [B,C,H,W]
    """
    C = logits.size(1)
    losses = []
    for c in range(C):
        logit_c = logits[:, c].contiguous().view(-1)
        label_c = labels[:, c].contiguous().view(-1)
        losses.append(lovasz_hinge_flat(logit_c, label_c))
    return torch.stack(losses).mean()


def localization_loss_deep_supervision(seg256, aux16, aux32, gt256):
    """
    Localization loss with deep supervision.

    seg256 : [B,8,256,256] logits
    aux32  : [B,8,32,32]   logits
    aux16  : [B,8,16,16]   logits
    gt256  : [B,8,256,256] target heatmaps
    """
    focal = FocalBCELoss(alpha=1.0, gamma=2.0)

    # Main 256×256 scale
    loss256 = focal(seg256, gt256)
    loss256 += soft_dice_loss(seg256, gt256)
    loss256 += lovasz_hinge(seg256, (gt256 > 0.3).float())

    # Aux 32×32
    gt32 = F.interpolate(gt256, size=(32, 32), mode="bilinear", align_corners=False)
    loss32 = focal(aux32, gt32)
    loss32 += soft_dice_loss(aux32, gt32)

    # Aux 16×16
    gt16 = F.interpolate(gt256, size=(16, 16), mode="bilinear", align_corners=False)
    loss16 = focal(aux16, gt16)
    loss16 += soft_dice_loss(aux16, gt16)

    # Weighted deep supervision
    return loss256 + 0.5 * loss32 + 0.25 * loss16


# ===============================================================
# SwinV2 Backbone + Spatial Attention + Progressive Sharp Decoder
# ===============================================================
class SpatialAttention(nn.Module):
    """Spatial attention to sharpen class-relevant regions."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.conv(x))
        return x * att


class ProgressiveSharpDecoder(nn.Module):
    """
    Progressive UNet-style decoder for sharper heatmaps.

    Inputs
    ------
    f32 : [B,384,32,32]     (stage-1)
    f16 : [B,768,16,16]     (stage-2)
    f8  : [B,1536, 8, 8]    (stage-3)

    Returns
    -------
    seg256 : [B,8,256,256]
    aux16  : [B,8,16,16]
    aux32  : [B,8,32,32]
    """

    def __init__(self, out_ch=8):
        super().__init__()

        # align channels
        self.p8 = nn.Conv2d(1536, 512, 1)
        self.p16 = nn.Conv2d(768, 256, 1)
        self.p32 = nn.Conv2d(384, 128, 1)

        # 8 → 16
        self.up8_16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(True),
        )
        self.aux16 = nn.Conv2d(256, out_ch, 1)

        # 16 → 32
        self.up16_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(True),
        )
        self.aux32 = nn.Conv2d(128, out_ch, 1)

        # 32 → 64 → 128 → 256
        self.up32_64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 96, 3, padding=1),
            nn.ReLU(True),
        )
        self.up64_128 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(True),
        )
        self.up128_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, out_ch, 3, padding=1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, f32, f16, f8):
        f8 = self.p8(f8)    # [B,512,8,8]
        f16 = self.p16(f16)  # [B,256,16,16]
        f32 = self.p32(f32)  # [B,128,32,32]

        # 8 → 16 + skip
        x16 = self.up8_16(f8)
        x16 = x16 + f16
        aux16 = self.aux16(x16)

        # 16 → 32 + skip
        x32 = self.up16_32(x16)
        x32 = x32 + f32
        aux32 = self.aux32(x32)

        # 32 → 256
        x64 = self.up32_64(x32)
        x128 = self.up64_128(x64)
        seg256 = self.up128_256(x128)
        seg256 = self.refine(seg256)

        return seg256, aux16, aux32


class SwinV2_AttentionSeg(nn.Module):
    """
    SwinV2 backbone (MIMIC-trained) + spatial attention + progressive sharp decoder.

    Forward
    -------
    x : torch.FloatTensor [B,3,256,256]

    Returns
    -------
    logits_14 : [B,14]
        Global 14-class logits (MIMIC ordering).
    seg256    : [B,8,256,256]
        Main localization maps for LOC_CLASSES.
    aux16     : [B,8,16,16]
        Deep-supervision maps @16×16.
    aux32     : [B,8,32,32]
        Deep-supervision maps @32×32.
    """

    def __init__(self, backbone_name="swinv2_large_window12to16_192to256"):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3],  # 32×32, 16×16, 8×8
        )

        self.att = SpatialAttention(1536)
        self.decoder = ProgressiveSharpDecoder(out_ch=len(LOC_CLASSES))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(1536)
        self.classifier = nn.Linear(1536, 14)

    def forward(self, x):
        feats = self.backbone(x)

        def to_nchw(f):
            if f.ndim == 4 and f.shape[-1] in (384, 768, 1536):
                return f.permute(0, 3, 1, 2).contiguous()
            return f

        f32 = to_nchw(feats[0])  # [B,384,32,32]
        f16 = to_nchw(feats[1])  # [B,768,16,16]
        f8 = to_nchw(feats[2])   # [B,1536,8,8]

        f8_att = self.att(f8)

        seg256, aux16, aux32 = self.decoder(f32, f16, f8_att)

        pooled = self.pool(f8_att).flatten(1)
        pooled = self.norm(pooled)
        logits_14 = self.classifier(pooled)

        return logits_14, seg256, aux16, aux32


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
LAMBDA_LOC = 3.0  # emphasis on localization

criterion_cls = WeightedBCELoss(pos_weight_8).to(DEVICE)

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"🔧 Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=1e-4,
    weight_decay=1e-4,
)

scheduler = None  # optional


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
      loc_loss → Focal BCE + Soft Dice + Lovasz @256
                 + weighted aux losses @32,16.
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

        logits_14, seg256, aux16, aux32 = model(imgs)

        logits_8 = logits_14[:, MAP_8_TO_14]
        loss_cls = criterion_cls(logits_8, y_cls)

        loss_loc = localization_loss_deep_supervision(seg256, aux16, aux32, y_hm)

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

    total = cls_loss + LAMBDA_LOC * loc_loss
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

        logits_14, seg256, aux16, aux32 = model(imgs)

        logits_8 = logits_14[:, MAP_8_TO_14]
        loss_cls = criterion_cls(logits_8, y_cls)

        loss_loc = localization_loss_deep_supervision(seg256, aux16, aux32, y_hm)

        total_loss += (loss_cls + LAMBDA_LOC * loss_loc).item() * imgs.size(0)

    return total_loss / len(loader.dataset)


# ===============================================================
# Training Loop (saves swinv2_loc_multiscale_unetlite_attention_deepDS_locLoss_best.pth)
# ===============================================================
EPOCHS = 20
best_val = float("inf")
CKPT_PATH = "swinv2_loc_multiscale_unetlite_attention_deepDS_locLoss_best.pth"

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

        logits_14, seg256, aux16, aux32 = model(imgs)

        pred_maps = seg256.sigmoid().cpu()  # [B,8,256,256]
        y_hm = y_hm.cpu()

        for b in range(imgs.size(0)):
            gt_hm = y_hm[b]       # [8,H,W]
            cam = pred_maps[b]    # [8,H,W]
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
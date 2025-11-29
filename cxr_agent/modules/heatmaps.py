"""Heatmap / localization model and utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms as T
import timm

from config import (
    DEVICE,
    HEATMAP_MODEL_CKPT,
    HEATMAP_LOC_CLASSES,
    HEATMAP_LABEL_TO_IDX,
    DEFAULT_HEATMAP_ALPHA,
    DEFAULT_HEATMAP_GAMMA,
    DISPLAY_DIR,
    LABEL_COLS,
)


# -------------------------------------------------------------------------
# Spatial attention gate
# -------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    """Spatial attention gate that reweights features with a learned mask."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.sigmoid(self.conv(x))  # [B,1,H,W]
        return x * att


class MultiScaleSegDecoder(nn.Module):
    """Multi-scale decoder for SwinV2 feature maps.

    Expects:
      - feat8  : [B,1536, 8, 8]   (stage-3)
      - feat16 : [B, 768,16,16]   (stage-2)

    Produces:
      - seg_256 : [B,num_loc,256,256]
      - aux16   : [B,num_loc,16,16]
    """

    def __init__(self, ch8: int = 1536, ch16: int = 768, out_ch: int = 8) -> None:
        super().__init__()

        # 8×8 → 16×16
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch8, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.aux16 = nn.Conv2d(512, out_ch, 1)

        # Fuse 16×16 Swin feature + upsampled 8×8 feature
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

    def forward(
        self,
        feat8: torch.Tensor,
        feat16: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        up16 = self.up1(feat8)         # [B,512,16,16]
        aux16 = self.aux16(up16)       # [B,out_ch,16,16]

        fused = torch.cat([up16, feat16], dim=1)  # [B,512+768,16,16]
        fused = self.fuse16(fused)     # [B,256,16,16]

        seg_256 = self.final_up(fused)  # [B,out_ch,256,256]
        return seg_256, aux16


class SwinV2_AttentionSeg(nn.Module):
    """SwinV2 backbone + spatial attention + multi-scale decoder."""

    def __init__(
        self,
        backbone_name: str = "swinv2_large_window12to16_192to256",
        num_cls: int = len(LABEL_COLS),
        num_loc: int = len(HEATMAP_LOC_CLASSES),
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(2, 3),  # stage-2 (16×16), stage-3 (8×8)
        )

        self.att8 = SpatialAttention(1536)
        self.decoder = MultiScaleSegDecoder(1536, 768, num_loc)

        # Classification head (not used directly in this module,
        # but kept for architectural completeness).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(1536)
        self.classifier = nn.Linear(1536, num_cls)

    def _to_nchw(self, feat: torch.Tensor) -> torch.Tensor:
        """Convert NHWC → NCHW if timm returns channels-last."""
        if feat.ndim == 4 and feat.shape[-1] in (384, 768, 1536):
            if feat.shape[1] not in (384, 768, 1536):
                feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        feat16 = self._to_nchw(feats[0])  # [B, 768,16,16]
        feat8 = self._to_nchw(feats[1])   # [B,1536, 8, 8]

        feat8_att = self.att8(feat8)
        seg_256, aux16 = self.decoder(feat8_att, feat16)

        pooled = self.pool(feat8_att).flatten(1)
        pooled = self.norm(pooled)
        logits_14 = self.classifier(pooled)

        return logits_14, seg_256, aux16


heatmap_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


_HEATMAP_MODEL: Optional[SwinV2_AttentionSeg] = None


def load_heatmap_model() -> SwinV2_AttentionSeg:
    """Load the SwinV2 attention segmentation model."""
    global _HEATMAP_MODEL
    if _HEATMAP_MODEL is None:
        model = SwinV2_AttentionSeg().to(DEVICE)
        state = torch.load(HEATMAP_MODEL_CKPT, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Heatmap missing keys:", missing)
        print("Heatmap unexpected keys:", unexpected)
        model.eval()
        _HEATMAP_MODEL = model
        print("✅ Loaded heatmap model from:", HEATMAP_MODEL_CKPT)
    return _HEATMAP_MODEL


def create_heatmap_overlay_image(
    base_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = DEFAULT_HEATMAP_ALPHA,
    gamma: float = DEFAULT_HEATMAP_GAMMA,
) -> Image.Image:
    """Create a JET-colored heatmap overlay on top of base_img."""
    h, w = heatmap.shape
    img = base_img.resize((w, h)).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0

    hm = heatmap.astype(np.float32)
    hm -= hm.min()
    if hm.max() > 0:
        hm /= hm.max()
    hm = np.clip(hm ** gamma, 0.0, 1.0)

    cmap = plt.get_cmap("jet")
    hm_color = cmap(hm)[..., :3]  # RGB in [0,1]

    blended = (1.0 - alpha) * img_np + alpha * hm_color
    blended = np.clip(blended, 0.0, 1.0)
    blended = (blended * 255).astype(np.uint8)

    return Image.fromarray(blended)


def select_top3_localization_labels(
    probs: Dict[str, float],
    threshold: float = 0.5,
):
    """Select up to 3 localisation labels (among 8) with prob >= threshold."""
    candidates = [
        (label, p)
        for label, p in probs.items()
        if label in HEATMAP_LABEL_TO_IDX and p >= threshold
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:3]


def generate_heatmaps_for_top3(
    image_path: str,
    image_loader,
    probs: Optional[Dict[str, float]] = None,
    classifier_fn=None,
    alpha: float = DEFAULT_HEATMAP_ALPHA,
    gamma: float = DEFAULT_HEATMAP_GAMMA,
):
    """Generate heatmaps for up to 3 highest-probability localisation labels.

    Parameters
    ----------
    image_path : str
        Path to the original uploaded image (DICOM or PNG/JPEG).
    image_loader : Callable
        Function like :func:`modules.qc.load_cxr_from_path`.
    probs : Optional[Dict[str, float]], optional
        Existing classifier probabilities; if ``None``, classifier_fn
        will be called to compute them, by default None.
    classifier_fn : Optional[Callable], optional
        Function for classification; if provided and probs is None,
        it is called as `classifier_fn(image)`, by default None.
    """
    heatmap_model = load_heatmap_model()

    image, _ = image_loader(image_path)
    image_rgb = image.convert("RGB")

    if probs is None and classifier_fn is not None:
        probs = classifier_fn(image)
    probs = probs or {}

    model_input = heatmap_transform(image_rgb).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        _, seg_256, _ = heatmap_model(model_input)
        seg_maps = seg_256[0].detach().cpu().numpy()  # [8,H,W]

    selected = select_top3_localization_labels(probs)
    if not selected:
        return [], (
            "No localization labels with probability > 0.5 among classifier predictions. "
        ), probs

    DISPLAY_DIR.mkdir(parents=True, exist_ok=True)

    gallery_items = []
    for label, p in selected:
        cidx = HEATMAP_LABEL_TO_IDX[label]
        hm = seg_maps[cidx]

        overlay_img = create_heatmap_overlay_image(
            image_rgb, hm, alpha=alpha, gamma=gamma
        )

        out_path = DISPLAY_DIR / f"heatmap_{Path(image_path).stem}_{label.replace(' ', '_')}.png"
        overlay_img.save(out_path)
        caption = f"{label} (p={p:.2f})"
        gallery_items.append((str(out_path), caption))

    status = "Generated heatmaps for: " + ", ".join(
        [f"{lbl} (p={p:.2f})" for lbl, p in selected]
    )
    return gallery_items, status, probs

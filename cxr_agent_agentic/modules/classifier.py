"""SwinV2-based 14-label chest X-ray classifier."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm

from config import DEVICE, CLASSIFIER_WEIGHTS, LABEL_COLS


class CXRClassifier(nn.Module):
    """Wrapper around a SwinV2-Large backbone for 14-label classification."""

    def __init__(self, num_labels: int = len(LABEL_COLS)) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
            pretrained=False,
            num_classes=num_labels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.backbone(x)


# Preprocessing transform for the classifier
classification_transform = T.Compose([
    T.Resize((256, 256)),
    T.Grayscale(num_output_channels=3),  # replicate to 3 channels
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_classifier_model() -> CXRClassifier:
    """Instantiate and load the SwinV2 classifier weights from disk."""
    from collections import OrderedDict

    model = CXRClassifier(num_labels=len(LABEL_COLS))
    state = torch.load(CLASSIFIER_WEIGHTS, map_location="cpu")

    # If keys do not have "backbone." prefix, add it.
    if any(k.startswith("patch_embed.") or k.startswith("layers.") for k in state.keys()):
        new_state = OrderedDict()
        for k, v in state.items():
            if not k.startswith("backbone."):
                new_state["backbone." + k] = v
            else:
                new_state[k] = v
        state = new_state

    incompat = model.load_state_dict(state, strict=False)
    print("Classifier missing keys:", incompat.missing_keys)
    print("Classifier unexpected keys:", incompat.unexpected_keys)

    model.to(DEVICE)
    model.eval()
    return model


def run_classifier(
    model: CXRClassifier,
    image: Image.Image,
) -> Dict[str, float]:
    """Run the SwinV2 classifier and return label -> probability mapping."""
    model_input = classification_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(model_input)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {label: float(p) for label, p in zip(LABEL_COLS, probs)}


def summarize_probs(probs: Dict[str, float], top_k: int = 5) -> str:
    """Return a short human-readable summary of top-k probabilities."""
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return ", ".join([f"{label}: {p*100:.1f}%" for label, p in items])

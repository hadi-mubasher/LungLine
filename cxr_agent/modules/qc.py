"""Image loading and quality control (QC) utilities for chest X-rays.

This module provides:

- :func:`load_cxr_from_path` to load DICOM / PNG / JPEG images.
- :func:`predict_cxr_prob` using a CLIP vision-language model to
  estimate P(image is a chest X-ray).
- :func:`run_quality_check` to run a set of heuristic QC rules.
- :func:`format_qc_message` to create a human-readable QC report.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import pydicom

import torch
from transformers import CLIPProcessor, CLIPModel

from config import DEVICE, DISPLAY_DIR


def load_cxr_from_path(path: str) -> Tuple[Image.Image, Optional[pydicom.Dataset]]:
    """Load a chest X-ray from disk, handling both DICOM and image files."""
    path = str(path)
    ext = Path(path).suffix.lower()
    dicom_ds: Optional[pydicom.Dataset] = None

    if ext in [".dcm", ".dicom"]:
        dicom_ds = pydicom.dcmread(path)
        arr = dicom_ds.pixel_array.astype(np.float32)

        # Simple rescaling / normalisation to [0, 255]
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        image = Image.fromarray(arr).convert("L")  # grayscale
    else:
        image = Image.open(path).convert("L")

    return image, dicom_ds


def save_display_image(image: Image.Image, original_path: str) -> str:
    """Save a PNG version of the image for UI display in DISPLAY_DIR."""
    orig_name = Path(original_path).stem
    out_path = DISPLAY_DIR / f"{orig_name}_display.png"
    image.save(out_path)
    return str(out_path)


@dataclass
class QCResult:
    """Container for quality-check results on a single CXR."""

    is_cxr: bool
    orientation_ok: bool
    inspiration_ok: bool
    artifacts_ok: bool
    severity: str          # "pass", "warning", "fail"
    reasons: List[str]

    def to_dict(self) -> Dict:
        """Return the QC result as a plain dictionary."""
        return asdict(self)


_CLIP_MODEL: Optional[CLIPModel] = None
_CLIP_PROCESSOR: Optional[CLIPProcessor] = None


def _load_cxr_clip_model() -> Tuple[CLIPModel, CLIPProcessor]:
    """Load and cache the CLIP model/processor used for CXR vs non-CXR."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        model_name = "openai/clip-vit-base-patch32"
        _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(DEVICE)
        _CLIP_MODEL.eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def predict_cxr_prob(image: Image.Image) -> float:
    """Estimate P(image is a chest X-ray) using CLIP."""
    model, processor = _load_cxr_clip_model()

    texts = [
        "a chest X-ray image",
        "an image that is not a chest X-ray",
    ]

    inputs = processor(
        text=texts,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # [1, 2]
        probs = logits_per_image.softmax(dim=-1)[0].detach().cpu().numpy()

    p_cxr = float(probs[0])  # index 0 corresponds to "a chest X-ray image"
    return p_cxr


def run_quality_check(
    image: Image.Image,
    dicom_ds: Optional[pydicom.Dataset] = None,
    cxr_prob_threshold: float = 0.5,
) -> QCResult:
    """Run CLIP-based and heuristic QC checks on a CXR."""
    reasons: List[str] = []

    # 1) CLIP-based CXR vs non-CXR
    cxr_prob = predict_cxr_prob(image)
    is_cxr_model = cxr_prob >= cxr_prob_threshold
    if not is_cxr_model:
        reasons.append(
            "Image classifier thinks this is *not* a chest X-ray "
            f"(P(CXR)={cxr_prob:.2f} < threshold {cxr_prob_threshold:.2f})."
        )

    # 2) DICOM modality / body-part checks (if available)
    is_cxr_meta = True
    if dicom_ds is not None:
        modality = getattr(dicom_ds, "Modality", "").upper()
        body_part = str(getattr(dicom_ds, "BodyPartExamined", "")).upper()

        if modality not in ("CR", "DX"):
            is_cxr_meta = False
            reasons.append(
                f"Modality={modality!r}, expected CR/DX for radiographic chest X-ray."
            )
        if "CHEST" not in body_part and "THORAX" not in body_part:
            reasons.append(
                f"BodyPartExamined={body_part!r} (cannot confidently confirm chest)."
            )
    else:
        reasons.append(
            "No DICOM header – cannot confirm modality or body part; "
            "relying on image-based classifier."
        )

    is_cxr = is_cxr_model and is_cxr_meta

    # 3) Orientation / aspect ratio
    w, h = image.size
    aspect = h / max(w, 1)
    orientation_ok = True
    if aspect < 0.7 or aspect > 1.8:
        orientation_ok = False
        reasons.append(
            f"Unusual aspect ratio {aspect:.2f} – possible rotation, "
            "cropping, or wrong projection."
        )

    # 4) Artifacts: large near-uniform regions
    arr = np.array(image, dtype=np.uint8)
    black_ratio = (arr < 5).mean()
    white_ratio = (arr > 250).mean()
    artifacts_ok = True
    if black_ratio > 0.5 or white_ratio > 0.5:
        artifacts_ok = False
        reasons.append(
            "Large uniform regions (very dark or very bright) – potential collimation, "
            "truncation, or exposure artifact."
        )

    # 5) Crude exposure / inspiration adequacy
    mean_intensity = arr.mean()
    inspiration_ok = True
    if mean_intensity < 40:
        inspiration_ok = False
        reasons.append(
            f"Overall image is very dark (mean intensity {mean_intensity:.1f}) – "
            "possible poor inspiration or underexposure."
        )

    severity = "pass"
    if not is_cxr:
        severity = "fail"
    elif not orientation_ok or not artifacts_ok:
        severity = "fail"
    elif not inspiration_ok or reasons:
        severity = "warning"

    return QCResult(
        is_cxr=is_cxr,
        orientation_ok=orientation_ok,
        inspiration_ok=inspiration_ok,
        artifacts_ok=artifacts_ok,
        severity=severity,
        reasons=reasons,
    )


def format_qc_message(qc: QCResult) -> str:
    """Format a QCResult into a markdown-friendly message."""
    msg = f"QC result: **{qc.severity.upper()}**\n\n"
    msg += f"- Is CXR modality: `{qc.is_cxr}`\n"
    msg += f"- Orientation OK: `{qc.orientation_ok}`\n"
    msg += f"- Inspiration / technical adequacy OK: `{qc.inspiration_ok}`\n"
    msg += f"- Artifacts OK: `{qc.artifacts_ok}`\n"
    if qc.reasons:
        msg += "\n**Notes:**\n"
        for r in qc.reasons:
            msg += f"- {r}\n"
    if qc.severity == "fail":
        msg += (
            "\n➡️ I will *not* provide a confident automated interpretation on this image. "
            "Please upload a better-quality chest X-ray or confirm that this is indeed the correct study.\n"
        )
    elif qc.severity == "warning":
        msg += (
            "\n⚠️ Some quality concerns were detected; please interpret AI outputs with extra caution.\n"
        )
    return msg

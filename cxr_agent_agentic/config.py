"""Global configuration for the CXR-Agent project.

All directories, model paths, label sets, and global constants are defined
here so they can be easily modified when running on Google Colab or any
other environment.
"""

import os
from pathlib import Path
from typing import List

import torch

# ------------------------------------------------------------
# Device configuration
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Base directories (Colab-friendly)
# ------------------------------------------------------------
# You can change BASE_DRIVE_DIR to point to your own Google Drive folder
# or any other persistent storage location.
BASE_DRIVE_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Agentic Systems Term Project")

# Temporary directory for generated PNGs / previews
TEMP_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Agentic Systems Term Project/temp")
TEMP_DIR.mkdir(exist_ok=True)
DISPLAY_DIR = TEMP_DIR / "display"
DISPLAY_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Model paths (update these to match your Drive layout)
# ------------------------------------------------------------
CLASSIFIER_WEIGHTS = str(
    BASE_DRIVE_DIR / "models" / "swinv2_large_14class_weightedbce.pth"
)

MEDGEMMA_BASE_ID = "google/medgemma-4b-it"
MEDGEMMA_PEFT_DIR = str(BASE_DRIVE_DIR / "models" / "medgemma")

HEATMAP_MODEL_CKPT = str(
    BASE_DRIVE_DIR / "models" / "swinv2_multiscale_deep_dice_best.pth"
)

# ------------------------------------------------------------
# Database path (SQLite by default, lives on Drive)
# ------------------------------------------------------------
DEFAULT_DB_URL = f"sqlite:///{BASE_DRIVE_DIR / 'cxr_patient_registry.db'}"
DATABASE_URL = os.environ.get("PATIENT_DB_URL", DEFAULT_DB_URL)

# ------------------------------------------------------------
# Label sets
# ------------------------------------------------------------
LABEL_COLS: List[str] = [
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

# 8 localization labels (order must match heatmap training)
HEATMAP_LOC_CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural Effusion",
    "Fracture",
    "Pneumothorax",
    "Lung Lesion",
    "Pleural Other",
]

HEATMAP_LABEL_TO_IDX = {name: i for i, name in enumerate(HEATMAP_LOC_CLASSES)}

# ------------------------------------------------------------
# OpenAI / GPT-4o-mini
# ------------------------------------------------------------
# For safety, we NEVER hard-code the API key in the source file.
# Set it in the environment before running:
#   os.environ["OPENAI_API_KEY"] = "sk-..."
OPENAI_MODEL_NAME = os.environ.get("CXR_AGENT_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# ------------------------------------------------------------
# Misc constants (heatmap visualisation)
# ------------------------------------------------------------
DEFAULT_HEATMAP_ALPHA = 0.45
DEFAULT_HEATMAP_GAMMA = 10.0
